"""Web search with timeouts, retries, endpoint failover and page fetching.

Querying a single hardcoded public instance with no timeout, retry or error
handling is fragile: public SearxNG instances are routinely rate-limited, and
a 429 with no fallback kills the run.

This module also fetches the result pages. Search-result *snippets* are
two-line engine summaries; reasoning over them means "web research" is really
reasoning over SERP previews, and leaves grounding nothing to check a citation
against. Fetching and extracting the top results makes web evidence real source
text, symmetric with local evidence.
"""

from __future__ import annotations

import concurrent.futures
import html
import logging
import re
import time
from collections.abc import Sequence
from urllib.parse import urljoin, urlparse

from .budget import CircuitBreaker
from .config import Settings
from .schemas import WebResult

log = logging.getLogger("ragverify.web")

_UA = "Mozilla/5.0 (compatible; RagVerify/1.0)"

# Bounds on a single fetch. A retrieved page is untrusted input, so its size
# and redirect depth are ours to cap, not the server's to choose.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>", re.DOTALL | re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Nav/footer boilerplate that dominates extracted text and pollutes BM25.
_BOILERPLATE = re.compile(
    r"^(cookie|accept all|subscribe|sign in|log in|menu|skip to|share this|advertisement)\b", re.I
)


class SearchUnavailable(RuntimeError):
    """Every configured backend failed. The caller degrades, not crashes."""


def _requests():
    import requests

    return requests


def search(query: str, settings: Settings, breaker: CircuitBreaker | None = None) -> list[WebResult]:
    """Query SearxNG instances in order, then fall back to DuckDuckGo.

    Returns an empty list rather than raising when everything fails, so a dead
    network downgrades the run to local-only instead of ending it. The caller
    records a warning and the verifier sees the reduced evidence.

    ``breaker`` short-circuits endpoints already known to be down *this run*.
    Without it, a run where every backend is unreachable pays the full ladder
    -- 4 backends x 3 retries x backoff -- on every sub-query, having already
    learned the answer on the first. An observed eval run spent roughly two
    minutes per web case doing exactly that.
    """
    for endpoint in settings.searx_endpoints:
        if breaker is not None and breaker.is_open(endpoint):
            log.info("skipping %s: circuit open", endpoint)
            continue
        try:
            results = _searxng(query, endpoint, settings)
            if results:
                log.info("searxng %s returned %d results", endpoint, len(results))
                if breaker is not None:
                    breaker.record_success(endpoint)
                return results
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            log.warning("searxng %s failed: %s", endpoint, exc)
            if breaker is not None:
                breaker.record_failure(endpoint)
            continue

    if breaker is not None and breaker.is_open("duckduckgo"):
        return []
    try:
        results = _duckduckgo(query, settings)
        if breaker is not None:
            breaker.record_success("duckduckgo")
        return results
    except Exception as exc:  # noqa: BLE001
        log.warning("duckduckgo fallback failed: %s", exc)
        if breaker is not None:
            breaker.record_failure("duckduckgo")
        return []


def _searxng(query: str, endpoint: str, settings: Settings) -> list[WebResult]:
    requests = _requests()
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}

    last: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=settings.request_timeout_s,
            )
            if response.status_code == 429:
                # Honour Retry-After when present; these instances mean it.
                wait = float(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 8.0))
                last = RuntimeError("rate limited")
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 * (2**attempt))
    else:
        raise SearchUnavailable(str(last))

    out: list[WebResult] = []
    for item in payload.get("results", [])[: settings.web_max_results]:
        url = item.get("url") or ""
        if not url:
            continue
        out.append(
            WebResult(
                url=url,
                title=(item.get("title") or "").strip(),
                snippet=(item.get("content") or "").strip(),
                engine=item.get("engine", "searxng"),
            )
        )
    return out


def _duckduckgo(query: str, settings: Settings) -> list[WebResult]:
    """HTML-endpoint fallback, used when every SearxNG instance is down."""
    requests = _requests()
    response = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _UA},
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()

    results: list[WebResult] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    for match in pattern.finditer(response.text):
        results.append(
            WebResult(
                url=html.unescape(match.group("url")),
                title=_strip_html(match.group("title")),
                snippet=_strip_html(match.group("snippet")),
                engine="duckduckgo",
            )
        )
        if len(results) >= settings.web_max_results:
            break
    return results


def _strip_html(fragment: str) -> str:
    return _WS.sub(" ", html.unescape(_TAGS.sub(" ", fragment))).strip()


class UnsafeURL(ValueError):
    """The URL resolves somewhere the fetcher must not go."""


def _assert_safe_url(url: str) -> None:
    """Reject URLs that would turn the fetcher into an SSRF primitive.

    Search results are attacker-influenceable: getting a page into a result
    set, or onto a page that redirects, is enough to choose the address this
    process connects to. Without validation that is a request forgery gadget
    aimed at whatever the host can reach -- cloud metadata endpoints
    (169.254.169.254), localhost admin panels, private RFC1918 services.

    Every resolved address is checked, not just the hostname, because DNS can
    return a private address for a public name. This narrows but does not
    close DNS rebinding, where the address changes between this check and the
    connection; defeating that needs connection-time pinning.
    """
    import ipaddress
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {parsed.scheme!r} not allowed")

    host = parsed.hostname
    if not host:
        raise UnsafeURL("no host")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve {host}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local      # 169.254.0.0/16 — cloud metadata
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeURL(f"{host} resolves to non-public address {address}")


def fetch_page(url: str, settings: Settings, max_chars: int = 20_000) -> str:
    """Fetch and extract readable text from ``url``. Empty string on failure.

    Deliberately dependency-free rather than pulling in trafilatura: the goal
    is enough clean text for BM25 and for grounding to check a claim against,
    not perfect article extraction.

    Redirects are followed manually so each hop is validated. Handing
    ``allow_redirects=True`` to requests means only the *first* URL is ever
    checked, and a public URL that 302s to ``127.0.0.1`` walks straight past
    the guard.
    """
    requests = _requests()
    current = url
    try:
        for _hop in range(MAX_REDIRECTS):
            _assert_safe_url(current)
            response = requests.get(
                current,
                headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=settings.request_timeout_s,
                allow_redirects=False,
                stream=True,
            )

            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    return ""
                current = urljoin(current, location)
                response.close()
                continue

            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", "").lower():
                return ""

            # Cap the body before it is in memory. Content-Length is a claim,
            # not a guarantee, so the stream is truncated as it arrives.
            declared = int(response.headers.get("Content-Length") or 0)
            if declared and declared > MAX_RESPONSE_BYTES:
                log.info("skipping %s: declared %d bytes", current, declared)
                return ""

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(8192):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    break
            body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
            return _extract_text(body)[:max_chars]

        log.info("too many redirects for %s", url)
        return ""
    except UnsafeURL as exc:
        log.warning("blocked unsafe fetch %s: %s", url, exc)
        return ""
    except Exception as exc:  # noqa: BLE001 - a dead link is not a run failure
        log.info("fetch failed for %s: %s", url, exc)
        return ""


def _extract_text(markup: str) -> str:
    body = _SCRIPT_STYLE.sub(" ", markup)
    # Preserve block boundaries as paragraph breaks so the chunker still has
    # structure to split on after tags are gone.
    body = re.sub(r"</(p|div|section|article|li|h[1-6]|tr)>", "\n\n", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    text = html.unescape(_TAGS.sub(" ", body))

    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line).strip()
        # Single words and nav labels are almost always chrome, not content.
        if len(line) > 40 and not _BOILERPLATE.match(line):
            lines.append(line)
    return "\n\n".join(lines)


def fetch_many(
    results: Sequence[WebResult],
    settings: Settings,
    limit: int = 5,
    max_workers: int = 5,
) -> dict[str, str]:
    """Fetch the top ``limit`` results concurrently.

    Sequential fetching of five pages at a 12s timeout is a 60s worst case on
    its own; in parallel it is bounded by the slowest single page.
    """
    targets = [r for r in results[:limit] if r.url.startswith(("http://", "https://"))]
    if not targets:
        return {}

    pages: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_page, r.url, settings): r.url for r in targets}
        for future in concurrent.futures.as_completed(futures, timeout=settings.request_timeout_s * 3):
            url = futures[future]
            try:
                text = future.result()
            except Exception:  # noqa: BLE001
                text = ""
            if text:
                pages[url] = text
    return pages


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or url
    except Exception:  # noqa: BLE001
        return url
