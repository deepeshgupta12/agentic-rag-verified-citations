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
# Short lines worth keeping: addresses, versions, quantities, ranges, dates.
_DATA_BEARING = re.compile(r"\d")


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


def fetch_page(
    url: str,
    settings: Settings,
    max_chars: int | None = None,
    breaker: CircuitBreaker | None = None,
    deadline: float | None = None,
    report: list[dict] | None = None,
) -> str:
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
    host = urlparse(url).hostname or url

    # A host that has already failed this run is not retried. Fetching is the
    # last external call that was completely unmetered: with N results per
    # round and a 12s timeout each, a set of dead hosts could consume the
    # whole run budget inside page retrieval alone.
    if breaker is not None and breaker.is_open(host):
        log.info("skipping fetch %s: circuit open", host)
        return ""

    try:
        for _hop in range(MAX_REDIRECTS):
            # A per-request timeout does not bound total time; the run
            # deadline does.
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.info("skipping fetch %s: run deadline passed", current)
                    return ""
            _assert_safe_url(current)
            response = requests.get(
                current,
                headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=min(settings.request_timeout_s, remaining)
                if deadline is not None
                else settings.request_timeout_s,
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
            if breaker is not None:
                breaker.record_success(host)

            extracted = _extract_text(body)
            limit = settings.fetch_max_chars if max_chars is None else max_chars

            # Truncation used to be silent. A long specification cut here loses
            # whole sections while the remaining text still reads perfectly, so
            # a question about a missing section gets a truthful "the sources
            # do not state this" about a document that does. Nothing downstream
            # can detect that, because by then the text is all there is.
            if len(extracted) > limit:
                log.warning(
                    "truncated %s: kept %d of %d chars (%.0f%%)",
                    current, limit, len(extracted), 100 * limit / len(extracted),
                )
                if report is not None:
                    report.append({
                        "url": current,
                        "kept": limit,
                        "total": len(extracted),
                        "truncated": True,
                    })
                # Cut on a paragraph boundary where one is close, so the tail
                # is not a fragment of a sentence.
                cut = extracted.rfind("\n\n", int(limit * 0.9), limit)
                return extracted[: cut if cut > 0 else limit]

            if report is not None:
                report.append({"url": current, "kept": len(extracted),
                               "total": len(extracted), "truncated": False})
            return extracted

        log.info("too many redirects for %s", url)
        return ""
    except UnsafeURL as exc:
        # Not a transport failure: do not count a blocked address against the
        # host's circuit, or one bad link would disable a working domain.
        log.warning("blocked unsafe fetch %s: %s", url, exc)
        return ""
    except Exception as exc:  # noqa: BLE001 - a dead link is not a run failure
        log.info("fetch failed for %s: %s", url, exc)
        if breaker is not None:
            breaker.record_failure(host)
        return ""


# Rendered mathematics carries the constants that matter -- "k1 is usually
# chosen in [1.2, 2.0]" is entirely inside the formula markup. Stripping tags
# blindly deletes them and leaves a sentence that trails off exactly where its
# value should be, so the page looks readable and has silently lost the fact.
# Found by asking for BM25's default parameters and getting a correct
# "the excerpts do not state this" from a page that visibly does.
_MATH_ANNOTATION = re.compile(
    r"<annotation[^>]*>(.*?)</annotation>", re.DOTALL | re.I
)
_MATH_ALT = re.compile(r'<(?:img|math)[^>]*\salt(?:text)?="([^"]{1,300})"', re.I)
_MATH_BLOCK = re.compile(r"<math[^>]*>.*?</math>", re.DOTALL | re.I)


def _recover_math(markup: str) -> str:
    """Replace math elements with their LaTeX/alt text before tags are stripped.

    MathML carries a TeX ``<annotation>`` and Wikipedia's fallback images carry
    ``alt``. Either is a usable textual rendering; without one the formula
    becomes whitespace.
    """
    def replace(match: re.Match) -> str:
        block = match.group(0)
        annotation = _MATH_ANNOTATION.search(block)
        if annotation:
            return f" {html.unescape(annotation.group(1))} "
        alt = _MATH_ALT.search(block)
        return f" {html.unescape(alt.group(1))} " if alt else " "

    return _MATH_BLOCK.sub(replace, markup)


def _extract_text(markup: str) -> str:
    markup = _recover_math(markup)
    body = _SCRIPT_STYLE.sub(" ", markup)
    # Preserve block boundaries as paragraph breaks so the chunker still has
    # structure to split on after tags are gone.
    # Cells are joined within a row before rows are split. Without a cell
    # separator the columns run together; without keeping the row intact the
    # address loses its description and vice versa.
    body = re.sub(r"</(td|th)>", " | ", body, flags=re.I)
    body = re.sub(r"</(p|div|section|article|li|h[1-6]|tr)>", "\n\n", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    text = html.unescape(_TAGS.sub(" ", body))

    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line).strip()
        if _BOILERPLATE.match(line):
            continue
        # Length alone is the wrong test for whether a line is content. Table
        # cells carrying the actual data are short -- "10.0.0.0/8" is ten
        # characters -- so a length filter keeps the prose column and silently
        # discards the column with the facts in it. The page still reads
        # coherently and has lost exactly what was being asked for. Short
        # lines are kept when they carry data.
        if len(line) > 40 or (len(line) > 3 and _DATA_BEARING.search(line)):
            lines.append(line)
    return "\n\n".join(lines)


def fetch_many(
    results: Sequence[WebResult],
    settings: Settings,
    limit: int = 5,
    max_workers: int = 5,
    breaker: CircuitBreaker | None = None,
    deadline: float | None = None,
    report: list[dict] | None = None,
) -> dict[str, str]:
    """Fetch the top ``limit`` results concurrently.

    Sequential fetching of five pages at a 12s timeout is a 60s worst case on
    its own; in parallel it is bounded by the slowest single page.
    """
    targets = [r for r in results[:limit] if r.url.startswith(("http://", "https://"))]
    if not targets:
        return {}

    pages: dict[str, str] = {}
    try:
        _fetch_into(pages, targets, settings, max_workers, breaker, deadline, report)
    except concurrent.futures.TimeoutError:
        # Whatever arrived before the deadline is still usable evidence.
        log.info("fetch_many hit the deadline with %d page(s) retrieved", len(pages))
    return pages


def _fetch_into(pages, targets, settings, max_workers, breaker, deadline, report=None) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_page, r.url, settings, None, breaker, deadline, report): r.url
            for r in targets
        }
        overall = settings.request_timeout_s * 3
        if deadline is not None:
            overall = min(overall, max(0.1, deadline - time.monotonic()))
        for future in concurrent.futures.as_completed(futures, timeout=overall):
            url = futures[future]
            try:
                text = future.result()
            except Exception:  # noqa: BLE001
                text = ""
            if text:
                pages[url] = text


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or url
    except Exception:  # noqa: BLE001
        return url
