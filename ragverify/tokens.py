"""Token counting with a graceful fallback.

``tiktoken`` is optional. When it is missing (or its encoding download fails
in an offline environment) everything degrades to a character-ratio estimate
rather than crashing -- chunk sizing and budget packing only need to be
approximately right.
"""

from __future__ import annotations

import functools

# Empirically close to cl100k/o200k for English prose.
_CHARS_PER_TOKEN = 4.0


@functools.lru_cache(maxsize=4)
def _encoder(model: str = "o200k_base"):  # pragma: no cover - env dependent
    try:
        import tiktoken

        return tiktoken.get_encoding(model)
    except Exception:  # noqa: BLE001 - any failure means "estimate instead"
        return None


def count_tokens(text: str, model: str = "o200k_base") -> int:
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))
    return len(enc.encode(text, disallowed_special=()))


def split_tokens(text: str, size: int, overlap: int = 0) -> list[str]:
    """Hard-split ``text`` into windows of ``size`` tokens with ``overlap``.

    Only used for a single paragraph too large to fit a chunk; ordinary
    splitting happens on paragraph boundaries in ``ingest.chunk_page``.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    stride = size - overlap

    enc = _encoder()
    if enc is None:
        # Approximate the same windowing over characters.
        csize, cstride = int(size * _CHARS_PER_TOKEN), int(stride * _CHARS_PER_TOKEN)
        return [text[i : i + csize] for i in range(0, len(text), cstride) if text[i : i + csize].strip()]

    ids = enc.encode(text, disallowed_special=())
    out: list[str] = []
    for start in range(0, len(ids), stride):
        window = ids[start : start + size]
        if not window:
            break
        out.append(enc.decode(window).strip())
        if start + size >= len(ids):
            break
    return [piece for piece in out if piece]


def truncate_to_tokens(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` tokens, on a token boundary."""
    if limit <= 0:
        return ""
    enc = _encoder()
    if enc is None:
        return text[: int(limit * _CHARS_PER_TOKEN)]
    ids = enc.encode(text, disallowed_special=())
    return text if len(ids) <= limit else enc.decode(ids[:limit])


def fits(text: str, budget: int, used: int = 0) -> bool:
    return used + count_tokens(text) <= budget


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: tuple[float, float] | None,
) -> float:
    """Cost in USD from per-million-token pricing; 0.0 when unpriced."""
    if not pricing:
        return 0.0
    p_in, p_out = pricing
    return (prompt_tokens / 1e6) * p_in + (completion_tokens / 1e6) * p_out
