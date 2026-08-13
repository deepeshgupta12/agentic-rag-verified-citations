"""Persistent embedding cache.

Embeddings are rebuilt from scratch on every process start. That is invisible
on a three-paragraph fixture and painful on anything real: a 500-page report
is thousands of chunks, and every restart re-embeds all of them, paying the
full latency and the full API bill to compute a value that has not changed.

The computation is deterministic in exactly the way a cache wants -- the same
text and the same model always produce the same vector -- so the content hash
is a complete key. Nothing about the corpus, the question or the run needs to
be part of it, which means a chunk shared between two documents is embedded
once, and re-uploading a file that differs by one page re-embeds one page.

Three properties this has to get right, because a corrupted or stale cache is
worse than no cache:

* **Model-keyed.** Vectors from different embedding models are not
  interchangeable, and mixing them silently produces a retrieval index whose
  distances mean nothing. The model name is stored and a mismatch invalidates
  the file rather than merging.
* **Atomic writes.** A process killed mid-save must not leave a half-written
  file that loads as garbage. Written to a temp file and renamed, which is
  atomic on POSIX.
* **Fail-open.** A cache is an optimisation. Any error reading or writing it
  degrades to computing embeddings normally, never to failing the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("ragverify.store")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragverify"
CACHE_VERSION = 1


def text_key(text: str) -> str:
    """Content hash identifying a chunk's embedding, independent of position."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class EmbeddingCache:
    """Disk-backed map from content hash to embedding vector.

    Loaded once on construction and written once on ``save``; there is no
    per-lookup I/O, because the whole point is to avoid work rather than to
    trade API latency for disk latency.
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path | str | None = None,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        # One file per model: mixing models in one file would make an
        # invalidation on model change throw away vectors that are still good.
        safe_model = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
        self.path = self.dir / f"emb-v{CACHE_VERSION}-{safe_model}.npz"

        self._vectors: dict[str, list[float]] = {}
        self._dirty = False
        self.hits = 0
        self.misses = 0

        if self.enabled:
            self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            import numpy as np

            with np.load(self.path, allow_pickle=False) as data:
                meta = json.loads(str(data["__meta__"].item()))
                if meta.get("model") != self.model or meta.get("version") != CACHE_VERSION:
                    log.info("embedding cache is for a different model/version; ignoring")
                    return
                keys = [str(k) for k in data["__keys__"]]
                matrix = data["__vectors__"]
                if len(keys) != len(matrix):
                    log.warning("embedding cache is inconsistent; ignoring")
                    return
                self._vectors = {key: matrix[i].tolist() for i, key in enumerate(keys)}
            log.info("loaded %d cached embeddings from %s", len(self._vectors), self.path)
        except Exception as exc:  # noqa: BLE001 - a cache must never fail a run
            log.warning("could not read embedding cache (%s); starting empty", exc)
            self._vectors = {}

    def save(self) -> bool:
        """Persist if anything changed. Returns whether a write happened."""
        if not self.enabled or not self._dirty or not self._vectors:
            return False
        try:
            import numpy as np

            self.dir.mkdir(parents=True, exist_ok=True)
            keys = list(self._vectors)
            matrix = np.asarray([self._vectors[k] for k in keys], dtype="float32")
            meta = json.dumps({"model": self.model, "version": CACHE_VERSION, "count": len(keys)})

            # Temp file in the same directory so the rename stays on one
            # filesystem and is therefore atomic.
            handle, tmp = tempfile.mkstemp(dir=self.dir, suffix=".npz.tmp")
            os.close(handle)
            np.savez_compressed(
                tmp, __keys__=np.asarray(keys), __vectors__=matrix, __meta__=np.asarray(meta)
            )
            # savez appends .npz when the name lacks it.
            written = Path(tmp if tmp.endswith(".npz") else tmp + ".npz")
            os.replace(written, self.path)
            self._dirty = False
            log.info("saved %d embeddings to %s", len(keys), self.path)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write embedding cache (%s)", exc)
            return False

    # -- lookup --------------------------------------------------------

    def split(self, texts: Sequence[str]) -> tuple[dict[int, list[float]], list[int]]:
        """Partition into ``(cached_by_position, positions_needing_embedding)``.

        Positions rather than texts, so the caller can reassemble the result
        in the original order -- retrieval indexes vectors positionally
        against its chunk list, and a reordering would silently attach every
        vector to the wrong chunk.
        """
        if not self.enabled:
            return {}, list(range(len(texts)))

        cached: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, text in enumerate(texts):
            vector = self._vectors.get(text_key(text))
            if vector is None:
                missing.append(index)
            else:
                cached[index] = vector

        self.hits += len(cached)
        self.misses += len(missing)
        return cached, missing

    def put(self, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if not self.enabled:
            return
        for text, vector in zip(texts, vectors, strict=False):
            self._vectors[text_key(text)] = list(vector)
            self._dirty = True

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0

    def clear(self) -> None:
        self._vectors.clear()
        self._dirty = True


def embed_cached(
    texts: Sequence[str],
    client,
    cache: EmbeddingCache | None,
) -> list[list[float]]:
    """Embed ``texts``, computing only what the cache lacks.

    Order is preserved by position, and only the missing texts are sent to the
    provider -- the saving is the whole point, so a partial hit must still
    avoid re-embedding the part that hit.
    """
    if cache is None or not cache.enabled:
        return client.embed(list(texts))

    cached, missing = cache.split(texts)
    if not missing:
        return [cached[i] for i in range(len(texts))]

    fresh = client.embed([texts[i] for i in missing])
    cache.put([texts[i] for i in missing], fresh)

    out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    for index, vector in cached.items():
        out[index] = vector
    for slot, index in enumerate(missing):
        out[index] = fresh[slot]
    return out
