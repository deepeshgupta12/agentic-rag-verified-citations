"""Hybrid retrieval: BM25 + optional dense embeddings, fused by RRF.

The obvious lexical scorer is ``len(query_tokens & chunk_tokens)`` -- the
number of distinct query words appearing anywhere in a chunk. It has three
defects that compound:

* **No IDF.** "what", "the" and "is" score exactly as much as the one rare
  term that actually identifies the answer, so a chunk matching only stopwords
  ties with a chunk matching the subject of the question.
* **No length normalisation.** Scoring on a *set* means a longer chunk holds
  more distinct tokens and therefore wins by size alone. With 800-word chunks
  the longest chunk in a document is close to unbeatable.
* **No term frequency.** A chunk mentioning the subject fifteen times scores
  identically to one mentioning it once in passing.

BM25 fixes all three. Dense retrieval is added on top because lexical matching
cannot answer a question phrased in different words than the document uses,
and the two are combined with Reciprocal Rank Fusion, which needs no score
calibration between the retrievers.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from .schemas import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Kept as prose rather than a quoted list: this is edited by hand when tuning
# recall, and a 100-element list literal is unreviewable in a diff.
_STOPWORD_TEXT = """
a an and are as at be been but by can could did do does for from had has have
he her his how i if in into is it its me my no nor not of on or our out over
she should so than that the their them then there these they this those to
too under until up was we were what when where which while who whom why will
with would you your about above after again against all also am any because
before being below between both during each few further here more most other
same some such only own s t just don now
"""

STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stem(token: str) -> str:
    """Very small suffix stripper.

    Deliberately not a full Porter stemmer: the goal is only to make
    "requirement"/"requirements" and "routing"/"route" collide, which is where
    most recall is lost on short document sets. Aggressive stemming hurts
    precision more than it helps here.
    """
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def analyze(text: str, *, drop_stopwords: bool = True) -> list[str]:
    out = []
    for tok in tokenize(text):
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


class BM25Index:
    """Okapi BM25 over the chunk corpus.

    Built once per corpus and reused across every round of the adaptive loop
    and every follow-up question, rather than re-scanning the whole chunk list
    per query with no precomputation.
    """

    __slots__ = ("chunks", "k1", "b", "_df", "_tf", "_len", "_avglen", "_idf")

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()

        for chunk in self.chunks:
            terms = analyze(chunk.text)
            tf = Counter(terms)
            self._tf.append(tf)
            self._len.append(len(terms))
            self._df.update(tf.keys())

        n = len(self.chunks)
        self._avglen = (sum(self._len) / n) if n else 0.0
        # Lucene's non-negative IDF variant: the raw Robertson formula goes
        # negative for terms in >half the corpus, which on a small single-topic
        # document set would actively penalise the topic word itself.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = analyze(query)
        if not terms or not self.chunks:
            return []

        # Candidate generation: only chunks sharing at least one query term can
        # score above zero, so the posting-list intersection bounds the work.
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1.0 - self.b + self.b * (self._len[i] / self._avglen or 1.0)
                scores[i] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


class DenseIndex:
    """Embedding retrieval over the same chunks.

    Optional by design: it needs network and spend, so every caller must cope
    with it being absent. When embedding fails the hybrid retriever falls back
    to BM25 alone and records a warning instead of failing the run.
    """

    def __init__(self, chunks: Sequence[Chunk], vectors) -> None:
        import numpy as np

        self.chunks = list(chunks)
        mat = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        # Pre-normalise so querying is a single matrix-vector product.
        self.matrix = mat / np.clip(norms, 1e-9, None)

    def search(self, query_vector, top_k: int) -> list[tuple[int, float]]:
        import numpy as np

        q = np.asarray(query_vector, dtype="float32")
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.matrix @ q
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        return sorted(((int(i), float(sims[i])) for i in idx), key=lambda kv: -kv[1])


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, float]]],
    k: int = 60,
) -> dict[int, tuple[float, list[str]]]:
    """Fuse ranked lists by ``sum(1 / (k + rank))``.

    RRF is used rather than score averaging because BM25 scores and cosine
    similarities live on incomparable scales; normalising them against each
    other requires calibration data this app does not have. RRF only reads
    rank order, so it needs none.
    """
    fused: dict[int, float] = defaultdict(float)
    provenance: dict[int, list[str]] = defaultdict(list)
    for name, ranked in rankings.items():
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
            provenance[doc_id].append(name)
    return {doc_id: (score, provenance[doc_id]) for doc_id, score in fused.items()}


def mmr_rerank(
    candidates: list[ScoredChunk],
    top_k: int,
    lambda_: float = 0.72,
) -> list[ScoredChunk]:
    """Maximal Marginal Relevance over token-set Jaccard similarity.

    Overlapping chunks mean the top hits are frequently near-duplicates of one
    another, which wastes the evidence budget on the same sentence three times
    and makes the verifier see false corroboration. MMR trades a little
    relevance for coverage of distinct passages.
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    sets = [set(analyze(c.chunk.text)) for c in candidates]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_i, best_score = remaining[0], float("-inf")
        for i in remaining:
            redundancy = max(
                (_jaccard(sets[i], sets[j]) for j in selected),
                default=0.0,
            )
            score = lambda_ * candidates[i].score - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class HybridRetriever:
    """BM25 + optional dense retrieval, fused and diversified."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        dense: DenseIndex | None = None,
        embed_query=None,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25Index(self.chunks)
        self.dense = dense
        self._embed_query = embed_query
        self.warnings: list[str] = []

    @property
    def has_dense(self) -> bool:
        return self.dense is not None and self._embed_query is not None

    def search(
        self,
        query: str,
        top_k: int = 6,
        *,
        expansions: Iterable[str] = (),
        diversify: bool = True,
    ) -> list[ScoredChunk]:
        """Retrieve for ``query``, optionally widened by ``expansions``.

        ``expansions`` are the verifier's gap statements on later rounds: each
        becomes its own ranked list in the fusion, so a gap the original
        phrasing missed can pull in chunks the first round never saw. This is
        how a "widen" round differs from simply re-running the same query.
        """
        if not self.chunks:
            return []

        # Overfetch before fusion and MMR so both have room to reorder.
        pool = max(top_k * 4, 20)
        rankings: dict[str, list[tuple[int, float]]] = {}

        queries = [query, *[e for e in expansions if e and e.strip()]]
        for i, q in enumerate(queries):
            hits = self.bm25.search(q, pool)
            if hits:
                rankings[f"bm25:{i}"] = hits

        if self.has_dense:
            try:
                for i, q in enumerate(queries):
                    vec = self._embed_query(q)
                    hits = self.dense.search(vec, pool)
                    if hits:
                        rankings[f"dense:{i}"] = hits
            except Exception as exc:  # noqa: BLE001 - degrade to lexical only
                self.warnings.append(f"Dense retrieval unavailable, using BM25 only ({exc}).")

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        scored = [
            ScoredChunk(chunk=self.chunks[i], score=score, retrievers=sorted(set(src)))
            for i, (score, src) in fused.items()
        ]
        scored.sort(key=lambda s: -s.score)

        head = scored[: max(top_k * 3, top_k)]
        return mmr_rerank(head, top_k) if diversify else head[:top_k]
