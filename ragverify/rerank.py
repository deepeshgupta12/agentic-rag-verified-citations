"""Reranking: judge passages against the question, not against each other.

BM25 and dense retrieval both score a passage *in isolation* -- how well its
terms or its vector match the query. Neither reads the passage as an answer to
the question, and that is where the last big slice of retrieval quality sits.
A passage can score highly for containing every query term while answering a
different question entirely, and a passage that answers the question perfectly
in different words can rank tenth.

Reranking closes that gap by scoring each candidate *for this question*, after
retrieval has narrowed the field. Two backends, same interface:

* **Cross-encoder** (``sentence-transformers``, optional). A model that reads
  query and passage together in one forward pass, which is exactly what
  bi-encoder embeddings cannot do -- they must encode the passage without
  knowing the query. Local, no per-call cost, but a real dependency and a
  model download.
* **LLM reranker**. Scores a batch of passages in one structured call. No new
  dependency and it reuses the configured client, at the cost of a call per
  round.

The order matters more than it looks. The evidence budget truncates: whatever
sits below the token limit never reaches the model at all. Getting the right
passage into position three instead of position eleven is often the difference
between a grounded answer and an abstention, and no amount of downstream
verification can recover a passage that was never shown.

Both backends degrade to the fused retrieval order on any failure. A reranker
that cannot run must not cost you the retrieval you already have.

**Contract:** a reranker must return passages with ``score`` set to its own
relevance judgement, not merely in a new order. MMR runs afterwards and reads
``score`` as its relevance term, so a reranker that reorders without rescoring
is silently undone by the diversity pass.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from .schemas import ScoredChunk

log = logging.getLogger("ragverify.rerank")

#: Passages scoring below this are dropped when ``drop_below`` is enabled.
#: Deliberately low: the reranker decides *order* well and *absolute quality*
#: only roughly, so it should discard only the clearly irrelevant.
DEFAULT_DROP_THRESHOLD = 0.15


class PassageScore(BaseModel):
    index: int
    relevance: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=200)


class RerankBatch(BaseModel):
    scores: list[PassageScore] = Field(default_factory=list)


LLM_SYSTEM = """\
You rank passages by how well each ANSWERS a question. You are not judging
writing quality, topic similarity, or whether the passage is interesting.

Score each passage 0.0 to 1.0:
  1.0  Contains the answer outright.
  0.7  Contains a substantial part of the answer.
  0.4  Related and useful context, but does not answer it.
  0.1  Same topic, does not help.
  0.0  Irrelevant.

A passage sharing many words with the question is not thereby relevant --
keyword overlap is what produced this candidate list, and your job is to
correct it. Conversely a passage that answers the question in entirely
different words scores high.

Return one score per passage, using the given index. Score every passage."""


def _passages_prompt(question: str, candidates: Sequence[ScoredChunk], max_chars: int) -> str:
    blocks = [
        f"### Passage {i}\n{c.chunk.text[:max_chars]}"
        for i, c in enumerate(candidates)
    ]
    return f"Question: {question}\n\n" + "\n\n".join(blocks)


def llm_rerank(
    question: str,
    candidates: Sequence[ScoredChunk],
    client,
    top_k: int,
    drop_below: float | None = None,
    max_chars: int = 1200,
) -> list[ScoredChunk]:
    """Rerank with a single structured call. Falls back to input order."""
    if not candidates:
        return []

    try:
        batch = client.structured(
            LLM_SYSTEM, _passages_prompt(question, candidates, max_chars), RerankBatch
        )
    except Exception as exc:  # noqa: BLE001 - never lose retrieval to a reranker
        log.warning("LLM rerank failed (%s); keeping fused order", exc)
        return list(candidates[:top_k])

    scored = {s.index: s.relevance for s in batch.scores if 0 <= s.index < len(candidates)}
    if not scored:
        log.warning("LLM rerank returned no usable scores; keeping fused order")
        return list(candidates[:top_k])

    ranked = sorted(
        enumerate(candidates),
        # Unscored passages keep their retrieval rank rather than being sent
        # to the bottom: a missing score is missing information, not evidence
        # of irrelevance.
        key=lambda pair: (scored.get(pair[0], 0.5), -pair[0]),
        reverse=True,
    )

    out: list[ScoredChunk] = []
    for index, candidate in ranked:
        relevance = scored.get(index, 0.5)
        if drop_below is not None and relevance < drop_below:
            continue
        out.append(
            candidate.model_copy(
                update={
                    "score": relevance,
                    "retrievers": [*candidate.retrievers, "llm-rerank"],
                }
            )
        )
        if len(out) >= top_k:
            break

    return out or list(candidates[:top_k])


class CrossEncoderReranker:
    """Local cross-encoder, loaded lazily and cached.

    Query and passage are encoded *together*, which is the whole point: a
    bi-encoder must embed the passage without knowing the query, so it cannot
    represent "relevant to this question" at all.
    """

    _model = None
    _model_name = ""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name

    @property
    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):
        # Cached on the class: the model is expensive to load and identical
        # across instances, and a per-question reload would dominate latency.
        if CrossEncoderReranker._model is None or CrossEncoderReranker._model_name != self.model_name:
            from sentence_transformers import CrossEncoder

            CrossEncoderReranker._model = CrossEncoder(self.model_name)
            CrossEncoderReranker._model_name = self.model_name
        return CrossEncoderReranker._model

    def rerank(
        self,
        question: str,
        candidates: Sequence[ScoredChunk],
        top_k: int,
        drop_below: float | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        try:
            model = self._load()
            raw = model.predict([(question, c.chunk.text) for c in candidates])
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-encoder rerank failed (%s); keeping fused order", exc)
            return list(candidates[:top_k])

        # ms-marco cross-encoders emit unbounded logits; squash to [0,1] so
        # the threshold means the same thing across backends.
        import math

        scores = [1.0 / (1.0 + math.exp(-float(value))) for value in raw]
        ranked = sorted(zip(candidates, scores, strict=False), key=lambda p: -p[1])

        out = [
            candidate.model_copy(
                update={"score": score, "retrievers": [*candidate.retrievers, "cross-encoder"]}
            )
            for candidate, score in ranked
            if drop_below is None or score >= drop_below
        ][:top_k]
        return out or list(candidates[:top_k])


def rerank(
    question: str,
    candidates: Sequence[ScoredChunk],
    top_k: int,
    method: str = "none",
    client=None,
    drop_below: float | None = None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> tuple[list[ScoredChunk], str]:
    """Rerank by the configured method. Returns ``(passages, method_used)``.

    ``method_used`` reports what actually ran, which will differ from what was
    asked for when a backend is unavailable -- silently falling back without
    saying so would make retrieval quality unexplainable.
    """
    if method == "none" or not candidates:
        return list(candidates[:top_k]), "none"

    if method == "cross-encoder":
        encoder = CrossEncoderReranker(model_name)
        if encoder.available:
            return encoder.rerank(question, candidates, top_k, drop_below), "cross-encoder"
        log.info("sentence-transformers not installed; falling back to LLM rerank")
        method = "llm" if client is not None else "none"

    if method == "llm" and client is not None:
        return llm_rerank(question, candidates, client, top_k, drop_below), "llm"

    return list(candidates[:top_k]), "none"
