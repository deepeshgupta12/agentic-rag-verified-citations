"""Reranking — judging passages against the question, not in isolation.

BM25 and dense retrieval score a passage on its own terms. Neither reads it as
an answer, so a passage full of query words can outrank the one that actually
answers. Order matters because the evidence budget truncates: a passage below
the token limit never reaches the model, and no downstream verification can
recover what was never shown.
"""

from __future__ import annotations

from ragverify.rerank import (
    CrossEncoderReranker,
    PassageScore,
    RerankBatch,
    llm_rerank,
    rerank,
)
from ragverify.schemas import Chunk, ScoredChunk


def sc(cid: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_name="d", ordinal=int(cid[1:]), text=text),
        score=score,
        retrievers=["bm25"],
    )


CANDIDATES = [
    sc("c1", "The question of margins is a question many ask about margins.", 0.9),
    sc("c2", "Operating margin was unchanged at 18% across all segments.", 0.5),
    sc("c3", "The cafeteria menu changes on Fridays.", 0.3),
]


class FakeClient:
    def __init__(self, batch=None, error=None):
        self.batch, self.error, self.calls = batch, error, 0

    def structured(self, system, user, schema, **kw):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return self.batch


class TestLLMRerank:
    def test_promotes_the_passage_that_answers(self):
        """c2 answers; c1 merely shares words with the question."""
        client = FakeClient(RerankBatch(scores=[
            PassageScore(index=0, relevance=0.1),
            PassageScore(index=1, relevance=0.95),
            PassageScore(index=2, relevance=0.0),
        ]))
        out = llm_rerank("What was the operating margin?", CANDIDATES, client, top_k=3)

        assert [c.chunk.chunk_id for c in out] == ["c2", "c1", "c3"]
        assert "llm-rerank" in out[0].retrievers

    def test_respects_top_k(self):
        client = FakeClient(RerankBatch(scores=[
            PassageScore(index=i, relevance=1.0 - i * 0.3) for i in range(3)
        ]))
        assert len(llm_rerank("q", CANDIDATES, client, top_k=2)) == 2

    def test_drop_below_removes_irrelevant(self):
        client = FakeClient(RerankBatch(scores=[
            PassageScore(index=0, relevance=0.9),
            PassageScore(index=1, relevance=0.8),
            PassageScore(index=2, relevance=0.01),
        ]))
        out = llm_rerank("q", CANDIDATES, client, top_k=3, drop_below=0.15)
        assert [c.chunk.chunk_id for c in out] == ["c1", "c2"]

    def test_unscored_passage_keeps_its_retrieval_rank(self):
        """A missing score is missing information, not evidence of irrelevance."""
        client = FakeClient(RerankBatch(scores=[PassageScore(index=2, relevance=0.99)]))
        out = llm_rerank("q", CANDIDATES, client, top_k=3)

        assert out[0].chunk.chunk_id == "c3", "the scored winner leads"
        assert {c.chunk.chunk_id for c in out} == {"c1", "c2", "c3"}, "none dropped"

    def test_out_of_range_indices_ignored(self):
        client = FakeClient(RerankBatch(scores=[PassageScore(index=99, relevance=1.0)]))
        out = llm_rerank("q", CANDIDATES, client, top_k=3)
        assert len(out) == 3


class TestFailureIsNeverCostly:
    def test_call_failure_keeps_fused_order(self):
        """A reranker that cannot run must not cost you the retrieval you have."""
        out = llm_rerank("q", CANDIDATES, FakeClient(error="provider down"), top_k=3)
        assert [c.chunk.chunk_id for c in out] == ["c1", "c2", "c3"]

    def test_empty_scores_keep_fused_order(self):
        out = llm_rerank("q", CANDIDATES, FakeClient(RerankBatch(scores=[])), top_k=3)
        assert [c.chunk.chunk_id for c in out] == ["c1", "c2", "c3"]

    def test_empty_candidates(self):
        assert llm_rerank("q", [], FakeClient(RerankBatch()), top_k=3) == []


class TestDispatch:
    def test_none_is_a_passthrough(self):
        out, method = rerank("q", CANDIDATES, top_k=2, method="none")
        assert method == "none"
        assert [c.chunk.chunk_id for c in out] == ["c1", "c2"]

    def test_llm_method_used(self):
        client = FakeClient(RerankBatch(scores=[PassageScore(index=1, relevance=0.99)]))
        _, method = rerank("q", CANDIDATES, top_k=3, method="llm", client=client)
        assert method == "llm"

    def test_llm_without_client_degrades_to_none(self):
        _, method = rerank("q", CANDIDATES, top_k=3, method="llm", client=None)
        assert method == "none"

    def test_cross_encoder_falls_back_when_unavailable(self, monkeypatch):
        """The method actually used is reported, never silently substituted."""
        monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: False))
        client = FakeClient(RerankBatch(scores=[PassageScore(index=1, relevance=0.9)]))

        _, method = rerank("q", CANDIDATES, top_k=3, method="cross-encoder", client=client)
        assert method == "llm", "fallback must be visible in the reported method"

    def test_cross_encoder_unavailable_and_no_client(self, monkeypatch):
        monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: False))
        _, method = rerank("q", CANDIDATES, top_k=3, method="cross-encoder")
        assert method == "none"


class TestRetrieverIntegration:
    def _retriever(self, reranker=None):
        from ragverify.retrieval import HybridRetriever

        chunks = [
            Chunk(chunk_id="c1", doc_name="d", ordinal=1,
                  text="Operating margin margin margin discussion of margin."),
            Chunk(chunk_id="c2", doc_name="d", ordinal=2,
                  text="Operating margin was unchanged at 18% across all reporting segments."),
        ]
        return HybridRetriever(chunks, reranker=reranker)

    def test_reranker_reorders_retrieval(self):
        # A reranker must RESCORE, not just reorder: MMR runs afterwards and
        # reads `score` as its relevance term, so reordering alone is undone
        # by the diversity pass. Both real backends set score.
        def fake(question, candidates, top_k):
            ordered = sorted(candidates, key=lambda c: c.chunk.chunk_id != "c2")
            rescored = [
                c.model_copy(update={"score": 1.0 - i * 0.5}) for i, c in enumerate(ordered)
            ]
            return rescored[:top_k], "llm"

        hits = self._retriever(fake).search("What was the operating margin?", top_k=2)
        assert hits[0].chunk.chunk_id == "c2"

    def test_reordering_without_rescoring_is_undone_by_mmr(self):
        """Documents why the rescoring contract exists."""
        def reorder_only(question, candidates, top_k):
            ordered = sorted(candidates, key=lambda c: c.chunk.chunk_id != "c2")
            return ordered[:top_k], "llm"

        hits = self._retriever(reorder_only).search("operating margin", top_k=2)
        assert hits[0].chunk.chunk_id == "c1", (
            "without rescoring, MMR falls back to retrieval scores"
        )

    def test_method_recorded_for_explainability(self):
        def fake(question, candidates, top_k):
            return list(candidates[:top_k]), "cross-encoder"

        retriever = self._retriever(fake)
        retriever.search("margin", top_k=2)
        assert retriever.last_rerank_method == "cross-encoder"

    def test_no_reranker_is_unchanged(self):
        retriever = self._retriever()
        assert retriever.search("margin", top_k=2)
        assert retriever.last_rerank_method == "none"

    def test_reranker_returning_nothing_falls_back(self):
        retriever = self._retriever(lambda q, c, k: ([], "llm"))
        assert retriever.search("margin", top_k=2), "must not lose all retrieval"
