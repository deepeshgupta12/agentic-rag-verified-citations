"""Retrieval quality tests, written against naive-scorer failure modes."""

from __future__ import annotations

from ragverify.retrieval import BM25Index, HybridRetriever, mmr_rerank, reciprocal_rank_fusion
from ragverify.schemas import Chunk, ScoredChunk


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_name="doc.txt", ordinal=int(cid[1:]), text=text)


def _naive_overlap_score(query: str, text: str) -> int:
    """The naive scorer: size of the query/chunk token-set intersection."""
    import re

    tok = lambda s: set(re.findall(r"[a-zA-Z0-9]+", s.lower()))  # noqa: E731
    return len(tok(query) & tok(text))


class TestNaiveScorerFailures:
    def test_length_bias_with_equal_term_frequency(self):
        """Same query terms, same frequency, 60x the length.

        Scoring a *set* intersection ignores length entirely, so the naive
        scorer ties these two -- and with a stable sort, the padded chunk wins
        the tie as often as not. BM25's length normalisation is what separates
        them: the dense short passage is far more likely to be about the query
        than a passing mention buried in filler.
        """
        query = "mitochondrial membrane potential gradient"
        relevant = "Mitochondrial membrane potential is a voltage gradient."
        filler = "The committee reviewed logistics and scheduling for the fiscal review session. " * 30
        padded = (
            filler
            + "A mitochondrial note, a membrane note, potential outcomes, and a gradient of options. "
            + filler
        )

        assert _naive_overlap_score(query, padded) == _naive_overlap_score(query, relevant), (
            "the naive scorer cannot tell these apart"
        )

        index = BM25Index([chunk("c1", relevant), chunk("c2", padded)])
        ranked = index.search(query, top_k=2)
        assert index.chunks[ranked[0][0]].chunk_id == "c1"
        assert ranked[0][1] > 2 * ranked[1][1], "the dense passage should win decisively"

    def test_term_frequency_is_used(self):
        query = "quantum decoherence"
        once = "The paper mentions quantum decoherence in passing, then moves on to other topics entirely."
        many = "Quantum decoherence dominates. Quantum decoherence rates rise with temperature; quantum decoherence is measured directly."
        # The naive scorer rates these identically: both contain both terms.
        assert _naive_overlap_score(query, once) == _naive_overlap_score(query, many)

        index = BM25Index([chunk("c1", once), chunk("c2", many)])
        scores = dict(index.search(query, top_k=2))
        assert scores[1] > scores[0]

    def test_stopwords_do_not_create_matches(self):
        index = BM25Index([chunk("c1", "The report is about the thing that was in the place.")])
        # A query sharing only stopwords must not retrieve anything.
        assert index.search("what is the of and", top_k=5) == []


class TestBM25:
    def test_empty_corpus_and_empty_query(self):
        assert BM25Index([]).search("anything", 5) == []
        assert BM25Index([chunk("c1", "text here")]).search("", 5) == []

    def test_rare_term_outweighs_common_term(self):
        common = [chunk(f"c{i}", "revenue growth revenue growth") for i in range(1, 9)]
        rare = chunk("c9", "revenue growth plus the unique tokenidentifier zzyzx")
        index = BM25Index([*common, rare])
        top = index.search("revenue zzyzx", top_k=1)
        assert index.chunks[top[0][0]].chunk_id == "c9"

    def test_stemming_bridges_plural(self):
        index = BM25Index([chunk("c1", "The requirements are documented in section four.")])
        assert index.search("requirement", top_k=1), "singular query should match plural text"

    def test_ranking_is_deterministic(self):
        chunks = [chunk(f"c{i}", f"alpha beta gamma item {i}") for i in range(1, 6)]
        index = BM25Index(chunks)
        assert index.search("alpha item", 5) == index.search("alpha item", 5)


class TestFusion:
    def test_rrf_rewards_agreement(self):
        rankings = {
            "bm25": [(1, 9.0), (2, 8.0), (3, 7.0)],
            "dense": [(3, 0.9), (1, 0.8), (9, 0.7)],
        }
        fused = reciprocal_rank_fusion(rankings)
        # Doc 1 is ranked highly by both; doc 9 only by one.
        assert fused[1][0] > fused[9][0]
        assert sorted(fused[1][1]) == ["bm25", "dense"]

    def test_rrf_ignores_score_scale(self):
        # Cosine similarities are ~1.0 while BM25 can be ~20. RRF reads rank
        # only, so inflating one retriever's scores changes nothing.
        a = reciprocal_rank_fusion({"x": [(1, 0.9), (2, 0.8)], "y": [(2, 30.0), (1, 25.0)]})
        b = reciprocal_rank_fusion({"x": [(1, 0.9), (2, 0.8)], "y": [(2, 3.0), (1, 2.5)]})
        assert a == b


class TestMMR:
    def test_drops_near_duplicates(self):
        # Scores are on the RRF scale the retriever actually produces
        # (1/(60+rank)), where rank gaps are small and redundancy dominates.
        text = "the adaptive router escalates to web search when local evidence is insufficient"
        candidates = [
            ScoredChunk(chunk=chunk("c1", text), score=1 / 61),
            ScoredChunk(chunk=chunk("c2", text), score=1 / 62),
            ScoredChunk(
                chunk=chunk("c3", "grounding verifies citations against source passages"),
                score=1 / 63,
            ),
        ]
        picked = [s.chunk.chunk_id for s in mmr_rerank(candidates, top_k=2)]
        assert picked[0] == "c1"
        assert picked[1] == "c3", "the distinct chunk must beat the exact duplicate"

    def test_keeps_duplicate_when_score_gap_is_large(self):
        # MMR trades relevance for diversity; it must not throw away a much
        # stronger hit to avoid mild redundancy.
        text = "adaptive routing escalates to web search"
        candidates = [
            ScoredChunk(chunk=chunk("c1", text), score=1.0),
            ScoredChunk(chunk=chunk("c2", text + " when evidence is thin"), score=0.95),
            ScoredChunk(chunk=chunk("c3", "unrelated boilerplate about cookies"), score=0.01),
        ]
        assert [s.chunk.chunk_id for s in mmr_rerank(candidates, top_k=2)] == ["c1", "c2"]

    def test_handles_single_and_empty(self):
        assert mmr_rerank([], 3) == []
        one = [ScoredChunk(chunk=chunk("c1", "solo"), score=1.0)]
        assert len(mmr_rerank(one, 3)) == 1


class TestHybridRetriever:
    def test_falls_back_when_dense_raises(self):
        chunks = [chunk("c1", "photosynthesis converts light into chemical energy")]

        class Boom:
            def search(self, *_a, **_k):
                raise RuntimeError("embedding service down")

        def bad_embed(_q):
            raise RuntimeError("embedding service down")

        retriever = HybridRetriever(chunks, dense=Boom(), embed_query=bad_embed)
        hits = retriever.search("photosynthesis", top_k=1)
        assert len(hits) == 1, "BM25 results must survive a dense failure"
        assert retriever.warnings and "BM25 only" in retriever.warnings[0]

    def test_expansions_widen_recall(self):
        chunks = [
            chunk("c1", "The company reported strong annual revenue."),
            chunk("c2", "Headcount in the Berlin office reached four hundred staff."),
        ]
        retriever = HybridRetriever(chunks)
        assert len(retriever.search("revenue", top_k=5)) == 1
        # A verifier gap about headcount pulls in the chunk the original
        # question never matched -- this is what makes a widen round useful.
        widened = retriever.search("revenue", top_k=5, expansions=["Berlin office headcount"])
        assert len(widened) == 2

    def test_empty_index_returns_nothing(self):
        assert HybridRetriever([]).search("anything", top_k=5) == []
