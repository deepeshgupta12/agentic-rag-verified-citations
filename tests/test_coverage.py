"""Coverage-measured routing.

Deciding local-vs-web from a filename list and a chunk count means routing on
a guess: no retrieved passage, no relevance score. These tests pin the routing
decision to a real retrieval measurement.
"""

from __future__ import annotations

from ragverify.coverage import STRONG, WEAK, measure, suggest_route
from ragverify.retrieval import HybridRetriever
from ragverify.schemas import Chunk, Route


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_name="doc.txt", ordinal=int(cid[1:]), text=text)


CORPUS = [
    chunk("c1", "European revenue grew 34% year over year to 2.1 billion euro in Q3 2024."),
    chunk("c2", "The Berlin engineering office reached 412 staff at the end of the quarter."),
    chunk("c3", "Operating margin was unchanged at 18% across all reporting segments."),
]


def retriever_for(chunks=CORPUS) -> HybridRetriever:
    return HybridRetriever(chunks)


class TestMeasure:
    def test_covered_question_scores_high(self):
        report = measure("What was European revenue growth in Q3?", retriever_for())
        assert report.score >= STRONG
        assert report.verdict == "strong"
        # Recall is 2/3, not 1.0: the corpus says "grew", the question says
        # "growth", and the light suffix stemmer cannot bridge irregular
        # morphology. Coverage still reads strong because score mass and
        # concentration compensate -- which is why the metric is a blend
        # rather than term recall alone.
        assert report.term_recall >= 0.66
        assert report.missing_terms == ["growth"]

    def test_uncovered_question_scores_low(self):
        report = measure("Who won the 2026 World Cup final?", retriever_for())
        assert report.score <= WEAK
        assert report.verdict == "weak"

    def test_missing_terms_are_reported(self):
        # These become the justification shown to the triage agent.
        report = measure("What is the mitochondrial membrane potential?", retriever_for())
        assert "mitochondri" in " ".join(report.missing_terms)

    def test_no_corpus_scores_zero(self):
        report = measure("anything at all", None)
        assert report.score == 0.0 and report.n_hits == 0

    def test_empty_retriever_scores_zero(self):
        assert measure("anything", HybridRetriever([])).score == 0.0

    def test_stopword_only_question_scores_zero(self):
        assert measure("what is the of and", retriever_for()).score == 0.0

    def test_probe_hits_are_returned_for_the_prompt(self):
        report = measure("European revenue growth", retriever_for())
        assert report.probe_hits, "triage must be shown the passages, not just a score"
        assert report.probe_hits[0].chunk.chunk_id == "c1"

    def test_partial_question_lands_in_the_middle(self):
        # Half the question is covered (revenue), half is not (CEO's tenure).
        report = measure("What was European revenue and how long has the CEO served?", retriever_for())
        assert WEAK < report.score < STRONG


class TestSuggestRoute:
    def test_strong_coverage_routes_local(self):
        report = measure("What was European revenue growth in Q3?", retriever_for())
        route, reason = suggest_route(report, web_available=True, has_corpus=True)
        assert route is Route.LOCAL
        assert "Strong local coverage" in reason

    def test_weak_coverage_routes_web(self):
        report = measure("Who won the 2026 World Cup final?", retriever_for())
        route, reason = suggest_route(report, web_available=True, has_corpus=True)
        assert route is Route.WEB
        assert "Weak local coverage" in reason

    def test_partial_coverage_routes_hybrid(self):
        report = measure("What was European revenue and how long has the CEO served?", retriever_for())
        route, _ = suggest_route(report, web_available=True, has_corpus=True)
        assert route is Route.HYBRID

    def test_no_corpus_routes_web(self):
        route, _ = suggest_route(measure("q", None), web_available=True, has_corpus=False)
        assert route is Route.WEB

    def test_no_corpus_no_web_stays_local(self):
        # Nothing is retrievable; the loop's no-evidence path handles it.
        route, _ = suggest_route(measure("q", None), web_available=False, has_corpus=False)
        assert route is Route.LOCAL

    def test_web_disabled_never_suggests_web(self):
        report = measure("Who won the 2026 World Cup final?", retriever_for())
        route, reason = suggest_route(report, web_available=False, has_corpus=True)
        assert route is Route.LOCAL
        assert "Web disabled" in reason

    def test_reason_is_always_stated(self):
        for question in ("European revenue growth", "unrelated quantum topic"):
            report = measure(question, retriever_for())
            _, reason = suggest_route(report, True, True)
            assert reason and reason[0].isupper()
