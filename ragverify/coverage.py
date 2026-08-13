"""Measure whether the local corpus actually covers a question.

Deciding "local vs web" from a filename list and a chunk count is guesswork:
without a retrieved passage or a relevance score, the model is inferring
topical coverage from strings like ``q3-report.pdf``.

The fix is to invert the order: retrieve *first* with a cheap lexical pass,
compute coverage from what actually came back, and route on that measurement.
Retrieval is BM25-only here and costs nothing, so a probe is far cheaper than
the round it saves.

Coverage combines three signals, because each is individually foolable:

* **Score mass** -- how strongly the best passages match. High scores on a
  single term are not enough on their own.
* **Term recall** -- what fraction of the question's content words appear in
  the retrieved set at all. Catches the case where retrieval returns confident
  matches for half the question and nothing for the rest.
* **Concentration** -- whether hits cluster in a few passages or scatter
  thinly. Scattered weak hits usually mean the corpus is adjacent to the topic
  rather than about it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import STOPWORDS, HybridRetriever, analyze
from .schemas import Route, ScoredChunk


@dataclass
class CoverageReport:
    """Measured local-corpus coverage for one question."""

    score: float  # 0..1 overall coverage
    term_recall: float  # share of question terms found
    top_score: float  # strength of the best passage
    n_hits: int
    missing_terms: list[str]
    probe_hits: list[ScoredChunk]

    @property
    def verdict(self) -> str:
        if self.score >= STRONG:
            return "strong"
        if self.score > WEAK:
            return "partial"
        return "weak"


# Routing thresholds. Deliberately conservative in the middle band: when
# coverage is ambiguous, hybrid is cheaper than being wrong and burning a
# whole round on the wrong source.
STRONG = 0.55
WEAK = 0.28


def measure(
    question: str,
    retriever: HybridRetriever | None,
    probe_k: int = 8,
) -> CoverageReport:
    """Probe the corpus and score how well it covers ``question``."""
    terms = [t for t in analyze(question) if t not in STOPWORDS and len(t) > 2]
    if retriever is None or not retriever.chunks or not terms:
        return CoverageReport(0.0, 0.0, 0.0, 0, terms, [])

    # Lexical only: the probe must be free, and dense retrieval costs an
    # embedding call per question.
    raw = retriever.bm25.search(question, probe_k)
    hits = [
        ScoredChunk(chunk=retriever.chunks[i], score=s, retrievers=["bm25"])
        for i, s in raw
    ]
    if not hits:
        return CoverageReport(0.0, 0.0, 0.0, 0, terms, [])

    retrieved_terms = set()
    for hit in hits:
        retrieved_terms.update(analyze(hit.chunk.text))
    found = [t for t in terms if t in retrieved_terms]
    term_recall = len(found) / len(terms)
    missing = [t for t in terms if t not in retrieved_terms]

    # BM25 scores are unbounded and scale with corpus size -- IDF is small
    # when there are few documents, so a strong match on a 3-chunk corpus
    # scores far lower than the same match against 10,000 chunks. A saturating
    # ratio is stable across both; an exponential tuned for one corpus size
    # systematically misreads the other.
    top = hits[0].score
    strength = top / (top + 2.0)

    # Concentration: a top hit well above the mean means the corpus has a
    # specific answer rather than diffuse topical similarity.
    mean = sum(h.score for h in hits) / len(hits)
    concentration = min(1.0, top / (mean * 2.0)) if mean > 0 else 0.0

    # Term recall dominates because it is the only scale-free signal: a corpus
    # missing the question's key nouns cannot answer it however strongly its
    # other terms match.
    score = 0.60 * term_recall + 0.25 * strength + 0.15 * concentration

    return CoverageReport(
        score=round(min(1.0, score), 3),
        term_recall=round(term_recall, 3),
        top_score=round(top, 3),
        n_hits=len(hits),
        missing_terms=missing,
        probe_hits=hits,
    )


def suggest_route(
    report: CoverageReport,
    web_available: bool,
    has_corpus: bool,
) -> tuple[Route, str]:
    """Map a coverage measurement onto a route, with a stated reason.

    This runs *before* the triage agent and constrains it: the model is asked
    to confirm or override a measurement rather than to guess from filenames.
    """
    if not has_corpus:
        return (
            (Route.WEB, "No local corpus; web is the only source.")
            if web_available
            else (Route.LOCAL, "No corpus and no web access.")
        )

    if not web_available:
        return Route.LOCAL, f"Web disabled; local coverage is {report.verdict} ({report.score:.2f})."

    if report.score >= STRONG:
        return Route.LOCAL, f"Strong local coverage ({report.score:.2f}), term recall {report.term_recall:.0%}."

    if report.score <= WEAK:
        missing = ", ".join(report.missing_terms[:5])
        return Route.WEB, (
            f"Weak local coverage ({report.score:.2f})"
            + (f"; corpus lacks: {missing}." if missing else ".")
        )

    return Route.HYBRID, (
        f"Partial local coverage ({report.score:.2f}); "
        f"combining local passages with web to fill gaps."
    )
