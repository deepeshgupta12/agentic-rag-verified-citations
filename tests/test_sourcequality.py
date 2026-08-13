"""Source quality, freshness, diversity and corroboration.

Web evidence was treated as uniformly trustworthy: eight results from one
content farm scored like eight independent primary sources. That matters more
here than in a plain search tool, because grounding then *certifies* whatever
those pages said — a claim correctly grounded in a bad source passes every
other check and is still wrong.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ragverify.schemas import EvidenceItem, Route
from ragverify.sourcequality import (
    assess,
    authority_score,
    extract_published_year,
    freshness_score,
    is_time_sensitive,
    rank,
    registrable_domain,
)

YEAR = datetime.now(tz=timezone.utc).year


def web(sid: str, url: str, text: str = "Revenue grew 34% to 2.1 billion euro.") -> EvidenceItem:
    return EvidenceItem(source_id=sid, label=url, text=text, origin=Route.WEB, url=url)


def local(sid: str, text: str = "Local content.") -> EvidenceItem:
    return EvidenceItem(source_id=sid, label="doc", text=text, origin=Route.LOCAL)


class TestAuthority:
    @pytest.mark.parametrize(
        "url,floor",
        [
            ("https://www.sec.gov/filings", 0.95),
            ("https://ec.europa.eu/stats", 0.95),
            ("https://arxiv.org/abs/2401.1", 0.85),
            ("https://mit.edu/research", 0.85),
            ("https://en.wikipedia.org/wiki/X", 0.7),
            ("https://reuters.com/article", 0.6),
        ],
    )
    def test_recognised_publishers_score_high(self, url, floor):
        assert authority_score(url) >= floor

    def test_content_farms_score_low(self):
        assert authority_score("https://foo.blogspot.com/p") <= 0.25
        assert authority_score("https://www.ehow.com/how") <= 0.25

    def test_unknown_domain_is_neutral_not_penalised(self):
        """An unrecognised domain is unmeasured, not bad."""
        score = authority_score("https://some-company-blog.xyz/post")
        assert 0.3 < score < 0.6

    def test_malformed_url_does_not_raise(self):
        assert authority_score("not a url") == pytest.approx(0.45)


class TestRegistrableDomain:
    def test_subdomains_collapse(self):
        """Five pages from one site are one source wearing five hats."""
        assert registrable_domain("https://a.example.com/x") == registrable_domain(
            "https://b.example.com/y"
        )

    def test_two_level_suffixes(self):
        assert registrable_domain("https://news.bbc.co.uk/x") == "bbc.co.uk"
        assert registrable_domain("https://x.gov.uk/y") == "x.gov.uk"

    def test_www_stripped(self):
        assert registrable_domain("https://www.reuters.com/a") == "reuters.com"


class TestFreshness:
    def test_time_sensitive_questions_detected(self):
        assert is_time_sensitive("Who is the current CEO?")
        assert is_time_sensitive("What is the latest guidance for 2026?")
        assert not is_time_sensitive("What is photosynthesis?")

    def test_old_page_penalised_only_when_it_matters(self):
        """A 2019 page is fine for a stable fact and disqualifying for 'current'."""
        assert freshness_score(2019, time_sensitive=True) < 0.15
        assert freshness_score(2019, time_sensitive=False) > 0.4

    def test_unknown_age_is_neutral(self):
        """Most pages state no date; undated must not mean old."""
        assert freshness_score(None, True) == pytest.approx(0.6)

    def test_current_year_scores_high(self):
        assert freshness_score(YEAR, True) > 0.95

    def test_year_extracted_from_text(self):
        assert extract_published_year("Published: 12 March 2023 by staff") == 2023
        assert extract_published_year("Updated 2024-06-01") == 2024
        assert extract_published_year("no date anywhere here") is None

    def test_implausible_years_ignored(self):
        assert extract_published_year("in the year 1650 the king") is None


class TestCorroborationAndDiversity:
    def test_independent_domains_corroborate(self):
        text = "Revenue grew 34% to 2.1 billion euro in Q3 2024."
        report = assess([web("W1", "https://a.test/x", text), web("W2", "https://b.test/y", text)], "q")

        assert report.distinct_domains == 2
        assert all(a.corroborated_by for a in report.assessments)

    def test_subdomains_are_not_independent_corroboration(self):
        text = "Revenue grew 34% to 2.1 billion euro in Q3 2024."
        report = assess(
            [web("W1", "https://a.same.test/x", text), web("W2", "https://b.same.test/y", text)], "q"
        )

        assert report.distinct_domains == 1
        assert not any(a.corroborated_by for a in report.assessments)
        assert any("domain" in w for w in report.warnings)

    def test_conflicting_figures_do_not_corroborate(self):
        report = assess(
            [
                web("W1", "https://a.test/x", "Revenue grew 34% to 2.1 billion euro."),
                web("W2", "https://b.test/y", "Revenue grew 47% to 9.9 billion euro."),
            ],
            "q",
        )
        assert not any(a.corroborated_by for a in report.assessments)

    def test_low_authority_is_warned_not_removed(self):
        """A weak source that is the only answer is still the answer."""
        report = assess([web("W1", "https://foo.blogspot.com/p")], "q")

        assert len(report.assessments) == 1, "nothing is discarded"
        assert any("low-authority" in w for w in report.warnings)

    def test_stale_sources_warned_on_time_sensitive_question(self):
        report = assess(
            [web("W1", "https://a.test/x", "Published 2015-01-01. The CEO is Alice.")],
            "Who is the current CEO?",
        )
        assert report.time_sensitive
        assert any("current information" in w for w in report.warnings)

    def test_local_evidence_is_not_assessed(self):
        """The user chose these; ranking their 'authority' is meaningless."""
        report = assess([local("S1"), local("S2")], "q")
        assert report.assessments == []


class TestRanking:
    def test_web_sorted_by_quality_local_position_preserved(self):
        text = "Revenue grew 34% to 2.1 billion euro."
        evidence = [
            local("S1"),
            web("W1", "https://foo.blogspot.com/p", text),
            web("W2", "https://www.sec.gov/filing", text),
        ]
        report = assess(evidence, "q")
        ordered = rank(evidence, report)

        assert ordered[0].source_id == "S1", "local evidence keeps its position"
        assert ordered[1].source_id == "W2", "the regulator outranks the content farm"

    def test_ranking_is_stable_with_no_web(self):
        evidence = [local("S1"), local("S2")]
        assert [e.source_id for e in rank(evidence, assess(evidence, "q"))] == ["S1", "S2"]


class TestInRun:
    def test_quality_reported_on_the_result(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import (
            CITED_ANSWER,
            GOOD_DRAFT,
            QUESTION,
            FakeLLM,
            corpus_for,
            settings,
            triage,
            verdict,
        )

        from ragverify.orchestrator import AdaptiveResearcher
        from ragverify.schemas import NextAction, Verdict

        cfg = settings(max_rounds=1)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        # Local-only run: no web sources to assess, so the field stays empty
        # rather than reporting fabricated quality numbers.
        assert result.source_quality == {}
