"""Citation grounding — verifying claims against source text."""

from __future__ import annotations

import pytest

from ragverify.grounding import check, claim_support, strip_unsupported
from ragverify.schemas import Claim, EvidenceItem, Route


def ev(sid: str, text: str) -> EvidenceItem:
    return EvidenceItem(source_id=sid, label=f"doc {sid}", text=text, origin=Route.LOCAL)


SOURCES = [
    ev("S1", "The Q3 report states that European revenue grew 34% year over year to 2.1 billion euro."),
    ev("S2", "Headcount in the Berlin engineering office reached 412 at the end of the quarter."),
]


class TestClaimSupport:
    def test_paraphrase_is_supported(self):
        supported, ratio = claim_support("European revenue grew 34% year over year", SOURCES[0].text)
        assert supported and ratio > 0.5

    def test_fabricated_number_is_rejected(self):
        # The prose matches almost perfectly; only the figure is invented.
        # This is the highest-value catch, and pure word overlap misses it.
        supported, ratio = claim_support("European revenue grew 47% year over year", SOURCES[0].text)
        assert not supported
        assert ratio > 0.5, "rejection must come from the number check, not low overlap"

    def test_unrelated_claim_rejected(self):
        assert not claim_support("The CEO resigned in October", SOURCES[0].text)[0]

    def test_empty_claim_rejected(self):
        assert claim_support("", SOURCES[0].text) == (False, 0.0)

    def test_number_present_elsewhere_in_source_passes(self):
        assert claim_support("Berlin headcount reached 412", SOURCES[1].text)[0]


class TestCheck:
    def test_supported_claim(self):
        report = check([Claim(text="European revenue grew 34%", citations=["S1"])], SOURCES)
        assert len(report.supported) == 1
        assert report.support_rate == 1.0

    def test_fabricated_citation_is_flagged(self):
        report = check([Claim(text="Revenue grew 34%", citations=["S9"])], SOURCES)
        assert report.hallucinated_citations == ["S9"]
        assert len(report.unsupported) == 1

    def test_real_citation_wrong_content(self):
        # S2 exists but says nothing about revenue: cited-but-irrelevant, the
        # failure mode a "does the citation resolve?" check alone would miss.
        report = check([Claim(text="European revenue grew 34%", citations=["S2"])], SOURCES)
        assert len(report.unsupported) == 1
        assert not report.hallucinated_citations

    def test_claim_with_no_citations_is_unsupported(self):
        report = check([Claim(text="Revenue grew 34%", citations=[])], SOURCES)
        assert len(report.unsupported) == 1

    def test_any_valid_citation_suffices(self):
        claim = Claim(text="European revenue grew 34%", citations=["S2", "S1"])
        assert len(check([claim], SOURCES).supported) == 1

    def test_bracket_format_drift_is_tolerated(self):
        # Models write [S1] or S1. constantly; treating those as fabrications
        # would drown the real signal.
        for cited in ("[S1]", "S1.", " s1 ", "(S1)"):
            report = check([Claim(text="European revenue grew 34%", citations=[cited])], SOURCES)
            assert len(report.supported) == 1, f"{cited!r} should resolve to S1"

    def test_support_rate_mixed(self):
        report = check(
            [
                Claim(text="European revenue grew 34%", citations=["S1"]),
                Claim(text="The CEO resigned in October", citations=["S1"]),
            ],
            SOURCES,
        )
        assert report.support_rate == 0.5

    def test_empty_inputs(self):
        report = check([], SOURCES)
        assert report.total == 0 and report.support_rate == 0.0
        assert len(check([Claim(text="anything", citations=["S1"])], []).unsupported) == 1


class TestStripUnsupported:
    def test_marks_rather_than_deletes(self):
        answer = "Revenue grew 34%. The CEO resigned."
        out = strip_unsupported(answer, [Claim(text="The CEO resigned.", citations=[])])
        assert "_[unverified]_" in out
        assert "Revenue grew 34%." in out, "supported text must be untouched"


class TestAnswerAudit:
    """The final answer is re-verified; these pin the parsing edge cases.

    A splitter bug here fails *correct* answers silently, which is the worst
    direction for this check to be wrong in — it would push the pipeline to
    discard good output and fall back unnecessarily.
    """

    def test_trailing_citation_belongs_to_its_sentence(self):
        from ragverify.grounding import verify_answer

        audit = verify_answer(
            "- European revenue grew 34% year over year to 2.1 billion euro. [S1]", SOURCES
        )
        assert audit.verified_rate == 1.0, "a citation after the period must not be orphaned"
        assert not audit.unverified_sentences

    def test_two_sentences_each_with_trailing_citations(self):
        from ragverify.grounding import verify_answer

        audit = verify_answer(
            "European revenue grew 34% year over year. [S1] "
            "Berlin headcount reached 412. [S2]",
            SOURCES,
        )
        assert len(audit.verified_sentences) == 2

    def test_fabricated_citation_still_caught(self):
        from ragverify.grounding import verify_answer

        audit = verify_answer("The CEO resigned in October. [S99]", SOURCES)
        assert audit.fabricated_citations == ["S99"]
        assert not audit.is_clean

    def test_absence_statements_are_not_failures(self):
        """Disclosure must not be penalised.

        A source cannot contain words confirming what it does not say, so
        failing these would push the synthesizer to omit its own gap
        disclosures — the opposite of the intended incentive.
        """
        from ragverify.grounding import verify_answer

        audit = verify_answer(
            "European revenue grew 34% year over year. [S1] "
            "The 2027 forecast is not provided in the source. [S1]",
            SOURCES,
        )
        assert len(audit.disclosure_sentences) == 1
        assert audit.verified_rate == 1.0
        assert audit.is_clean

    def test_uncited_answer_is_never_clean(self):
        """`is_clean` must not pass vacuously on an answer citing nothing."""
        from ragverify.grounding import verify_answer

        audit = verify_answer("Revenue grew substantially over the period.", SOURCES)
        assert audit.total_cited == 0
        assert not audit.is_clean, "nothing to verify is not the same as nothing failed"

    def test_headings_and_tables_are_ignored(self):
        from ragverify.grounding import verify_answer

        audit = verify_answer(
            "## Summary\n\n| a | b |\n\n"
            "European revenue grew 34% year over year. [S1]",
            SOURCES,
        )
        assert audit.verified_rate == 1.0


class TestValueNormalization:
    """Figures must compare by value, not by spelling.

    Raw string comparison failed in both directions: it rejected correct
    claims written in a different format, and accepted a percentage validated
    by an unrelated headcount.
    """

    @pytest.mark.parametrize(
        "claim,source",
        [
            ("Revenue was 2.1 billion euro", "Revenue was EUR 2,100,000,000"),
            ("Revenue was €2.1B", "Revenue was 2.1 billion euro"),
            ("Margin was 34%", "Margin was 34 percent"),
            ("Cash was $1.5M", "Cash was 1,500,000 USD"),
            ("Revenue was €2.1 billion", "Revenue reached 2.1 billion"),
            ("Reported in October 2024", "Reported on 2024-10-15"),
            ("Results for Q3 2024", "Results for the third quarter of 2024"),
            ("Guidance for FY2025", "Guidance for fiscal 2025"),
        ],
    )
    def test_equivalent_forms_match(self, claim, source):
        from ragverify.normalize import unsupported_values

        assert not unsupported_values(claim, source), f"{claim!r} should match {source!r}"

    @pytest.mark.parametrize(
        "claim,source",
        [
            ("Margin was 34%", "Headcount was 34 staff"),      # type confusion
            ("Margin was 47%", "Margin was 34%"),               # wrong figure
            ("Revenue was 2.1 billion", "Revenue was 2.1 million"),  # 1000x error
            ("Cash was $5M", "Cash was €5M"),                   # wrong currency
            ("Headcount was 412", "Headcount was 380"),
            ("Results for Q3 2024", "Results for Q4 2024"),
            ("Guidance for FY2027", "Guidance for fiscal 2025"),
        ],
    )
    def test_different_values_are_rejected(self, claim, source):
        from ragverify.normalize import unsupported_values

        assert unsupported_values(claim, source), f"{claim!r} must NOT match {source!r}"

    def test_grounding_uses_normalized_values(self):
        # End-to-end: the same fact in two formats must ground.
        item = EvidenceItem(
            source_id="S1", label="a",
            text="European revenue grew 34 percent to EUR 2,100,000,000 in the third quarter of 2024.",
            origin=Route.LOCAL,
        )
        report = check(
            [Claim(text="European revenue grew 34% to €2.1 billion in Q3 2024", citations=["S1"])],
            [item],
        )
        assert len(report.supported) == 1

    def test_grounding_still_rejects_a_wrong_figure(self):
        item = EvidenceItem(
            source_id="S1", label="a",
            text="European revenue grew 34 percent year over year.", origin=Route.LOCAL,
        )
        report = check(
            [Claim(text="European revenue grew 43 percent year over year", citations=["S1"])],
            [item],
        )
        assert len(report.unsupported) == 1, "a transposed figure must still fail"
