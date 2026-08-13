"""Citation grounding — verifying claims against source text."""

from __future__ import annotations

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
