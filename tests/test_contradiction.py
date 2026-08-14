"""Deterministic cross-source contradiction detection.

Grounding checks a claim against its own citation; entailment does the same
semantically. Neither ever compares one source against another, so a corpus
holding "revenue was 2.1bn" and "revenue was 1.6bn" produces a fully verified
answer built on whichever passage ranked first. Every check passes and the
answer is still wrong.

Precision matters more than recall here: a false contradiction erodes trust
in every real one, so the "must not detect" cases are the important half.
"""

from __future__ import annotations

from ragverify.contradiction import as_prompt_block, detect, summarize
from ragverify.schemas import EvidenceItem, Route


def ev(sid: str, text: str) -> EvidenceItem:
    return EvidenceItem(source_id=sid, label=sid, text=text, origin=Route.LOCAL)


class TestValueConflicts:
    def test_percentage_conflict(self):
        conflicts = detect([
            ev("S1", "Acme reported European revenue of 2.1 billion euro for Q3 2024, "
                     "representing 34% year-over-year growth."),
            ev("S2", "Our reconstruction puts Acme European revenue at 1.6 billion euro "
                     "for Q3 2024, implying 19% growth."),
        ])
        kinds = {(c.value_type, c.value_a, c.value_b) for c in conflicts}
        assert ("pct", "pct:34", "pct:19") in kinds

    def test_money_conflict(self):
        conflicts = detect([
            ev("S1", "Acme reported European revenue of 2.1 billion euro for Q3 2024 "
                     "across all reporting segments."),
            ev("S2", "Acme European revenue was 1.6 billion euro for Q3 2024 across "
                     "all reporting segments."),
        ])
        assert any(c.value_type == "money" for c in conflicts)

    def test_headcount_conflict(self):
        conflicts = detect([
            ev("S1", "The Berlin office employs 412 engineers at the end of the reporting quarter."),
            ev("S2", "We count 380 engineers at the Berlin office site at quarter end."),
        ])
        assert any(c.value_type == "num" for c in conflicts)

    def test_conflict_names_both_sources(self):
        conflicts = detect([
            ev("A", "European revenue reached 2.1 billion euro in the third quarter of 2024."),
            ev("B", "European revenue reached 1.6 billion euro in the third quarter of 2024."),
        ])
        assert conflicts
        assert {conflicts[0].source_a, conflicts[0].source_b} == {"A", "B"}


class TestPolarityConflicts:
    def test_assertion_versus_denial(self):
        conflicts = detect([
            ev("S1", "The chief executive resigned from the company in October following the audit."),
            ev("S2", "The chief executive did not resign from the company in October despite the audit."),
        ])
        assert any(c.kind == "polarity" for c in conflicts)


class TestPrecision:
    """The important half: a false conflict discredits every real one."""

    def test_different_subjects_are_not_compared(self):
        assert not detect([
            ev("S1", "European revenue grew 34% year over year to 2.1 billion euro in Q3 2024."),
            ev("S2", "The Berlin engineering office reached 412 staff at the end of the quarter."),
        ])

    def test_same_fact_spelled_differently_is_not_a_conflict(self):
        """Canonical comparison: 2.1 billion euro == EUR 2,100,000,000."""
        assert not detect([
            ev("S1", "European revenue grew 34% year over year to 2.1 billion euro in Q3 2024."),
            ev("S2", "European revenue grew 34 percent year over year to EUR 2,100,000,000 in Q3 2024."),
        ])

    def test_agreement_on_one_measure_is_not_conflict(self):
        """Sharing any value of a type means they agree on that measure."""
        conflicts = detect([
            ev("S1", "European revenue grew 34% in the third quarter of 2024 across segments."),
            ev("S2", "European revenue grew 34% in the third quarter of 2024 in every segment."),
        ])
        assert not [c for c in conflicts if c.value_type == "pct"]

    def test_currencies_are_not_compared(self):
        """Different units are a mismatch, not a disagreement."""
        conflicts = detect([
            ev("S1", "The contract value was 5 million USD for the reporting period covered."),
            ev("S2", "The contract value was 5 million EUR for the reporting period covered."),
        ])
        assert not [c for c in conflicts if c.value_type == "money"]

    def test_a_source_is_not_compared_with_itself(self):
        assert not detect([
            ev("S1", "Revenue was 2.1 billion euro in Q3 2024 for the European segment. "
                     "Revenue was 1.6 billion euro in Q3 2024 for the Nordic segment."),
        ])

    def test_short_fragments_are_ignored(self):
        assert not detect([ev("S1", "Grew 34%."), ev("S2", "Grew 19%.")])

    def test_empty_input(self):
        assert detect([]) == []
        assert summarize([]) == ""
        assert as_prompt_block([]) == ""


class TestReporting:
    def test_summary_counts_pairs(self):
        conflicts = detect([
            ev("S1", "European revenue reached 2.1 billion euro in the third quarter of 2024."),
            ev("S2", "European revenue reached 1.6 billion euro in the third quarter of 2024."),
        ])
        text = summarize(conflicts)
        assert "contradiction" in text
        assert "rather than choosing" in text, "detection must not imply resolution"

    def test_prompt_block_forbids_picking_a_side(self):
        conflicts = detect([
            ev("S1", "European revenue reached 2.1 billion euro in the third quarter of 2024."),
            ev("S2", "European revenue reached 1.6 billion euro in the third quarter of 2024."),
        ])
        block = as_prompt_block(conflicts)
        assert "Do not choose a side" in block
        assert "[S1]" in block and "[S2]" in block


class TestInRun:
    def test_contradictions_reach_the_result(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import (
            CITED_ANSWER,
            GOOD_DRAFT,
            QUESTION,
            FakeLLM,
            settings,
            triage,
            verdict,
        )

        from ragverify.ingest import Document
        from ragverify.orchestrator import AdaptiveResearcher, Corpus
        from ragverify.schemas import NextAction, Verdict

        # Two documents, so the statements land in separate evidence items.
        # A single passage disagreeing with itself is a different problem and
        # is deliberately not compared.
        conflicting = [
            Document(name="press.txt", pages=[
                "Acme press release. European revenue grew 34% year over year to "
                "2.1 billion euro in the third quarter of 2024 across all segments."
            ]),
            Document(name="analyst.txt", pages=[
                "Independent analyst note. European revenue grew 19% year over year "
                "to 1.6 billion euro in the third quarter of 2024 across all segments."
            ]),
        ]
        cfg = settings(max_rounds=1)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, Corpus(conflicting, cfg)).run(QUESTION)

        assert result.contradictions, "a self-conflicting corpus must be reported"
        assert any("contradiction" in w.lower() for w in result.warnings)

    def test_disabled_by_setting(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import settings

        assert settings(detect_contradictions=False).detect_contradictions is False
