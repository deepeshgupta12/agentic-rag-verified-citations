"""Entailment gate — the failures lexical overlap structurally cannot catch.

"revenue grew in most regions" cited for "revenue grew in all regions" shares
every content word and contains no figures. Lexical grounding passes it and
the claim is false.
"""

from __future__ import annotations

import pytest

from ragverify.entailment import (
    ClaimVerdict,
    EntailmentBatch,
    EntailmentLabel,
    _quote_present,
    check_entailment,
)
from ragverify.schemas import Claim, EvidenceItem, Route

SOURCE = "Revenue grew in most European regions during the third quarter of 2024."


def ev(sid: str = "S1", text: str = SOURCE) -> EvidenceItem:
    return EvidenceItem(source_id=sid, label="doc", text=text, origin=Route.LOCAL)


class FakeClient:
    """Returns a scripted batch, or raises, without touching the network."""

    def __init__(self, batch=None, error=None):
        self.batch, self.error = batch, error
        self.prompts: list[str] = []

    def structured(self, system, user, schema, **kw):
        self.prompts.append(user)
        if self.error:
            raise RuntimeError(self.error)
        return self.batch


def verdict(index, label, quote="", reason="") -> ClaimVerdict:
    return ClaimVerdict(index=index, label=label, quote=quote, reason=reason)


class TestLabelling:
    def test_entailed_claim_is_kept(self):
        claims = [Claim(text="Revenue grew in most European regions", citations=["S1"])]
        client = FakeClient(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.ENTAILED, quote="Revenue grew in most European regions")
        ]))
        report = check_entailment(claims, [ev()], client)

        assert report.ran and report.entailment_rate == 1.0
        assert len(report.entailed) == 1

    def test_overgeneralization_is_not_entailed(self):
        """The headline case: every word matches, the claim is still false."""
        claims = [Claim(text="Revenue grew in all European regions", citations=["S1"])]
        client = FakeClient(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.NEUTRAL, quote="most European regions",
                    reason="'most' does not establish 'all'")
        ]))
        report = check_entailment(claims, [ev()], client)

        assert len(report.neutral) == 1
        assert not report.entailed

    def test_contradiction_is_flagged_separately(self):
        """A contradiction is worse than thin evidence: no retrieval fixes it."""
        claims = [Claim(text="Revenue fell in all European regions", citations=["S1"])]
        client = FakeClient(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.CONTRADICTED, quote="Revenue grew in most European regions")
        ]))
        report = check_entailment(claims, [ev()], client)

        assert report.has_contradiction
        assert len(report.contradicted) == 1

    def test_missing_verdict_defaults_to_neutral(self):
        claims = [Claim(text="Something", citations=["S1"])]
        report = check_entailment(claims, [ev()], FakeClient(EntailmentBatch(verdicts=[])))
        assert len(report.neutral) == 1, "no judgement must not become approval"


class TestQuoteVerification:
    def test_quote_not_in_source_downgrades(self):
        """A quote absent from the passage means the judgement was not made from it."""
        claims = [Claim(text="Revenue grew in most regions", citations=["S1"])]
        client = FakeClient(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.ENTAILED, quote="Revenue doubled across every market")
        ]))
        report = check_entailment(claims, [ev()], client)

        assert not report.entailed, "an unverifiable quote must not carry an entailment"
        assert report.unverifiable_quotes

    @pytest.mark.parametrize(
        "quote,expected",
        [
            ("Revenue grew in most European regions", True),
            ("revenue  grew   in most european regions", True),   # whitespace/case drift
            ("Revenue grew in most... third quarter", True),      # elided
            ("Revenue doubled everywhere", False),
            ("short", True),                                      # too short to weigh
        ],
    )
    def test_quote_matching_tolerates_drift(self, quote, expected):
        assert _quote_present(quote, SOURCE) is expected


class TestResilience:
    def test_failure_is_not_fatal(self):
        claims = [Claim(text="Revenue grew", citations=["S1"])]
        report = check_entailment(claims, [ev()], FakeClient(error="provider down"))

        assert not report.ran
        assert "provider down" in report.error
        assert report.total == 0, "a failed check must not produce verdicts"

    def test_claims_without_resolvable_citations_are_skipped(self):
        claims = [Claim(text="Revenue grew", citations=["S99"])]
        report = check_entailment(claims, [ev()], FakeClient(EntailmentBatch()))
        assert not report.ran

    def test_empty_input(self):
        assert not check_entailment([], [ev()], FakeClient(EntailmentBatch())).ran


class TestOrchestratorIntegration:
    """The gate must only ever downgrade."""

    def _run(self, batch, **cfg_kw):
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
        )
        from test_orchestrator import (
            verdict as vdict,
        )

        from ragverify.orchestrator import AdaptiveResearcher
        from ragverify.schemas import NextAction, Verdict

        cfg = settings(max_rounds=1, use_entailment=True, **cfg_kw)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, vdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        llm.structured_override = batch
        original = llm.structured

        def structured(system, user, schema, **kw):
            if schema is EntailmentBatch:
                return batch
            return original(system, user, schema, **kw)

        llm.structured = structured
        return AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)

    def test_disabled_by_default(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import settings

        assert not settings().use_entailment, "costs a call per round; opt-in"

    def test_neutral_verdict_removes_the_claim(self):
        result = self._run(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.NEUTRAL, quote="European revenue grew 34%")
        ]))
        assert result.rounds[0].entailment, "entailment must be recorded on the round"
        assert not result.rounds[0].grounding.supported, "a neutral claim must be dropped"

    def test_entailed_verdict_keeps_the_claim(self):
        result = self._run(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.ENTAILED, quote="European revenue grew 34%")
        ]))
        assert result.rounds[0].grounding.supported

    def test_lenient_mode_keeps_neutral_claims(self):
        result = self._run(
            EntailmentBatch(verdicts=[
                verdict(0, EntailmentLabel.NEUTRAL, quote="European revenue grew 34%")
            ]),
            entailment_strict=False,
        )
        assert result.rounds[0].grounding.supported, "lenient mode keeps neutral"

    def test_contradiction_warns(self):
        result = self._run(EntailmentBatch(verdicts=[
            verdict(0, EntailmentLabel.CONTRADICTED, quote="European revenue grew 34%")
        ]))
        assert any("contradicted" in w.lower() for w in result.warnings)
