"""The adaptive loop.

Every assertion here is about the loop *acting* on the verifier's verdict
rather than treating it as a label nothing reads. They run with a scripted fake LLM, so they need no
API key and no network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ragverify.config import Settings
from ragverify.ingest import Document
from ragverify.llm import LLMClient, LLMError
from ragverify.orchestrator import AdaptiveResearcher, Corpus
from ragverify.schemas import (
    AnswerClaim,
    AnswerClaimKind,
    Claim,
    NextAction,
    Outcome,
    ResearchDraft,
    Route,
    StructuredAnswer,
    TriageDecision,
    Verdict,
    VerifierReport,
)
from ragverify.trace import EventKind, Tracer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM(LLMClient):
    """LLMClient with the network replaced by a scripted queue.

    Subclasses the real client so retry, usage accounting and the structured
    path are all exercised rather than stubbed out.
    """

    def __init__(self, settings: Settings, script: list[Any], budget=None):
        self.settings = settings
        from ragverify.schemas import Usage

        self.usage = Usage()
        self.budget = budget
        self._structured_supported = None
        self._client = None
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        # Mirror the real client: budget is enforced per call, not per round.
        if self.budget is not None:
            self.budget.check()
        self.prompts.append(user)
        if not self.script:
            return "Final synthesized answer."
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(user)
        out = (
            json.dumps(item.model_dump(mode="json"))
            if hasattr(item, "model_dump")
            else (item if isinstance(item, str) else json.dumps(item))
        )
        # Mirror the real client's accounting so usage assertions are meaningful.
        from ragverify.schemas import Usage
        from ragverify.tokens import count_tokens

        self.usage = self.usage.add(
            Usage(
                prompt_tokens=count_tokens(system + user),
                completion_tokens=count_tokens(out),
                calls=1,
            )
        )
        if self.budget is not None:
            self.budget.calls = self.usage.calls
            self.budget.cost_usd = self.usage.cost_usd
        return out

    def embed(self, texts, batch_size: int = 96):
        raise RuntimeError("embeddings disabled in tests")


def settings(**kw) -> Settings:
    base: dict[str, Any] = dict(
        api_key="test", web_enabled=False, use_embeddings=False, max_rounds=3, top_k=4
    )
    base.update(kw)
    return Settings(**base)


DOC = Document(
    name="report.txt",
    pages=[
        "European revenue grew 34% year over year to 2.1 billion euro in Q3.\n\n"
        "The Berlin engineering office reached 412 staff at quarter end.\n\n"
        "Operating margin was unchanged at 18%."
    ],
)


def corpus_for(cfg: Settings) -> Corpus:
    return Corpus([DOC], cfg)


def triage(route=Route.LOCAL, **kw) -> TriageDecision:
    return TriageDecision(route=route, confidence=kw.pop("confidence", 0.9), rationale="test", **kw)


def draft(text: str, citations: list[str], **kw) -> ResearchDraft:
    return ResearchDraft(claims=[Claim(text=text, citations=citations)], draft_answer=text, **kw)


def verdict(v: Verdict, action: NextAction, **kw) -> VerifierReport:
    return VerifierReport(verdict=v, next_action=action, rationale="test", **kw)


GOOD_DRAFT = draft("European revenue grew 34% year over year", ["S1"])

QUESTION = "How did European revenue change this quarter?"

# Synthesis returns CLAIMS now, not prose: the answer is rendered from what
# verifies. A bare string no longer represents a successful synthesis.
def answer_claims(*claims) -> StructuredAnswer:
    """Build a structured synthesis result for the scripted client."""
    return StructuredAnswer(claims=list(claims))


def assertion(text: str, *citations: str) -> AnswerClaim:
    return AnswerClaim(text=text, kind=AnswerClaimKind.ASSERTION, citations=list(citations))


def disclosure(text: str, *citations: str) -> AnswerClaim:
    return AnswerClaim(text=text, kind=AnswerClaimKind.DISCLOSURE, citations=list(citations))


CITED_ANSWER = answer_claims(
    assertion("European revenue grew 34% year over year", "S1")
)


# ---------------------------------------------------------------------------
# The core regression: the verdict must change behaviour
# ---------------------------------------------------------------------------


class TestVerdictDrivesControlFlow:
    def test_sufficient_stops_after_one_round(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(),
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run("How did revenue change?")

        assert len(result.rounds) == 1
        assert not result.stopped_early
        assert result.confidence == "high"

    def test_insufficient_triggers_another_round(self):
        """The failure this design exists to prevent: answering regardless."""
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(),
            draft("European revenue grew 34%", ["S1"]),
            verdict(Verdict.INSUFFICIENT, NextAction.WIDEN_LOCAL, gaps=["Berlin headcount"]),
            draft("The Berlin engineering office reached 412 staff", ["S1"]),
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run("Revenue and headcount?")

        assert len(result.rounds) == 2, "an insufficient verdict must cause a second round"
        assert result.rounds[0].verifier.verdict is Verdict.INSUFFICIENT
        assert result.rounds[1].verifier.verdict is Verdict.SUFFICIENT
        assert not result.stopped_early

    def test_widen_local_increases_top_k(self):
        cfg = settings(top_k=4, widen_factor=2)
        llm = FakeLLM(cfg, [
            triage(),
            GOOD_DRAFT,
            verdict(Verdict.INSUFFICIENT, NextAction.WIDEN_LOCAL, gaps=["margin"]),
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert result.rounds[0].top_k == 4
        assert result.rounds[1].top_k == 8

    def test_refine_query_changes_the_query(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(),
            GOOD_DRAFT,
            verdict(
                Verdict.INSUFFICIENT,
                NextAction.REFINE_QUERY,
                refined_query="operating margin percentage",
            ),
            draft("Operating margin was unchanged at 18%", ["S1"]),
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        question = "How profitable was the quarter?"
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(question)
        assert result.rounds[0].query == question
        assert result.rounds[1].query == "operating margin percentage"

    def test_gaps_become_retrieval_expansions(self):
        """A gap must actually reach the retriever, not just be logged."""
        cfg = settings()
        captured: list[Any] = []
        corpus = corpus_for(cfg)
        original = corpus.retriever.search

        def spy(query, top_k=6, *, expansions=(), **kw):
            captured.append(list(expansions))
            return original(query, top_k, expansions=expansions, **kw)

        corpus.retriever.search = spy  # type: ignore[assignment]

        llm = FakeLLM(cfg, [
            triage(),
            GOOD_DRAFT,
            verdict(Verdict.INSUFFICIENT, NextAction.WIDEN_LOCAL, gaps=["Berlin office headcount"]),
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        AdaptiveResearcher(cfg, llm, corpus).run(QUESTION)

        assert captured[0] == []
        assert "Berlin office headcount" in captured[1]

    def test_round_budget_is_respected_and_flagged(self):
        cfg = settings(max_rounds=2)
        never_happy = [
            triage(),
            GOOD_DRAFT,
            verdict(Verdict.INSUFFICIENT, NextAction.WIDEN_LOCAL, gaps=["more"]),
            GOOD_DRAFT,
            verdict(Verdict.INSUFFICIENT, NextAction.WIDEN_LOCAL, gaps=["still more"]),
            CITED_ANSWER,
        ]
        result = AdaptiveResearcher(cfg, FakeLLM(cfg, never_happy), corpus_for(cfg)).run(QUESTION)

        assert len(result.rounds) == 2, "must not exceed the round budget"
        assert result.stopped_early, "an unresolved run must be flagged, not presented as complete"
        assert result.open_gaps == ["still more"]
        assert result.final_answer


class TestGroundingOverridesVerifier:
    def test_lenient_verifier_is_overruled_by_grounding(self):
        """A verifier waving through fabricated citations must not end the loop."""
        cfg = settings(min_support_rate=0.6)
        bogus = ResearchDraft(
            claims=[
                Claim(text="Revenue grew 34%", citations=["S99"]),
                Claim(text="The CEO resigned in October", citations=["S98"]),
            ],
            draft_answer="Revenue grew 34%. The CEO resigned in October.",
        )
        llm = FakeLLM(cfg, [
            triage(),
            bogus,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),  # verifier says ship it
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        tracer = Tracer()
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg), tracer).run(QUESTION)

        assert len(result.rounds) == 2, "0% grounding must override a 'sufficient' verdict"
        assert result.rounds[0].grounding.support_rate == 0.0
        assert set(result.rounds[0].grounding.hallucinated_citations) == {"S98", "S99"}
        assert any(e.kind is EventKind.WARNING for e in tracer.events)

    def test_high_grounding_yields_high_confidence(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        assert AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION).confidence == "high"

    def test_only_cited_sources_are_returned(self):
        cfg = settings(top_k=4)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert [c.source_id for c in result.citations] == ["S1"]


class TestRouting:
    def test_web_route_forced_off_when_web_disabled(self):
        """Naive routing falls through to an empty local search here."""
        cfg = settings(web_enabled=False)
        llm = FakeLLM(cfg, [
            triage(route=Route.WEB),
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert result.rounds[0].route is Route.LOCAL

    def test_no_documents_forces_web(self):
        cfg = settings(web_enabled=True)
        researcher = AdaptiveResearcher(cfg, FakeLLM(cfg, []), corpus=None)
        assert researcher._initial_route(triage(route=Route.LOCAL)) is Route.WEB

    def test_escalate_to_web_becomes_hybrid_when_docs_exist(self):
        cfg = settings(web_enabled=True)
        researcher = AdaptiveResearcher(cfg, FakeLLM(cfg, []), corpus_for(cfg))
        route, *_ = researcher._escalate(
            verdict(Verdict.INSUFFICIENT, NextAction.ESCALATE_TO_WEB, gaps=["g"]),
            "q", "q", Route.LOCAL, 4, [], 1,
        )
        assert route is Route.HYBRID, "local evidence should be kept, not discarded"

    def test_impossible_escalation_still_makes_progress(self):
        # Verifier asks for web while web is off. The round must still differ
        # from the last one, or the budget is burned repeating a failure.
        cfg = settings(web_enabled=False, top_k=4)
        researcher = AdaptiveResearcher(cfg, FakeLLM(cfg, []), corpus_for(cfg))
        route, query, top_k, _ = researcher._escalate(
            verdict(Verdict.INSUFFICIENT, NextAction.ESCALATE_TO_WEB),
            "q", "q", Route.LOCAL, 4, [], 1,
        )
        assert (route, query, top_k) != (Route.LOCAL, "q", 4)
        assert top_k == 8


class TestResilience:
    def test_triage_failure_does_not_kill_the_run(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            LLMError("triage provider 500"),
            LLMError("triage provider 500"),
            LLMError("triage provider 500"),
            GOOD_DRAFT,
            verdict(Verdict.SUFFICIENT, NextAction.ANSWER),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert result.final_answer
        assert any("Triage agent failed" in w for w in result.warnings)

    def test_verifier_failure_degrades_to_partial(self):
        cfg = settings(max_rounds=1)
        llm = FakeLLM(cfg, [
            triage(),
            GOOD_DRAFT,
            LLMError("verifier down"), LLMError("verifier down"), LLMError("verifier down"),
            CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert result.rounds[0].verifier.verdict is Verdict.PARTIAL
        assert result.confidence != "high", "an unverified answer must not claim high confidence"

    def test_no_evidence_abstains_rather_than_answering(self):
        """Zero evidence must not report `answered`.

        The message body said evidence could not be found while the outcome
        said ANSWERED, so any caller branching on `is_answer` treated a
        non-answer as a result.
        """
        cfg = settings(web_enabled=False)
        llm = FakeLLM(cfg, [triage()])
        result = AdaptiveResearcher(cfg, llm, corpus=None).run(QUESTION)

        assert result.rounds == []
        assert result.outcome is Outcome.ABSTAINED
        assert not result.is_answer, "outcome must not contradict the message body"
        assert result.confidence == "low"

    def test_usage_is_accumulated(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        assert result.usage.calls == 4


class TestTrace:
    def test_events_cover_every_stage(self):
        cfg = settings()
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        tracer = Tracer()
        AdaptiveResearcher(cfg, llm, corpus_for(cfg), tracer).run(QUESTION)
        kinds = {e.kind for e in tracer.events}
        assert {
            EventKind.START, EventKind.TRIAGE, EventKind.RETRIEVE,
            EventKind.RESEARCH, EventKind.GROUND, EventKind.VERIFY,
            EventKind.SYNTHESIZE, EventKind.DONE,
        } <= kinds

    def test_broken_ui_callback_cannot_break_the_run(self):
        cfg = settings()

        def explode(_event):
            raise RuntimeError("UI thread died")

        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg), Tracer(on_event=explode)).run(QUESTION)
        assert result.final_answer


class TestSettings:
    @pytest.mark.parametrize(
        "model,expected", [("gpt-5-nano", False), ("gpt-5", False), ("o3-mini", False),
                           ("gpt-4o-mini", True), ("gpt-4.1", True)]
    )
    def test_temperature_support_by_model(self, model, expected):
        # Reasoning models reject an explicit temperature parameter.
        assert Settings(model=model).supports_temperature() is expected

    def test_blank_overrides_do_not_clobber_env(self, monkeypatch):
        monkeypatch.setenv("RAGVERIFY_MODEL", "gpt-4o")
        assert Settings.from_env(model="").model == "gpt-4o"
        assert Settings.from_env(model="gpt-5").model == "gpt-5"

    def test_api_key_never_touches_environ(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Settings(api_key="sk-secret")
        llm = FakeLLM(cfg, [triage(), GOOD_DRAFT,
                            verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER])
        AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)
        import os

        assert "OPENAI_API_KEY" not in os.environ


class TestDotenv:
    """`.env` is the documented setup path, so it must actually be read."""

    def test_dotenv_is_loaded(self, tmp_path, monkeypatch):
        import ragverify.config as config

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_DOTENV_LOADED", False)

        assert Settings.from_env().api_key == "sk-from-dotenv"

    def test_real_env_beats_dotenv(self, tmp_path, monkeypatch):
        # A stale checked-out .env must never shadow a CI secret.
        import ragverify.config as config

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environ")
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_DOTENV_LOADED", False)

        assert Settings.from_env().api_key == "sk-from-environ"

    def test_missing_dotenv_is_not_fatal(self, tmp_path, monkeypatch):
        import ragverify.config as config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_DOTENV_LOADED", False)
        Settings.from_env()  # must not raise
