"""OpenTelemetry integration — optional, and no-op by default.

A project that never enables tracing should not pay for the ability to, and
observability must never break the thing it observes.
"""

from __future__ import annotations

import pytest

from ragverify import telemetry


class TestNoOpByDefault:
    def test_disabled_until_configured(self):
        assert not telemetry.enabled()

    def test_span_is_a_working_context_manager_when_disabled(self):
        with telemetry.span("x", foo="bar") as current:
            assert current is None

    def test_helpers_are_safe_when_disabled(self):
        telemetry.set_attributes(None, a=1)
        telemetry.add_event("evt", a=1)

    def test_exceptions_propagate_through_a_noop_span(self):
        """Telemetry observes failures; it must not swallow them."""
        with pytest.raises(ValueError), telemetry.span("x"):
            raise ValueError("boom")

    def test_bridge_is_inert_when_disabled(self):
        from ragverify.trace import Tracer

        seen = []
        tracer = Tracer(on_event=seen.append)
        telemetry.bridge(tracer)
        tracer.emit(__import__("ragverify.trace", fromlist=["EventKind"]).EventKind.START, "hi")

        assert len(seen) == 1, "the original callback must keep working"


class TestAttributeCoercion:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("s", "s"), (1, 1), (1.5, 1.5), (True, True),
            (["a", "b"], ["a", "b"]),
            ((1, 2), ["1", "2"]),
        ],
    )
    def test_scalars_and_sequences(self, value, expected):
        assert telemetry._coerce(value) == expected

    def test_objects_become_strings(self):
        """OTel accepts only scalars and homogeneous sequences."""
        assert isinstance(telemetry._coerce({"a": 1}), str)


class TestConfigure:
    def test_configure_without_dependency_returns_false(self, monkeypatch):
        monkeypatch.setattr(telemetry, "available", lambda: False)
        assert telemetry.configure() is False
        assert not telemetry.enabled()


class TestRunStillWorks:
    def test_orchestrator_runs_with_telemetry_off(self):
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
        assert AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION).final_answer
