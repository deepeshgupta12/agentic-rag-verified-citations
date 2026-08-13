"""Multi-hop planning: retrieval chained on what earlier retrieval found.

Distinct from the parallel sub-query widening the loop already does. A
question is multi-hop only when one sub-question's *query text* cannot be
written until another has been answered.
"""

from __future__ import annotations

import pytest

from ragverify.planner import (
    MAX_SUB_QUESTIONS,
    ResearchPlan,
    SubQuestion,
    _topological_order,
    hop_coverage,
    plan_research,
    resolve_query,
    validate_plan,
)
from ragverify.schemas import Claim, EvidenceItem, Route


def sq(sid: int, text: str, deps=()) -> SubQuestion:
    return SubQuestion(id=sid, text=text, depends_on=list(deps))


class FakeClient:
    def __init__(self, plan=None, error=None):
        self.plan, self.error = plan, error

    def structured(self, system, user, schema, **kw):
        if self.error:
            raise RuntimeError(self.error)
        return self.plan


class TestValidation:
    def test_single_hop_plan_degrades(self):
        """A one-hop 'plan' is ordinary retrieval wearing a costume."""
        plan, _ = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[sq(0, "one")]))
        assert not plan.multi_hop

    def test_valid_dag_is_ordered(self):
        plan, warnings = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[
            sq(1, "growth for {{0}}", [0]),
            sq(0, "which offices exist"),
        ]))
        assert plan.multi_hop
        assert [s.id for s in plan.sub_questions] == [0, 1], "dependency must run first"
        assert not warnings

    def test_cycle_is_cut_not_executed(self):
        """An unvalidated cycle is an infinite loop."""
        plan, warnings = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[
            sq(0, "a", [1]), sq(1, "b", [0]),
        ]))
        assert any("cycle" in w for w in warnings)
        assert plan.multi_hop and len(plan.sub_questions) == 2
        assert all(not s.depends_on for s in plan.sub_questions)

    def test_dangling_dependency_dropped_hop_kept(self):
        """A hop with a missing dependency is still answerable."""
        plan, warnings = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[
            sq(0, "a"), sq(1, "b", [99]),
        ]))
        assert any("unknown" in w for w in warnings)
        assert len(plan.sub_questions) == 2
        assert plan.sub_questions[1].depends_on == []

    def test_self_dependency_removed(self):
        plan, warnings = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[
            sq(0, "a"), sq(1, "b", [1]),
        ]))
        assert any("self" in w for w in warnings)
        assert all(s.id not in s.depends_on for s in plan.sub_questions)

    def test_duplicate_ids_dropped(self):
        plan, warnings = validate_plan(ResearchPlan(multi_hop=True, sub_questions=[
            sq(0, "a"), sq(0, "duplicate"), sq(1, "b"),
        ]))
        assert any("Duplicate" in w for w in warnings)
        assert len({s.id for s in plan.sub_questions}) == len(plan.sub_questions)

    def test_runaway_breadth_truncated(self):
        plan, warnings = validate_plan(ResearchPlan(
            multi_hop=True, sub_questions=[sq(i, f"q{i}") for i in range(12)]
        ))
        assert len(plan.sub_questions) <= MAX_SUB_QUESTIONS
        assert any("truncated" in w for w in warnings)

    def test_non_multi_hop_passes_through(self):
        plan, _ = validate_plan(ResearchPlan(multi_hop=False, rationale="single"))
        assert not plan.multi_hop and plan.rationale == "single"


class TestTopologicalOrder:
    def test_chain(self):
        order, cyclic = _topological_order([sq(2, "c", [1]), sq(0, "a"), sq(1, "b", [0])])
        assert order == [0, 1, 2] and not cyclic

    def test_detects_cycle(self):
        _, cyclic = _topological_order([sq(0, "a", [1]), sq(1, "b", [0])])
        assert cyclic == {0, 1}

    def test_independent_nodes_are_deterministic(self):
        order, _ = _topological_order([sq(2, "c"), sq(0, "a"), sq(1, "b")])
        assert order == [0, 1, 2], "ordering must be stable across runs"


class TestQueryResolution:
    def test_placeholder_substituted(self):
        """This is what makes a hop chained rather than merely ordered."""
        findings = {0: [Claim(text="Berlin, Paris and Madrid", citations=["S1"])]}
        assert resolve_query(sq(1, "growth for {{0}}", [0]), findings) == (
            "growth for Berlin, Paris and Madrid"
        )

    @pytest.mark.parametrize("form", ["{{0}}", "{{ 0 }}", "{{hop 0}}", "{{answer #0}}"])
    def test_placeholder_forms(self, form):
        findings = {0: [Claim(text="Berlin", citations=["S1"])]}
        assert "Berlin" in resolve_query(sq(1, f"growth for {form}", [0]), findings)

    def test_dependency_without_placeholder_gets_context_appended(self):
        """Otherwise the hop searches as if nothing had been learned."""
        findings = {0: [Claim(text="Berlin office", citations=["S1"])]}
        resolved = resolve_query(sq(1, "what was growth", [0]), findings)
        assert "Berlin office" in resolved

    def test_unresolved_dependency_falls_back_to_text(self):
        resolved = resolve_query(sq(1, "growth for {{0}}", [0]), {})
        assert "{{0}}" not in resolved and resolved

    def test_no_dependencies_is_verbatim(self):
        assert resolve_query(sq(0, "which offices exist"), {}) == "which offices exist"


class TestCoverage:
    def _ev(self, n):
        return [
            EvidenceItem(source_id=f"S{i}", label="d", text="x", origin=Route.LOCAL)
            for i in range(n)
        ]

    def test_no_findings_is_zero(self):
        assert hop_coverage([], self._ev(3)) == 0.0

    def test_no_evidence_is_zero(self):
        assert hop_coverage([Claim(text="x", citations=["S0"])], []) == 0.0

    def test_grounded_findings_score(self):
        claims = [Claim(text="a", citations=["S0"]), Claim(text="b", citations=["S1"])]
        assert hop_coverage(claims, self._ev(3)) > 0.5


class TestPlanning:
    def test_failure_degrades_to_single_hop(self):
        plan = plan_research("q", "corpus", FakeClient(error="planner down"))
        assert not plan.multi_hop
        assert "unavailable" in plan.rationale


class TestInRun:
    """Per-sub-question coverage, not one aggregate that hides a failed hop."""

    def _run(self, plan, **kw):
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
        from ragverify.schemas import NextAction, ResearchDraft, Verdict

        cfg = settings(max_rounds=1, use_multi_hop=True, **kw)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        original = llm.structured

        def structured(system, user, schema, **kwargs):
            if schema is ResearchPlan:
                return plan
            if schema is ResearchDraft and "Question:" in user:
                return GOOD_DRAFT
            return original(system, user, schema, **kwargs)

        llm.structured = structured
        return AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)

    def test_single_hop_plan_records_nothing(self):
        result = self._run(ResearchPlan(multi_hop=False, rationale="single-hop"))
        assert result.plan == {}, "a single-hop question must not carry a plan"

    def test_multi_hop_records_per_hop_coverage(self):
        result = self._run(ResearchPlan(multi_hop=True, sub_questions=[
            SubQuestion(id=0, text="What was European revenue?"),
            SubQuestion(id=1, text="What was the margin for {{0}}?", depends_on=[0]),
        ]))
        assert result.plan["multi_hop"]
        assert len(result.plan["hops"]) == 2
        for hop in result.plan["hops"]:
            assert "coverage" in hop and "answered" in hop

    def test_disabled_by_default(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import settings

        assert not settings().use_multi_hop, "most questions are single-hop"
