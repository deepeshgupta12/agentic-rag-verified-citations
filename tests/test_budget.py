"""Budget caps and the circuit breaker.

A fixed-length pipeline is implicitly bounded by its fixed call count. An
adaptive loop removes that accidental safety: each round costs
more than the last, and a corpus that never satisfies the verifier keeps
buying rounds.
"""

from __future__ import annotations

import time

import pytest

from ragverify.budget import Budget, BudgetExceeded, CircuitBreaker


class TestBudget:
    def test_fresh_budget_has_room(self):
        assert Budget().exhausted() is None
        Budget().check()  # must not raise

    def test_cost_cap(self):
        budget = Budget(max_cost_usd=0.10)
        budget.record(calls=1, cost_usd=0.11)
        assert "cost budget exhausted" in budget.exhausted()
        with pytest.raises(BudgetExceeded):
            budget.check()

    def test_call_cap(self):
        budget = Budget(max_calls=3)
        budget.record(calls=3)
        assert "call budget exhausted" in budget.exhausted()

    def test_time_cap(self):
        budget = Budget(max_seconds=0.01)
        time.sleep(0.02)
        assert "time budget exhausted" in budget.exhausted()

    def test_can_afford_round_is_predictive(self):
        # Checked BEFORE escalating, so the loop stops at a clean boundary
        # with a usable answer rather than dying mid-round.
        budget = Budget(max_calls=10)
        budget.record(calls=8)
        assert budget.can_afford_round(estimated_calls=2)
        assert not budget.can_afford_round(estimated_calls=3)

    def test_can_afford_round_respects_time(self):
        budget = Budget(max_seconds=10.0)
        assert not budget.can_afford_round(estimated_seconds=20.0)

    def test_snapshot_reports_percentages(self):
        budget = Budget(max_cost_usd=1.0, max_calls=10)
        budget.record(calls=5, cost_usd=0.25)
        snap = budget.snapshot()
        assert snap["calls"] == 5
        assert snap["cost_usd"] == 0.25
        assert snap["pct_cost"] == 25.0

    def test_zero_cost_cap_does_not_divide_by_zero(self):
        assert Budget(max_cost_usd=0.0).snapshot()["pct_cost"] == 0.0


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(threshold=3)
        for _ in range(2):
            breaker.record_failure("searx")
        assert not breaker.is_open("searx")
        breaker.record_failure("searx")
        assert breaker.is_open("searx")

    def test_guard_raises_when_open(self):
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure("searx")
        with pytest.raises(BudgetExceeded, match="circuit breaker open"):
            breaker.guard("searx")

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=2)
        breaker.record_failure("searx")
        breaker.record_success("searx")
        breaker.record_failure("searx")
        assert not breaker.is_open("searx"), "the counter must reset on success"

    def test_half_open_after_cooldown(self):
        breaker = CircuitBreaker(threshold=1, cooldown_s=0.01)
        breaker.record_failure("searx")
        assert breaker.is_open("searx")
        time.sleep(0.02)
        assert not breaker.is_open("searx"), "cooldown must allow a probe through"

    def test_circuits_are_independent(self):
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure("searx")
        assert breaker.is_open("searx")
        assert not breaker.is_open("openai"), "one dead dependency must not disable others"

    def test_open_circuits_listed(self):
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure("searx")
        breaker.record_failure("ddg")
        assert set(breaker.open_circuits()) == {"searx", "ddg"}


class TestBreakerWiredIntoSearch:
    """The breaker existed but was never connected to a dependency."""

    def test_open_circuit_skips_the_endpoint(self, monkeypatch):
        from ragverify import websearch
        from ragverify.config import Settings

        attempted: list[str] = []

        def fake_searx(query, endpoint, settings):
            attempted.append(endpoint)
            raise RuntimeError("down")

        monkeypatch.setattr(websearch, "_searxng", fake_searx)
        monkeypatch.setattr(websearch, "_duckduckgo", lambda q, s: [])

        cfg = Settings(searx_endpoints=("https://a.test", "https://b.test"))
        breaker = CircuitBreaker(threshold=1)

        websearch.search("q1", cfg, breaker)
        first_round = len(attempted)
        websearch.search("q2", cfg, breaker)

        assert first_round == 2, "both endpoints tried on the first query"
        assert len(attempted) == 2, "second query must skip endpoints already known down"

    def test_success_keeps_the_circuit_closed(self, monkeypatch):
        from ragverify import websearch
        from ragverify.config import Settings
        from ragverify.schemas import WebResult

        monkeypatch.setattr(
            websearch, "_searxng",
            lambda q, e, s: [WebResult(url="https://x.test", title="t")],
        )
        cfg = Settings(searx_endpoints=("https://a.test",))
        breaker = CircuitBreaker(threshold=1)

        assert websearch.search("q1", cfg, breaker)
        assert websearch.search("q2", cfg, breaker), "a working endpoint stays available"


class TestFetchPathGuards:
    """Fetching was the last completely unmetered external call."""

    def test_open_circuit_skips_fetch(self, monkeypatch):
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        called = []
        monkeypatch.setattr(
            "ragverify.websearch._assert_safe_url", lambda u: called.append(u)
        )
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure("dead.example")

        assert fetch_page("https://dead.example/page", Settings(), breaker=breaker) == ""
        assert not called, "must not even validate a URL on an open circuit"

    def test_expired_deadline_skips_fetch(self, monkeypatch):
        import time

        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        called = []
        monkeypatch.setattr(
            "ragverify.websearch._assert_safe_url", lambda u: called.append(u)
        )
        past = time.monotonic() - 1
        assert fetch_page("https://example.com/x", Settings(), deadline=past) == ""
        assert not called

    def test_blocked_url_does_not_trip_the_circuit(self):
        """One bad link must not disable an otherwise working domain."""
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        breaker = CircuitBreaker(threshold=1)
        fetch_page("http://127.0.0.1/admin", Settings(), breaker=breaker)
        assert not breaker.is_open("127.0.0.1"), "an SSRF block is not a transport failure"

    def test_budget_exposes_a_single_deadline(self):
        # deadline is started_at + max_seconds exactly, so this is an equality
        # check, not a range. Float subtraction returns 30.000000000000004,
        # which a `<= 30` bound rejects for no real reason.
        budget = Budget(max_seconds=30.0)
        assert budget.deadline - budget.started_at == pytest.approx(30.0)
        assert budget.deadline > budget.started_at


class TestBudgetIsPerRun:
    """A client is reused across questions; a budget is not.

    Binding the budget once at construction let the first question's spend
    follow the client into every later one. Observed on a 12-question run:
    questions 3 onward each got 0.1s and abstained instantly, having
    inherited an already-exhausted clock.
    """

    def _researcher(self, client, cfg):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import corpus_for

        from ragverify.orchestrator import AdaptiveResearcher

        return AdaptiveResearcher(cfg, client, corpus_for(cfg))

    def _script(self, cfg):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import CITED_ANSWER, GOOD_DRAFT, FakeLLM, triage, verdict

        from ragverify.schemas import NextAction, Verdict

        return FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ] * 4)

    def test_second_question_gets_a_fresh_budget(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import QUESTION, settings

        cfg = settings(max_rounds=1, max_calls=6)
        client = self._script(cfg)

        first = self._researcher(client, cfg).run(QUESTION)
        second = self._researcher(client, cfg).run(QUESTION)

        assert first.is_answer, "first question must answer"
        assert second.is_answer, "second must not inherit the first's spent budget"
        assert second.usage.calls <= 4, "usage must be per-run, not cumulative"

    def test_client_budget_is_rebound_each_run(self):
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import QUESTION, settings

        cfg = settings(max_rounds=1)
        client = self._script(cfg)

        r1 = self._researcher(client, cfg)
        r1.run(QUESTION)
        r2 = self._researcher(client, cfg)
        r2.run(QUESTION)

        assert client.budget is r2.budget, "the client must track the current run's budget"
        assert client.budget is not r1.budget


class TestTruncationIsReported:
    """Truncation used to be silent.

    A long specification cut at the character cap loses whole sections while
    the remaining text still reads perfectly, so a question about a missing
    section gets a truthful "the sources do not state this" about a document
    that does. Nothing downstream can detect it, because by then the truncated
    text is all there is.
    """

    def _page(self, monkeypatch, body: str):
        from ragverify import websearch

        class Resp:
            status_code, is_redirect, encoding = 200, False, "utf-8"
            headers = {"Content-Type": "text/html"}

            def raise_for_status(self): ...
            def iter_content(self, n): yield body.encode()
            def close(self): ...

        monkeypatch.setattr(websearch, "_assert_safe_url", lambda u: None)
        monkeypatch.setattr(websearch, "_requests", lambda: type("R", (), {"get": lambda *a, **k: Resp()}))

    def test_truncation_is_recorded(self, monkeypatch):
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        self._page(monkeypatch, "<p>" + ("word " * 4000) + "</p>")
        report: list[dict] = []
        text = fetch_page("https://x.test/doc", Settings(fetch_max_chars=500), report=report)

        assert len(text) <= 500
        assert report and report[0]["truncated"] is True
        assert report[0]["total"] > report[0]["kept"]

    def test_short_page_is_not_flagged(self, monkeypatch):
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        self._page(monkeypatch, "<p>" + ("word " * 30) + "</p>")
        report: list[dict] = []
        fetch_page("https://x.test/doc", Settings(fetch_max_chars=20_000), report=report)

        assert report and report[0]["truncated"] is False

    def test_report_is_optional(self, monkeypatch):
        """Default None must preserve the previous behaviour exactly."""
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        self._page(monkeypatch, "<p>" + ("word " * 4000) + "</p>")
        assert fetch_page("https://x.test/doc", Settings(fetch_max_chars=500))

    def test_explicit_max_chars_still_honoured(self, monkeypatch):
        """The old positional argument keeps working."""
        from ragverify.config import Settings
        from ragverify.websearch import fetch_page

        self._page(monkeypatch, "<p>" + ("word " * 4000) + "</p>")
        assert len(fetch_page("https://x.test/doc", Settings(), 300)) <= 300

    def test_setting_default_matches_previous_hardcoded_value(self):
        from ragverify.config import Settings

        assert Settings().fetch_max_chars == 20_000
