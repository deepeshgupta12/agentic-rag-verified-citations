"""Hard budget caps and a circuit breaker.

A fixed-length pipeline caps cost implicitly by having a fixed number of
calls. That stops being true the moment the loop can iterate, and an adaptive
system that escalates on failure is precisely the kind that can run away: each round retrieves more, prompts
grow, and a corpus that never satisfies the verifier keeps buying rounds.

Three independent limits, because they fail differently. Time protects the
user's patience, cost protects their bill, and calls protect against a bug in
the loop itself.

The breaker exists for a different failure: when a dependency is genuinely
down, retrying it once per round converts one failure into `max_rounds`
timeouts. After a threshold the breaker opens and calls fail immediately, so
degradation is fast rather than slow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a hard limit is hit. Callers finish with partial results."""


@dataclass
class Budget:
    """Tracks spend against caps for a single run."""

    max_seconds: float = 180.0
    max_cost_usd: float = 1.00
    max_calls: int = 40

    started_at: float = field(default_factory=time.monotonic)
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def deadline(self) -> float:
        """Absolute monotonic time this run must stop by.

        A single deadline propagated to every external call is what makes the
        time cap real. Per-request timeouts bound one call each and compose
        into an unbounded total: five fetches at 12s is a 60s worst case that
        no per-call setting can prevent.
        """
        return self.started_at + self.max_seconds

    def exhausted(self) -> str | None:
        """Why the budget is spent, or None while it still has room."""
        if self.elapsed >= self.max_seconds:
            return f"time budget exhausted ({self.elapsed:.0f}s of {self.max_seconds:.0f}s)"
        if self.cost_usd >= self.max_cost_usd:
            return f"cost budget exhausted (${self.cost_usd:.4f} of ${self.max_cost_usd:.2f})"
        if self.calls >= self.max_calls:
            return f"call budget exhausted ({self.calls} of {self.max_calls})"
        return None

    def check(self) -> None:
        reason = self.exhausted()
        if reason:
            raise BudgetExceeded(reason)

    def can_afford_round(self, estimated_calls: int = 3, estimated_seconds: float = 20.0) -> bool:
        """Is there room for another round?

        Checked before escalating rather than after, so the loop stops at a
        clean boundary with a usable answer instead of dying mid-round.
        """
        return (
            self.calls + estimated_calls <= self.max_calls
            and self.elapsed + estimated_seconds <= self.max_seconds
            and self.cost_usd < self.max_cost_usd
        )

    def record(self, calls: int = 1, cost_usd: float = 0.0) -> None:
        self.calls += calls
        self.cost_usd += cost_usd

    def snapshot(self) -> dict[str, float]:
        return {
            "elapsed_s": round(self.elapsed, 1),
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 4),
            "pct_time": round(100 * self.elapsed / self.max_seconds, 1),
            "pct_cost": round(100 * self.cost_usd / self.max_cost_usd, 1) if self.max_cost_usd else 0.0,
        }


class CircuitBreaker:
    """Per-dependency breaker with a cooldown.

    Deliberately simple: closed → open after ``threshold`` consecutive
    failures, half-open after ``cooldown``, and a single success closes it
    again. There is no rolling window because a run is short enough that
    consecutive failures are the signal that matters.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        opened = self._opened_at.get(name)
        if opened is None:
            return False
        if time.monotonic() - opened >= self.cooldown_s:
            # Half-open: allow one probe through to test recovery.
            del self._opened_at[name]
            self._failures[name] = self.threshold - 1
            return False
        return True

    def record_success(self, name: str) -> None:
        self._failures.pop(name, None)
        self._opened_at.pop(name, None)

    def record_failure(self, name: str) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count >= self.threshold:
            self._opened_at[name] = time.monotonic()

    def guard(self, name: str) -> None:
        if self.is_open(name):
            raise BudgetExceeded(f"circuit breaker open for {name}; skipping")

    def open_circuits(self) -> list[str]:
        return [name for name in list(self._opened_at) if self.is_open(name)]
