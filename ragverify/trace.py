"""Run event stream.

Wrapping a multi-step pipeline in one spinner shows nothing until every step
has finished, so a run that takes a minute is indistinguishable from a hung
one. The orchestrator emits events instead, and any frontend (Streamlit, CLI,
tests) subscribes.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    START = "start"
    TRIAGE = "triage"
    RETRIEVE = "retrieve"
    RESEARCH = "research"
    GROUND = "ground"
    VERIFY = "verify"
    ESCALATE = "escalate"
    SYNTHESIZE = "synthesize"
    WARNING = "warning"
    DONE = "done"


@dataclass
class Event:
    kind: EventKind
    message: str
    round_index: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


class Tracer:
    """Collects events and forwards them to an optional live callback."""

    def __init__(self, on_event: Callable[[Event], None] | None = None) -> None:
        self.events: list[Event] = []
        self._on_event = on_event
        self._t0 = time.time()

    def emit(self, kind: EventKind, message: str, round_index: int = 0, **data: Any) -> Event:
        event = Event(kind=kind, message=message, round_index=round_index, data=data)
        self.events.append(event)
        if self._on_event is not None:
            # A broken UI callback must never take down the research run.
            with contextlib.suppress(Exception):
                self._on_event(event)
        return event

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def warnings(self) -> list[str]:
        return [e.message for e in self.events if e.kind is EventKind.WARNING]
