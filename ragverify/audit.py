"""Append-only audit log: who asked what, and which documents answered.

The evidence ledger records what a run *used* — hashes, spans, claim-to-source
edges — so an answer can be reconstructed. That is a debugging artefact. It
answers "how did this answer come about" and nothing about who was entitled
to it.

An audit log answers a different question, asked later and usually by someone
else: *who saw this document, when, and under what authority?* It is written
for the case where that question is contested, which drives every decision
here.

**Append-only.** Records are written and never edited. A log that can be
corrected after the fact is evidence of nothing, because the correction and
the falsification are indistinguishable.

**Hash-chained.** Each record carries the hash of the one before it, so
removing or altering a record breaks the chain at that point and every
verification afterwards reports where. Tamper-evident, not tamper-proof:
anyone who can rewrite the file can recompute the chain. It raises the cost
of silent alteration and makes casual alteration detectable, which is what a
local file can honestly offer. Real integrity needs an append-only store the
writer cannot rewrite.

**No document content.** Names, hashes and ids only. An audit log that quotes
the material it is protecting becomes a second copy of it, outside whatever
controls the original had.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AuditEvent(str, Enum):
    QUERY = "query"                 # a principal asked a question
    ACCESS_GRANTED = "access"       # documents entered retrieval
    ACCESS_DENIED = "denied"        # documents withheld by policy
    ANSWERED = "answered"           # an answer was returned
    ABSTAINED = "abstained"         # the pipeline declined
    INJECTION = "injection"         # a source attempted prompt injection
    CONTRADICTION = "contradiction" # sources disagreed


@dataclass
class AuditRecord:
    event: AuditEvent
    subject: str
    tenant: str
    at: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)
    prev_hash: str = ""
    record_hash: str = ""

    def payload(self) -> dict:
        return {
            "event": self.event.value,
            "subject": self.subject,
            "tenant": self.tenant,
            "at": round(self.at, 3),
            "detail": self.detail,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        # sort_keys so the digest does not depend on dict ordering: a chain
        # that breaks on a serialisation detail is worse than no chain,
        # because it cries tamper at every reader.
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class AuditLog:
    """Append-only, hash-chained JSONL.

    Every record is flushed immediately. Buffering would lose exactly the
    records that matter most -- the ones written just before a crash.
    """

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self.records: list[AuditRecord] = []
        self._last_hash = ""

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._last_hash = self._tail_hash()

    def _tail_hash(self) -> str:
        """Continue an existing chain rather than starting a new one."""
        if not self.path or not self.path.exists():
            return ""
        last = ""
        with self.path.open() as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return ""
        try:
            return json.loads(last).get("record_hash", "")
        except json.JSONDecodeError:
            return ""

    def write(
        self,
        event: AuditEvent,
        subject: str,
        tenant: str,
        **detail,
    ) -> AuditRecord | None:
        record = AuditRecord(
            event=event, subject=subject, tenant=tenant,
            detail=detail, prev_hash=self._last_hash,
        )
        record.record_hash = record.compute_hash()
        self.records.append(record)
        self._last_hash = record.record_hash

        if self.enabled:
            line = json.dumps({**record.payload(), "record_hash": record.record_hash})
            with self.path.open("a") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record

    # -- verification --------------------------------------------------

    @staticmethod
    def verify(path: str | Path) -> dict:
        """Check the chain. Reports where it breaks, not merely that it did.

        A verifier that says only "invalid" is unusable in the situation it
        exists for: someone needs to know which records are still trustworthy
        and from which point the log stopped being evidence.
        """
        path = Path(path)
        if not path.exists():
            return {"valid": False, "error": "log not found", "records": 0}

        expected_prev = ""
        count = 0
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return {"valid": False, "records": count,
                        "broken_at": number, "error": "unparseable record"}

            stored = data.pop("record_hash", "")
            if data.get("prev_hash", "") != expected_prev:
                return {"valid": False, "records": count, "broken_at": number,
                        "error": "chain does not link to the previous record"}

            recomputed = hashlib.sha256(
                json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if recomputed != stored:
                return {"valid": False, "records": count, "broken_at": number,
                        "error": "record content does not match its hash"}
            expected_prev = stored

        return {"valid": True, "records": count, "broken_at": None}

    # -- querying ------------------------------------------------------

    @staticmethod
    def read(path: str | Path, tenant: str | None = None) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tenant is None or record.get("tenant") == tenant:
                out.append(record)
        return out


def record_run(
    log: AuditLog,
    principal,
    question: str,
    result,
    granted: Sequence[str] = (),
    denied: Iterable[str] = (),
) -> None:
    """Write the audit trail for one completed run.

    The question is hashed rather than stored. It is frequently sensitive in
    its own right -- "why was X fired", "does Y have a diagnosis of Z" -- and
    an audit log that records it verbatim becomes a second copy of exactly
    what the access controls were protecting.
    """
    question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    subject, tenant = principal.subject, principal.tenant

    log.write(AuditEvent.QUERY, subject, tenant,
              question_hash=question_hash, question_chars=len(question))

    if granted:
        log.write(AuditEvent.ACCESS_GRANTED, subject, tenant,
                  documents=sorted(set(granted)), question_hash=question_hash)
    denied = sorted(set(denied))
    if denied:
        log.write(AuditEvent.ACCESS_DENIED, subject, tenant,
                  documents=denied, question_hash=question_hash)

    if result.injections_detected:
        log.write(AuditEvent.INJECTION, subject, tenant,
                  kinds=result.injections_detected, question_hash=question_hash)
    if result.contradictions:
        log.write(AuditEvent.CONTRADICTION, subject, tenant,
                  count=len(result.contradictions), question_hash=question_hash)

    event = AuditEvent.ANSWERED if result.is_answer else AuditEvent.ABSTAINED
    log.write(
        event, subject, tenant,
        question_hash=question_hash,
        outcome=result.outcome.value,
        confidence=result.confidence,
        cited=[c.source_id for c in result.citations],
        documents=sorted({c.label.split(" ")[0] for c in result.citations}),
        cost_usd=round(result.usage.cost_usd, 4),
    )
