"""Immutable evidence ledger.

Provenance metadata existed -- page numbers, URLs, chunk ids -- but a finished
run could not be *audited*. Asked six months later "what exactly did S3 say
when this answer was produced, and which claims rested on it?", nothing could
answer: the evidence lived only in memory, the sanitiser rewrote passages in
place with no record of the original, and the claim-to-source relationship
was implicit in a list of ids.

That matters for three concrete reasons.

* **Sources change.** A web page cited today may say something different
  tomorrow, or be gone. Without a content hash and a snapshot there is no way
  to tell a shifted source from a fabricated citation after the fact.
* **Sanitisation edits the evidence.** Injection neutralisation rewrites text
  that grounding then checks against. Keeping only the rewritten form means
  the thing that was verified is not the thing the document said.
* **Disputes need edges, not lists.** "Which claims did this passage support,
  and which citations were dropped for it?" is a graph query, and a flat list
  of ids cannot answer it.

The ledger records all of it, append-only, and serialises to JSON so a run can
be replayed, diffed, or handed to someone who was not there.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schemas import EvidenceItem, GroundingReport


def content_hash(text: str) -> str:
    """Stable SHA-256 prefix identifying an exact passage body."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceRecord:
    """An immutable snapshot of one passage as it was when used.

    Both the raw and sanitised bodies are kept. Grounding checks against the
    sanitised text, so that is what a verification result actually refers to;
    the raw text is what the document said. Auditing a disputed claim needs
    both, and a diff between them shows exactly what the injection filter
    changed.
    """

    source_id: str
    label: str
    origin: str
    retrieved_at: float
    raw_text: str
    raw_hash: str
    sanitized_text: str
    sanitized_hash: str
    url: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    # Character offsets of this passage within its parent document, so a
    # citation can be resolved back to a span rather than a whole file.
    span_start: int | None = None
    span_end: int | None = None
    retrieval_score: float = 0.0
    retrievers: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.raw_hash != self.sanitized_hash


@dataclass(frozen=True)
class ClaimEdge:
    """One claim-to-source relationship, with its verification outcome.

    ``verdict`` distinguishes the three ways a citation can fail, which a list
    of surviving ids collapses into one: ``supported`` passed, ``dropped``
    resolved to a real passage that did not support the claim, and
    ``fabricated`` named a source that was never retrieved. Only the last is
    a hallucination.
    """

    claim_text: str
    source_id: str
    verdict: str  # supported | dropped | fabricated
    round_index: int
    overlap: float = 0.0
    unsupported_values: list[str] = field(default_factory=list)


class EvidenceLedger:
    """Append-only record of everything a run relied on."""

    def __init__(self, question: str) -> None:
        self.question = question
        self.started_at = time.time()
        self.records: dict[str, EvidenceRecord] = {}
        self.edges: list[ClaimEdge] = []
        self._frozen = False

    # -- writing -------------------------------------------------------

    def record_evidence(
        self,
        item: EvidenceItem,
        raw_text: str | None = None,
        injection_flags: list[str] | None = None,
        span: tuple[int, int] | None = None,
        chunk_id: str | None = None,
        page: int | None = None,
        retrievers: list[str] | None = None,
    ) -> EvidenceRecord:
        """Snapshot one passage. Re-recording the same id is a no-op.

        Ids are stable within a run, and a later round retrieving the same
        passage must not overwrite the version an earlier claim was checked
        against -- that would make the ledger disagree with what actually
        happened.
        """
        if self._frozen:
            raise RuntimeError("ledger is frozen; a completed run cannot be amended")
        if item.source_id in self.records:
            return self.records[item.source_id]

        raw = raw_text if raw_text is not None else item.text
        record = EvidenceRecord(
            source_id=item.source_id,
            label=item.label,
            origin=item.origin.value,
            retrieved_at=time.time(),
            raw_text=raw,
            raw_hash=content_hash(raw),
            sanitized_text=item.text,
            sanitized_hash=content_hash(item.text),
            url=item.url,
            page=page,
            chunk_id=chunk_id,
            span_start=span[0] if span else None,
            span_end=span[1] if span else None,
            retrieval_score=item.score,
            retrievers=retrievers or [],
            injection_flags=injection_flags or [],
        )
        self.records[item.source_id] = record
        return record

    def record_grounding(self, report: GroundingReport, round_index: int) -> None:
        """Write one edge per claim-citation pair, including the failures."""
        if self._frozen:
            raise RuntimeError("ledger is frozen; a completed run cannot be amended")

        from . import normalize
        from .grounding import claim_support

        for claim in report.supported:
            for cid in claim.citations:
                record = self.records.get(cid)
                overlap = (
                    claim_support(claim.text, record.sanitized_text)[1] if record else 0.0
                )
                self.edges.append(
                    ClaimEdge(claim.text, cid, "supported", round_index, round(overlap, 3))
                )

        for claim in report.unsupported:
            for cid in claim.citations:
                self._record_failed_edge(claim, cid, report, round_index, normalize, claim_support)

        # Dropped citations attached to otherwise-supported claims.
        for cid in report.dropped_citations:
            if not any(e.source_id == cid for e in self.edges):
                self.edges.append(ClaimEdge("", cid, "dropped", round_index))

    def _record_failed_edge(self, claim, cid, report, round_index, normalize, claim_support) -> None:
        verdict = "fabricated" if cid in report.hallucinated_citations else "dropped"
        record = self.records.get(cid)
        overlap, missing = 0.0, []
        if record is not None:
            overlap = claim_support(claim.text, record.sanitized_text)[1]
            missing = sorted(normalize.unsupported_values(claim.text, record.sanitized_text))
        self.edges.append(
            ClaimEdge(claim.text, cid, verdict, round_index, round(overlap, 3), missing)
        )

    def freeze(self) -> None:
        self._frozen = True

    # -- reading -------------------------------------------------------

    def sources_for(self, claim_text: str, verdict: str = "supported") -> list[str]:
        return [e.source_id for e in self.edges if e.claim_text == claim_text and e.verdict == verdict]

    def claims_from(self, source_id: str, verdict: str = "supported") -> list[str]:
        return [e.claim_text for e in self.edges if e.source_id == source_id and e.verdict == verdict]

    def modified_sources(self) -> list[EvidenceRecord]:
        """Passages the injection filter rewrote before verification."""
        return [r for r in self.records.values() if r.was_modified]

    def verify_integrity(self) -> list[str]:
        """Re-hash every snapshot and report any that no longer match.

        Cheap insurance against a passage being mutated in place after it was
        recorded -- the ledger's whole value is that its contents are what was
        actually used.
        """
        broken = []
        for record in self.records.values():
            if content_hash(record.raw_text) != record.raw_hash:
                broken.append(f"{record.source_id}: raw text does not match its hash")
            if content_hash(record.sanitized_text) != record.sanitized_hash:
                broken.append(f"{record.source_id}: sanitized text does not match its hash")
        return broken

    # -- serialisation -------------------------------------------------

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        records = []
        for record in self.records.values():
            payload = asdict(record)
            if not include_text:
                # Hashes alone still prove which passage was used, without
                # copying document content into a shareable artifact.
                payload.pop("raw_text")
                payload.pop("sanitized_text")
            records.append(payload)

        return {
            "question": self.question,
            "started_at": self.started_at,
            "evidence": records,
            "edges": [asdict(e) for e in self.edges],
            "summary": {
                "sources": len(self.records),
                "edges": len(self.edges),
                "supported": sum(e.verdict == "supported" for e in self.edges),
                "dropped": sum(e.verdict == "dropped" for e in self.edges),
                "fabricated": sum(e.verdict == "fabricated" for e in self.edges),
                "modified_by_sanitizer": len(self.modified_sources()),
            },
        }

    def save(self, path: str | Path, include_text: bool = True) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(include_text), indent=2), encoding="utf-8")
        return target
