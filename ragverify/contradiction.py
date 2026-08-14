"""Deterministic cross-source contradiction detection.

Grounding checks a claim against the source it cites. Entailment checks the
same pair semantically. Neither ever compares one source against another, so
a corpus containing "revenue was 2.1 billion" and "revenue was 1.6 billion"
produces a perfectly verified answer built on whichever passage retrieval
happened to rank first. Every check passes. The answer is still wrong, and
nothing in the pipeline is in a position to notice.

The synthesizer is instructed to surface conflicts, but an instruction is not
a detector: it only sees the passages that reached the prompt, and it decides
whether to mention a disagreement it may not have registered.

This module answers a narrower question mechanically -- *do two sources
assert different values for the same thing?* -- and it answers it without a
model, so a confident draft cannot talk it out of its verdict.

**What it detects**

* **Value conflicts.** Two passages about the same subject asserting
  different figures of the same type. Comparison is by canonical value, so
  "2.1 billion euro" and "EUR 2,100,000,000" agree while 2.1bn and 1.6bn do
  not, and a percentage is never compared against a headcount.
* **Polarity conflicts.** One passage asserts what another denies.

**What it deliberately does not do**

It does not adjudicate. Deciding which source is right needs recency,
authority and domain rules that vary by corpus, and guessing wrong is worse
than reporting the disagreement -- a silently chosen side is indistinguishable
from a verified fact. Detection is reported; the choice stays with the reader.

Precision is preferred to recall throughout. A false contradiction erodes
trust in every real one, so the topical bar is deliberately high and a pair
that is merely adjacent in subject is not compared at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .normalize import canonical_values
from .retrieval import STOPWORDS, analyze
from .schemas import EvidenceItem

# Two passages must be this topically similar before their figures are
# compared at all. Below it they are discussing different things, and
# different things are allowed to carry different numbers.
DEFAULT_MIN_OVERLAP = 0.22

# Content terms each side needs before a comparison means anything. Two
# three-word fragments can score high overlap by accident.
MIN_CONTENT_TERMS = 4

# Value types worth comparing, and how much topical agreement each needs.
# Percentages and money are high-signal: a document rarely carries two
# different percentages about the same subject innocently. Bare counts are
# noisy -- page numbers, section numbers, list sizes -- so they need more
# topical agreement before a difference counts as a conflict.
_TYPE_THRESHOLDS: dict[str, float] = {
    "pct": DEFAULT_MIN_OVERLAP,
    "money": DEFAULT_MIN_OVERLAP,
    "date": 0.30,
    "quarter": 0.30,
    "num": 0.42,
}

_NEGATION = re.compile(
    r"\b(?:not|never|no|none|neither|nor|without|failed\s+to|declined\s+to|"
    r"denied|refused|rejected|did\s?n[o']t|does\s?n[o']t|was\s?n[o']t|"
    r"were\s?n[o']t|is\s?n[o']t|are\s?n[o']t|cannot|can\s?not)\b",
    re.I,
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|\n+")


@dataclass
class Contradiction:
    """Two sources disagreeing about the same thing."""

    kind: str  # "value" | "polarity"
    source_a: str
    source_b: str
    value_type: str
    value_a: str
    value_b: str
    excerpt_a: str
    excerpt_b: str
    overlap: float
    shared_terms: list[str] = field(default_factory=list)

    def describe(self) -> str:
        subject = ", ".join(self.shared_terms[:5]) or "the same subject"
        if self.kind == "polarity":
            return (
                f"[{self.source_a}] and [{self.source_b}] disagree on whether "
                f"{subject} holds."
            )
        return (
            f"[{self.source_a}] and [{self.source_b}] give different "
            f"{self.value_type} values for {subject}: "
            f"{_readable(self.value_a)} vs {_readable(self.value_b)}."
        )


def _readable(token: str) -> str:
    """Turn a canonical token back into something a human reads."""
    parts = token.split(":")
    if parts[0] == "pct":
        return f"{parts[1]}%"
    if parts[0] == "money":
        return f"{parts[2]} {parts[1]}"
    return parts[-1]


def _content_terms(text: str) -> set[str]:
    return {t for t in analyze(text) if t not in STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _by_type(values: set[str]) -> dict[str, set[str]]:
    """Group canonical tokens by their type prefix.

    Money keeps its currency in the key, so USD and EUR amounts are never
    compared -- differing currencies are a unit mismatch, not a disagreement.
    """
    grouped: dict[str, set[str]] = {}
    for value in values:
        parts = value.split(":")
        key = f"{parts[0]}:{parts[1]}" if parts[0] == "money" and len(parts) > 2 else parts[0]
        grouped.setdefault(key, set()).add(value)
    return grouped


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) > 30]


def detect(
    evidence: Sequence[EvidenceItem],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_pairs: int = 4000,
) -> list[Contradiction]:
    """Find pairs of sources that assert different things about one subject.

    Comparison is sentence-level rather than passage-level: two long passages
    will almost always share *some* topical terms and *some* differing
    numbers, which at document granularity produces conflicts that are not
    there.
    """
    # Index sentences once, with their terms and values precomputed.
    units: list[tuple[str, str, set[str], set[str]]] = []
    for item in evidence:
        for sentence in _sentences(item.text):
            terms = _content_terms(sentence)
            if len(terms) >= MIN_CONTENT_TERMS:
                units.append((item.source_id, sentence, terms, canonical_values(sentence)))

    found: list[Contradiction] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    comparisons = 0

    for i, (src_a, sent_a, terms_a, values_a) in enumerate(units):
        for src_b, sent_b, terms_b, values_b in units[i + 1 :]:
            if src_a == src_b:
                continue  # a source disagreeing with itself is a different problem
            comparisons += 1
            if comparisons > max_pairs:
                return found

            overlap = _jaccard(terms_a, terms_b)
            if overlap < min_overlap:
                continue

            shared = sorted(terms_a & terms_b)

            # --- value conflicts ---------------------------------------
            grouped_a, grouped_b = _by_type(values_a), _by_type(values_b)
            for value_type, side_a in grouped_a.items():
                side_b = grouped_b.get(value_type)
                if not side_b:
                    continue
                if overlap < _TYPE_THRESHOLDS.get(value_type.split(":")[0], DEFAULT_MIN_OVERLAP):
                    continue
                # Any shared value means they agree on this measure; only a
                # fully disjoint set is a disagreement.
                if side_a & side_b:
                    continue

                key = (src_a, src_b, value_type, min(side_a), min(side_b))
                if key in seen:
                    continue
                seen.add(key)
                found.append(Contradiction(
                    kind="value", source_a=src_a, source_b=src_b,
                    value_type=value_type.split(":")[0],
                    value_a=sorted(side_a)[0], value_b=sorted(side_b)[0],
                    excerpt_a=sent_a[:220], excerpt_b=sent_b[:220],
                    overlap=round(overlap, 3), shared_terms=shared,
                ))

            # --- polarity conflicts ------------------------------------
            neg_a, neg_b = bool(_NEGATION.search(sent_a)), bool(_NEGATION.search(sent_b))
            if neg_a != neg_b and overlap >= 0.34 and not (values_a & values_b):
                key = (src_a, src_b, "polarity", "", "")
                if key not in seen:
                    seen.add(key)
                    found.append(Contradiction(
                        kind="polarity", source_a=src_a, source_b=src_b,
                        value_type="polarity",
                        value_a="asserted" if not neg_a else "denied",
                        value_b="asserted" if not neg_b else "denied",
                        excerpt_a=sent_a[:220], excerpt_b=sent_b[:220],
                        overlap=round(overlap, 3), shared_terms=shared,
                    ))

    return found


def summarize(conflicts: Sequence[Contradiction]) -> str:
    """One line for the run warnings."""
    if not conflicts:
        return ""
    pairs = {(c.source_a, c.source_b) for c in conflicts}
    return (
        f"{len(conflicts)} contradiction(s) detected across {len(pairs)} source pair(s). "
        "Sources disagree; the pipeline reports this rather than choosing between them."
    )


def as_prompt_block(conflicts: Sequence[Contradiction], limit: int = 6) -> str:
    """Conflicts rendered for the synthesis prompt.

    Given to the synthesizer so a disagreement it might not have noticed is
    stated explicitly, with the instruction to attribute rather than resolve.
    """
    if not conflicts:
        return ""
    lines = [
        "DETECTED CONTRADICTIONS — sources disagree. Do not choose a side.",
        "Attribute each figure to its source, or state that the sources conflict.",
        "",
    ]
    for conflict in conflicts[:limit]:
        lines.append(f"- {conflict.describe()}")
        lines.append(f"    [{conflict.source_a}] {conflict.excerpt_a}")
        lines.append(f"    [{conflict.source_b}] {conflict.excerpt_b}")
    return "\n".join(lines)
