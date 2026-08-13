"""Entailment checking: does the source *mean* what the claim says?

Lexical grounding answers a narrow question -- do the claim's words and
figures appear in the cited passage -- and it answers that question without a
model, which is why it cannot be argued out of its verdict. But it is a bag of
words, and bags of words have a specific blind spot:

    source: "revenue grew in most European regions"
    claim:  "revenue grew in all European regions"

Every content word matches. There are no figures to compare. Lexical grounding
passes it, and the claim is false. The same hole covers negation ("did not
approve" vs "approved"), scope creep ("the Berlin office" vs "the company"),
tense and modality ("expects to launch" vs "launched"), and attribution ("the
CEO said X" vs "X is true").

Those are exactly the errors a careful reader catches and a keyword matcher
never can, so catching them needs something that reads. This module asks a
model the narrow, checkable question -- *does this passage entail this claim?*
-- rather than the open-ended "is this answer good?", which is the question
models are worst at judging about their own team's work.

Design constraints that follow from cost and trust:

* **It runs second.** Lexical grounding is free and rejects fabricated
  citations and invented numbers outright. Entailment only sees claims that
  already passed, so the expensive check runs on the smallest set.
* **It can only downgrade.** A claim rejected lexically is never revived by
  entailment. The deterministic layer stays the floor, so adding this cannot
  make the pipeline accept something it previously refused.
* **Failure is neutral, not fatal.** If the call errors, claims keep their
  lexical verdict and the run records a warning. An unavailable optional
  check must not fail a run or silently strengthen its claims.
* **It is optional.** Off by default: it costs a call per round and roughly
  doubles verification latency.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from .schemas import Claim, EvidenceItem

log = logging.getLogger("ragverify.entailment")


class EntailmentLabel(str, Enum):
    """Standard three-way NLI labels.

    ``NEUTRAL`` is the one that matters most here: it is where
    over-generalisation lands. "Most" does not entail "all", but neither does
    it contradict it, and collapsing neutral into either bucket loses exactly
    the distinction this module exists to draw.
    """

    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"


class ClaimVerdict(BaseModel):
    index: int = Field(description="Index of the claim being judged")
    label: EntailmentLabel
    # The span the judgement rests on. Requiring a quote makes the verdict
    # checkable -- a quote that does not appear in the passage exposes a
    # judgement that was not actually grounded in it.
    quote: str = Field(default="", max_length=400)
    reason: str = Field(default="", max_length=300)


class EntailmentBatch(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class EntailmentReport(BaseModel):
    """Outcome of the entailment pass over one round's claims."""

    entailed: list[Claim] = Field(default_factory=list)
    contradicted: list[Claim] = Field(default_factory=list)
    neutral: list[Claim] = Field(default_factory=list)
    # Verdicts whose quote could not be found in the cited passage. The model
    # judged, but not demonstrably from the source.
    unverifiable_quotes: list[str] = Field(default_factory=list)
    ran: bool = False
    error: str = ""

    @property
    def total(self) -> int:
        return len(self.entailed) + len(self.contradicted) + len(self.neutral)

    @property
    def entailment_rate(self) -> float:
        return len(self.entailed) / self.total if self.total else 0.0

    @property
    def has_contradiction(self) -> bool:
        """A contradiction is qualitatively worse than weak support.

        Weak support means the evidence is thin; a contradiction means the
        answer asserts the opposite of its own source, and no amount of
        further retrieval fixes that claim.
        """
        return bool(self.contradicted)


SYSTEM = """\
You are a natural-language-inference judge. For each claim you are given the
exact passage it cites. Decide only whether that passage supports the claim.

Labels:
- "entailed"      A careful reader of the passage must conclude the claim is
                  true. The passage may paraphrase; wording need not match.
- "contradicted"  The passage states something incompatible with the claim.
- "neutral"       The passage neither establishes nor refutes the claim.

Judge ONLY against the passage. Your own knowledge of the subject is
irrelevant and must not influence the label — a claim you know to be true is
still "neutral" if this passage does not establish it.

These are the distinctions that matter, and all of them are "neutral" or
"contradicted", never "entailed":
- Scope: "most regions" does not establish "all regions". "The Berlin office"
  does not establish "the company".
- Quantity: a larger or rounder number is not established by a smaller or
  more precise one.
- Modality and tense: "expects to launch" does not establish "launched".
  "May reduce" does not establish "reduces".
- Attribution: "the CEO said profits rose" establishes that the CEO said it,
  not that profits rose.
- Causation: two facts stated together do not establish that one caused the
  other.

Quote the exact span your judgement rests on, copied verbatim from the
passage. If no span is relevant, the label is "neutral" and the quote empty.
Return one verdict per claim, using the claim's given index."""


def _prompt(pairs: Sequence[tuple[int, Claim, str]]) -> str:
    blocks = []
    for index, claim, source in pairs:
        blocks.append(
            f"### Claim {index}\n{claim.text}\n\n"
            f"Cited passage:\n\"\"\"\n{source[:2500]}\n\"\"\""
        )
    return "\n\n".join(blocks)


def check_entailment(
    claims: Sequence[Claim],
    evidence: Sequence[EvidenceItem],
    client,
    max_claims: int = 12,
) -> EntailmentReport:
    """Judge whether each claim is entailed by the passage it cites.

    Only claims that already passed lexical grounding should be passed in --
    this is the second, expensive stage of a two-stage filter.
    """
    by_id = {e.source_id: e for e in evidence}

    pairs: list[tuple[int, Claim, str]] = []
    for index, claim in enumerate(claims[:max_claims]):
        source = next((by_id[c].text for c in claim.citations if c in by_id), None)
        if source:
            pairs.append((index, claim, source))

    if not pairs:
        return EntailmentReport(ran=False)

    try:
        batch = client.structured(SYSTEM, _prompt(pairs), EntailmentBatch)
    except Exception as exc:  # noqa: BLE001 - optional check, never fatal
        log.warning("entailment check failed: %s", exc)
        return EntailmentReport(ran=False, error=str(exc))

    verdicts = {v.index: v for v in batch.verdicts}
    report = EntailmentReport(ran=True)

    for index, claim, source in pairs:
        verdict = verdicts.get(index)
        if verdict is None:
            # No judgement returned: leave the lexical verdict standing rather
            # than inventing one.
            report.neutral.append(claim)
            continue

        # A quote that is not in the passage means the judgement was not made
        # from the source. Downgrade rather than trust it.
        if verdict.quote and not _quote_present(verdict.quote, source):
            report.unverifiable_quotes.append(verdict.quote[:120])
            report.neutral.append(claim)
            continue

        if verdict.label is EntailmentLabel.ENTAILED:
            report.entailed.append(claim)
        elif verdict.label is EntailmentLabel.CONTRADICTED:
            report.contradicted.append(claim)
        else:
            report.neutral.append(claim)

    return report


def _quote_present(quote: str, source: str) -> bool:
    """Is ``quote`` actually in ``source``, allowing whitespace drift?

    Models normalise whitespace and ellipsise when copying, so an exact match
    is too strict; a normalised substring check is the useful middle. Very
    short quotes are accepted because they carry no evidential weight either
    way and rejecting them would only add noise.
    """
    if len(quote.strip()) < 12:
        return True
    normalise = lambda s: " ".join(s.lower().split())  # noqa: E731
    needle, haystack = normalise(quote), normalise(source)
    if needle in haystack:
        return True
    # Elided quotes: check the head, which is where models truncate least.
    head = needle.split("...")[0].strip()
    return len(head) >= 12 and head in haystack
