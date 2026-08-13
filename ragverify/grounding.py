"""Deterministic verification that citations point at real supporting text.

Asking a synthesizer for "citations to the evidence" while handing it
model-written ``{source, summary}`` pairs rather than source text fails two
ways:

* A citation could name a source that was never retrieved, and nothing
  checked. The label looked identical to a real one in the UI.
* The verifier judged the draft against the researcher's *own summary* of the
  evidence, so a fact invented during the research step was invisible: the
  summary agreed with the draft because the same model wrote both.

This module runs before the verifier and answers, without an LLM, a narrower
question the model should not be trusted to self-report: does each cited
source exist, and does its text actually contain the claim's content? Being
deterministic matters -- it is the one signal in the loop that cannot be
talked out of its answer by a confident draft.

The check is intentionally a *recall* filter, not an entailment model. It
catches fabricated citations, cited-but-irrelevant sources, and invented
numbers. It cannot catch a subtle misreading of text that is genuinely there,
and does not claim to.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .retrieval import STOPWORDS, analyze
from .schemas import Claim, EvidenceItem, GroundingReport

# Numbers, percentages, years, money. These are what models most often
# fabricate while otherwise paraphrasing the source correctly, and they are
# cheap to verify exactly.
_NUMERIC = re.compile(r"\$?\d[\d,]*\.?\d*%?")

# Default share of a claim's content words that must appear in the cited
# source. Tuned on the eval set in ``evals/``: lower admits paraphrase but
# lets unrelated citations through, higher rejects legitimate rewording.
DEFAULT_OVERLAP_THRESHOLD = 0.45


def _content_terms(text: str) -> set[str]:
    return {t for t in analyze(text) if t not in STOPWORDS and len(t) > 2}


def _numbers(text: str) -> set[str]:
    return {m.group(0).rstrip(".").replace(",", "") for m in _NUMERIC.finditer(text)}


def claim_support(
    claim_text: str,
    source_text: str,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> tuple[bool, float]:
    """Does ``source_text`` support ``claim_text``?

    Returns ``(supported, overlap_ratio)``. Support requires both that enough
    of the claim's content words appear in the source, and that every number
    in the claim appears in the source -- a claim asserting "revenue grew 34%"
    against a source that never says 34 is rejected regardless of how well the
    surrounding prose matches.
    """
    claim_terms = _content_terms(claim_text)
    if not claim_terms:
        return False, 0.0

    source_terms = _content_terms(source_text)
    overlap = len(claim_terms & source_terms) / len(claim_terms)

    claim_numbers = _numbers(claim_text)
    if claim_numbers and not claim_numbers.issubset(_numbers(source_text)):
        return False, overlap

    return overlap >= threshold, overlap


def check(
    claims: Sequence[Claim],
    evidence: Sequence[EvidenceItem],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> GroundingReport:
    """Partition ``claims`` into supported and unsupported.

    A claim is supported when at least one of its citations both resolves to a
    real evidence item and passes ``claim_support``. Citations naming an id
    that was never retrieved are collected separately -- those are outright
    fabrications and are reported to the verifier as such, because they mean
    something stronger than "weak evidence".
    """
    by_id: dict[str, EvidenceItem] = {e.source_id: e for e in evidence}
    supported: list[Claim] = []
    unsupported: list[Claim] = []
    hallucinated: list[str] = []

    for claim in claims:
        if not claim.citations:
            unsupported.append(claim)
            continue

        resolved: list[EvidenceItem] = []
        for cid in claim.citations:
            item = by_id.get(cid) or by_id.get(_normalize_id(cid, by_id))
            if item is None:
                hallucinated.append(cid)
            else:
                resolved.append(item)

        if any(claim_support(claim.text, item.text, threshold)[0] for item in resolved):
            supported.append(claim)
        else:
            unsupported.append(claim)

    return GroundingReport(
        supported=supported,
        unsupported=unsupported,
        hallucinated_citations=sorted(set(hallucinated)),
    )


def _normalize_id(cid: str, by_id: dict[str, EvidenceItem]) -> str:
    """Recover from cosmetic citation-format drift.

    Models reliably write ``[S3]`` or ``S3.`` when asked for ``S3``. Rejecting
    those as hallucinations would swamp the real signal, so bracket and
    punctuation noise is stripped before a citation is declared fabricated.
    Anything that still fails to resolve is a genuine invention.
    """
    stripped = cid.strip().strip("[]()<>.,; ").upper()
    if stripped in by_id:
        return stripped
    for key in by_id:
        if key.upper() == stripped:
            return key
    return cid


def strip_unsupported(answer: str, unsupported: Iterable[Claim]) -> str:
    """Annotate rather than delete unsupported sentences.

    Silently dropping them would leave a fluent answer with holes the reader
    cannot see. Marking them keeps the reader's trust calibrated to what the
    evidence actually shows.
    """
    out = answer
    for claim in unsupported:
        needle = claim.text.strip()
        if needle and needle in out:
            out = out.replace(needle, f"{needle} _[unverified]_")
    return out
