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
from .schemas import AnswerAudit, Claim, EvidenceItem, GroundingReport

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

    Verification is **per citation**, not per claim. Accepting a claim because
    *any* one of its citations supports it lets an irrelevant source ride along
    on a good one and still be displayed to the user as a source for that
    claim. Each citation is therefore tested independently, and a supported
    claim keeps only the citations that actually carry it.

    Citations are also rewritten to their resolved, canonical ids. Grounding
    tolerates ``[S1]`` and ``S1.``; without rewriting, a downstream exact-match
    lookup silently drops those and the citation disappears from the answer.

    Citations naming an id that was never retrieved are collected separately --
    those are outright fabrications, which is a stronger signal than "weak
    evidence" and is reported to the verifier as such.
    """
    by_id: dict[str, EvidenceItem] = {e.source_id: e for e in evidence}
    supported: list[Claim] = []
    unsupported: list[Claim] = []
    hallucinated: list[str] = []
    dropped: list[str] = []

    for claim in claims:
        if not claim.citations:
            unsupported.append(claim)
            continue

        supporting: list[str] = []
        for cid in claim.citations:
            canonical = cid if cid in by_id else _normalize_id(cid, by_id)
            item = by_id.get(canonical)
            if item is None:
                hallucinated.append(cid)
            elif claim_support(claim.text, item.text, threshold)[0]:
                if canonical not in supporting:
                    supporting.append(canonical)
            else:
                # Resolves to a real passage that does not support the claim.
                dropped.append(canonical)

        if supporting:
            supported.append(claim.model_copy(update={"citations": supporting}))
        else:
            unsupported.append(claim)

    return GroundingReport(
        supported=supported,
        unsupported=unsupported,
        hallucinated_citations=sorted(set(hallucinated)),
        dropped_citations=sorted(set(dropped)),
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


# Sentence boundaries, with one subtlety that matters a great deal here: a
# citation following terminal punctuation -- "revenue grew 34%. [S1]" -- belongs
# to the sentence BEFORE it. Treating "[" as the start of a new sentence orphans
# every trailing citation, leaving the claim uncited and the bracket stranded as
# a contentless "sentence" that can never verify. That silently fails correct
# answers, which is the worst direction for this check to be wrong in.
_SENTENCE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'])"   # ordinary sentence end
    r"|(?<=\])\s+(?=[A-Z\"'])"     # end of a trailing citation
    r"|\n+"                        # explicit line breaks
)

# Statements asserting that the sources do NOT contain something. Unverifiable
# by construction, and the pipeline is supposed to produce them.
_ABSENCE = re.compile(
    r"\b(?:not\s+(?:provided|available|disclosed|stated|mentioned|included|covered|specified|"
    r"reported|given|found|present)|does\s+not|do\s+not|did\s+not|doesn't|don't|didn't|"
    r"no\s+(?:forecast|guidance|data|information|figure|mention|estimate|projection|detail)|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|cannot\s+be|could\s+not\s+be|"
    r"absent\s+from|silent\s+on|beyond\s+the\s+scope|outside\s+the\s+scope)\b",
    re.I,
)
_INLINE_CITE = re.compile(r"\[([A-Za-z]{1,2}\d{1,3}(?:\s*,\s*[A-Za-z]{1,2}\d{1,3})*)\]")


def extract_inline_citations(text: str) -> list[str]:
    """Pull ``[S1]`` / ``[S1, W2]`` style citation ids out of prose."""
    out: list[str] = []
    for match in _INLINE_CITE.finditer(text):
        out.extend(part.strip().upper() for part in match.group(1).split(","))
    return out


def verify_answer(
    answer: str,
    evidence: Sequence[EvidenceItem],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> AnswerAudit:
    """Re-verify the *final answer text* against the evidence.

    This closes the gap that otherwise voids the entire guarantee. Grounding
    validates the research agent's structured claims, and those claims shape
    the synthesis prompt -- but the synthesizer emits free text, and a prompt
    is a request, not a constraint. Nothing downstream re-read that text, so a
    synthesizer that ignored its instructions could return an entirely
    fabricated answer and the run would still report ``answered`` at ``high``
    confidence, with a legitimate-looking source list beside it.

    Every factual sentence is therefore re-parsed and re-checked here, against
    the same immutable evidence, by the same deterministic test. A sentence
    carrying a citation must be supported by at least one cited passage; a
    citation that names an unretrieved source is a fabrication regardless of
    what the sentence says.

    Sentences with no citation at all are reported but not failed: headings,
    transitions and the "what this doesn't cover" section legitimately carry
    none. It is *cited* sentences that make verifiable assertions.
    """
    by_id = {e.source_id.upper(): e for e in evidence}
    verified: list[str] = []
    unverified: list[str] = []
    uncited: list[str] = []
    disclosures: list[str] = []
    fabricated: list[str] = []

    for raw in _SENTENCE.split(answer):
        sentence = raw.strip()
        if not sentence:
            continue

        cited = extract_inline_citations(sentence)

        # A citation makes a sentence an assertion, whatever its length.
        # Applying a length filter before this check let "Revenue fell 80%
        # [S42]." -- 23 characters, entirely fabricated -- bypass the gate.
        # Length only decides whether *uncited* prose is worth reporting.
        if not cited:
            if len(sentence) >= 25 and not sentence.lstrip().startswith(("#", "|", "---", "```")):
                uncited.append(sentence)
            continue

        resolved = []
        for cid in cited:
            item = by_id.get(cid)
            if item is None:
                fabricated.append(cid)
            else:
                resolved.append(item)

        prose = _INLINE_CITE.sub("", sentence)

        # A statement of absence cannot be verified by overlap: a source
        # cannot contain words confirming what it does not say. Failing these
        # penalises the pipeline for disclosing its own gaps, which is the
        # opposite of the incentive this gate exists to create -- it would
        # push the synthesizer to omit the "what this doesn't cover" section
        # rather than write one. They are recorded separately and excluded
        # from the rate.
        if _ABSENCE.search(prose):
            disclosures.append(sentence)
            continue

        if resolved and any(claim_support(prose, item.text, threshold)[0] for item in resolved):
            verified.append(sentence)
        else:
            unverified.append(sentence)

    return AnswerAudit(
        verified_sentences=verified,
        unverified_sentences=unverified,
        uncited_sentences=uncited,
        disclosure_sentences=disclosures,
        fabricated_citations=sorted(set(fabricated)),
    )


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
