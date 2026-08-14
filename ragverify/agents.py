"""Agent roles and their prompts.

Instantiating agent objects and calling ``generate_reply`` on each exactly
once -- no shared conversation, no handoffs -- makes them an expensive wrapper
around independent one-shot completions.

This version is explicit about that trade-off instead of hiding it. Each role
is a prompt plus an output schema, executed through ``LLMClient.structured``;
the *orchestration* lives in ``orchestrator.py`` where the loop can actually
inspect and branch on each result. An optional AG2 group-chat backend is
provided in ``ag2_team.py`` for the cases where free-form agent debate helps.

Rebuilding agents per query is also avoided: roles are stateless prompt
constants.
"""

from __future__ import annotations

from collections.abc import Sequence

from .sanitize import BOUNDARY_PREAMBLE
from .schemas import (
    EvidenceItem,
    GroundingReport,
    ResearchDraft,
    Route,
    TriageDecision,
    VerifierReport,
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BOUNDARY = BOUNDARY_PREAMBLE + "\n\n"

TRIAGE_SYSTEM = _BOUNDARY + """\
You are the triage agent of a research team. A lexical retrieval probe has
already run against the local corpus, and you are shown its measured coverage
score plus the passages it actually returned. Your job is to confirm or
override that measurement, not to guess from filenames.

Choose:
- "local"     the corpus covers the question.
- "web"       the question needs current information, or the corpus is empty
              or unrelated.
- "hybrid"    the corpus is partly relevant and outside context fills real gaps.
- "clarify"   the question is too ambiguous to retrieve for — it has several
              materially different readings that would need different evidence.
              You MUST supply `clarifying_question` if you choose this.
- "no_answer" the question is unanswerable in principle (asks for private,
              future, or non-existent information) and no source could help.

Override the measured suggestion when you can see something the probe cannot:
lexical retrieval has no notion of synonyms, so a corpus that uses different
vocabulary for the same topic will score low despite covering the question.
Say so in your rationale when you override.

Use "clarify" and "no_answer" sparingly — they end the run. A merely broad
question is not ambiguous; retrieve for it.

Also decompose the question into up to 4 sub-questions that must each be
answered; these widen retrieval. Confidence is the probability your chosen
route yields sufficient evidence."""

RESEARCH_SYSTEM = _BOUNDARY + """\
You are a research agent. You are given numbered evidence passages. Extract
what the evidence actually supports.

Hard rules:
- Use ONLY the supplied passages. You have no other knowledge for this task.
- Break your findings into atomic claims. One fact per claim.
- Every claim MUST cite the source ids it comes from, e.g. ["S2","S5"].
- Copy names, numbers, dates and quantities exactly as they appear. Never
  round, infer or interpolate a figure that is not written in a passage.
- If the evidence does not answer part of the question, put that part in
  `unanswered` rather than filling the hole from memory. An honest gap is more
  useful to this pipeline than a confident guess -- a later round can go and
  retrieve it, but only if you report it.
- `draft_answer` must be assembled only from your claims."""

VERIFIER_SYSTEM = _BOUNDARY + """\
You are the verifier. You decide whether the team may answer yet, and if not,
what it should do next. You are the quality gate, not a summariser.

You receive the question, the draft, and a deterministic grounding report that
already checked every citation against its source text. Trust that report over
the draft's own confidence: `unsupported` claims failed a mechanical check, and
`hallucinated_citations` name sources that were never retrieved at all.

Set `verdict`:
- "sufficient"   the evidence genuinely answers the question.
- "partial"      the core is answered but named gaps remain.
- "insufficient" the question is not answered, or key claims are unsupported.

Set `next_action` — this is dispatched by the orchestrator, so choose the one
that would most improve the next round:
- "answer"           stop and write the final answer.
- "widen_local"      the corpus likely holds it; retrieve more/differently.
- "escalate_to_web"  the corpus cannot contain it (external, recent, or absent).
- "refine_query"     retrieval missed because of phrasing; supply
                     `refined_query` using the vocabulary the source would use.

List concrete, retrievable gaps -- "2024 revenue figure for the EU segment",
not "more detail". Gaps are used verbatim as search queries."""

STRUCTURED_SYNTHESIZER_SYSTEM = _BOUNDARY + """\
You produce the final answer as a list of CLAIMS. Prose is rendered from your
claims afterwards, so do not write paragraphs.

Each claim has:
- text       one statement, in plain language, no citation markers inside it
- kind       "assertion" for a fact drawn from the sources
             "disclosure" for something the sources do NOT establish
- citations  the source ids supporting it, e.g. ["S1","S3"]

Rules:
- EVERY claim needs at least one citation. An assertion without one is
  discarded, and a disclosure without one is treated as an unsourced claim.
- Every citation on an assertion must independently support it. Do not attach
  a source that is merely nearby or on the same topic; extra citations are
  removed and count against the answer.
- Use "disclosure" for a genuine gap, and phrase it as what the SOURCES lack:
  "the sources do not give a 2027 forecast", not "there is no 2027 forecast".
  A disclosure is a statement about the evidence, not a fact about the world.
- Never restate a claim that failed verification.
- `lead` is one optional framing sentence. It must contain no facts.

A short, fully supported answer is the correct outcome. Claims that cannot be
supported belong in `disclosure` or nowhere."""

SYNTHESIZER_SYSTEM = _BOUNDARY + """\
You are the synthesizer. Write the final answer for the user.

- Build only on verified claims. Claims marked unsupported were checked against
  their sources and failed; you may not restate them as fact.
- Cite inline with the source ids, e.g. [S3]. EVERY factual sentence must carry
  at least one citation. An uncited assertion cannot be verified and will be
  rejected.
- If gaps remain, state them plainly in a short "What this doesn't cover"
  section. Do not paper over them with hedged prose.
- Match the answer's length to the evidence available. A well-evidenced
  two-sentence answer beats a padded page.
- Write clean markdown. No preamble about being an AI or about the process."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def triage_prompt(
    question: str,
    corpus_summary: str,
    corpus_terms: Sequence[str],
    coverage=None,
    suggested=None,
    reason: str = "",
) -> str:
    terms = ", ".join(corpus_terms[:40]) if corpus_terms else "(no local corpus)"

    from .sanitize import sanitize

    probe = "(no probe run — no local corpus)"
    if coverage is not None and coverage.probe_hits:
        # Show what the probe actually retrieved. This is the evidence
        # a filename-only triage step never has. The
        # previews are document text, so they are sanitized like any other
        # untrusted passage before being embedded in the prompt.
        previews = "\n".join(
            f"  {i}. [{h.score:.2f}] {h.chunk.label}: "
            f"{sanitize(h.chunk.text[:220]).text.strip()}…"
            for i, h in enumerate(coverage.probe_hits[:5], start=1)
        )
        probe = (
            f"Measured coverage: {coverage.score:.2f} ({coverage.verdict})\n"
            f"Question-term recall: {coverage.term_recall:.0%}\n"
            f"Terms absent from the corpus: "
            f"{', '.join(coverage.missing_terms[:8]) or '(none)'}\n\n"
            f"Top passages the probe returned:\n{previews}"
        )
    elif coverage is not None:
        probe = (
            f"Measured coverage: {coverage.score:.2f} ({coverage.verdict}) — "
            f"the probe returned no passages at all."
        )

    suggestion = (
        f"Measured suggestion: {suggested.value} — {reason}" if suggested is not None else ""
    )

    return f"""Question:
{question}

Local corpus:
{corpus_summary}

Most distinctive terms in the corpus:
{terms}

--- Retrieval probe ---
{probe}

{suggestion}"""


def format_evidence(evidence: Sequence[EvidenceItem], token_budget: int) -> str:
    """Render evidence for a prompt, fenced and packed against a token budget.

    Two things happen here.

    Whole passages are included until the budget is reached, and only the last
    one is truncated, on a token boundary. Slicing every hit to a fixed
    character count -- ``chunk.text[:300]`` on an 800-word chunk -- shows the
    model a sliver of each passage however much budget is free.

    Every passage is also wrapped in an explicit untrusted-data fence, because
    this text comes from uploaded files and fetched web pages and is therefore
    attacker-controllable.
    """
    from .sanitize import wrap_passage
    from .tokens import count_tokens, truncate_to_tokens

    blocks: list[str] = []
    used = 0
    for item in evidence:
        header_cost = count_tokens(f"{item.source_id} {item.label} {item.url or ''}") + 24
        cost = count_tokens(item.text) + header_cost
        if used + cost > token_budget:
            remaining = token_budget - used - header_cost
            if remaining > 120:
                blocks.append(
                    wrap_passage(
                        item.source_id,
                        item.label,
                        truncate_to_tokens(item.text, remaining) + " […truncated]",
                        item.url or "",
                    )
                )
            break
        blocks.append(wrap_passage(item.source_id, item.label, item.text, item.url or ""))
        used += cost
    return "\n\n".join(blocks) if blocks else "(no evidence retrieved)"


def research_prompt(question: str, evidence: Sequence[EvidenceItem], token_budget: int) -> str:
    return f"""Question:
{question}

Evidence passages:

{format_evidence(evidence, token_budget)}"""


def verifier_prompt(
    question: str,
    draft: ResearchDraft,
    grounding: GroundingReport,
    route: Route,
    round_index: int,
    max_rounds: int,
    web_available: bool,
) -> str:
    supported = "\n".join(f"  ✓ {c.text}  {c.citations}" for c in grounding.supported) or "  (none)"
    unsupported = "\n".join(f"  ✗ {c.text}  {c.citations}" for c in grounding.unsupported) or "  (none)"
    fabricated = ", ".join(grounding.hallucinated_citations) or "(none)"
    unanswered = "\n".join(f"  - {u}" for u in draft.unanswered) or "  (none reported)"

    # The verifier is told the remaining budget so it can prefer "answer" when
    # no further round is available, instead of requesting work that can never
    # happen and leaving the user with nothing.
    rounds_left = max_rounds - round_index
    budget_note = (
        f"This is round {round_index} of {max_rounds}; {rounds_left} further round(s) available."
        if rounds_left > 0
        else f"This is the FINAL round ({round_index} of {max_rounds}). No further retrieval is "
        "possible, so choose 'answer' and report the remaining gaps honestly."
    )
    if not web_available:
        budget_note += " Web search is unavailable, so do not choose 'escalate_to_web'."

    return f"""Question:
{question}

Evidence route used this round: {route.value}
{budget_note}

Draft answer:
{draft.draft_answer or "(empty)"}

Grounding report (mechanical check, {grounding.support_rate:.0%} of claims supported):
Supported claims:
{supported}

Unsupported claims:
{unsupported}

Citations naming sources that do not exist: {fabricated}

Gaps the research agent itself reported:
{unanswered}"""


def synthesis_prompt(
    question: str,
    grounding: GroundingReport,
    evidence: Sequence[EvidenceItem],
    verifier: VerifierReport,
    token_budget: int,
) -> str:
    supported = "\n".join(f"- {c.text}  {c.citations}" for c in grounding.supported) or "- (none)"
    unsupported = "\n".join(f"- {c.text}" for c in grounding.unsupported) or "- (none)"
    gaps = "\n".join(f"- {g}" for g in verifier.gaps) or "- (none)"

    return f"""Question:
{question}

Verified claims (cite these):
{supported}

Claims that FAILED verification (do not assert these as fact):
{unsupported}

Remaining gaps to disclose:
{gaps}

Verifier verdict: {verifier.verdict.value}

Source passages for reference:

{format_evidence(evidence, token_budget // 2)}"""


# Schema bindings, so the orchestrator names a role rather than a schema.
ROLE_SCHEMAS = {
    "triage": TriageDecision,
    "research": ResearchDraft,
    "verifier": VerifierReport,
}


def resynthesis_prompt(original_prompt: str, audit) -> str:
    """Corrective turn after the final answer failed verification.

    Shows the model the specific sentences that failed and why, rather than
    re-asking the same question and hoping for a better sample.
    """
    failed = "\n".join(f"  - {s}" for s in audit.unverified_sentences[:8]) or "  (none)"
    fabricated = ", ".join(audit.fabricated_citations) or "(none)"
    return f"""{original_prompt}

--- YOUR PREVIOUS ANSWER FAILED VERIFICATION ---

Each sentence below carried a citation, but the cited passage does not support
it. Every one was checked mechanically against the source text:

{failed}

Citations naming sources that do not exist: {fabricated}

Rewrite the answer. Rules:
- State only what the verified claims above support.
- Cite the source id that actually contains each statement.
- Never cite an id that is not in the evidence list.
- If a point cannot be supported, omit it or move it to what this doesn't cover.
- A shorter, fully supported answer is the correct outcome here.
"""
