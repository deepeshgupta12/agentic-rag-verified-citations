"""Typed contracts for every hop in the pipeline.

Passing untyped ``dict`` objects between steps and calling
``dict.get(key, default)`` at each boundary lets a malformed LLM response
degrade silently into an empty dict, and the run continues on nothing. Every
inter-agent payload here is a Pydantic model instead: a bad response raises at
the boundary that produced it, where the repair loop in ``llm.py`` can see it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Route(str, Enum):
    """Where evidence for the current round comes from.

    ``CLARIFY`` and ``NO_ANSWER`` are terminal: they end the run without
    burning rounds on a question that retrieval cannot serve. With only
    local/web there is no way to say "this question is too vague to retrieve
    for" or "nothing here can answer this" -- every question yields a
    confident-looking answer.
    """

    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"
    CLARIFY = "clarify"
    NO_ANSWER = "no_answer"

    @property
    def is_terminal(self) -> bool:
        return self in (Route.CLARIFY, Route.NO_ANSWER)


class Verdict(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class NextAction(str, Enum):
    """What the verifier wants the orchestrator to do next.

    This is the field that makes the loop adaptive: the verdict carries an
    instruction the orchestrator dispatches on, rather than being a label
    nothing reads.
    """

    ANSWER = "answer"
    WIDEN_LOCAL = "widen_local"
    ESCALATE_TO_WEB = "escalate_to_web"
    REFINE_QUERY = "refine_query"


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable span of a source document.

    ``chunk_id`` is stable and globally unique across the corpus because it is
    what citations point at; grounding resolves a citation by looking the id
    back up in the index.
    """

    chunk_id: str
    doc_name: str
    ordinal: int
    text: str
    n_tokens: int = 0
    page: int | None = None

    @property
    def label(self) -> str:
        loc = f"p.{self.page}" if self.page is not None else f"#{self.ordinal}"
        return f"{self.doc_name} {loc}"


class WebResult(BaseModel):
    """A single web hit, normalised across search backends."""

    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""

    @property
    def label(self) -> str:
        return self.title or self.url


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    # Which retrievers found this chunk, for debugging fusion behaviour.
    retrievers: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """One piece of evidence handed to the drafting agent.

    ``source_id`` is the citation key the agent must quote. ``text`` is the
    verbatim source span -- never a model-written summary -- so that grounding
    can check a claim against what the document actually says.
    """

    source_id: str
    label: str
    text: str
    origin: Route
    url: str | None = None
    score: float = 0.0


# --------------------------------------------------------------------------
# Agent outputs (each is a structured-output schema)
# --------------------------------------------------------------------------


class TriageDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    # Sub-questions the triage agent thinks must be answered. Used to seed
    # query expansion on the first retrieval round.
    sub_queries: list[str] = Field(default_factory=list, max_length=5)
    # Populated only when route is CLARIFY: what the user must disambiguate.
    clarifying_question: str | None = None
    # Measured coverage the decision was made against, for the trace.
    measured_coverage: float | None = None

    @field_validator("rationale")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v.strip()


class Claim(BaseModel):
    """An atomic assertion plus the sources the drafter says support it.

    Splitting the draft into claims is what makes grounding possible at all: a
    prose blob can only be checked as a whole, whereas a claim can be accepted
    or rejected individually and the unsupported ones stripped.
    """

    text: str
    citations: list[str] = Field(default_factory=list)


class ResearchDraft(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    draft_answer: str = ""
    # Things the agent looked for in the evidence and did not find.
    unanswered: list[str] = Field(default_factory=list)


class VerifierReport(BaseModel):
    verdict: Verdict
    next_action: NextAction
    gaps: list[str] = Field(default_factory=list)
    # Query the verifier suggests for the next round when it asks for a refine.
    refined_query: str | None = None
    rationale: str = ""


class GroundingReport(BaseModel):
    """Deterministic (non-LLM) check that each claim's citations are real.

    Runs before the verifier so the verifier judges a draft whose citation
    validity is already known, rather than taking the drafter's word for it.
    """

    supported: list[Claim] = Field(default_factory=list)
    unsupported: list[Claim] = Field(default_factory=list)
    # Citations naming a source that was never retrieved: outright fabrication.
    hallucinated_citations: list[str] = Field(default_factory=list)
    # Citations resolving to a real passage that does not support the claim.
    # Distinct from fabrication and a weaker signal, but still removed.
    dropped_citations: list[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.supported) + len(self.unsupported)

    @property
    def support_rate(self) -> float:
        return len(self.supported) / self.total if self.total else 0.0


# --------------------------------------------------------------------------
# Run-level results
# --------------------------------------------------------------------------


class AnswerAudit(BaseModel):
    """Re-verification of the final answer text itself.

    Grounding checks the research agent's structured claims. Those claims shape
    the synthesis prompt, but a prompt is a request, not a constraint -- the
    synthesizer emits free text. This audits that text against the same
    evidence so the user-visible answer is mechanically verified, not merely
    verified upstream of the thing the user reads.
    """

    verified_sentences: list[str] = Field(default_factory=list)
    unverified_sentences: list[str] = Field(default_factory=list)
    # Headings, transitions and disclosure sections carry no citations by
    # design; reported for transparency, not counted as failures.
    uncited_sentences: list[str] = Field(default_factory=list)
    # Statements that the sources do NOT contain something. Unverifiable by
    # construction and actively desirable, so excluded from the rate rather
    # than counted as failures.
    disclosure_sentences: list[str] = Field(default_factory=list)
    fabricated_citations: list[str] = Field(default_factory=list)

    @property
    def total_cited(self) -> int:
        """Sentences making a *verifiable* assertion -- the rate's denominator.

        Excludes disclosures, which assert an absence and cannot be checked
        by overlap.
        """
        return len(self.verified_sentences) + len(self.unverified_sentences)

    @property
    def cited_sentences(self) -> int:
        """Sentences carrying a citation at all, disclosures included.

        Distinct from ``total_cited``, and conflating the two is a real bug:
        an answer whose every cited sentence is a legitimate disclosure has
        citations but a ``total_cited`` of zero, so it reads as uncited and
        triggers regeneration it does not need. "Did the answer cite
        anything?" and "how much of what it asserted verified?" are different
        questions.
        """
        return self.total_cited + len(self.disclosure_sentences)

    @property
    def verified_rate(self) -> float:
        return len(self.verified_sentences) / self.total_cited if self.total_cited else 0.0

    @property
    def is_clean(self) -> bool:
        """Nothing failed AND something was actually checked.

        An answer citing nothing is trivially free of failures, so treating
        that as clean would let the least verifiable output pass the
        strictest gate. Disclosures count as citing: they carry a source and
        are exactly what an honest answer to an unanswerable question looks
        like.
        """
        return (
            not self.unverified_sentences
            and not self.fabricated_citations
            and self.cited_sentences > 0
        )


class RoundRecord(BaseModel):
    """Everything that happened in one pass of the adaptive loop."""

    index: int
    route: Route
    query: str
    top_k: int
    n_evidence: int
    # Populated only when the optional entailment stage ran this round.
    entailment: dict[str, Any] = Field(default_factory=dict)
    draft: ResearchDraft | None = None
    grounding: GroundingReport | None = None
    verifier: VerifierReport | None = None
    elapsed_s: float = 0.0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


class Outcome(str, Enum):
    """How the run ended. Distinguishing these is the point of abstention.

    A pipeline with exactly one outcome answers regardless of whether the
    evidence supports one.
    """

    ANSWERED = "answered"
    PARTIAL = "partial"  # answered, but gaps remain
    ABSTAINED = "abstained"  # evidence did not support any answer
    CLARIFY = "clarify"  # question too ambiguous to retrieve for
    BUDGET = "budget"  # stopped by a cost/time/call cap


class ResearchResult(BaseModel):
    question: str
    final_answer: str
    confidence: Literal["high", "medium", "low"]
    outcome: Outcome = Outcome.ANSWERED
    citations: list[EvidenceItem] = Field(default_factory=list)
    rounds: list[RoundRecord] = Field(default_factory=list)
    triage: TriageDecision | None = None
    usage: Usage = Field(default_factory=Usage)
    # Populated when the loop hit its round budget without reaching
    # "sufficient". The answer is still returned, but flagged.
    stopped_early: bool = False
    open_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Prompt-injection patterns found in retrieved text and neutralized.
    injections_detected: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    # Re-verification of the answer text the user actually sees.
    answer_audit: AnswerAudit | None = None
    # Audit trail: content hashes, claim-to-source edges and sanitiser
    # modifications. Text is excluded here so the result stays shareable;
    # the full ledger with bodies is on the researcher.
    ledger: dict[str, Any] = Field(default_factory=dict)
    # Publisher class, recency and cross-domain corroboration for web sources.
    source_quality: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, float] = Field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def is_answer(self) -> bool:
        return self.outcome in (Outcome.ANSWERED, Outcome.PARTIAL)
