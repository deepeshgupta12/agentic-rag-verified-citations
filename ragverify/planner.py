"""Multi-hop planning: retrieval chained on what earlier retrieval found.

Some questions cannot be answered by retrieving harder. "Which of our EU
offices grew fastest?" needs two retrievals in sequence: first *which offices
exist*, then *the growth figure for each of those offices*. The second query
cannot be written until the first has an answer, because the office names are
in it.

The existing loop widens retrieval in *parallel* -- sub-queries and verifier
gaps all become extra ranked lists fused into one search. That is the right
shape for "I need more on this topic" and the wrong shape for "I need the
answer to X before I can ask Y". Fusing a query containing an unresolved
placeholder retrieves nothing useful however many rounds it runs.

So this module adds a genuinely different capability rather than more of the
same one:

* A **plan** is a small DAG of sub-questions, each declaring which others it
  depends on.
* Execution is **topological**: a hop runs only once its dependencies have
  resolved, and their findings are substituted into its query text.
* Coverage is tracked **per sub-question**, not for the question as a whole.
  A question can be 80% covered overall while the one sub-question that
  actually determines the answer is entirely unanswered, and a single
  aggregate number hides exactly that.

Three constraints keep it from being worse than the single-hop path:

* **Planning is itself a decision.** Most questions are single-hop, and
  running a DAG over them costs calls for nothing. The planner is asked to
  say so, and a single-hop plan degrades to ordinary retrieval.
* **The DAG is validated, not trusted.** Cycles, dangling dependencies and
  runaway breadth come back from models regularly. An invalid plan is
  repaired or rejected, never executed.
* **Budget is checked per hop.** Multi-hop multiplies retrieval, which is
  precisely the shape of run that exhausts a budget mid-question.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from .schemas import Claim, EvidenceItem

log = logging.getLogger("ragverify.planner")

MAX_SUB_QUESTIONS = 6
MAX_DEPTH = 3

# Placeholder a planner uses to mark "the answer to hop N goes here".
_SLOT = re.compile(r"\{\{\s*(?:hop|answer)?\s*#?(\d+)\s*\}\}", re.I)


class SubQuestion(BaseModel):
    id: int = Field(ge=0)
    text: str
    # Ids whose findings must be known before this one can be asked. Empty
    # means it can run immediately.
    depends_on: list[int] = Field(default_factory=list)
    why: str = Field(default="", max_length=200)


class ResearchPlan(BaseModel):
    multi_hop: bool = False
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=300)


class HopResult(BaseModel):
    sub_question_id: int
    question: str
    query_used: str
    evidence_ids: list[str] = Field(default_factory=list)
    findings: list[Claim] = Field(default_factory=list)
    answered: bool = False
    coverage: float = 0.0
    note: str = ""


class PlanReport(BaseModel):
    """Outcome of executing a plan, with coverage per sub-question."""

    planned: bool = False
    multi_hop: bool = False
    hops: list[HopResult] = Field(default_factory=list)
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def answered_count(self) -> int:
        return sum(h.answered for h in self.hops)

    @property
    def coverage(self) -> float:
        """Mean per-sub-question coverage.

        Reported alongside the per-hop figures, never instead of them: the
        mean is exactly the number that hides an unanswered decisive hop.
        """
        return round(sum(h.coverage for h in self.hops) / len(self.hops), 3) if self.hops else 0.0

    @property
    def unanswered(self) -> list[str]:
        return [h.question for h in self.hops if not h.answered]


PLANNER_SYSTEM = """\
You decide whether a question needs SEQUENTIAL retrieval, and if so you break
it into sub-questions.

A question is multi-hop only when one sub-question's *query text* cannot be
written until another has been answered. The test is concrete: could you type
both searches right now? If yes, it is single-hop.

  single-hop: "What were revenue and headcount?"
              Two facts, both searchable immediately. NOT multi-hop.

  multi-hop:  "Which of our EU offices grew fastest?"
              You cannot search for growth per office until you know which
              offices exist.

  multi-hop:  "What did the CEO who signed the 2024 filing previously run?"
              You must identify the person before you can research them.

Most questions are single-hop. Say so — a plan for a question that does not
need one costs retrieval and buys nothing.

When it IS multi-hop:
- Give each sub-question an id starting at 0.
- List in `depends_on` the ids whose ANSWERS its text requires.
- Write dependent sub-questions with a {{0}} placeholder where the earlier
  answer belongs: "What was revenue growth for {{0}}?"
- Keep it to 4 sub-questions or fewer. Depth beats breadth; parallel
  sub-questions that could all be searched now belong in ONE sub-question.
- No cycles. A dependency graph that loops cannot be executed."""


def plan_research(question: str, corpus_summary: str, client) -> ResearchPlan:
    """Ask for a plan. Any failure degrades to single-hop."""
    prompt = f"Question:\n{question}\n\nAvailable corpus:\n{corpus_summary}"
    try:
        plan = client.structured(PLANNER_SYSTEM, prompt, ResearchPlan)
    except Exception as exc:  # noqa: BLE001 - planning is an optimisation
        log.warning("planning failed (%s); treating as single-hop", exc)
        return ResearchPlan(multi_hop=False, rationale=f"planner unavailable: {exc}")
    return plan


def validate_plan(plan: ResearchPlan) -> tuple[ResearchPlan, list[str]]:
    """Repair or reject a plan. Never execute one that was merely returned.

    Models produce cycles, dangling ids and twelve-way fan-outs often enough
    that trusting the structure is not an option -- an unvalidated cycle is an
    infinite loop, and a dangling dependency is a hop that can never run.
    """
    warnings: list[str] = []
    if not plan.multi_hop or not plan.sub_questions:
        return ResearchPlan(multi_hop=False, rationale=plan.rationale), warnings

    subs = plan.sub_questions[:MAX_SUB_QUESTIONS]
    if len(plan.sub_questions) > MAX_SUB_QUESTIONS:
        warnings.append(
            f"Plan had {len(plan.sub_questions)} sub-questions; truncated to {MAX_SUB_QUESTIONS}."
        )

    # Deduplicate ids: two hops sharing an id makes dependencies ambiguous.
    seen: set[int] = set()
    unique: list[SubQuestion] = []
    for sub in subs:
        if sub.id in seen:
            warnings.append(f"Duplicate sub-question id {sub.id} dropped.")
            continue
        seen.add(sub.id)
        unique.append(sub)

    # Drop dangling dependencies rather than the hop: a hop whose dependency
    # does not exist is still answerable, just without substitution.
    for sub in unique:
        dangling = [d for d in sub.depends_on if d not in seen or d == sub.id]
        if dangling:
            warnings.append(f"Sub-question {sub.id} depends on unknown/self id(s) {dangling}; ignored.")
            sub.depends_on = [d for d in sub.depends_on if d in seen and d != sub.id]

    order, cyclic = _topological_order(unique)
    if cyclic:
        warnings.append(
            f"Plan contains a dependency cycle involving {sorted(cyclic)}; "
            "those dependencies were cut."
        )
        for sub in unique:
            if sub.id in cyclic:
                sub.depends_on = []
        order, _ = _topological_order(unique)

    if _depth(unique) > MAX_DEPTH:
        warnings.append(f"Plan deeper than {MAX_DEPTH} hops; deeper dependencies cut.")

    ordered = [next(s for s in unique if s.id == sid) for sid in order]
    if len(ordered) < 2:
        # A one-hop "plan" is ordinary retrieval wearing a costume.
        return ResearchPlan(multi_hop=False, rationale=plan.rationale), warnings

    return (
        ResearchPlan(multi_hop=True, sub_questions=ordered, rationale=plan.rationale),
        warnings,
    )


def _topological_order(subs: Sequence[SubQuestion]) -> tuple[list[int], set[int]]:
    """Kahn's algorithm. Returns ``(order, ids_in_cycles)``."""
    ids = {s.id for s in subs}
    indegree = {s.id: len([d for d in s.depends_on if d in ids]) for s in subs}
    dependents: dict[int, list[int]] = {s.id: [] for s in subs}
    for sub in subs:
        for dep in sub.depends_on:
            if dep in ids:
                dependents[dep].append(sub.id)

    queue = sorted(sid for sid, deg in indegree.items() if deg == 0)
    order: list[int] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
        queue.sort()

    return order, ids - set(order)


def _depth(subs: Sequence[SubQuestion]) -> int:
    by_id = {s.id: s for s in subs}
    memo: dict[int, int] = {}

    def depth(sid: int, seen: frozenset[int]) -> int:
        if sid in seen:
            return 0  # cycle guard; cycles are handled separately
        if sid in memo:
            return memo[sid]
        sub = by_id.get(sid)
        value = 1 + max((depth(d, seen | {sid}) for d in sub.depends_on if d in by_id), default=0) if sub else 0
        memo[sid] = value
        return value

    return max((depth(s.id, frozenset()) for s in subs), default=0)


# A run of capitalised words, allowing lowercase connectors ("Bank of England").
_PROPER_RUN = re.compile(r"\b[A-Z][\w'-]*(?:\s+(?:of|de|van|der|and|for|the)\s+[A-Z][\w'-]*|\s+[A-Z][\w'-]*)*")
_QUOTED = re.compile(r"[\"“']([^\"”']{3,60})[\"”']")
_MEASURE = re.compile(r"\b\d[\d,.]*\s?(?:%|percent\b|million\b|billion\b|bn\b|m\b|k\b)", re.I)

# Words that start a sentence and carry no retrieval signal. A capitalised
# "The" is punctuation, not an entity.
_SENTENCE_STARTER_TEXT = (
    "the this that these those it he she they there here a an in on at for and but "
    "his her their its we you i our your"
)
_SENTENCE_STARTERS = frozenset(_SENTENCE_STARTER_TEXT.split())


def salient_terms(claims: Sequence[Claim], limit: int = 6) -> list[str]:
    """Pull the entities out of findings, for use in a follow-up query.

    Splicing whole claim sentences into the next query technically works --
    retrieval still matches on the entity buried inside -- but it drags in the
    prose around it, and BM25 scores every one of those words. "What did The
    2024 statutory audit letter was signed by Ingrid Halvorsen, Chief Financial
    Officer. previously run?" retrieves the right passage for the wrong
    reasons and dilutes the term that actually matters.

    Proper nouns, quoted strings and measurements are what a follow-up query
    needs. Sentence-initial capitals are dropped because a capitalised "The"
    is punctuation, not an entity.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip(" .,;:")
        key = value.lower()
        if len(value) > 2 and key not in seen:
            seen.add(key)
            terms.append(value)

    for claim in claims:
        for quoted in _QUOTED.findall(claim.text):
            add(quoted)
        # Sentence by sentence, so "first word" is meaningful.
        for sentence in re.split(r"(?<=[.!?])\s+", claim.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            for match in _PROPER_RUN.finditer(sentence):
                run = match.group(0)
                # Single capitalised word at the very start is sentence case.
                if match.start() == 0 and " " not in run and run.lower() in _SENTENCE_STARTERS:
                    continue
                if match.start() == 0 and run.split()[0].lower() in _SENTENCE_STARTERS:
                    run = run.split(" ", 1)[1] if " " in run else ""
                if run:
                    add(run)
        for measure in _MEASURE.findall(claim.text):
            add(measure)

    return terms[:limit]


def resolve_query(sub: SubQuestion, findings: dict[int, list[Claim]]) -> str:
    """Substitute earlier findings into a dependent sub-question's text.

    This is the step that makes a hop *chained* rather than merely ordered.
    Without substitution a dependent query still contains its placeholder and
    retrieves noise, which looks like a multi-hop failure but is really a
    plumbing one.

    Entities are substituted rather than whole sentences: a query is scored
    term by term, so the surrounding prose competes with the entity that
    actually matters.
    """

    def _terms_for(dep: int) -> str:
        claims = findings.get(dep, [])
        if not claims:
            return ""
        terms = salient_terms(claims)
        # Nothing entity-shaped: fall back to the claim text, truncated. A
        # verbose query beats an empty one.
        return ", ".join(terms) if terms else claims[0].text[:120]

    text = _SLOT.sub(lambda m: _terms_for(int(m.group(1))), sub.text)

    # A dependency with no placeholder still needs its context appended, or
    # the hop searches as if nothing had been learned.
    if not _SLOT.search(sub.text) and sub.depends_on:
        context = " ".join(filter(None, (_terms_for(dep) for dep in sub.depends_on)))
        if context:
            text = f"{text} {context}"

    return " ".join(text.split()).strip() or sub.text


def hop_coverage(findings: Sequence[Claim], evidence: Sequence[EvidenceItem]) -> float:
    """How well this hop was answered: grounded findings against evidence seen.

    Per-hop rather than per-run, because an aggregate hides the case this
    module exists for -- a decisive sub-question left unanswered while the
    others carry the mean.
    """
    if not evidence:
        return 0.0
    if not findings:
        return 0.0
    cited = {c for claim in findings for c in claim.citations}
    return round(min(1.0, len(cited) / max(1, min(len(evidence), 3))), 3)
