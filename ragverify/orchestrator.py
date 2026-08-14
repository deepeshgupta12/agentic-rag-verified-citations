"""The adaptive research loop.

This module is the reason v2 exists.

The common shape is a straight line: triage once, retrieve once, draft once,
verify once, synthesize. The verifier produces ``{"verdict": ..., "gaps":
[...]}``, that object is passed to the synthesizer as *text*, and no control
flow ever reads it. Nothing branches on it, so "insufficient" and "sufficient"
produce exactly the same behaviour: write the final answer anyway. Such a
pipeline computes an adaptation signal and discards it.

The loop here dispatches on that signal:

    triage → [ retrieve → draft → ground → verify ] × N → synthesize
                   ↑                            │
                   └──── escalate on verdict ───┘

Escalation is a ladder, so each round is strictly more capable than the last
rather than a retry of the same failing strategy:

  1. widen_local      more passages, plus the verifier's gaps as extra queries
  2. refine_query     re-retrieve using the verifier's rewritten phrasing
  3. escalate_to_web  add live web evidence to the local pool
  4. budget exhausted answer with what exists and disclose the gaps

The loop also refuses to accept "sufficient" when the mechanical grounding
rate is below ``min_support_rate``: a model marking its own team's work as
sufficient is exactly the failure grounding exists to catch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from . import agents, sanitize, sourcequality, telemetry, websearch
from . import coverage as coverage_mod
from . import entailment as entailment_mod
from . import grounding as grounding_mod
from . import planner as planner_mod
from . import rerank as rerank_mod
from . import store as store_mod
from .budget import Budget, BudgetExceeded, CircuitBreaker
from .config import Settings
from .ingest import Document, build_index, corpus_summary, top_terms
from .ledger import EvidenceLedger
from .llm import LLMClient, LLMError, SchemaError
from .retrieval import DenseIndex, HybridRetriever
from .schemas import (
    AnswerAudit,
    Chunk,
    EvidenceItem,
    GroundingReport,
    NextAction,
    Outcome,
    ResearchDraft,
    ResearchResult,
    RoundRecord,
    Route,
    TriageDecision,
    Usage,
    Verdict,
    VerifierReport,
    WebResult,
)
from .trace import EventKind, Tracer

log = logging.getLogger("ragverify.orchestrator")

# Above this measured coverage, a pre-retrieval NO_ANSWER is not credible.
WEAK_ENOUGH = 0.15


class Corpus:
    """A prepared, reusable local index.

    Built once and cached by the caller. Re-parsing every uploaded PDF and
    rebuilding the chunk index per question would parse a 200-page report five
    times to answer five questions.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        settings: Settings,
        client: LLMClient | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.documents = list(documents)
        self.chunks: list[Chunk] = build_index(
            self.documents, settings.chunk_tokens, settings.chunk_overlap_tokens
        )
        self.summary = corpus_summary(self.chunks)
        self.terms = top_terms(self.chunks)
        self.warnings: list[str] = []

        dense: DenseIndex | None = None
        embed_query = None
        self.embedding_cache: store_mod.EmbeddingCache | None = None
        if settings.use_embeddings and client and self.chunks:
            try:
                self.embedding_cache = store_mod.EmbeddingCache(
                    settings.embed_model, settings.cache_dir, enabled=settings.cache_embeddings
                )
                with telemetry.span(
                    "ragverify.embed_corpus",
                    **{"ragverify.chunks": len(self.chunks),
                       "gen_ai.request.model": settings.embed_model},
                ) as sp:
                    vectors = store_mod.embed_cached(
                        [c.text for c in self.chunks], client, self.embedding_cache
                    )
                    telemetry.set_attributes(
                        sp,
                        **{"ragverify.cache_hits": self.embedding_cache.hits,
                           "ragverify.cache_hit_rate": self.embedding_cache.hit_rate},
                    )
                self.embedding_cache.save()
                if self.embedding_cache.hits:
                    self.warnings.append(
                        f"Reused {self.embedding_cache.hits} cached embedding(s) "
                        f"({self.embedding_cache.hit_rate:.0%} hit rate)."
                    )
                dense = DenseIndex(self.chunks, vectors)

                # Query embeddings are cached too: refined and repeated
                # queries recur constantly across rounds and sessions.
                def embed_query(q, _client=client, _cache=self.embedding_cache):
                    return store_mod.embed_cached([q], _client, _cache)[0]
            except Exception as exc:  # noqa: BLE001 - lexical-only is a fine corpus
                self.warnings.append(f"Embeddings unavailable, using BM25 only ({exc}).")
                if tracer:
                    tracer.emit(EventKind.WARNING, self.warnings[-1])

        reranker = None
        if settings.rerank_method != "none":
            def reranker(question, candidates, top_k):
                return rerank_mod.rerank(
                    question, candidates, top_k,
                    method=settings.rerank_method,
                    client=client,
                    drop_below=settings.rerank_drop_below,
                    model_name=settings.rerank_model,
                )

        self.retriever = HybridRetriever(
            self.chunks, dense=dense, embed_query=embed_query, reranker=reranker
        )

    def __len__(self) -> int:
        return len(self.chunks)


def _to_evidence(scored, prefix: str = "S") -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id=f"{prefix}{i}",
            label=s.chunk.label,
            text=s.chunk.text,
            origin=Route.LOCAL,
            score=s.score,
        )
        for i, s in enumerate(scored, start=1)
    ]


def _web_evidence(
    results: Sequence[WebResult],
    pages: dict,
    settings: Settings,
    start_index: int,
) -> list[EvidenceItem]:
    """Turn search results into evidence, preferring fetched page text.

    Falls back to the SERP snippet when a page could not be fetched, and marks
    it so grounding is judged against what was actually available.
    """
    from .tokens import truncate_to_tokens

    out: list[EvidenceItem] = []
    for offset, result in enumerate(results):
        body = pages.get(result.url, "").strip()
        if body:
            text = truncate_to_tokens(body, settings.chunk_tokens * 4)
        elif result.snippet:
            text = f"(search snippet only, page not retrieved) {result.snippet}"
        else:
            continue
        out.append(
            EvidenceItem(
                source_id=f"W{start_index + offset}",
                label=f"{result.label} [{websearch.domain(result.url)}]",
                text=text,
                origin=Route.WEB,
                url=result.url,
            )
        )
    return out


class AdaptiveResearcher:
    """Runs the adaptive loop for one question against one corpus."""

    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        corpus: Corpus | None = None,
        tracer: Tracer | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.corpus = corpus
        self.tracer = tracer or Tracer()
        self.warnings: list[str] = list(corpus.warnings) if corpus else []
        self.budget = budget or Budget(
            max_seconds=settings.max_seconds,
            max_cost_usd=settings.max_cost_usd,
            max_calls=settings.max_calls,
        )
        self.breaker = CircuitBreaker()
        self.injections: list[str] = []
        # Set when synthesis had to be replaced by a deterministic
        # rendering. The content is verified, but generation proved
        # unreliable, and confidence should say so.
        self.synthesis_degraded = False
        # Latest web source-quality assessment, surfaced on the result.
        self.quality: sourcequality.QualityReport | None = None
        # Evidence gathered by multi-hop, merged into the main loop so the
        # answer is written over everything the hops found.
        self._hop_evidence: list[EvidenceItem] = []
        # Append-only audit trail. Built during the run so it captures each
        # passage as it was actually used, including the pre-sanitisation body.
        self.ledger: EvidenceLedger | None = None

    # ------------------------------------------------------------------

    def run(self, question: str) -> ResearchResult:
        """Public entry point: wraps the loop in a span when tracing is on."""
        # Claim the client's budget for the duration of THIS run, and restart
        # the clock. A researcher is one question; a client is usually reused
        # across many (a Streamlit session, a batch eval, a server). Binding
        # the budget once at construction let the first question's spent
        # budget follow the client into every later one -- observed on a
        # 12-question run where questions 3 onward each got 0.1s and abstained
        # instantly, having inherited an already-exhausted clock.
        self.budget.started_at = time.monotonic()
        self.budget.calls = 0
        self.budget.cost_usd = 0.0
        self.client.budget = self.budget
        # Usage is per-run too, or cost accounting reports the session total
        # as though one question had spent it.
        self.client.usage = Usage()

        if self.settings.telemetry and not telemetry.enabled():
            telemetry.configure(endpoint=self.settings.otlp_endpoint)

        with telemetry.span("ragverify.run", **{"ragverify.question": question[:200]}) as sp:
            telemetry.bridge(self.tracer)
            result = self._run(question)
            telemetry.set_attributes(
                sp,
                **{"ragverify.outcome": result.outcome.value,
                   "ragverify.confidence": result.confidence,
                   "ragverify.rounds": len(result.rounds),
                   "gen_ai.usage.input_tokens": result.usage.prompt_tokens,
                   "gen_ai.usage.output_tokens": result.usage.completion_tokens,
                   "ragverify.cost_usd": result.usage.cost_usd},
            )
            return result

    def _run(self, question: str) -> ResearchResult:
        started = time.time()
        settings = self.settings
        tracer = self.tracer
        tracer.emit(EventKind.START, f"Researching: {question}")

        rounds: list[RoundRecord] = []
        self.ledger = EvidenceLedger(question)

        # Measure coverage BEFORE routing. Judging "document coverage" from
        # filenames and a chunk count is guesswork; this probes the index with
        # a free BM25 pass so the decision is made against retrieved content
        # and real scores.
        cov = coverage_mod.measure(question, self.corpus.retriever if self.corpus else None)
        suggested, reason = coverage_mod.suggest_route(
            cov, self._web_available(), bool(self.corpus and len(self.corpus))
        )
        self.tracer.emit(
            EventKind.RETRIEVE,
            f"Coverage probe: {cov.verdict} ({cov.score:.2f}) — term recall {cov.term_recall:.0%}"
            + (f", missing: {', '.join(cov.missing_terms[:4])}" if cov.missing_terms else ""),
            data={"coverage": cov.score, "suggested": suggested.value, "reason": reason},
        )

        # Multi-hop planning runs before triage: chained hops gather evidence
        # the single-shot loop could not have queried for, and the loop then
        # reasons over it.
        plan_report = self._plan_and_execute(question)

        triage = self._triage(question, cov, suggested, reason)

        # Terminal routes end the run without spending a round.
        if triage.route is Route.CLARIFY:
            return self._clarify_result(question, triage, started)
        if triage.route is Route.NO_ANSWER:
            return self._abstain_result(question, triage, started, Outcome.ABSTAINED)

        route = self._initial_route(triage)
        query = question
        top_k = settings.top_k
        expansions: list[str] = list(triage.sub_queries)
        web_pool: list[EvidenceItem] = []

        draft = ResearchDraft()
        grounding = GroundingReport()
        verifier = VerifierReport(
            verdict=Verdict.INSUFFICIENT,
            next_action=NextAction.ANSWER,
            rationale="No round completed.",
        )
        evidence: list[EvidenceItem] = []
        stopped_early = True
        budget_stopped = False

        for index in range(1, settings.max_rounds + 1):
            round_started = time.time()
            tracer.emit(
                EventKind.RETRIEVE,
                f"Round {index}: retrieving via {route.value} (top_k={top_k})",
                index,
                route=route.value,
                query=query,
            )

            evidence = self._gather(question, query, route, top_k, expansions, web_pool, index)
            if not evidence:
                self._warn("No evidence retrieved this round.", index)
                if route is not Route.WEB and self._web_available():
                    route = Route.WEB
                    continue
                break

            # Retrieved text is untrusted input. Neutralize model-directed
            # instructions before it reaches any prompt. The raw body is kept
            # first: grounding checks the sanitised text, so an audit needs
            # both to show what the filter changed.
            raw_by_id = {e.source_id: e.text for e in evidence}
            flags_by_id: dict[str, list[str]] = {}
            if settings.sanitize_sources:
                evidence, detections = sanitize.sanitize_evidence(evidence)
                if detections:
                    note = sanitize.summarize(detections)
                    self.injections.extend(sorted({d.kind for d in detections}))
                    self._warn(note, index)
                    for det in detections:
                        flags_by_id.setdefault(det.source_id, []).append(det.kind)

            # Rank web evidence by publisher class, recency and cross-domain
            # corroboration before it reaches the drafting prompt, so the
            # strongest sources are the ones that fit the token budget.
            if settings.assess_source_quality:
                quality = sourcequality.assess(
                    evidence, question,
                    min_authority=settings.min_source_authority,
                    min_domains=settings.min_distinct_domains,
                )
                if quality.assessments:
                    evidence = sourcequality.rank(evidence, quality)
                    self.quality = quality
                    for note in quality.warnings:
                        self._warn(note, index)

            for item in evidence:
                self.ledger.record_evidence(
                    item,
                    raw_text=raw_by_id.get(item.source_id),
                    injection_flags=sorted(set(flags_by_id.get(item.source_id, []))),
                )

            tracer.emit(
                EventKind.RESEARCH,
                f"Round {index}: drafting from {len(evidence)} passages",
                index,
                n_evidence=len(evidence),
            )
            draft = self._draft(question, evidence)

            grounding = grounding_mod.check(draft.claims, evidence)

            # Second stage: semantic entailment over claims that already
            # passed the lexical check. It can only DOWNGRADE -- a claim
            # rejected lexically is never revived here, so the deterministic
            # layer stays the floor and enabling this can never make the
            # pipeline accept something it previously refused.
            entail_report = self._entail(grounding, evidence, index)

            self.ledger.record_grounding(grounding, index)
            tracer.emit(
                EventKind.GROUND,
                f"Round {index}: {len(grounding.supported)}/{grounding.total} claims grounded"
                f"{f', {len(grounding.hallucinated_citations)} fabricated citation(s)' if grounding.hallucinated_citations else ''}",
                index,
                support_rate=grounding.support_rate,
                hallucinated=grounding.hallucinated_citations,
            )

            verifier = self._verify(question, draft, grounding, route, index)
            tracer.emit(
                EventKind.VERIFY,
                f"Round {index}: verdict={verifier.verdict.value}, next={verifier.next_action.value}",
                index,
                verdict=verifier.verdict.value,
                gaps=verifier.gaps,
            )

            rounds.append(
                RoundRecord(
                    index=index,
                    route=route,
                    query=query,
                    top_k=top_k,
                    n_evidence=len(evidence),
                    draft=draft,
                    grounding=grounding,
                    entailment=entail_report.model_dump(mode="json") if entail_report.ran else {},
                    verifier=verifier,
                    elapsed_s=round(time.time() - round_started, 2),
                )
            )

            if self._can_answer(verifier, grounding):
                stopped_early = False
                break

            if index == settings.max_rounds:
                break

            # Check the budget before escalating, not after: stopping at a
            # round boundary leaves a usable answer, whereas running out
            # mid-round leaves nothing.
            self._sync_budget()
            if not self.budget.can_afford_round():
                self._warn(
                    f"Stopping early — {self.budget.exhausted() or 'insufficient budget for another round'}.",
                    index,
                )
                budget_stopped = True
                break

            route, query, top_k, expansions = self._escalate(
                verifier, question, query, route, top_k, expansions, index
            )

        # Abstain rather than answer when nothing survived grounding. An
        # answer built on zero verified claims is worse than none: it reads
        # exactly like a good one.
        # No evidence at all is an abstention, not an answer. Reporting
        # ANSWERED here contradicts the message body, and any caller
        # branching on `is_answer` would treat a non-answer as a result.
        if not rounds or not evidence:
            self._warn("No evidence was retrieved; abstaining.")
            return self._abstain_result(
                question, triage, started, Outcome.ABSTAINED,
                rounds=rounds, gaps=verifier.gaps,
            )

        if self._should_abstain(rounds, grounding, evidence):
            return self._abstain_result(
                question,
                triage,
                started,
                Outcome.ABSTAINED,
                rounds=rounds,
                gaps=verifier.gaps,
            )

        final_answer, confidence, audit = self._synthesize(question, grounding, evidence, verifier)

        if budget_stopped:
            outcome = Outcome.BUDGET
        elif stopped_early and rounds:
            outcome = Outcome.PARTIAL
        else:
            outcome = Outcome.ANSWERED

        self._sync_budget()
        self.ledger.freeze()
        result = ResearchResult(
            question=question,
            final_answer=final_answer,
            confidence=confidence,
            outcome=outcome,
            citations=self._cited(evidence, grounding),
            rounds=rounds,
            triage=triage,
            usage=self.client.usage,
            stopped_early=stopped_early and bool(rounds),
            open_gaps=verifier.gaps,
            warnings=self.warnings + (self.corpus.retriever.warnings if self.corpus else []),
            injections_detected=sorted(set(self.injections)),
            answer_audit=audit,
            ledger=self.ledger.to_dict(include_text=False),
            plan=plan_report.model_dump(mode="json") if plan_report.planned else {},
            source_quality=(
                {
                    "mean_authority": self.quality.mean_authority,
                    "distinct_domains": self.quality.distinct_domains,
                    "diversity": self.quality.diversity,
                    "time_sensitive": self.quality.time_sensitive,
                    "sources": [
                        {
                            "source_id": a.source_id, "domain": a.domain,
                            "authority": a.authority, "freshness": a.freshness,
                            "published_year": a.published_year,
                            "corroborated_by": a.corroborated_by, "score": a.score,
                        }
                        for a in self.quality.assessments
                    ],
                }
                if self.quality
                else {}
            ),
            budget=self.budget.snapshot(),
            elapsed_s=round(time.time() - started, 2),
        )
        tracer.emit(
            EventKind.DONE,
            f"{outcome.value} in {result.elapsed_s}s over {len(rounds)} round(s) — "
            f"${result.usage.cost_usd:.4f}",
            data={"confidence": confidence, "outcome": outcome.value},
        )
        return result

    # ------------------------------------------------------------------
    # Terminal outcomes
    # ------------------------------------------------------------------

    def _should_abstain(
        self,
        rounds: list[RoundRecord],
        grounding: GroundingReport,
        evidence: Sequence[EvidenceItem],
    ) -> bool:
        if not rounds or not evidence:
            return False  # handled by the no-evidence path in _synthesize
        if self.settings.abstain_below_support <= 0:
            return False
        if not grounding.total:
            return True
        return grounding.support_rate < self.settings.abstain_below_support

    def _clarify_result(self, question: str, triage: TriageDecision, started: float) -> ResearchResult:
        ask = triage.clarifying_question or "Could you narrow the question? It is too broad to retrieve for."
        self.tracer.emit(EventKind.DONE, "Asking for clarification", data={"outcome": "clarify"})
        self._sync_budget()
        return ResearchResult(
            question=question,
            final_answer=f"**I need one clarification before researching this.**\n\n{ask}",
            confidence="low",
            outcome=Outcome.CLARIFY,
            triage=triage,
            clarifying_question=ask,
            usage=self.client.usage,
            warnings=self.warnings,
            budget=self.budget.snapshot(),
            elapsed_s=round(time.time() - started, 2),
        )

    def _abstain_result(
        self,
        question: str,
        triage: TriageDecision | None,
        started: float,
        outcome: Outcome,
        rounds: list[RoundRecord] | None = None,
        gaps: list[str] | None = None,
    ) -> ResearchResult:
        gaps = gaps or []
        reason = (
            triage.rationale
            if triage and triage.route is Route.NO_ANSWER and triage.rationale
            else "No claim survived verification against the retrieved sources."
        )
        body = [
            "**I can't answer this from the available evidence.**",
            "",
            reason,
        ]
        if gaps:
            body += ["", "Specifically, these remain unresolved:", *[f"- {g}" for g in gaps]]
        body += [
            "",
            "_Answering anyway would mean presenting unverified claims as fact, "
            "which is the failure this pipeline exists to prevent._",
        ]

        self.tracer.emit(EventKind.DONE, "Abstained — evidence insufficient", data={"outcome": outcome.value})
        self._sync_budget()
        return ResearchResult(
            question=question,
            final_answer="\n".join(body),
            confidence="low",
            outcome=outcome,
            rounds=rounds or [],
            triage=triage,
            usage=self.client.usage,
            open_gaps=gaps,
            warnings=self.warnings + (self.corpus.retriever.warnings if self.corpus else []),
            injections_detected=sorted(set(self.injections)),
            budget=self.budget.snapshot(),
            elapsed_s=round(time.time() - started, 2),
        )

    def _sync_budget(self) -> None:
        """Mirror the client's usage into the budget tracker."""
        usage = self.client.usage
        self.budget.calls = usage.calls
        self.budget.cost_usd = usage.cost_usd

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _plan_and_execute(self, question: str) -> planner_mod.PlanReport:
        """Decompose into a sub-question DAG and run the hops in order.

        Each hop retrieves, drafts and grounds independently, and its findings
        are substituted into dependent hops' queries. Coverage is recorded per
        sub-question: an aggregate would hide a decisive hop left unanswered
        while the others carry the mean.
        """
        report = planner_mod.PlanReport()
        if not self.settings.use_multi_hop or not self.corpus:
            return report

        raw = planner_mod.plan_research(question, self.corpus.summary, self.client)
        plan, warnings = planner_mod.validate_plan(raw)
        for note in warnings:
            self._warn(f"Plan: {note}")
        report.rationale = plan.rationale
        report.warnings = warnings

        if not plan.multi_hop:
            self.tracer.emit(
                EventKind.TRIAGE,
                f"Single-hop question — {plan.rationale or 'no chained retrieval needed'}",
            )
            return report

        report.planned = True
        report.multi_hop = True
        self.tracer.emit(
            EventKind.TRIAGE,
            f"Multi-hop plan: {len(plan.sub_questions)} sub-question(s) — {plan.rationale}",
            data={"sub_questions": [sq.text for sq in plan.sub_questions]},
        )

        findings: dict[int, list] = {}
        total = len(plan.sub_questions)
        for hop_index, sub in enumerate(plan.sub_questions[: self.settings.max_hops], start=1):
            self._sync_budget()
            if not self.budget.can_afford_round(estimated_calls=2, estimated_seconds=15.0):
                self._warn(f"Stopped multi-hop after {hop_index - 1} hop(s): budget exhausted.")
                break

            query = planner_mod.resolve_query(sub, findings)
            self.tracer.emit(
                EventKind.RETRIEVE,
                f"Hop {hop_index}/{total}: {query[:110]}"
                + (f"  (depends on {sub.depends_on})" if sub.depends_on else ""),
                data={"sub_question_id": sub.id},
            )

            scored = self.corpus.retriever.search(query, self.settings.top_k)
            evidence = _to_evidence(scored, prefix=f"H{sub.id}x")
            if self.settings.sanitize_sources and evidence:
                evidence, detections = sanitize.sanitize_evidence(evidence)
                if detections:
                    self.injections.extend(sorted({d.kind for d in detections}))

            hop = planner_mod.HopResult(
                sub_question_id=sub.id,
                question=sub.text,
                query_used=query,
                evidence_ids=[e.source_id for e in evidence],
            )

            if not evidence:
                hop.note = "no evidence retrieved"
                report.hops.append(hop)
                continue

            try:
                draft = self.client.structured(
                    agents.RESEARCH_SYSTEM,
                    agents.research_prompt(sub.text, evidence, self.settings.evidence_token_budget),
                    ResearchDraft,
                )
            except (LLMError, SchemaError, BudgetExceeded) as exc:
                hop.note = f"research failed: {exc}"
                report.hops.append(hop)
                continue

            grounded = grounding_mod.check(draft.claims, evidence)
            hop.findings = grounded.supported
            hop.coverage = planner_mod.hop_coverage(grounded.supported, evidence)
            hop.answered = bool(grounded.supported)
            if not hop.answered:
                hop.note = "nothing grounded"

            findings[sub.id] = grounded.supported
            self._hop_evidence.extend(evidence)
            if self.ledger is not None:
                for item in evidence:
                    self.ledger.record_evidence(item)

            self.tracer.emit(
                EventKind.GROUND,
                f"Hop {hop_index}: {'answered' if hop.answered else 'UNANSWERED'} "
                f"({len(grounded.supported)} grounded claim(s), coverage {hop.coverage:.0%})",
                data={"sub_question_id": sub.id},
            )
            report.hops.append(hop)

        if report.unanswered:
            self._warn(
                f"{len(report.unanswered)} sub-question(s) unanswered: "
                + "; ".join(q[:70] for q in report.unanswered)
            )
        return report

    def _triage(
        self,
        question: str,
        cov: coverage_mod.CoverageReport,
        suggested: Route,
        reason: str,
    ) -> TriageDecision:
        """Ask the agent to confirm or override a measured route.

        The model is given the coverage measurement, the passages the probe
        actually retrieved, and a suggested route with its justification. It
        can still override -- it sees the question's semantics, which BM25
        does not -- but it is now correcting a measurement rather than
        guessing from a filename.
        """
        summary = self.corpus.summary if self.corpus else "No local documents provided."
        terms = self.corpus.terms if self.corpus else []
        try:
            decision = self.client.structured(
                agents.TRIAGE_SYSTEM,
                agents.triage_prompt(question, summary, terms, cov, suggested, reason),
                TriageDecision,
            )
        except (LLMError, SchemaError, BudgetExceeded) as exc:
            # A failed triage is recoverable: fall back to the measurement,
            # which needed no model call in the first place.
            self._warn(f"Triage agent failed ({exc}); using the measured route.")
            decision = TriageDecision(
                route=suggested,
                confidence=0.0,
                rationale=f"Triage unavailable; using measured coverage. {reason}",
            )

        decision.measured_coverage = cov.score

        # Guard the terminal routes: a model that answers CLARIFY for every
        # slightly-broad question makes the tool useless, so clarification
        # requires an actual question to ask.
        if decision.route is Route.CLARIFY and not decision.clarifying_question:
            decision.route = suggested
            decision.rationale += " (clarify requested without a question; using measured route)"

        # NO_ANSWER before any retrieval is only credible when there is
        # genuinely nothing to search.
        if decision.route is Route.NO_ANSWER and (self._web_available() or cov.score > WEAK_ENOUGH):
            decision.route = suggested
            decision.rationale += " (sources are available; proceeding rather than abstaining)"

        self.tracer.emit(
            EventKind.TRIAGE,
            f"Route: {decision.route.value} (confidence {decision.confidence:.0%}) — {decision.rationale}"
            + (f" [measured suggestion was {suggested.value}]" if decision.route is not suggested else ""),
            data=decision.model_dump(mode="json"),
        )
        return decision

    def _initial_route(self, triage: TriageDecision) -> Route:
        """Reconcile the model's route with what is physically available.

        Handling only "no documents → web" leaves the inverse case broken:
        with web disabled and a web verdict, the run silently falls through to
        the local branch and searches an empty index.
        """
        has_local = bool(self.corpus and len(self.corpus))
        has_web = self._web_available()

        if not has_local and not has_web:
            return Route.LOCAL
        if not has_local:
            return Route.WEB
        if not has_web:
            return Route.LOCAL
        return triage.route

    def _gather(
        self,
        question: str,
        query: str,
        route: Route,
        top_k: int,
        expansions: Sequence[str],
        web_pool: list[EvidenceItem],
        round_index: int,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []

        if route in (Route.LOCAL, Route.HYBRID) and self.corpus and len(self.corpus):
            scored = self.corpus.retriever.search(query, top_k, expansions=expansions)
            evidence.extend(_to_evidence(scored))

        # Hop evidence leads: it answers sub-questions the top-level query
        # could not express, so dropping it would waste the hops entirely.
        if round_index == 1 and self._hop_evidence:
            seen = {e.source_id for e in evidence}
            evidence = [e for e in self._hop_evidence if e.source_id not in seen] + evidence

        if route in (Route.WEB, Route.HYBRID) and self._web_available():
            # Reuse pages already fetched in an earlier round rather than
            # paying the network cost again for the same URLs.
            if web_pool and route is Route.HYBRID:
                evidence.extend(web_pool)
            else:
                fetched = self._search_web(query, expansions, len(evidence) + 1, round_index)
                web_pool.clear()
                web_pool.extend(fetched)
                evidence.extend(fetched)

        return evidence

    def _search_web(
        self,
        query: str,
        expansions: Sequence[str],
        start_index: int,
        round_index: int,
    ) -> list[EvidenceItem]:
        queries = [query, *list(expansions)[:2]]
        results: list[WebResult] = []
        seen: set[str] = set()
        for q in queries:
            for result in websearch.search(q, self.settings, self.breaker):
                if result.url not in seen:
                    seen.add(result.url)
                    results.append(result)
            if len(results) >= self.settings.web_max_results:
                break

        if not results:
            self._warn("Web search returned no results (all backends failed or empty).", round_index)
            return []

        # Truncation is collected and surfaced. A page cut mid-document still
        # reads coherently, so an answer drawn from it looks complete while
        # the section holding the fact was never fetched.
        fetch_report: list[dict] = []
        pages = websearch.fetch_many(
            results, self.settings, breaker=self.breaker,
            deadline=self.budget.deadline, report=fetch_report,
        )
        truncated = [r for r in fetch_report if r.get("truncated")]
        if truncated:
            worst = min(r["kept"] / r["total"] for r in truncated)
            self._warn(
                f"{len(truncated)} page(s) truncated at {self.settings.fetch_max_chars:,} chars "
                f"(smallest fraction kept: {worst:.0%}); raise fetch_max_chars if an answer "
                "seems to be missing.",
                round_index,
            )
        if not pages:
            self._warn("Could not fetch any result pages; falling back to search snippets.", round_index)
        return _web_evidence(results[: self.settings.web_max_results], pages, self.settings, start_index)

    def _draft(self, question: str, evidence: Sequence[EvidenceItem]) -> ResearchDraft:
        try:
            return self.client.structured(
                agents.RESEARCH_SYSTEM,
                agents.research_prompt(question, evidence, self.settings.evidence_token_budget),
                ResearchDraft,
            )
        except (LLMError, SchemaError, BudgetExceeded) as exc:
            self._warn(f"Research agent failed: {exc}")
            return ResearchDraft(unanswered=["The research step failed to produce a draft."])

    def _entail(self, grounding, evidence, round_index: int):
        """Run the entailment stage and fold its verdicts back into grounding.

        Downgrade-only by construction: it reads ``grounding.supported`` and
        can move claims out of it, never in.
        """
        report = entailment_mod.EntailmentReport(ran=False)
        if not self.settings.use_entailment or not grounding.supported:
            return report

        report = entailment_mod.check_entailment(grounding.supported, evidence, self.client)
        if not report.ran:
            if report.error:
                self._warn(
                    f"Entailment check unavailable ({report.error}); lexical verdicts stand.",
                    round_index,
                )
            return report

        rejected = list(report.contradicted)
        if self.settings.entailment_strict:
            rejected += report.neutral

        if rejected:
            keep = {c.text for c in report.entailed}
            grounding.unsupported.extend(rejected)
            grounding.supported = [c for c in grounding.supported if c.text in keep]

        self.tracer.emit(
            EventKind.GROUND,
            f"Round {round_index}: entailment {len(report.entailed)}/{report.total} entailed"
            + (f", {len(report.contradicted)} CONTRADICTED" if report.has_contradiction else "")
            + (f", {len(report.neutral)} neutral" if report.neutral else ""),
            round_index,
            data={"entailment_rate": report.entailment_rate},
        )
        if report.has_contradiction:
            self._warn(
                f"{len(report.contradicted)} claim(s) contradicted by their own cited source.",
                round_index,
            )
        if report.unverifiable_quotes:
            self._warn(
                f"Entailment judge quoted text absent from the passage "
                f"({len(report.unverifiable_quotes)}); those claims downgraded.",
                round_index,
            )
        return report

    def _verify(
        self,
        question: str,
        draft: ResearchDraft,
        grounding: GroundingReport,
        route: Route,
        round_index: int,
    ) -> VerifierReport:
        try:
            return self.client.structured(
                agents.VERIFIER_SYSTEM,
                agents.verifier_prompt(
                    question,
                    draft,
                    grounding,
                    route,
                    round_index,
                    self.settings.max_rounds,
                    self._web_available(),
                ),
                VerifierReport,
            )
        except (LLMError, SchemaError, BudgetExceeded) as exc:
            # Fail closed on the *quality* judgement but open on control flow:
            # without a verdict we cannot claim sufficiency, yet blocking the
            # loop entirely would return nothing at all.
            self._warn(f"Verifier failed ({exc}); treating as partial.", round_index)
            return VerifierReport(
                verdict=Verdict.PARTIAL,
                next_action=NextAction.ANSWER,
                rationale="Verifier unavailable; answer not independently checked.",
            )

    def _can_answer(self, verifier: VerifierReport, grounding: GroundingReport) -> bool:
        """Gate on both the verifier's verdict and the mechanical support rate.

        The second condition is what stops a lenient verifier from waving
        through a draft whose citations do not hold up. If the verifier says
        "sufficient" but only a third of claims survived grounding, that
        disagreement is resolved in favour of the deterministic check.
        """
        if verifier.next_action is not NextAction.ANSWER and verifier.verdict is not Verdict.SUFFICIENT:
            return False
        if verifier.verdict is Verdict.INSUFFICIENT:
            return False
        if grounding.total and grounding.support_rate < self.settings.min_support_rate:
            self.tracer.emit(
                EventKind.WARNING,
                f"Verifier said {verifier.verdict.value} but only "
                f"{grounding.support_rate:.0%} of claims are grounded; continuing.",
            )
            return False
        return True

    def _escalate(
        self,
        verifier: VerifierReport,
        question: str,
        query: str,
        route: Route,
        top_k: int,
        expansions: list[str],
        round_index: int,
    ) -> tuple[Route, str, int, list[str]]:
        """Pick the next round's strategy from the verifier's instruction.

        Gaps are always folded into ``expansions`` regardless of the branch
        taken: they are the concrete, retrievable descriptions of what is
        missing, and they widen retrieval even when the route is unchanged.
        """
        action = verifier.next_action
        new_expansions = list(dict.fromkeys([*expansions, *verifier.gaps]))[:6]
        new_route, new_query, new_top_k = route, query, top_k

        if action is NextAction.ESCALATE_TO_WEB and self._web_available():
            new_route = Route.HYBRID if (self.corpus and len(self.corpus)) else Route.WEB
        elif action is NextAction.REFINE_QUERY and verifier.refined_query:
            new_query = verifier.refined_query
        elif action is NextAction.WIDEN_LOCAL:
            new_top_k = min(top_k * self.settings.widen_factor, 24)
        else:
            # The verifier asked for something impossible (web escalation with
            # web off, or refine with no query). Widening is the only strategy
            # guaranteed to be available, so the round is never wasted.
            new_top_k = min(top_k * self.settings.widen_factor, 24)
            if self._web_available() and route is Route.LOCAL:
                new_route = Route.HYBRID

        # A round identical to the previous one would burn budget for nothing.
        if (new_route, new_query, new_top_k) == (route, query, top_k) and new_expansions == expansions:
            new_top_k = min(top_k * self.settings.widen_factor, 24)

        self.tracer.emit(
            EventKind.ESCALATE,
            f"Escalating: {route.value}→{new_route.value}, top_k {top_k}→{new_top_k}"
            + (", query refined" if new_query != query else ""),
            round_index,
            gaps=verifier.gaps,
        )
        return new_route, new_query, new_top_k, new_expansions

    def _synthesize(
        self,
        question: str,
        grounding: GroundingReport,
        evidence: Sequence[EvidenceItem],
        verifier: VerifierReport,
    ) -> tuple[str, str, AnswerAudit | None]:
        if not grounding.supported and not evidence:
            return (
                "I could not find evidence to answer this question. "
                + ("No local documents were searchable and web search was unavailable."
                   if not self._web_available()
                   else "Neither the uploaded documents nor web search returned relevant material."),
                "low",
                None,
            )

        self.tracer.emit(EventKind.SYNTHESIZE, "Writing final answer")
        prompt = agents.synthesis_prompt(
            question, grounding, evidence, verifier, self.settings.evidence_token_budget
        )

        answer = ""
        audit: AnswerAudit | None = None

        # The synthesis prompt is a request, not a constraint. Audit the text
        # that actually comes back, and give the model one corrective turn
        # before falling back to something deterministic.
        for attempt in range(2):
            try:
                answer = self.client.complete(agents.SYNTHESIZER_SYSTEM, prompt)
            except (LLMError, BudgetExceeded) as exc:
                self._warn(f"Synthesis failed ({exc}); rendering verified claims directly.")
                self.synthesis_degraded = True
                return self._render_claims(grounding), "low", None

            audit = grounding_mod.verify_answer(answer, evidence)
            self.tracer.emit(
                EventKind.VERIFY,
                f"Answer audit: {len(audit.verified_sentences)}/{audit.total_cited} cited "
                f"sentences verified"
                + (f", fabricated: {', '.join(audit.fabricated_citations)}"
                   if audit.fabricated_citations else ""),
                data={"verified_rate": audit.verified_rate},
            )

            if audit.is_clean or attempt == 1:
                break

            if not audit.cited_sentences and grounding.supported:
                # An answer that cites nothing cannot be verified at all. It
                # is the least trustworthy output the pipeline can emit and
                # the easiest to mistake for a good one.
                self._warn("Final answer contained no citations; regenerating with sources required.")
            else:
                self._warn(
                    f"Final answer failed verification "
                    f"({audit.verified_rate:.0%} of cited sentences supported); regenerating."
                )
            prompt = agents.resynthesis_prompt(prompt, audit)

        if audit is not None and (
            (audit.total_cited and audit.verified_rate < 0.5)
            or (not audit.cited_sentences and grounding.supported)
        ):
            # Two attempts produced text the evidence does not support. Falling
            # back to a deterministic rendering of the verified claims is the
            # only option that keeps the guarantee intact.
            self._warn(
                "Synthesis could not be verified after 2 attempts "
                + (f"({audit.verified_rate:.0%} supported)" if audit.total_cited else "(no citations emitted)")
                + "; rendering verified claims instead."
            )
            self.synthesis_degraded = True
            answer = self._render_claims(grounding)
            audit = grounding_mod.verify_answer(answer, evidence)

        return answer, self._confidence(grounding, verifier, audit), audit

    @staticmethod
    def _render_claims(grounding: GroundingReport) -> str:
        """Deterministic answer built only from verified claims.

        The escape hatch when generation cannot be trusted: every line here
        came through grounding, so it cannot contain an unsupported assertion.
        """
        if not grounding.supported:
            return (
                "No claim in the drafted answer could be verified against the retrieved "
                "sources, so there is nothing here I can state as fact."
            )
        lines = ["Based only on claims verified against the sources:", ""]
        lines += [f"- {c.text} {' '.join(f'[{cid}]' for cid in c.citations)}" for c in grounding.supported]
        return "\n".join(lines)

    def _confidence(
        self,
        grounding: GroundingReport,
        verifier: VerifierReport,
        audit: AnswerAudit | None = None,
    ) -> str:
        """Confidence reflects the text the user reads, not just the draft.

        A high upstream support rate means nothing if the answer that was
        actually emitted does not hold up, so the audit can only lower the
        grade, never raise it.
        """
        if verifier.verdict is Verdict.SUFFICIENT and grounding.support_rate >= 0.8:
            level = "high"
        elif verifier.verdict is Verdict.INSUFFICIENT or grounding.support_rate < 0.5:
            level = "low"
        else:
            level = "medium"

        if audit is not None and audit.total_cited:
            if audit.fabricated_citations or audit.verified_rate < 0.5:
                return "low"
            if audit.verified_rate < 0.8 and level == "high":
                level = "medium"

        # Every line of a fallback rendering is verified, but the run reached
        # it only because generation twice produced unsupported text. Reporting
        # "high" would hide that.
        if self.synthesis_degraded and level == "high":
            return "medium"
        return level

    @staticmethod
    def _cited(evidence: Sequence[EvidenceItem], grounding: GroundingReport) -> list[EvidenceItem]:
        """Only return sources a surviving claim actually cited."""
        used = {cid for claim in grounding.supported for cid in claim.citations}
        by_id = {e.source_id: e for e in evidence}
        return [by_id[cid] for cid in sorted(used) if cid in by_id]

    # ------------------------------------------------------------------

    def _web_available(self) -> bool:
        return self.settings.web_enabled

    def _warn(self, message: str, round_index: int = 0) -> None:
        self.warnings.append(message)
        self.tracer.emit(EventKind.WARNING, message, round_index)


def research(
    question: str,
    documents: Sequence[Document] = (),
    settings: Settings | None = None,
    on_event: Callable | None = None,
) -> ResearchResult:
    """One-call entry point used by the CLI and tests."""
    settings = settings or Settings.from_env()
    tracer = Tracer(on_event=on_event)
    client = LLMClient(settings)
    corpus = Corpus(documents, settings, client, tracer) if documents else None
    return AdaptiveResearcher(settings, client, corpus, tracer).run(question)
