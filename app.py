"""RagVerify — Streamlit front end.

UI decisions that matter in use:

* The corpus is parsed, chunked and embedded once and cached in session state.
  Re-parsing every PDF on each button press would make follow-ups as slow
  as the first question.
* Steps stream as they happen. A single spinner over a multi-step pipeline
  makes a slow run look identical to a hung one.
* The adaptive loop is visible: each round shows its route, what the grounding
  check found, and why the verifier escalated.
* Follow-up questions reuse the built index instead of starting over.
* The API key stays in session state and is never written to ``os.environ``,
  which in a shared deployment leaked one visitor's key to the next.
"""

from __future__ import annotations

import streamlit as st

from ragverify.config import DEFAULT_MODEL, MODEL_PRICING, Settings
from ragverify.ingest import load_documents
from ragverify.llm import LLMClient, LLMError
from ragverify.orchestrator import AdaptiveResearcher, Corpus
from ragverify.schemas import Outcome, Verdict
from ragverify.trace import Event, EventKind, Tracer

st.set_page_config(page_title="RagVerify", page_icon="◆", layout="wide")

_BADGE = {"high": "🟢", "medium": "🟡", "low": "🔴"}
_VERDICT = {Verdict.SUFFICIENT: "🟢", Verdict.PARTIAL: "🟡", Verdict.INSUFFICIENT: "🔴"}
_OUTCOME = {
    Outcome.ANSWERED: "✅",
    Outcome.PARTIAL: "🟡",
    Outcome.ABSTAINED: "🚫",
    Outcome.CLARIFY: "❓",
    Outcome.BUDGET: "⏱",
}


def _init_state() -> None:
    st.session_state.setdefault("corpus", None)
    st.session_state.setdefault("corpus_key", None)
    st.session_state.setdefault("history", [])


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◆ RagVerify")
    st.caption("Evidence-gated multi-agent research")

    api_key = st.text_input("OpenAI API key", type="password", help="Kept in session only.")
    model = st.selectbox("Model", list(MODEL_PRICING), index=list(MODEL_PRICING).index(DEFAULT_MODEL))

    st.divider()
    st.markdown("**Adaptive loop**")
    max_rounds = st.slider(
        "Max rounds", 1, 5, 3,
        help="How many times the team may escalate retrieval before it must answer.",
    )
    min_support = st.slider(
        "Min grounding rate", 0.0, 1.0, 0.6, 0.05,
        help="Share of claims that must survive the citation check before an answer is allowed.",
    )

    st.divider()
    st.markdown("**Safety & budget**")
    sanitize_sources = st.toggle(
        "Sanitize retrieved text", value=True,
        help="Neutralize prompt-injection attempts in uploaded files and web pages.",
    )
    abstain_below = st.slider(
        "Abstain below grounding", 0.0, 1.0, 0.25, 0.05,
        help="Decline to answer when fewer than this share of claims verify. 0 disables abstention.",
    )
    max_cost = st.number_input("Max spend per run ($)", 0.05, 20.0, 1.0, 0.05)

    st.divider()
    st.markdown("**Retrieval**")
    top_k = st.slider("Passages per round", 3, 20, 6)
    use_embeddings = st.toggle("Hybrid (BM25 + embeddings)", value=True)
    web_enabled = st.toggle("Web search", value=True)

    st.divider()
    if st.session_state.corpus:
        st.caption(st.session_state.corpus.summary)
    if st.button("Clear index & history", use_container_width=True):
        st.session_state.corpus = None
        st.session_state.corpus_key = None
        st.session_state.history = []
        st.rerun()

settings = Settings.from_env(
    api_key=api_key,
    model=model,
    max_rounds=max_rounds,
    min_support_rate=min_support,
    top_k=top_k,
    use_embeddings=use_embeddings,
    web_enabled=web_enabled,
    sanitize_sources=sanitize_sources,
    abstain_below_support=abstain_below,
    max_cost_usd=float(max_cost),
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("RagVerify")
st.caption(
    "A research team that escalates until its citations hold up — "
    "or tells you exactly what it could not verify."
)

files = st.file_uploader(
    "Documents (optional — leave empty to research from the web)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

question = st.text_area("Research question", height=90, placeholder="What do you want to know?")
run = st.button("Research", type="primary", disabled=not question.strip())


def _corpus_key(uploaded) -> str:
    return "|".join(sorted(f"{f.name}:{f.size}" for f in uploaded)) if uploaded else ""


def _ensure_corpus(client: LLMClient, tracer: Tracer):
    """Build the index only when the uploaded set has actually changed."""
    key = _corpus_key(files)
    if not key:
        return None
    if st.session_state.corpus_key == key and st.session_state.corpus is not None:
        return st.session_state.corpus

    with st.status("Indexing documents…", expanded=False) as status:
        documents = load_documents(files)
        for doc in documents:
            if "unreadable" in doc.name:
                st.warning(doc.name)
        corpus = Corpus(documents, settings, client, tracer)
        status.update(label=corpus.summary, state="complete")

    st.session_state.corpus = corpus
    st.session_state.corpus_key = key
    return corpus


if run:
    if not settings.api_key:
        st.error("Add your OpenAI API key in the sidebar.")
        st.stop()

    trace_box = st.container()
    lines: list[str] = []
    placeholder = trace_box.empty()

    def on_event(event: Event) -> None:
        if event.kind is EventKind.DONE:
            return
        prefix = f"`R{event.round_index}`" if event.round_index else "`—`"
        mark = "⚠️" if event.kind is EventKind.WARNING else ""
        lines.append(f"{prefix} **{event.kind.value}** · {mark}{event.message}")
        placeholder.markdown("\n\n".join(lines[-14:]))

    tracer = Tracer(on_event=on_event)

    try:
        client = LLMClient(settings)
        corpus = _ensure_corpus(client, tracer)
        with st.spinner("Researching…"):
            result = AdaptiveResearcher(settings, client, corpus, tracer).run(question)
    except LLMError as exc:
        st.error(f"{exc}")
        st.stop()

    placeholder.empty()
    st.session_state.history.append(result)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

for result in reversed(st.session_state.history):
    st.divider()
    st.markdown(f"#### {result.question}")

    cols = st.columns(5)
    cols[0].metric("Outcome", f"{_OUTCOME[result.outcome]} {result.outcome.value}")
    cols[1].metric("Confidence", f"{_BADGE[result.confidence]} {result.confidence}")
    last = result.rounds[-1] if result.rounds else None
    cols[2].metric(
        "Grounded",
        f"{last.grounding.support_rate:.0%}" if last and last.grounding else "—",
    )
    cols[3].metric("Rounds", len(result.rounds))
    cols[4].metric("Cost", f"${result.usage.cost_usd:.4f}")

    if result.injections_detected:
        st.error(
            "**A source tried to give this pipeline instructions.** Detected and neutralized: "
            + ", ".join(f"`{k}`" for k in result.injections_detected)
            + ". Treat that document's contribution as low-trust."
        )

    if result.outcome is Outcome.ABSTAINED:
        st.warning("Declined to answer: no claim survived verification against the sources.")
    elif result.outcome is Outcome.BUDGET:
        st.warning(
            f"Stopped by a budget cap ({result.budget.get('calls', 0)} calls, "
            f"${result.usage.cost_usd:.4f}). The answer reflects evidence gathered so far."
        )
    elif result.stopped_early:
        st.warning(
            "Round budget was exhausted before the evidence was judged sufficient. "
            "The answer below is what the evidence supports; open gaps are listed."
        )

    st.markdown(result.final_answer)

    if result.citations:
        st.markdown("**Sources**")
        for item in result.citations:
            label = f"[{item.source_id}] {item.label}"
            st.markdown(f"- {f'[{label}]({item.url})' if item.url else label}")

    if result.open_gaps:
        with st.expander(f"Open gaps ({len(result.open_gaps)})"):
            for gap in result.open_gaps:
                st.markdown(f"- {gap}")

    with st.expander(f"Adaptive trace — {len(result.rounds)} round(s)"):
        if result.triage:
            coverage = result.triage.measured_coverage
            st.markdown(
                f"**Triage** → `{result.triage.route.value}` "
                f"({result.triage.confidence:.0%}) — {result.triage.rationale}"
            )
            if coverage is not None:
                st.caption(f"Measured local coverage before routing: {coverage:.2f}")
        for record in result.rounds:
            verifier, ground = record.verifier, record.grounding
            icon = _VERDICT.get(verifier.verdict, "•") if verifier else "•"
            st.markdown(
                f"---\n**Round {record.index}** · route `{record.route.value}` · "
                f"top_k {record.top_k} · {record.n_evidence} passages · {record.elapsed_s}s"
            )
            if record.query != result.question:
                st.caption(f"Query used: _{record.query}_")
            if ground:
                st.markdown(
                    f"{icon} Grounding: **{len(ground.supported)}/{ground.total}** claims verified"
                    + (
                        f" · fabricated citations: `{', '.join(ground.hallucinated_citations)}`"
                        if ground.hallucinated_citations
                        else ""
                    )
                )
                if ground.unsupported:
                    st.caption("Rejected: " + " · ".join(c.text[:90] for c in ground.unsupported[:3]))
            if verifier:
                st.markdown(
                    f"Verdict **{verifier.verdict.value}** → next action "
                    f"`{verifier.next_action.value}` — {verifier.rationale}"
                )

    if result.warnings:
        with st.expander(f"Warnings ({len(result.warnings)})"):
            for warning in result.warnings:
                st.markdown(f"- {warning}")

if not st.session_state.history:
    st.info(
        "Upload documents and ask a question, or ask with no documents to research from the web.\n\n"
        "RagVerify checks every citation against its source text before answering. "
        "If the check fails, it retrieves again — widening the search, rephrasing the query, "
        "or going to the web — until the evidence holds or the round budget runs out."
    )
