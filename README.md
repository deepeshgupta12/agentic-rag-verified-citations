# Agentic RAG with Verified Citations

**A retrieval pipeline that proves its citations before it answers — and tells you when it can't.**

[![CI](https://github.com/deepeshgupta12/agentic-rag-verified-citations/actions/workflows/ci.yml/badge.svg)](https://github.com/deepeshgupta12/agentic-rag-verified-citations/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-343%20passing-brightgreen)](tests/)

*Agentic RAG · citation verification · NLI entailment · hallucination resistance · self-correcting retrieval · multi-hop*

---

## The problem

A RAG system that cites its sources feels trustworthy. Usually it isn't.

The citation is produced by the same model that wrote the claim, in the same breath, with no mechanism connecting the two. `[S3]` is a *token the model chose*, not a fact anyone checked. Five failure modes follow, and every one looks identical to a correct answer:

| Failure | What it looks like |
|---|---|
| **Fabricated citation** | Cites `[S7]` when only 5 passages were retrieved |
| **Cited-but-irrelevant** | `[S2]` is real, but says nothing about the claim |
| **Invented number** | Source says 34%, answer says 47%, everything else matches |
| **Over-generalisation** | Source says "most regions", answer says "all regions" |
| **Attribution slip** | Source says "the CEO claimed X", answer states X as fact |

The usual mitigation is a "verifier" agent asked whether the evidence is sufficient. That check is almost always decorative: the pipeline calls it, receives `{"verdict": "insufficient"}`, and writes the answer anyway, because nothing in the control flow reads the verdict.

## What this does instead

**Verification is a gate, not a label** — and it runs on the answer you actually read, not only on an intermediate draft.

```
                    ┌──────────────────────────────────┐
                    │  coverage probe (BM25, free)     │
                    └────────────────┬─────────────────┘
                                     ▼
                route: local │ web │ hybrid │ clarify │ no_answer
                                     │
   ┌─────────────────────────────────▼──────────────────────────────────┐
   │ retrieve ─► rerank ─► draft claims ─► GROUND ─► ENTAIL ─► verify   │
   │     ▲                             (mechanical)  (semantic)     │   │
   │     └────────── escalate: widen · rephrase · web ──────────────┘   │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     ▼
                          synthesize ─► AUDIT THE ANSWER
                                            │
                                  fails ────┴──► regenerate ──► render
                                                                verified
              answer │ partial │ abstain │ budget-stopped        claims
```

Three checks in series, each catching what the previous cannot:

1. **Lexical grounding** — deterministic and free. Does the cited passage contain the claim's words and figures? Catches fabricated citations, irrelevant sources, invented numbers. Cannot be argued out of its verdict by a confident draft.
2. **Entailment** — optional, semantic. Does the passage *mean* what the claim says? Catches "most"→"all", negation, scope, modality, attribution.
3. **Answer audit** — deterministic. Is the final text itself supported, sentence by sentence?

Grounding can overrule the verifier:

```python
# orchestrator.py — a lenient verifier does not get the last word
if grounding.total and grounding.support_rate < settings.min_support_rate:
    return False  # verifier said "sufficient"; the citations disagree
```

If nothing survives, it **abstains** rather than writing a fluent answer over nothing.

---

## Quickstart

```bash
git clone https://github.com/deepeshgupta12/agentic-rag-verified-citations
cd agentic-rag-verified-citations
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
```

```bash
streamlit run app.py                                      # web app
ragverify "What changed in the 2024 policy?" -d ./docs    # CLI
ragverify "Q3 margin vs peers" -d ./docs --entailment --rerank llm
```

```python
from ragverify import Settings, research
from ragverify.ingest import Document

result = research(
    "What was European revenue growth?",
    documents=[Document(name="q3.txt", pages=[report_text])],
    settings=Settings.from_env(use_entailment=True),
)

result.outcome        # answered | partial | abstained | clarify | budget
result.confidence     # high | medium | low
result.citations      # only sources a *verified* claim cited
result.answer_audit   # per-sentence verification of the final text
result.ledger         # content hashes + claim→source edges
result.plan           # per-sub-question coverage, if multi-hop ran
```

Exit code `3` from the CLI means it abstained. Any OpenAI-compatible endpoint works via `OPENAI_BASE_URL`.

> **`.env` is resolved from the current directory upward**, which is standard
> dotenv behaviour. Running `ragverify` from outside the project will not see
> the project's `.env` and will report no API key. Either run from the project
> root, or `export OPENAI_API_KEY=...` in your shell.

---

## How it works

### Routing is measured, not guessed
A free BM25 probe runs first, scoring term recall, match strength and concentration. Triage then *confirms or overrides a measurement* with the retrieved passages in view — it can see synonymy that lexical retrieval cannot. Two routes terminate immediately: `clarify` (several readings needing different evidence) and `no_answer` (unanswerable in principle), both guarded so a model cannot stall on every broad question.

### Retrieval is hybrid, then reranked
BM25 with IDF and length normalisation, optional dense embeddings, fused with Reciprocal Rank Fusion, diversified with MMR. RRF because BM25 scores and cosine similarities live on incomparable scales; it reads rank order only.

Optional reranking scores each passage as an *answer to the question* — retrieval scores it in isolation. This matters because the evidence budget truncates: a passage below the token limit never reaches the model, and no downstream verification recovers what was never shown.

### Values compare by meaning, not spelling
`2.1 billion euro` and `EUR 2,100,000,000` are one fact. `34%` and `34 staff` are not — percentages, money and bare counts are separately typed. Dates canonicalise across ISO, month-name, quarter, spelled-out-quarter and fiscal-year forms. Deliberately strict where ambiguous: an over-eager normaliser makes fabrication *easier*, so a false rejection (costing one round) beats a false accept.

### Evidence is verbatim, and auditable afterwards
Stable chunk ids, page numbers, URLs. Web results are fetched and extracted, not reasoned about from snippets.

The ledger records a content hash and snapshot of every passage **both before and after sanitisation** — grounding checks the sanitised text, so an audit needs both — plus one claim→source edge per citation carrying its verdict: `supported`, `dropped` (real passage, does not support) or `fabricated` (never retrieved). Only the last is a hallucination, and a flat list of surviving ids collapses all three into one.

### Web sources are ranked, never silently dropped
Publisher class, recency (half-life chosen by whether the question asks for current information), domain diversity, and cross-domain corroboration. Corroboration counts *registrable domains*, so five pages from one site are one source wearing five hats. Nothing is discarded — a weak source that is the only one answering the question is still the answer.

### Multi-hop when the question needs it
*"What did the person who signed the 2024 audit letter previously run?"* cannot be answered by retrieving harder: the second query cannot be written until the first is answered. Plans are a validated DAG — cycles cut, dangling ids dropped, breadth truncated — and hops execute topologically with **entities** substituted into dependent queries.

Coverage is tracked **per sub-question**, because one aggregate hides the case this exists for: a decisive hop unanswered while the others carry the mean. Most questions are single-hop, and the planner is asked to say so.

---

## Safety

Uploaded documents and fetched pages are **attacker-controllable text** that ends up in prompts. Three layers: passages fenced as untrusted, six classes of model-directed instruction neutralised in place (marked, not deleted, so a document *discussing* prompt injection stays usable as evidence), and detections surfaced in the result. Local files get identical treatment — a poisoned PDF works as well as a poisoned page.

The web fetcher validates **every redirect hop** against resolved addresses, refusing private, loopback, link-local (cloud metadata) and reserved ranges, with 5 MB and 5-redirect caps. An SSRF block deliberately does *not* trip the host circuit breaker: it is a policy decision, not a transport failure, and one bad link must not disable a working domain.

Every external call — LLM, embedding, search, fetch — is metered against one shared budget with a single propagated deadline. Per-request timeouts bound one call each and compose into an unbounded total.

---

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `max_rounds` | 3 | Escalation rounds before it must answer |
| `min_support_rate` | 0.6 | Grounding floor to accept a "sufficient" verdict |
| `abstain_below_support` | 0.25 | Below this after the last round, decline |
| `use_entailment` | `False` | Semantic check; one call/round, ~2× verification latency |
| `rerank_method` | `none` | `llm` or `cross-encoder` |
| `use_multi_hop` | `False` | Chained retrieval for genuinely sequential questions |
| `assess_source_quality` | `True` | Rank and warn on web evidence |
| `cache_embeddings` | `True` | Content-hash cache — **146× faster** on a warm corpus |
| `sanitize_sources` | `True` | Neutralise injections in retrieved text |
| `max_cost_usd` / `max_seconds` / `max_calls` | 1.00 / 180 / 40 | Hard caps, per run |

**Raise `max_seconds` when enabling the optional stages.** A single question
with reranking and entailment takes roughly 110 seconds, so the 180-second
default — calibrated before those existed — leaves no room for a second round.
The loop then abstains where it would otherwise have escalated. 420 seconds is
a realistic ceiling for the full pipeline.

Budgets are per *run*, not per client. A client reused across many questions
(a Streamlit session, a batch job, a server) gets a fresh budget for each
one.
| `telemetry` | `False` | OpenTelemetry spans |

Optional extras: `pip install ragverify[rerank]` (cross-encoder), `ragverify[otel]` (tracing).

---

## Evaluation

```bash
python evals/run_eval.py --out baseline.json
python evals/run_eval.py --baseline baseline.json --compare   # non-zero on regression
```

**25 cases across 9 domains**: finance, contradictions between sources, tabular data, temporal (superseded facts), multilingual, OCR-damaged scans, multi-hop, adversarial/poisoned corpora, and low-authority sources.

Route accuracy is **exact match**, with hybrid tracked separately as over-retrieval — counting hybrid as correct means a router that always says "hybrid" scores 100%. Abstention is scored **in both directions**: answering when it should abstain is hallucination, abstaining when it could answer is uselessness, and optimising either alone is trivially gamed.

Live evals run on manual dispatch, on PRs labelled `run-evals`, and weekly. The schedule catches drift in the *model behind the API* — the failure a commit-triggered run cannot see, because nothing in the repository changed.

## Testing

```bash
pip install -e ".[dev]"
pytest                    # 343 tests, no API key, no network
```

Tests run against a scripted fake LLM that *subclasses the real client*, so retries, usage accounting and the structured-output path are exercised rather than stubbed.

---

## Architecture

```
ragverify/
├── orchestrator.py   the adaptive loop
├── coverage.py       retrieve-first routing measurement
├── grounding.py      lexical verification + final-answer audit
├── entailment.py     semantic NLI gate (downgrade-only)
├── normalize.py      canonical numbers, money, dates, units
├── ledger.py         immutable evidence record, claim→source edges
├── planner.py        multi-hop DAG, per-sub-question coverage
├── rerank.py         cross-encoder / LLM reranking
├── sourcequality.py  authority, freshness, diversity, corroboration
├── retrieval.py      BM25 + dense + RRF + MMR
├── sanitize.py       untrusted-source boundary
├── budget.py         caps, deadline, circuit breaker
├── store.py          persistent embedding cache
├── telemetry.py      OpenTelemetry spans (optional)
├── llm.py            structured outputs, retries, cost accounting
├── jsonx.py          balanced-span JSON extraction
├── schemas.py        typed contracts for every hop
├── agents.py         role prompts and evidence formatting
├── config.py         settings, .env loading, model pricing
├── websearch.py      failover, SSRF-guarded fetch, extraction
├── ingest.py         loading, cleaning, token-aware chunking
├── tokens.py         token counting with a character-ratio fallback
├── trace.py          in-process event stream for the UI
├── ag2_team.py       optional AG2 GroupChat adversarial review
└── cli.py
```

Deeper notes in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Limitations

- **With entailment off (the default), grounding is recall-based.** "Most regions" cited for "all regions" still passes. Turn it on with `use_entailment=True` for the semantic check.
- **Entailment is itself a model judgement** and can be wrong. It only ever *downgrades* — a claim rejected lexically is never revived — so enabling it cannot make the pipeline accept something it previously refused.
- **Source authority is a coarse heuristic**: domain suffix plus a small known-domain table. An unrecognised domain scores neutral rather than bad, but a well-presented unreliable site scores the same as a good one.
- **Contradictions are surfaced, not adjudicated.** When two sources disagree, both are reported; the pipeline does not decide which is right.
- **Multi-hop planning is opt-in and imperfect.** The planner sometimes judges a question single-hop when chaining would have helped.
- **SSRF protection resolves addresses at validation time.** Every redirect hop is re-checked, but DNS rebinding — the address changing between check and connection — needs connection-time pinning, which this does not do.
- **25 synthetic cases is a smoke test, not a benchmark.** There is no human-labelled groundedness set, and coverage thresholds are tuned on these fixtures. Re-tune them on your data.

## Roadmap

- Human-labelled groundedness set with inter-annotator agreement
- Contradiction *resolution* rather than disclosure
- Table-aware chunking with cell-level provenance
- Streaming answers with incremental verification
- Connection-time DNS pinning to close rebinding

## License

MIT — see [LICENSE](LICENSE).
