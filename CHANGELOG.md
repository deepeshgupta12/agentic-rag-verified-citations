# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-13

Initial release.

### Core

- **Adaptive research loop** (`orchestrator.py`) — the verifier's verdict is a
  control-flow gate, not a label. A failed verification sends the pipeline back
  to retrieval with a concrete list of what's missing, escalating through widen
  → rephrase → web across a bounded number of rounds.
- **Deterministic citation grounding** (`grounding.py`) — every claim is checked
  against the verbatim text of the sources it cites, with no model in the loop.
  Catches fabricated citations, cited-but-irrelevant sources, and invented
  numbers. Can overrule a lenient verifier.
- **Explicit abstention** — when the round budget is spent and nothing verifies,
  the pipeline declines to answer and lists its open gaps rather than writing a
  fluent answer over unverified claims.
- **Coverage-measured routing** (`coverage.py`) — a free BM25 probe runs before
  any model call and scores term recall, match strength and concentration.
  Routing follows the measurement; the triage agent confirms or overrides it
  with the retrieved passages in view. Adds `clarify` and `no_answer` as
  terminal routes.

### Retrieval

- **Hybrid retrieval** (`retrieval.py`) — BM25 with IDF and length
  normalisation, optional dense embeddings, Reciprocal Rank Fusion, and MMR
  diversification to stop near-duplicate passages consuming the evidence budget.
- **Token-aware chunking** (`ingest.py`) — chunks sized in tokens and split on
  paragraph boundaries, with stable chunk ids, page numbers and URLs so every
  citation resolves to a specific location.
- **Full-page web evidence** (`websearch.py`) — result pages are fetched and
  extracted rather than reasoned about from search snippets, making web evidence
  checkable on the same terms as local evidence.

### Safety

- **Prompt-injection defence** (`sanitize.py`) — retrieved text is fenced as
  untrusted data, six classes of model-directed instruction are neutralised in
  place, and detections are surfaced in the result. Applied to uploaded files
  and web pages alike.
- **Budget caps and circuit breaker** (`budget.py`) — hard limits on time, cost
  and call count, checked before escalating so the loop stops at a clean
  boundary with a usable answer.

### Reliability

- **Structured outputs** (`llm.py`) — Pydantic validation with repair retries;
  a malformed response raises at the boundary that produced it instead of
  degrading into an empty dict.
- **Balanced-span JSON extraction** (`jsonx.py`) — string- and escape-aware
  scanning that survives prose containing braces, fenced blocks and multiple
  objects. Returns `None` on failure rather than a misleading `{}`.
- Bounded retries with jittered backoff, SearxNG endpoint failover, DuckDuckGo
  fallback, and graceful degradation on every external dependency.

### Interfaces

- Streamlit app with a live trace of the adaptive loop, per-round grounding
  results, and cost/latency metrics.
- CLI (`ragverify`) with JSON output and a distinct exit code for abstention.
- Python library API via `ragverify.research()`.
- Optional AG2 `GroupChat` backend for adversarial critic/defender review.

### Quality

- 143 offline tests — no API key, no network — run against a scripted fake LLM
  that subclasses the real client, so retries, usage accounting and the
  structured-output path are exercised rather than stubbed.
- Evaluation harness with regression comparison, measuring route accuracy,
  citation precision, answer recall, injection resistance, and abstention
  scored in both directions.
- CI across Python 3.10–3.12 with pinned linting.
