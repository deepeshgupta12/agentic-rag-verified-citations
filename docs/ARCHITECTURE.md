# Architecture

## Why the loop is shaped this way

The pipeline exists to answer one question honestly: *does the retrieved
evidence actually support an answer?* Everything follows from making that
question load-bearing rather than advisory.

### 1. Measure before routing

A cheap BM25 probe runs before any model call. Routing on a measurement, rather
than on a model's guess from filenames, means the expensive decision is
constrained by evidence. The triage agent can still override — it understands
synonymy, which lexical retrieval does not — but it is correcting a number, not
inventing one.

### 2. Ground before verifying

Grounding is deterministic and runs *before* the verifier. This ordering is the
whole design:

- The verifier sees which citations already failed a mechanical check, so it
  cannot be talked into "sufficient" by a confident draft.
- The grounding result can overrule the verifier (`min_support_rate`). A model
  marking its own team's work as sufficient is exactly the failure mode
  grounding exists to catch.
- Because it reads only source text, injected prose in a document cannot
  influence it.

### 3. Escalate, don't retry

Each round is strictly more capable than the last: widen retrieval → rephrase
the query using the verifier's gap statements → add web evidence. Repeating a
failing strategy burns budget for nothing, so `_escalate` guarantees the next
round differs from the last even when the verifier asks for something
impossible.

### 4. Abstain

If the round budget runs out and nothing verifies, the run returns an
abstention rather than an answer. An answer built on zero verified claims is
worse than none: it reads exactly like a good one.

## Trust boundaries

```
operator prompts        ← trusted
        │
        ▼
┌───────────────────────────────────┐
│ BOUNDARY_PREAMBLE                 │  explicit rules that override passages
├───────────────────────────────────┤
│ │││ BEGIN UNTRUSTED SOURCE S1     │  ← uploaded files, fetched pages
│ ...sanitized passage text...      │     attacker-controllable
│ │││ END UNTRUSTED SOURCE S1       │
└───────────────────────────────────┘
```

Local documents are treated exactly like web pages. A poisoned PDF is as
effective as a poisoned page, and trusting uploads because the user chose them
assumes the user knows what is in them.

## Failure modes and their handling

| Failure | Behaviour |
|---|---|
| Triage call fails | Fall back to the measured route (needed no model call) |
| Research call fails | Round yields no draft; loop continues or reports |
| Verifier call fails | Degrade to `partial`; confidence cannot be `high` |
| Embedding fails | Retriever drops to BM25, records a warning |
| All SearxNG endpoints fail | DuckDuckGo fallback, then local-only |
| Page fetch fails | Fall back to the search snippet, marked as such |
| Budget exhausted | Stop at a round boundary, return what verified |
| Nothing verifies | Abstain |

Every one of these degrades. None raises out of the UI.
