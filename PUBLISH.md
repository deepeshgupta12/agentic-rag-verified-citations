# Remaining publish steps

The repo is live at
<https://github.com/deepeshgupta12/agentic-rag-verified-citations> and `main`
is pushed. Two things still need `gh` (not authenticated in the build
environment):

```bash
gh auth login          # GitHub.com -> HTTPS -> browser
cd ~/agentic-rag-verified-citations
```

## 1. Topics — the biggest in-platform discovery lever

```bash
gh repo edit --add-topic "$(paste -sd, .github/topics.txt)"
```

## 2. Repo description

```bash
gh repo edit --description "Agentic RAG that verifies every citation against source text before answering, and abstains when the evidence does not hold. Self-correcting retrieval loop, deterministic citation grounding, prompt-injection defence."
```

## 3. Verify

```bash
gh repo view --web
gh run list          # CI should be green
```

## Optional

```bash
gh repo edit --enable-discussions
```

For PyPI, the package name in `pyproject.toml` is `ragverify` — check
availability with `pip index versions ragverify` before publishing.
