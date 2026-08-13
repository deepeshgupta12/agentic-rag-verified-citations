"""Command line interface.

A Streamlit-only interface cannot be scripted, batched or run in CI. Same
engine, no browser.

    ragverify "what changed in the 2024 policy?" -d ./docs
    ragverify "latest on the EU AI Act" --no-local --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .config import DEFAULT_MODEL, Settings
from .ingest import Document, clean_text, load_documents
from .llm import LLMError
from .orchestrator import AdaptiveResearcher, Corpus
from .trace import Event, EventKind, Tracer

_ICONS = {
    EventKind.START: "▸",
    EventKind.TRIAGE: "◆",
    EventKind.RETRIEVE: "⌕",
    EventKind.RESEARCH: "✎",
    EventKind.GROUND: "⚖",
    EventKind.VERIFY: "✓",
    EventKind.ESCALATE: "↑",
    EventKind.SYNTHESIZE: "★",
    EventKind.WARNING: "!",
    EventKind.DONE: "●",
}


def _read_paths(paths: list[str]) -> list[Document]:
    """Load documents from files and directories on disk."""
    documents: list[Document] = []
    for raw in paths:
        path = pathlib.Path(raw).expanduser()
        files = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
            if path.is_dir()
            else [path]
        )
        for file_path in files:
            if not file_path.is_file():
                print(f"skipping {file_path}: not a file", file=sys.stderr)
                continue
            if file_path.suffix.lower() == ".pdf":
                with file_path.open("rb") as handle:
                    handle.name = str(file_path)  # type: ignore[attr-defined]
                    documents.extend(load_documents([handle]))
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                documents.append(Document(name=file_path.name, pages=[clean_text(text)]))
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragverify",
        description="Evidence-gated multi-agent research.",
    )
    parser.add_argument("question", help="the research question")
    parser.add_argument("-d", "--docs", nargs="*", default=[], help="files or directories to search")
    parser.add_argument("-m", "--model", default=None, help=f"model id (default: {DEFAULT_MODEL})")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="passages per round")
    parser.add_argument("-r", "--max-rounds", type=int, default=None, help="max adaptive rounds")
    parser.add_argument("--no-web", action="store_true", help="disable web search")
    parser.add_argument("--no-local", action="store_true", help="ignore local documents")
    parser.add_argument("--no-embeddings", action="store_true", help="BM25 only, no embedding spend")
    parser.add_argument("--no-sanitize", action="store_true", help="skip prompt-injection neutralization")
    parser.add_argument(
        "--entailment", action="store_true",
        help="semantic entailment check on verified claims (catches 'most' cited as 'all'); "
             "costs one extra call per round",
    )
    parser.add_argument("--max-cost", type=float, default=None, help="hard spend cap in USD")
    parser.add_argument("--always-answer", action="store_true", help="never abstain, label confidence instead")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the live trace")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {"model": args.model, "top_k": args.top_k, "max_rounds": args.max_rounds}
    if args.no_web:
        overrides["web_enabled"] = False
    if args.no_embeddings:
        overrides["use_embeddings"] = False
    if args.no_sanitize:
        overrides["sanitize_sources"] = False
    if args.entailment:
        overrides["use_entailment"] = True
    if args.always_answer:
        overrides["abstain_below_support"] = 0.0
    if args.max_cost:
        overrides["max_cost_usd"] = args.max_cost
    settings = Settings.from_env(**overrides)

    if not settings.api_key:
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    # The trace goes to stderr so `--json` on stdout stays pipeable.
    def on_event(event: Event) -> None:
        if not args.quiet:
            icon = _ICONS.get(event.kind, "·")
            print(f"  {icon} {event.message}", file=sys.stderr, flush=True)

    documents = [] if args.no_local else _read_paths(args.docs)
    if args.docs and not documents and not args.no_local:
        print("warning: no readable documents found", file=sys.stderr)

    tracer = Tracer(on_event=on_event)
    try:
        from .llm import LLMClient

        client = LLMClient(settings)
        corpus = Corpus(documents, settings, client, tracer) if documents else None
        result = AdaptiveResearcher(settings, client, corpus, tracer).run(args.question)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    if result.injections_detected:
        print(
            "warning: a source attempted prompt injection ("
            + ", ".join(result.injections_detected)
            + "); neutralized.",
            file=sys.stderr,
        )

    print(f"\n{result.final_answer}\n")

    if result.citations:
        print("Sources")
        for item in result.citations:
            location = f" — {item.url}" if item.url else ""
            print(f"  [{item.source_id}] {item.label}{location}")

    if result.open_gaps:
        print("\nOpen gaps")
        for gap in result.open_gaps:
            print(f"  - {gap}")

    for warning in result.warnings:
        print(f"\nwarning: {warning}", file=sys.stderr)

    usage = result.usage
    print(
        f"\noutcome={result.outcome.value}  confidence={result.confidence}  "
        f"rounds={len(result.rounds)}  "
        f"{usage.calls} calls  {usage.prompt_tokens + usage.completion_tokens:,} tokens  "
        f"${usage.cost_usd:.4f}  {result.elapsed_s}s"
        + ("  [stopped early — gaps remain]" if result.stopped_early else ""),
        file=sys.stderr,
    )
    return 0 if result.is_answer else 3


if __name__ == "__main__":
    raise SystemExit(main())
