#!/usr/bin/env python3
"""Evaluation harness.

Without an eval set there is no way to tell whether a change made the
pipeline better or worse. This measures the dimensions that actually matter for a research
system, all of which are cheap to check because the run result is structured:

* **Route accuracy** -- did coverage-measured routing pick the right source?
* **Citation precision** -- what share of claims survived grounding?
* **Answer recall** -- do required facts appear in the final answer?
* **Abstention correctness** -- did it decline exactly when it should have?
  Scored both ways: answering when it should abstain is a hallucination, and
  abstaining when it could answer is uselessness. A system optimised for only
  one of these is trivially gamed.
* **Injection resistance** -- was the payload neutralised and kept out of the
  answer?
* **Cost and latency** per question.

Usage:
    python evals/run_eval.py                  # full set
    python evals/run_eval.py --filter route   # subset
    python evals/run_eval.py --baseline b.json --compare  # regression check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals import corpora  # noqa: E402
from ragverify.config import Settings  # noqa: E402
from ragverify.llm import LLMClient, LLMError  # noqa: E402
from ragverify.orchestrator import AdaptiveResearcher, Corpus  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    route: str = ""
    expected_route: str = ""
    outcome: str = ""
    expected_outcome: str = ""
    support_rate: float = 0.0
    rounds: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""
    route_exact: bool = False
    route_over_retrieved: bool = False
    answer_verified_rate: float = 0.0
    fabricated_citations: int = 0
    notes: list[str] = field(default_factory=list)


def load_cases(path: pathlib.Path, filter_: str | None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]
    return [c for c in cases if not filter_ or filter_ in c["id"]] if filter_ else cases


def run_case(case: dict[str, Any], settings: Settings) -> CaseResult:
    result = CaseResult(
        id=case["id"],
        question=case["question"],
        passed=False,
        expected_route=case.get("expect_route", ""),
        expected_outcome=case.get("expect_outcome", ""),
    )

    try:
        client = LLMClient(settings)
        documents = corpora.get(case.get("corpus", "empty"))
        corpus = Corpus(documents, settings, client) if documents else None
        run = AdaptiveResearcher(settings, client, corpus).run(case["question"])
    except LLMError as exc:
        result.error = str(exc)
        return result

    answer_lower = run.final_answer.lower()
    last = run.rounds[-1] if run.rounds else None

    result.route = run.rounds[0].route.value if run.rounds else (
        run.triage.route.value if run.triage else "none"
    )
    result.outcome = run.outcome.value
    result.support_rate = round(last.grounding.support_rate, 3) if last and last.grounding else 0.0
    result.rounds = len(run.rounds)
    result.cost_usd = round(run.usage.cost_usd, 5)
    result.elapsed_s = run.elapsed_s

    checks = result.checks

    if expected := case.get("expect_route"):
        # Counting hybrid as correct for a local-or-web expectation inflates
        # route accuracy: hybrid always retrieves from both, so it can never
        # be scored wrong, and a router that always answered "hybrid" would
        # report 100%. Exact match is the metric; hybrid is tracked separately
        # as over-retrieval, which is a cost problem rather than an error.
        checks["route"] = result.route == expected
        result.route_exact = result.route == expected
        result.route_over_retrieved = result.route == "hybrid" and expected != "hybrid"
        if result.route_over_retrieved:
            result.notes.append(f"routed hybrid where {expected} would have sufficed")

    if expected := case.get("expect_outcome"):
        checks["outcome"] = result.outcome == expected
        if expected == "abstained" and run.is_answer:
            result.notes.append("ANSWERED when it should have abstained (hallucination risk)")
        elif expected == "answered" and not run.is_answer:
            result.notes.append("ABSTAINED when the evidence supported an answer")

    # Required facts must appear in the answer the user actually sees.
    for needle in case.get("must_cite_text", []):
        checks[f"cites:{needle}"] = needle.lower() in answer_lower

    for needle in case.get("must_not_contain", []):
        present = needle.lower() in answer_lower
        checks[f"excludes:{needle}"] = not present
        if present:
            result.notes.append(f"LEAKED forbidden text: {needle!r}")

    if case.get("expect_injection"):
        checks["injection_detected"] = bool(run.injections_detected)
        if not run.injections_detected:
            result.notes.append("injection payload was NOT detected")

    # A grounded answer should clear the support floor.
    if run.is_answer and last and last.grounding and last.grounding.total:
        checks["grounded"] = last.grounding.support_rate >= settings.min_support_rate

    # Citation VALIDITY, not just fact presence. A substring check confirms a
    # number appears somewhere in the prose; it says nothing about whether the
    # inline citations resolve or whether the cited passage supports the
    # sentence. That is the guarantee this project makes, so it is measured.
    if run.is_answer and run.answer_audit is not None:
        audit = run.answer_audit
        result.answer_verified_rate = round(audit.verified_rate, 3)
        result.fabricated_citations = len(audit.fabricated_citations)
        checks["no_fabricated_citations"] = not audit.fabricated_citations
        if audit.total_cited:
            checks["answer_citations_supported"] = audit.verified_rate >= 0.8
        if case.get("require_clean_audit"):
            checks["clean_audit"] = audit.is_clean

    result.passed = all(checks.values()) if checks else False
    return result


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(r.passed for r in results)

    routed = [r for r in results if r.expected_route]
    abstain_cases = [r for r in results if r.expected_outcome == "abstained"]
    answer_cases = [r for r in results if r.expected_outcome == "answered"]
    injection_cases = [r for r in results if "injection_detected" in r.checks]

    return {
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "route_accuracy": round(
            sum(r.route_exact for r in routed) / len(routed), 3
        ) if routed else None,
        "route_over_retrieval": round(
            sum(r.route_over_retrieved for r in routed) / len(routed), 3
        ) if routed else None,
        "answer_citation_validity": round(
            sum(r.answer_verified_rate for r in answered) / len(answered), 3
        ) if (answered := [r for r in results if r.answer_verified_rate or r.outcome == "answered"]) else None,
        "fabricated_citation_cases": sum(bool(r.fabricated_citations) for r in results),
        "abstain_precision": round(
            sum(r.outcome == "abstained" for r in abstain_cases) / len(abstain_cases), 3
        ) if abstain_cases else None,
        "false_abstention": round(
            sum(r.outcome in ("abstained", "clarify") for r in answer_cases) / len(answer_cases), 3
        ) if answer_cases else None,
        "injection_detection": round(
            sum(r.checks.get("injection_detected", False) for r in injection_cases)
            / len(injection_cases), 3
        ) if injection_cases else None,
        "mean_support_rate": round(
            sum(r.support_rate for r in results) / total, 3
        ) if total else 0.0,
        "mean_rounds": round(sum(r.rounds for r in results) / total, 2) if total else 0.0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "mean_latency_s": round(sum(r.elapsed_s for r in results) / total, 2) if total else 0.0,
        "errors": sum(bool(r.error) for r in results),
    }


def print_report(results: list[CaseResult], summary: dict[str, Any]) -> None:
    print(f"\n{'id':<12} {'result':<7} {'route':<8} {'outcome':<10} {'grnd':<6} {'rds':<4} {'cost':<9} checks")
    print("-" * 100)
    for r in results:
        mark = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
        failed = [k for k, v in r.checks.items() if not v]
        detail = r.error[:40] if r.error else (", ".join(failed) if failed else "all ok")
        print(
            f"{r.id:<12} {mark:<7} {r.route:<8} {r.outcome:<10} "
            f"{r.support_rate:<6.2f} {r.rounds:<4} ${r.cost_usd:<8.5f} {detail}"
        )
        for note in r.notes:
            print(f"{'':>12}   ! {note}")

    print("\nSummary")
    print("-" * 40)
    for key, value in summary.items():
        if value is not None:
            print(f"  {key:<22} {value}")


def compare(summary: dict[str, Any], baseline_path: pathlib.Path) -> int:
    """Fail the run on a regression against a stored baseline."""
    baseline = json.loads(baseline_path.read_text())["summary"]
    # Metrics where lower is better; everything else is higher-is-better.
    lower_better = {"false_abstention", "errors", "mean_latency_s", "total_cost_usd"}
    regressions: list[str] = []

    print("\nCompared to baseline")
    print("-" * 60)
    for key, new in summary.items():
        old = baseline.get(key)
        if not isinstance(new, int | float) or not isinstance(old, int | float):
            continue
        delta = new - old
        worse = (delta > 1e-9) if key in lower_better else (delta < -1e-9)
        # Cost and latency drift within 20% is noise, not a regression.
        if key in ("total_cost_usd", "mean_latency_s") and old and abs(delta) / old < 0.2:
            worse = False
        flag = "REGRESSED" if worse else ""
        print(f"  {key:<22} {old:>8} -> {new:<8} ({delta:+.3f}) {flag}")
        if worse:
            regressions.append(key)

    if regressions:
        print(f"\n{len(regressions)} regression(s): {', '.join(regressions)}")
        return 1
    print("\nNo regressions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RagVerify eval set.")
    parser.add_argument("--dataset", type=pathlib.Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--filter", help="only run cases whose id contains this")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, help="write results as JSON")
    parser.add_argument("--baseline", type=pathlib.Path, help="baseline JSON to compare against")
    parser.add_argument("--compare", action="store_true", help="exit non-zero on regression")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {"model": args.model, "max_rounds": args.max_rounds}
    if args.no_web:
        overrides["web_enabled"] = False
    settings = Settings.from_env(**overrides)

    if not settings.api_key:
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    cases = load_cases(args.dataset, args.filter)
    print(f"Running {len(cases)} case(s) against {settings.model}…", file=sys.stderr)

    started = time.time()
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['id']}", file=sys.stderr, flush=True)
        results.append(run_case(case, settings))

    summary = summarize(results)
    summary["wall_clock_s"] = round(time.time() - started, 1)
    print_report(results, summary)

    if args.out:
        args.out.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]},
                indent=2,
            )
        )
        print(f"\nWrote {args.out}")

    if args.baseline and args.compare:
        return compare(summary, args.baseline)
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
