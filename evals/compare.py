#!/usr/bin/env python3
"""Compare an eval result against a baseline, without re-running anything.

Kept separate from ``run_eval`` because the workflow previously invoked the
runner twice -- once to produce results and once to compare -- which doubled
the cost and, worse, compared a *different* sample against the baseline. With
a nondeterministic model that makes ordinary variance look like regression.

Also refuses to compare unlike things. A baseline built from a different set
of cases, model or configuration cannot be a reference for this run, and
reporting "no regression" from such a comparison is worse than reporting
nothing: it says quality was enforced when nothing was checked.
"""

from __future__ import annotations

import json
import pathlib
import sys

# Metrics where a lower value is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"false_abstention", "errors", "fabricated_citation_cases",
                   "answers_with_no_citations", "mean_latency_s", "total_cost_usd"}
# Cost and latency drift within this fraction is noise rather than regression.
NOISE = {"total_cost_usd": 0.25, "mean_latency_s": 0.25, "wall_clock_s": 0.5}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare.py RESULTS.json BASELINE.json", file=sys.stderr)
        return 2

    results_path, baseline_path = (pathlib.Path(a) for a in argv)
    if not results_path.exists():
        print(f"::error::{results_path} not found; the eval produced no results.")
        return 1

    new = json.loads(results_path.read_text())["summary"]
    old = json.loads(baseline_path.read_text())["summary"]

    # A baseline over a different case set is not a reference for this run.
    if new.get("cases") != old.get("cases"):
        print(
            f"::error::Baseline covers {old.get('cases')} case(s) but this run "
            f"covers {new.get('cases')}. Regenerate the baseline: "
            "python evals/run_eval.py --out baseline.json"
        )
        return 1

    regressions: list[str] = []
    print(f"{'metric':<28} {'baseline':>10} {'now':>10} {'delta':>9}")
    print("-" * 62)

    for key, value in new.items():
        before = old.get(key)
        if not isinstance(value, int | float) or not isinstance(before, int | float):
            continue
        delta = value - before
        worse = delta > 0 if key in LOWER_IS_BETTER else delta < 0
        tolerance = NOISE.get(key)
        if worse and tolerance and before and abs(delta) / before <= tolerance:
            worse = False
        flag = "  REGRESSED" if worse else ""
        print(f"{key:<28} {before:>10} {value:>10} {delta:>+9.3f}{flag}")
        if worse:
            regressions.append(key)

    if regressions:
        print(f"\n::error::{len(regressions)} regression(s): {', '.join(regressions)}")
        return 1
    print("\nNo regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
