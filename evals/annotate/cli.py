#!/usr/bin/env python3
"""Groundedness annotation: extract, label, agree, adjudicate, score.

    python -m evals.annotate.cli extract    --questions q.txt --docs ./corpus
    python -m evals.annotate.cli label      --annotator alice
    python -m evals.annotate.cli agreement
    python -m evals.annotate.cli adjudicate --annotator lead
    python -m evals.annotate.cli score

The point of the exercise is to obtain a judgement the pipeline did not make.
Every quality number reported so far came from the same lexical rules the
product uses to decide, which measures self-consistency rather than
correctness. A human label is the only thing that breaks that circularity.

Two design choices follow from it:

* The pipeline's own verdict is **stored but never shown** during labelling.
  An annotator told what the system decided will agree with it more often,
  and the resulting agreement measures suggestion rather than judgement.
* Items are presented in a **stable shuffled order**, seeded per annotator.
  Labelling in retrieval order means the easy citations arrive first and
  fatigue lands entirely on the hard ones.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter

from .agreement import agreement_report
from .schema import Annotation, AnnotationItem, GoldItem, Label

DATA = pathlib.Path("evals/annotate/data")
ITEMS = DATA / "items.jsonl"
GOLD = DATA / "gold.jsonl"


def _annotations_path(annotator: str) -> pathlib.Path:
    return DATA / f"labels-{annotator}.jsonl"


def _read(path: pathlib.Path, model):
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: pathlib.Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r.model_dump_json() for r in records) + "\n")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    """Run the pipeline over questions and emit every claim-citation pair.

    Pairs are emitted whether the pipeline accepted or rejected them.
    Extracting only accepted ones would measure precision and leave recall
    unmeasurable -- the citations it wrongly threw away would never be seen.
    """
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from ragverify.config import Settings
    from ragverify.ingest import Document, clean_text
    from ragverify.llm import LLMClient
    from ragverify.orchestrator import AdaptiveResearcher, Corpus

    questions = [
        q.strip() for q in pathlib.Path(args.questions).read_text().splitlines() if q.strip()
    ]
    docs = []
    for path in sorted(pathlib.Path(args.docs).glob("*.txt")):
        docs.append(Document(name=path.name, pages=[clean_text(path.read_text())]))
    if not docs:
        print(f"no .txt documents in {args.docs}", file=sys.stderr)
        return 1

    settings = Settings.from_env(web_enabled=False, max_rounds=args.max_rounds)
    client = LLMClient(settings)
    corpus = Corpus(docs, settings, client)

    existing = {i.item_id for i in _read(ITEMS, AnnotationItem)}
    items: list[AnnotationItem] = _read(ITEMS, AnnotationItem)

    for n, question in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] {question}", file=sys.stderr, flush=True)
        # The researcher is held rather than discarded because the ledger it
        # owns still carries passage text. `result.ledger` is serialised with
        # include_text=False, which is right for a result payload -- hashes
        # prove which passage was used without copying document content into a
        # shareable artifact -- and useless here, where the text is the thing
        # a human has to read.
        researcher = AdaptiveResearcher(settings, client, corpus)
        run = researcher.run(question)
        if not run.rounds:
            continue

        by_id = {}
        for record in run.rounds:
            for claim_list, verdict in (
                (record.grounding.supported, "accepted"),
                (record.grounding.unsupported, "rejected"),
            ):
                for claim in claim_list:
                    for cid in claim.citations or ["(none)"]:
                        by_id[(claim.text, cid)] = verdict

        # Every passage the run SAW, not only those a surviving claim cited.
        # `run.citations` returns sources cited by supported claims alone, so
        # using it drops every rejected pair -- leaving precision measurable
        # and recall not, which is the opposite of the intent stated above.
        sources = {}
        ledger = getattr(researcher, "ledger", None)
        if ledger is not None:
            for record in ledger.records.values():
                sources[record.source_id] = record

        for (claim_text, cid), verdict in by_id.items():
            record = sources.get(cid)
            if record is None:
                continue
            # The sanitized text is what grounding actually checked against,
            # so it is what a human should judge.
            source_text = record.sanitized_text or record.raw_text or ""
            source_label = getattr(record, "label", cid)
            if not source_text:
                continue
            item_id = AnnotationItem.make_id(claim_text, cid, source_text)
            if item_id in existing:
                continue
            existing.add(item_id)
            items.append(AnnotationItem(
                item_id=item_id, question=question, claim=claim_text,
                source_id=cid, source_label=source_label, source_text=source_text,
                pipeline_verdict=verdict,
            ))

    _write(ITEMS, items)
    accepted = sum(i.pipeline_verdict == "accepted" for i in items)
    rejected = len(items) - accepted
    print(f"\n{len(items)} item(s) in {ITEMS}", file=sys.stderr)
    print(f"  {accepted} accepted, {rejected} rejected by the pipeline", file=sys.stderr)

    if len(items) < 300:
        print(
            f"note: {len(items)} pairs is below the ~300-500 needed for a stable kappa.",
            file=sys.stderr,
        )
    # Recall is measured over the pairs the pipeline THREW AWAY. Without a
    # reasonable number of them, the score can only report precision, and a
    # precision-only number flatters a system tuned to refuse.
    if rejected < max(20, len(items) // 10):
        print(
            f"warning: only {rejected} rejected pair(s). Recall is measured over "
            "citations the pipeline discarded, so it cannot be estimated from this "
            "set. Add questions the corpus answers poorly to produce more.",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

_KEYS = {
    "s": Label.SUPPORTED, "p": Label.PARTIAL, "u": Label.UNSUPPORTED,
    "c": Label.CONTRADICTED, "?": Label.UNCLEAR,
}


def cmd_label(args: argparse.Namespace) -> int:
    items = _read(ITEMS, AnnotationItem)
    if not items:
        print(f"no items; run extract first ({ITEMS} missing)", file=sys.stderr)
        return 1

    done = {a.item_id for a in _read(_annotations_path(args.annotator), Annotation)}

    # Per-annotator shuffle, seeded by name so a session can be resumed in the
    # same order. Retrieval order front-loads the easy pairs, which puts all
    # the fatigue on the hard ones.
    ordered = list(items)
    random.Random(args.annotator).shuffle(ordered)
    queue = [i for i in ordered if i.item_id not in done][: args.limit]

    if not queue:
        print(f"{args.annotator}: nothing left to label ({len(done)} done)", file=sys.stderr)
        return 0

    return _label_queue(queue, args.annotator, args.chars)


def _label_queue(queue: list, annotator: str, chars: int) -> int:
    """The interactive labelling loop, shared by label and retest."""
    path = _annotations_path(annotator)
    annotations = _read(path, Annotation)

    print(f"\n{len(queue)} item(s) to label. Keys: "
          "[s]upported [p]artial [u]nsupported [c]ontradicted [?]unclear "
          "[n]ote [b]ack [q]uit\n")

    index = 0
    while index < len(queue):
        item = queue[index]
        print("=" * 78)
        print(f"({index + 1}/{len(queue)})  QUESTION: {item.question}")
        print(f"\nCLAIM:\n  {item.claim}")
        print(f"\nCITED PASSAGE  [{item.source_id}] {item.source_label}:\n")
        for line in item.source_text[:chars].splitlines():
            print(f"  {line}")
        if len(item.source_text) > chars:
            print(f"  … ({len(item.source_text) - chars} more characters)")
        print("\nDoes this passage support this claim?")

        started = time.time()
        note = ""
        while True:
            try:
                key = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"
            if key == "q":
                _write(path, annotations)
                print(f"\nsaved {len(annotations)} label(s) to {path}", file=sys.stderr)
                return 0
            if key == "b" and index > 0:
                index -= 1
                break
            if key == "n":
                note = input("  note> ").strip()
                continue
            if key in _KEYS:
                annotations.append(Annotation(
                    item_id=item.item_id, annotator=annotator,
                    label=_KEYS[key], note=note,
                    seconds=round(time.time() - started, 1),
                ))
                # Written every item: an interrupted session must not lose
                # an hour of judgements.
                _write(path, annotations)
                index += 1
                break
            print("  keys: s p u c ? n b q")

    print(f"\ndone — {len(annotations)} label(s) in {path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# agreement / adjudicate / score
# ---------------------------------------------------------------------------


def _load_all_labels() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in sorted(DATA.glob("labels-*.jsonl")):
        name = path.stem.removeprefix("labels-")
        out[name] = {a.item_id: a.label.value for a in _read(path, Annotation)}
    return out


def cmd_retest(args: argparse.Namespace) -> int:
    """Re-label a sample already labelled, to measure self-consistency.

    A single annotator produces a usable gold set but no inter-annotator
    agreement, so there is no way to tell whether the guidelines are working
    or whether judgement drifted across a long session. Re-labelling a sample
    after a gap gives test-retest reliability, which plays the role kappa
    would: it will not catch a rule you consistently misread, but it does
    catch drift and coin-flipping on the hard cases.

    Labels are written to a separate annotator name, so the original pass is
    never overwritten and the two can be compared.
    """
    original = _annotations_path(args.annotator)
    done = _read(original, Annotation)
    if not done:
        print(f"no labels for {args.annotator} to retest", file=sys.stderr)
        return 1

    sample = list(done)
    random.Random(f"retest-{args.annotator}").shuffle(sample)
    sample = sample[: args.sample]

    items = {i.item_id: i for i in _read(ITEMS, AnnotationItem)}
    retest_name = f"{args.annotator}-retest"
    already = {a.item_id for a in _read(_annotations_path(retest_name), Annotation)}
    queue = [items[a.item_id] for a in sample if a.item_id in items and a.item_id not in already]

    if not queue:
        print("retest sample already complete", file=sys.stderr)
        return 0

    print(f"\nRe-labelling {len(queue)} item(s) you have already judged.")
    print("Judge them fresh — do not try to recall your previous answer.\n")
    return _label_queue(queue, retest_name, args.chars)


def cmd_agreement(args: argparse.Namespace) -> int:
    by_annotator = _load_all_labels()
    if not by_annotator:
        print("no labels found", file=sys.stderr)
        return 1

    report = agreement_report(by_annotator)
    print(json.dumps(report, indent=2))

    if report.get("kappa") is not None and not report["reliable"]:
        print(
            f"\nnote: kappa over {report['items_compared']} item(s) is not yet reliable; "
            "aim for 100+ jointly labelled.",
            file=sys.stderr,
        )
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    """Resolve disagreements into gold labels.

    Unanimous items become gold automatically. Disagreements are shown with
    each annotator's choice and settled by a third party -- not by majority,
    because a 2-1 split on a genuinely ambiguous item usually means the
    guideline is unclear rather than that one annotator was careless.
    """
    items = {i.item_id: i for i in _read(ITEMS, AnnotationItem)}
    by_annotator = _load_all_labels()
    if len(by_annotator) < 2:
        print("need at least two annotators", file=sys.stderr)
        return 1

    gold = {g.item_id: g for g in _read(GOLD, GoldItem)}
    names = sorted(by_annotator)
    shared = set.intersection(*(set(by_annotator[n]) for n in names))

    disputed = []
    for item_id in sorted(shared):
        if item_id in gold:
            continue
        labels = [by_annotator[n][item_id] for n in names]
        if len(set(labels)) == 1:
            gold[item_id] = GoldItem(
                item_id=item_id, label=Label(labels[0]), n_annotators=len(names),
                agreement=1.0, labels_given=[Label(x) for x in labels],
            )
        else:
            disputed.append((item_id, labels))

    _write(GOLD, list(gold.values()))
    print(f"{len(gold)} unanimous item(s) promoted to gold", file=sys.stderr)

    if not disputed:
        print("no disagreements to adjudicate", file=sys.stderr)
        return 0
    if not args.annotator:
        print(f"{len(disputed)} disagreement(s); rerun with --annotator NAME to resolve",
              file=sys.stderr)
        return 0

    print(f"\n{len(disputed)} disagreement(s). Keys: s p u c ? [k]eep unresolved [q]uit\n")
    for item_id, labels in disputed:
        item = items.get(item_id)
        if item is None:
            continue
        print("=" * 78)
        print(f"QUESTION: {item.question}\n\nCLAIM:\n  {item.claim}")
        print(f"\nPASSAGE [{item.source_id}]:\n  {item.source_text[:900]}")
        print("\nANNOTATORS:")
        for name, label in zip(names, labels, strict=True):
            print(f"  {name:12} {label}")
        try:
            key = input("  adjudicate > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            key = "q"
        if key == "q":
            break
        if key == "k":
            continue
        if key in _KEYS:
            counts = Counter(labels)
            gold[item_id] = GoldItem(
                item_id=item_id, label=_KEYS[key], n_annotators=len(names),
                agreement=round(counts.most_common(1)[0][1] / len(labels), 3),
                adjudicated=True, labels_given=[Label(x) for x in labels],
            )
            _write(GOLD, list(gold.values()))

    print(f"\n{len(gold)} gold label(s) in {GOLD}", file=sys.stderr)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score the pipeline's citation decisions against the gold labels."""
    gold = {g.item_id: g for g in _read(GOLD, GoldItem)}
    items = {i.item_id: i for i in _read(ITEMS, AnnotationItem)}
    if not gold:
        print("no gold labels; run adjudicate first", file=sys.stderr)
        return 1

    tp = fp = tn = fn = 0
    for item_id, judgement in gold.items():
        item = items.get(item_id)
        if item is None or not item.pipeline_verdict:
            continue
        kept = item.pipeline_verdict == "accepted"
        should_keep = judgement.label.is_positive
        if kept and should_keep:
            tp += 1
        elif kept and not should_keep:
            fp += 1
        elif not kept and should_keep:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    if not total:
        print("no scorable items", file=sys.stderr)
        return 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    adjudicated = sum(g.adjudicated for g in gold.values())

    print(json.dumps({
        "scored_items": total,
        "gold_labels": len(gold),
        "adjudicated_share": round(adjudicated / len(gold), 3),
        # Precision: of the citations kept, how many a human agrees with.
        # Recall: of the citations a human would keep, how many survived.
        # A verification-first system should hold precision high and may
        # trade recall; reporting only one hides that trade.
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="annotate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="run the pipeline and emit claim-citation pairs")
    p.add_argument("--questions", required=True, help="file with one question per line")
    p.add_argument("--docs", required=True, help="directory of .txt documents")
    p.add_argument("--max-rounds", type=int, default=2)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("label", help="label items interactively")
    p.add_argument("--annotator", required=True)
    p.add_argument("--limit", type=int, default=100, help="items this session")
    p.add_argument("--chars", type=int, default=1200, help="passage characters to show")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser(
        "retest",
        help="re-label a sample to measure self-consistency (single annotator)",
    )
    p.add_argument("--annotator", required=True)
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--chars", type=int, default=1200)
    p.set_defaults(func=cmd_retest)

    p = sub.add_parser("agreement", help="inter-annotator agreement")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("adjudicate", help="resolve disagreements into gold labels")
    p.add_argument("--annotator", help="adjudicator name; omit to see the count only")
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("score", help="score the pipeline against gold")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
