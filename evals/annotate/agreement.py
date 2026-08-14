"""Inter-annotator agreement.

Raw percent agreement is not a usable number. With a skewed label
distribution -- and groundedness sets are skewed, because most citations a
pipeline emits are fine -- two annotators who both answer "supported" for
everything agree 90% of the time while carrying no information at all. Kappa
subtracts the agreement expected from chance, so that pair scores near zero,
which is the honest reading.

Cohen's kappa for exactly two annotators, Fleiss' for three or more. Both are
implemented here rather than pulled in, because the whole dependency would be
these forty lines and a scoring harness should not need scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

# Landis & Koch (1977). Conventional and worth stating, because "kappa 0.55"
# means nothing to a reader who has not memorised the bands.
_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "poor — worse than chance"),
)


def interpret(kappa: float) -> str:
    for floor, label in _BANDS:
        if kappa >= floor:
            return label
    return "poor"


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two annotators, corrected for chance.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, and negative
    when the pair disagrees more than random labelling would.
    """
    if len(a) != len(b):
        raise ValueError("annotators must have labelled the same items")
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n

    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in set(count_a) | set(count_b)
    )

    # Perfect agreement on a single label: expected is 1.0 and kappa is
    # undefined. Report 1.0 rather than dividing by zero, but note that such
    # a set carries no discriminating information.
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float:
    """Agreement among three or more annotators.

    ``ratings`` is one sequence of labels per item. Items must all have the
    same number of raters; an item nobody could be bothered to label twice is
    not comparable to one everybody labelled.
    """
    items = [r for r in ratings if r]
    if not items:
        return 0.0

    n_raters = len(items[0])
    if any(len(r) != n_raters for r in items):
        raise ValueError("every item must have the same number of ratings")
    if n_raters < 2:
        return 0.0

    categories = sorted({label for item in items for label in item})
    n_items = len(items)

    # Per-item agreement: the proportion of rater PAIRS that agree.
    p_i = []
    for item in items:
        counts = Counter(item)
        agreeing_pairs = sum(c * (c - 1) for c in counts.values())
        p_i.append(agreeing_pairs / (n_raters * (n_raters - 1)))

    observed = sum(p_i) / n_items

    # Chance agreement from the marginal distribution of each category.
    totals = Counter(label for item in items for label in item)
    expected = sum((totals[c] / (n_items * n_raters)) ** 2 for c in categories)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_report(
    by_annotator: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Kappa across annotators, plus what could not be compared.

    ``by_annotator`` maps annotator -> {item_id: label}. Only items every
    annotator labelled are scored; partially labelled items are counted and
    reported rather than silently dropped, because a small overlap makes any
    kappa unreliable and the reader should see that.
    """
    names = sorted(by_annotator)
    if len(names) < 2:
        return {
            "annotators": names,
            "kappa": None,
            "note": "at least two annotators are required",
        }

    shared = set.intersection(*(set(by_annotator[n]) for n in names))
    if not shared:
        return {"annotators": names, "kappa": None, "note": "no items labelled by everyone"}

    ordered = sorted(shared)
    if len(names) == 2:
        kappa = cohens_kappa(
            [by_annotator[names[0]][i] for i in ordered],
            [by_annotator[names[1]][i] for i in ordered],
        )
        method = "cohen"
    else:
        kappa = fleiss_kappa([[by_annotator[n][i] for n in names] for i in ordered])
        method = "fleiss"

    labelled_totals = {n: len(by_annotator[n]) for n in names}
    return {
        "annotators": names,
        "method": method,
        "kappa": round(kappa, 3),
        "interpretation": interpret(kappa),
        "items_compared": len(ordered),
        "items_labelled": labelled_totals,
        # A kappa over a handful of items is noise; say so rather than
        # letting a confident-looking number stand alone.
        "reliable": len(ordered) >= 100,
    }
