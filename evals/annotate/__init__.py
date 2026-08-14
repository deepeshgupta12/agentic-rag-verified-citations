"""Human-labelled groundedness: schema, labelling tool, agreement, scoring.

The pipeline's own verification is the thing under test, so it cannot also be
the thing that grades it. Every number reported so far -- support rates,
citation validity, the real-document suites -- was produced by the same
lexical rules the product uses. That measures self-consistency, not
correctness.

This package produces an independent reference: humans judge whether a cited
passage supports a claim, disagreements are measured rather than hidden, and
the pipeline is scored against the resolved labels.
"""

from .agreement import cohens_kappa, fleiss_kappa, interpret, retest_report
from .schema import Annotation, AnnotationItem, GoldItem, Label

__all__ = [
    "Annotation",
    "AnnotationItem",
    "GoldItem",
    "Label",
    "cohens_kappa",
    "fleiss_kappa",
    "interpret",
    "retest_report",
]
