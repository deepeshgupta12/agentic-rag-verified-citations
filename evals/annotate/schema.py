"""Data model for groundedness annotation.

One record per *claim-citation pair*, not per answer. An answer is a bundle
of judgements and averaging them hides which one was wrong; the pair is the
smallest unit a human can judge consistently and the same unit the pipeline
decides on, so the two are directly comparable.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field


class Label(str, Enum):
    """Judgement for one claim against one cited passage.

    ``PARTIAL`` exists because forcing it into supported or unsupported is
    where annotator agreement collapses. "Revenue grew" against a passage
    saying "revenue grew in most regions" is neither wholly right nor wrong,
    and a scheme without a middle category produces low agreement for a
    reason that has nothing to do with the annotators.

    ``UNCLEAR`` is for items an annotator cannot judge -- ambiguous wording,
    missing context, a passage that needs domain knowledge they lack. Kept
    separate from a disagreement so adjudication can tell "we disagree" from
    "this item is unjudgeable" and drop the latter rather than force a
    verdict onto it.
    """

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCLEAR = "unclear"

    @property
    def is_positive(self) -> bool:
        """Does this label mean the citation should stand?

        The pipeline makes a binary decision, so scoring needs a binary
        projection. ``PARTIAL`` counts as positive: the citation is relevant
        and keeping it is defensible, which is the decision being graded.
        """
        return self in (Label.SUPPORTED, Label.PARTIAL)


class AnnotationItem(BaseModel):
    """One claim-citation pair awaiting judgement."""

    item_id: str
    question: str
    claim: str
    source_id: str
    source_label: str
    source_text: str
    # What the pipeline decided, recorded at extraction time and hidden from
    # the annotator. Showing it would anchor the judgement to the answer
    # being tested.
    pipeline_verdict: str = ""

    @staticmethod
    def make_id(claim: str, source_id: str, source_text: str) -> str:
        """Stable id from content, so re-extraction does not renumber items.

        Keyed on the source *text* rather than only its id: the same S1 in a
        different run may be a different passage entirely.
        """
        digest = hashlib.sha1(
            f"{claim}\x00{source_id}\x00{source_text[:2000]}".encode()
        ).hexdigest()
        return digest[:16]


class Annotation(BaseModel):
    """One annotator's judgement of one item."""

    item_id: str
    annotator: str
    label: Label
    note: str = ""
    seconds: float = 0.0


class GoldItem(BaseModel):
    """A resolved label, with how it was reached.

    ``agreement`` and ``adjudicated`` are kept because a gold label that
    needed a tiebreak is weaker evidence than a unanimous one, and a scorer
    that cannot tell them apart reports a confidence the data does not carry.
    """

    item_id: str
    label: Label
    n_annotators: int
    agreement: float
    adjudicated: bool = False
    labels_given: list[Label] = Field(default_factory=list)
