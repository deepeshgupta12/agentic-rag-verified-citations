"""Ranking conflicting sources, without deciding for the reader.

``contradiction.py`` answers *do these sources disagree?*. This answers the
next question -- *which one should you believe?* -- and it answers it with a
stated rationale rather than a verdict.

The distinction is the whole design. A pipeline that silently picks a side
produces an answer indistinguishable from one where the sources agreed: the
reader sees a confident figure and has no way to know a conflict existed or
that something chose for them. That is worse than the disagreement, because
it converts a visible problem into an invisible one.

So resolution here produces a **ranking with reasons**, and the reader still
sees both figures. The pipeline says "the filing is more authoritative and
more recent than the blog, and two other sources corroborate it" -- it does
not delete the blog.

Four signals, in the order they are trusted:

* **Corroboration.** Independent domains agreeing is the strongest available
  evidence, and the only one that comes from the corpus rather than from a
  prior about publishers.
* **Authority.** A regulator's filing outranks an unattributed blog. Coarse,
  and honest about being coarse.
* **Recency.** For a figure that changes, the newer statement usually wins.
  For a stable fact it means nothing, so it is weighted by whether the
  question is time-sensitive.
* **Specificity.** "2.1 billion euro" is a stronger claim than "around two
  billion", and a source that commits to precision is easier to check.

When the signals do not separate the sources, that is reported as
``inconclusive`` rather than resolved on a coin-flip. An arbitrary
tie-broken answer carries the same confident tone as a well-supported one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .contradiction import Contradiction
from .schemas import EvidenceItem
from .sourcequality import (
    QualityReport,
    authority_score,
    extract_published_year,
    freshness_score,
    is_time_sensitive,
    registrable_domain,
)

# Below this margin the sources are not meaningfully separated and the
# conflict is reported unresolved. Deliberately wide: a near-tie broken by
# rounding reads to the user exactly like a decisive result.
DECISIVE_MARGIN = 0.15

# Hedging language. A source that will not commit to a figure is weaker
# evidence for it than one that does.
_HEDGE = re.compile(
    r"\b(?:around|about|approximately|roughly|circa|near(?:ly)?|some|"
    r"estimated|reportedly|allegedly|rumou?red|claims?|unverified|"
    r"unconfirmed|may|might|could|appears?|seems?)\b",
    re.I,
)
_PRECISE = re.compile(r"\d+[.,]\d|\d{3,}")


@dataclass
class SourceStanding:
    """How well one side of a conflict is supported."""

    source_id: str
    domain: str
    corroboration: float
    authority: float
    recency: float
    specificity: float
    published_year: int | None = None
    corroborated_by: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(
            0.40 * self.corroboration
            + 0.30 * self.authority
            + 0.18 * self.recency
            + 0.12 * self.specificity,
            3,
        )

    def reasons(self) -> list[str]:
        """Why this source ranks where it does, in plain terms."""
        out = []
        if self.corroborated_by:
            out.append(f"corroborated by {len(self.corroborated_by)} other domain(s)")
        if self.authority >= 0.85:
            out.append("primary or academic publisher")
        elif self.authority <= 0.25:
            out.append("low-authority publisher")
        if self.published_year:
            out.append(f"dated {self.published_year}")
        if self.specificity >= 0.8:
            out.append("states a precise figure")
        elif self.specificity <= 0.3:
            out.append("hedged or imprecise")
        return out


@dataclass
class Resolution:
    """A ranked conflict, with the reasoning kept attached."""

    conflict: Contradiction
    ranked: list[SourceStanding]
    decisive: bool
    margin: float

    @property
    def preferred(self) -> SourceStanding | None:
        return self.ranked[0] if self.decisive and self.ranked else None

    def describe(self) -> str:
        if not self.decisive:
            return (
                f"{self.conflict.describe()} The available signals do not separate "
                f"these sources (margin {self.margin:.2f}); both are reported."
            )
        best, rest = self.ranked[0], self.ranked[1]
        why = ", ".join(best.reasons()) or "higher combined standing"
        return (
            f"{self.conflict.describe()} [{best.source_id}] is better supported "
            f"({why}); [{rest.source_id}] is retained but ranked lower."
        )


def _specificity(text: str) -> float:
    """How firmly a passage commits to its figure."""
    hedges = len(_HEDGE.findall(text))
    precise = len(_PRECISE.findall(text))
    if hedges and not precise:
        return 0.1
    if hedges:
        return 0.45
    return 0.9 if precise else 0.6


def _standing(
    source_id: str,
    evidence: dict[str, EvidenceItem],
    quality: QualityReport | None,
    excerpt: str,
    time_sensitive: bool,
) -> SourceStanding:
    item = evidence.get(source_id)
    url = (item.url if item else "") or ""
    domain = registrable_domain(url) if url else (item.label.split(" ")[0] if item else source_id)

    assessed = quality.by_id().get(source_id) if quality else None
    if assessed is not None:
        return SourceStanding(
            source_id=source_id, domain=assessed.domain,
            corroboration=assessed.corroboration, authority=assessed.authority,
            recency=assessed.freshness, specificity=_specificity(excerpt),
            published_year=assessed.published_year,
            corroborated_by=assessed.corroborated_by,
        )

    # Local documents are never quality-assessed -- the user chose them -- so
    # authority is neutral and only text-derived signals apply.
    year = extract_published_year(item.text) if item else None
    return SourceStanding(
        source_id=source_id, domain=domain,
        corroboration=0.0,
        authority=authority_score(url) if url else 0.5,
        recency=freshness_score(year, time_sensitive),
        specificity=_specificity(excerpt),
        published_year=year,
    )


def resolve(
    conflicts: Sequence[Contradiction],
    evidence: Sequence[EvidenceItem],
    question: str,
    quality: QualityReport | None = None,
    margin: float = DECISIVE_MARGIN,
) -> list[Resolution]:
    """Rank each conflict's sides. Never removes a source."""
    by_id = {e.source_id: e for e in evidence}
    time_sensitive = is_time_sensitive(question)
    out: list[Resolution] = []

    for conflict in conflicts:
        a = _standing(conflict.source_a, by_id, quality, conflict.excerpt_a, time_sensitive)
        b = _standing(conflict.source_b, by_id, quality, conflict.excerpt_b, time_sensitive)
        ranked = sorted([a, b], key=lambda s: -s.score)
        gap = round(ranked[0].score - ranked[1].score, 3)
        out.append(Resolution(
            conflict=conflict, ranked=ranked, decisive=gap >= margin, margin=gap,
        ))

    return out


def summarize(resolutions: Sequence[Resolution]) -> str:
    if not resolutions:
        return ""
    decisive = sum(r.decisive for r in resolutions)
    return (
        f"{len(resolutions)} conflict(s): {decisive} ranked by source standing, "
        f"{len(resolutions) - decisive} inconclusive. Both sides are reported either way."
    )


def as_prompt_block(resolutions: Sequence[Resolution], limit: int = 6) -> str:
    """Rendered for synthesis, with the instruction not to delete a side."""
    if not resolutions:
        return ""
    lines = [
        "CONTRADICTIONS WITH SOURCE STANDING",
        "",
        "Sources disagree. Where one is better supported, lead with it and say why,",
        "but ALWAYS report the other figure and its source. Never present a ranked",
        "conflict as though the sources agreed -- the reader must be able to see that",
        "a disagreement existed.",
        "",
    ]
    for resolution in resolutions[:limit]:
        lines.append(f"- {resolution.describe()}")
        for standing in resolution.ranked:
            reasons = ", ".join(standing.reasons()) or "no distinguishing signals"
            lines.append(f"    [{standing.source_id}] standing {standing.score:.2f} — {reasons}")
    return "\n".join(lines)
