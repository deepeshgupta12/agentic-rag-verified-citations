"""Source quality, freshness, domain diversity and corroboration.

Web evidence was treated as uniformly trustworthy: eight results from one
content farm scored exactly like eight independent primary sources, and a
2011 page scored like today's. That matters more here than in a plain search
tool, because grounding then *certifies* whatever those pages said. A claim
correctly grounded in a bad source passes every check this pipeline has and
is still wrong.

Four signals, each answering a different question:

* **Authority** -- what kind of publisher is this? A regulator's own site and
  an SEO aggregator are not interchangeable, and the domain suffix carries
  real information (.gov, .edu, .ac.uk) that costs nothing to read.
* **Freshness** -- how old is the page, and does the question care? A
  five-year-old page is fine for "what is photosynthesis" and disqualifying
  for "who is the current CEO".
* **Diversity** -- how many *independent* domains does the evidence span?
  Five pages from one site are one source wearing five hats.
* **Corroboration** -- do independent domains agree on the specific figures
  and dates? Two unrelated sites reporting the same number is the strongest
  signal available without leaving the retrieved text.

Everything is heuristic and says so. The goal is to *rank and warn*, never to
silently discard: a low-quality source that is the only one answering the
question is still the answer, and hiding it would be worse than flagging it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from .normalize import canonical_values
from .schemas import EvidenceItem, Route

# Publisher classes, most authoritative first. Deliberately coarse: a finer
# ranking would encode opinions this module has no basis for.
TIER_PRIMARY = 1.0      # regulators, statistics offices, standards bodies
TIER_ACADEMIC = 0.9     # universities, journals, preprint servers
TIER_REFERENCE = 0.75   # encyclopaedias, official documentation
TIER_ESTABLISHED = 0.65 # major outlets and known publishers
TIER_UNKNOWN = 0.45     # anything unrecognised — the default, not a penalty
TIER_LOW = 0.2          # content farms, scrapers, aggregators

_SUFFIX_TIERS = (
    ((".gov", ".gov.uk", ".mil", ".europa.eu", ".int"), TIER_PRIMARY),
    ((".edu", ".ac.uk", ".edu.au", ".ac.jp"), TIER_ACADEMIC),
)

_DOMAIN_TIERS: dict[str, float] = {
    # Reference and documentation
    "wikipedia.org": TIER_REFERENCE, "britannica.com": TIER_REFERENCE,
    "docs.python.org": TIER_REFERENCE, "developer.mozilla.org": TIER_REFERENCE,
    # Academic
    "arxiv.org": TIER_ACADEMIC, "nature.com": TIER_ACADEMIC,
    "science.org": TIER_ACADEMIC, "pubmed.ncbi.nlm.nih.gov": TIER_ACADEMIC,
    "doi.org": TIER_ACADEMIC, "jstor.org": TIER_ACADEMIC,
    # Established outlets
    "reuters.com": TIER_ESTABLISHED, "apnews.com": TIER_ESTABLISHED,
    "bbc.co.uk": TIER_ESTABLISHED, "bbc.com": TIER_ESTABLISHED,
    "ft.com": TIER_ESTABLISHED, "wsj.com": TIER_ESTABLISHED,
    "economist.com": TIER_ESTABLISHED, "bloomberg.com": TIER_ESTABLISHED,
    "nytimes.com": TIER_ESTABLISHED, "theguardian.com": TIER_ESTABLISHED,
}

# Patterns that indicate republished or machine-generated content rather than
# a primary account. Matched on the host only.
_LOW_QUALITY = re.compile(
    r"(^|\.)(answers|ehow|wikihow|buzzfeed|contentfarm|articlebase|ezinearticles"
    r"|scraper|aggregator|blogspot|wordpress|medium|substack)\.",
    re.I,
)

# Questions whose answers decay. "Current", "latest", "2026" all mean an old
# page is not merely less useful but actively misleading.
_TIME_SENSITIVE = re.compile(
    r"\b(current|latest|now|today|recent|recently|this\s+(?:year|month|quarter)|"
    r"as\s+of|up[\s-]?to[\s-]?date|newest|202[4-9]|nowadays|still)\b",
    re.I,
)

_DATE_IN_TEXT = re.compile(
    r"\b(20[0-2]\d|19[89]\d)-(\d{2})-(\d{2})\b"
    r"|\b(?:published|updated|posted|last\s+modified)[:\s]+[^\n]{0,40}?(20[0-2]\d)\b",
    re.I,
)


def domain_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def registrable_domain(url: str) -> str:
    """Approximate eTLD+1, so ``a.example.com`` and ``b.example.com`` are one source.

    Not a public-suffix-list implementation: it special-cases the common
    two-level suffixes and otherwise takes the last two labels. Good enough to
    stop subdomains of one site counting as independent corroboration, which
    is the only thing it is used for.
    """
    host = domain_of(url)
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    two_level = {"co.uk", "ac.uk", "gov.uk", "com.au", "co.jp", "co.in", "com.br", "co.za"}
    return ".".join(parts[-3:]) if ".".join(parts[-2:]) in two_level else ".".join(parts[-2:])


def authority_score(url: str) -> float:
    """Coarse publisher-class score in [0, 1]."""
    host = domain_of(url)
    if not host:
        return TIER_UNKNOWN

    if _LOW_QUALITY.search(host):
        return TIER_LOW
    for suffixes, tier in _SUFFIX_TIERS:
        if host.endswith(suffixes):
            return tier

    registrable = registrable_domain(url)
    for known, tier in _DOMAIN_TIERS.items():
        if registrable == known or host.endswith("." + known):
            return tier
    return TIER_UNKNOWN


def extract_published_year(text: str) -> int | None:
    """Best-effort publication year from page text. None when unknown."""
    years: list[int] = []
    for match in _DATE_IN_TEXT.finditer(text[:4000]):
        year = match.group(1) or match.group(4)
        if year:
            years.append(int(year))
    # The most recent plausible year: pages cite older dates in their body,
    # so the maximum is the better estimate of when this was written.
    current = datetime.now(tz=timezone.utc).year
    plausible = [y for y in years if 1990 <= y <= current + 1]
    return max(plausible) if plausible else None


def is_time_sensitive(question: str) -> bool:
    return bool(_TIME_SENSITIVE.search(question))


def freshness_score(published_year: int | None, time_sensitive: bool) -> float:
    """Recency score in [0, 1]; neutral when the age is unknown.

    Unknown age returns 0.6 rather than 0: most pages do not state a date, and
    treating "undated" as "old" would penalise the majority of the web for a
    property we simply failed to observe.
    """
    if published_year is None:
        return 0.6
    age = max(0, datetime.now(tz=timezone.utc).year - published_year)
    # Time-sensitive questions decay fast (half-life ~1.5y); otherwise slowly
    # (~8y), because a stable fact does not expire.
    half_life = 1.5 if time_sensitive else 8.0
    return round(math.exp(-age * math.log(2) / half_life), 3)


@dataclass
class SourceAssessment:
    source_id: str
    domain: str
    authority: float
    freshness: float
    published_year: int | None
    corroborated_by: list[str] = field(default_factory=list)

    @property
    def corroboration(self) -> float:
        """Saturating bonus: the second independent domain matters most."""
        return min(1.0, len(self.corroborated_by) / 2.0)

    @property
    def score(self) -> float:
        return round(
            0.45 * self.authority + 0.25 * self.freshness + 0.30 * self.corroboration, 3
        )


@dataclass
class QualityReport:
    assessments: list[SourceAssessment] = field(default_factory=list)
    distinct_domains: int = 0
    time_sensitive: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_authority(self) -> float:
        return (
            round(sum(a.authority for a in self.assessments) / len(self.assessments), 3)
            if self.assessments
            else 0.0
        )

    @property
    def diversity(self) -> float:
        """Independent domains as a fraction of web sources."""
        return round(self.distinct_domains / len(self.assessments), 3) if self.assessments else 0.0

    def by_id(self) -> dict[str, SourceAssessment]:
        return {a.source_id: a for a in self.assessments}


def assess(
    evidence: Sequence[EvidenceItem],
    question: str,
    min_authority: float = 0.3,
    min_domains: int = 2,
) -> QualityReport:
    """Score web evidence and flag what a reader should be told.

    Local documents are skipped: the user chose them, so ranking their
    "authority" would be presumptuous and meaningless.
    """
    web = [e for e in evidence if e.origin is Route.WEB and e.url]
    report = QualityReport(time_sensitive=is_time_sensitive(question))
    if not web:
        return report

    values_by_domain: dict[str, set[str]] = {}
    for item in web:
        domain = registrable_domain(item.url or "")
        values_by_domain.setdefault(domain, set()).update(canonical_values(item.text))

    for item in web:
        domain = registrable_domain(item.url or "")
        year = extract_published_year(item.text)
        own_values = canonical_values(item.text)

        # Corroboration: another *registrable domain* asserting the same
        # figures. Subdomains of one site are not independent confirmation.
        corroborators = [
            other
            for other, values in values_by_domain.items()
            if other != domain and own_values and len(own_values & values) >= 2
        ]

        report.assessments.append(
            SourceAssessment(
                source_id=item.source_id,
                domain=domain,
                authority=authority_score(item.url or ""),
                freshness=freshness_score(year, report.time_sensitive),
                published_year=year,
                corroborated_by=sorted(corroborators),
            )
        )

    report.distinct_domains = len({a.domain for a in report.assessments})

    # Warnings describe risk; they never remove evidence.
    if report.distinct_domains < min_domains and len(web) > 1:
        report.warnings.append(
            f"All {len(web)} web sources come from {report.distinct_domains} domain(s) "
            f"({', '.join(sorted({a.domain for a in report.assessments}))}). "
            "Independent corroboration is limited."
        )

    weak = [a for a in report.assessments if a.authority <= min_authority]
    if weak:
        report.warnings.append(
            f"{len(weak)} web source(s) are from low-authority domains: "
            + ", ".join(sorted({a.domain for a in weak}))
        )

    if report.time_sensitive:
        stale = [a for a in report.assessments if a.published_year and a.freshness < 0.4]
        if stale:
            oldest = min(a.published_year for a in stale)
            report.warnings.append(
                f"The question asks for current information, but {len(stale)} source(s) "
                f"appear to date from {oldest} or earlier."
            )

    uncorroborated = [a for a in report.assessments if not a.corroborated_by]
    if uncorroborated and report.distinct_domains >= min_domains:
        report.warnings.append(
            f"{len(uncorroborated)} web source(s) report figures no other domain confirms."
        )

    return report


def rank(evidence: Sequence[EvidenceItem], report: QualityReport) -> list[EvidenceItem]:
    """Reorder web evidence by quality, retrieval score preserved as tiebreak.

    Local evidence keeps its position: retrieval already ranked it, and this
    module has nothing to say about documents the user supplied.
    """
    scores = report.by_id()
    local = [e for e in evidence if e.origin is not Route.WEB]
    web = [e for e in evidence if e.origin is Route.WEB]
    web.sort(
        key=lambda e: (scores[e.source_id].score if e.source_id in scores else 0.0, e.score),
        reverse=True,
    )
    return local + web
