"""Canonical forms for numbers, money, dates and units.

Grounding rejects a claim whose numbers do not appear in the cited source.
That check compared raw strings, so it was defeated by formatting alone: a
source saying "2.1 billion euro" and a claim saying "EUR 2,100,000,000" are
the same fact and looked like different ones. The failure is silent and lands
on both sides --

* **False rejection.** A correct claim gets dropped because the source spells
  the figure differently, so the pipeline discards good evidence and escalates
  or abstains for no reason.
* **False acceptance.** "34" in "34%" matched "34" in "34 staff", so a
  percentage could be validated by an unrelated headcount.

Everything here is deliberately conservative. A normaliser that is too eager
invents equivalences and *weakens* the check it exists to strengthen, which is
worse than doing nothing: an over-matching normaliser makes fabrication
easier, not harder. Where a form is ambiguous it is left alone rather than
guessed at.
"""

from __future__ import annotations

import re
from datetime import date

# --------------------------------------------------------------------------
# Magnitudes and currency
# --------------------------------------------------------------------------

MAGNITUDES: dict[str, int] = {
    "k": 10**3, "thousand": 10**3,
    "m": 10**6, "mm": 10**6, "million": 10**6, "mn": 10**6,
    "b": 10**9, "bn": 10**9, "billion": 10**9,
    "t": 10**12, "tn": 10**12, "trillion": 10**12,
    # Indian numbering, common in financial reporting.
    "lakh": 10**5, "lac": 10**5, "crore": 10**7,
}

CURRENCY_SYMBOLS = {
    "$": "USD", "us$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP",
    "¥": "JPY", "jpy": "JPY",
    "₹": "INR", "inr": "INR", "rs": "INR", "rs.": "INR",
}

# --- date patterns (defined early: quantity masking depends on them) ---
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.I)
_QUARTER = re.compile(r"\b(?:Q([1-4])\s*(?:FY)?\s*(\d{4})|(?:FY)?\s*(\d{4})\s*Q([1-4]))\b", re.I)
# Financial prose writes "the third quarter of 2024" as often as "Q3 2024";
# treating them as different facts is a false rejection on a very common form.
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_QUARTER_WORDS = re.compile(
    r"\b(" + "|".join(_ORDINALS) + r")\s+quarter\s*(?:of|,|in)?\s*(?:FY)?\s*(\d{4})\b", re.I
)
_FISCAL_YEAR = re.compile(r"\b(?:FY|fiscal\s+year|fiscal)\s*(\d{4})\b", re.I)
# A quarter with no year. Masking it as an identifier avoids a spurious
# "3", but discarding it entirely makes Q3 and Q4 indistinguishable, so it
# gets its own year-less token.
_BARE_QUARTER = re.compile(r"\bQ([1-4])\b(?!\s*(?:FY)?\s*\d{4})", re.I)
_BARE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

_CURRENCY_ALT = "|".join(
    sorted((re.escape(s) for s in CURRENCY_SYMBOLS), key=len, reverse=True)
)
_MAGNITUDE_WORDS = "|".join(
    sorted((m for m in MAGNITUDES if len(m) > 2), key=len, reverse=True)
)
_MAGNITUDE_LETTERS = "|".join(
    sorted((m for m in MAGNITUDES if len(m) <= 2), key=len, reverse=True)
)

# A quantity: optional leading currency, a number, optional magnitude word,
# optional trailing currency/percent. Ordered so longer tokens win.
_QUANTITY = re.compile(
    rf"(?P<pre>{_CURRENCY_ALT})?\s*"
    # (?<![\w-]) so a hyphen after a word is not a minus sign: "GPT-4" is an
    # identifier, not negative four, and "cost-5" is not -5.
    rf"(?P<num>(?<![\w-])-?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|(?<![\w-])-?\d+(?:\.\d+)?)"
    # Spelled-out magnitudes may be separated ("2.1 billion"); single letters
    # must be attached ("2.1B"), because "25 b" is a number beside a variable
    # named b, not twenty-five billion.
    rf"(?:\s*(?P<mag>{_MAGNITUDE_WORDS})|(?P<magl>{_MAGNITUDE_LETTERS}))?"
    rf"\s*(?P<post>%|percent|per\s+cent|{_CURRENCY_ALT})?",
    re.I,
)

# Percent must be distinguishable from a bare count: "34%" and "34 staff" are
# different facts and previously collided.
_PCT_SUFFIX = re.compile(r"^(%|percent|per\s+cent)$", re.I)


# Identifiers that merely CONTAIN digits: PEP 8, BM25, IPv4, GPT-4, RFC 2616,
# ISO 9001, Q3, H1, COVID-19, Section 5. The digits name a thing rather than
# assert a quantity, so requiring the source to contain them rejects a claim
# for citing the document it came from. Found by running against real
# documents; no synthetic fixture contained an identifier with a digit in it.
_IDENTIFIER = re.compile(
    r"\b(?:"
    r"[A-Za-z]{2,}[-\u2011]?\d+(?:\.\d+)*"          # BM25, GPT-4, IPv4, COVID-19
    r"|(?:PEP|RFC|ISO|IEEE|ANSI|BS|EN|DIN|NIST|CVE|SP)\s+\d+(?:[-.]\d+)*"
    r"|(?:section|chapter|figure|table|appendix|part|clause|article|item|step)"
    r"\s+\d+(?:\.\d+)*"
    r"|[QH]\d\b"                                     # Q3, H1
    r")",
    re.I,
)


def _mask_identifiers(text: str) -> str:
    """Blank out identifiers whose digits are part of a name.

    "PEP 8 recommends 4 spaces" asserts one quantity, not two. Treating the 8
    as a claim means a correct statement is rejected unless the cited passage
    happens to repeat the standard's own number.
    """
    return _IDENTIFIER.sub(" ", text)


def _mask_dates(text: str) -> str:
    """Blank out date expressions before scanning for quantities.

    "Q3 2024" otherwise yields a bare quantity 3 and a quantity 2024, so a
    claim about Q3 would demand the source contain the number 3 -- and a
    source saying "third quarter" would fail. Dates are handled by their own
    extractor; masking keeps the two from contaminating each other.
    """
    for pattern in (
        _ISO, _MONTH_DAY_YEAR, _DAY_MONTH_YEAR, _MONTH_YEAR,
        _QUARTER, _QUARTER_WORDS, _FISCAL_YEAR,
    ):
        text = pattern.sub(" ", text)
    return text


def _to_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _fmt(value: float) -> str:
    """Stable string for a numeric value, integral where possible."""
    if value == int(value) and abs(value) < 10**18:
        return str(int(value))
    # Bound precision so 0.1+0.2 style drift cannot split one fact in two.
    return f"{value:.6g}"


def quantity_groups(text: str) -> list[set[str]]:
    """One set of *acceptable forms* per quantity found in ``text``.

    Each group is the alternatives that would all express the same figure, so
    a quantity is satisfied when the source contains **any** member. Returning
    a flat set instead would silently turn every alternative into a separate
    requirement: emitting both ``num:2100000000`` and ``num:2.1`` for
    "2.1 billion" would then reject a source that writes "2,100,000,000",
    because it does not also contain a bare 2.1.
    """
    groups: list[set[str]] = []

    # Dates first: 'Q3 2024' must be consumed whole by the quarter pattern.
    # Masking identifiers first strips the 'Q3', leaving a bare 2024 that
    # then reads as a quantity rather than part of the date.
    for match in _QUANTITY.finditer(_mask_identifiers(_mask_dates(text))):
        number = _to_number(match.group("num"))
        if number is None:
            continue

        magnitude = (match.group("mag") or match.group("magl") or "").lower()
        pre = (match.group("pre") or "").lower().strip()
        post = (match.group("post") or "").lower().strip()

        scaled = number * MAGNITUDES[magnitude] if magnitude in MAGNITUDES else number

        # Percentages are their own type: "34%" must never be satisfied by
        # "34 staff", which raw string matching happily allowed.
        if _PCT_SUFFIX.match(post):
            groups.append({f"pct:{_fmt(scaled)}"})
            continue

        # The unscaled form is deliberately NOT offered as an alternative.
        # Accepting bare 2.1 for "2.1 billion" makes "2.1 billion" and
        # "2.1 million" indistinguishable -- a thousand-fold error waved
        # through. The "in millions" table case is real but rarer, and this
        # module errs strict: a false accept helps fabrication, a false
        # rejection only costs a retrieval round.
        currency = CURRENCY_SYMBOLS.get(pre) or CURRENCY_SYMBOLS.get(post)
        forms = {f"money:{currency}:{_fmt(scaled)}"} if currency else {f"num:{_fmt(scaled)}"}
        groups.append(forms)

    return groups


def canonical_quantities(text: str) -> set[str]:
    """Flat set of every quantity form in ``text`` — the source-side view."""
    return {form for group in quantity_groups(text) for form in group}


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------



def canonical_dates(text: str) -> set[str]:
    """Canonical date tokens: ``date:YYYY-MM-DD``, ``month:YYYY-MM``,
    ``quarter:YYYY-Qn``, ``fy:YYYY``, ``year:YYYY``.

    Granularity is preserved rather than collapsed. "October 2024" is not the
    same assertion as "1 October 2024", and treating them as equal would let a
    claim invent a precision the source never stated.
    """
    out: set[str] = set()

    for year, month, day in _ISO.findall(text):
        try:
            out.add(f"date:{date(int(year), int(month), int(day)).isoformat()}")
            out.add(f"month:{year}-{int(month):02d}")
            out.add(f"year:{year}")
        except ValueError:
            continue

    for pattern, order in ((_MONTH_DAY_YEAR, "mdy"), (_DAY_MONTH_YEAR, "dmy")):
        for groups in pattern.findall(text):
            month_name, day_str, year = groups if order == "mdy" else (groups[1], groups[0], groups[2])
            month = _MONTHS[month_name.lower()]
            try:
                out.add(f"date:{date(int(year), month, int(day_str)).isoformat()}")
            except ValueError:
                continue
            out.add(f"month:{year}-{month:02d}")
            out.add(f"year:{year}")

    for month_name, year in _MONTH_YEAR.findall(text):
        out.add(f"month:{year}-{_MONTHS[month_name.lower()]:02d}")
        out.add(f"year:{year}")

    for word, year in _QUARTER_WORDS.findall(text):
        out.add(f"quarter:{year}-Q{_ORDINALS[word.lower()]}")
        out.add(f"year:{year}")

    for q1, y1, y2, q2 in _QUARTER.findall(text):
        quarter, year = (q1, y1) if q1 else (q2, y2)
        if quarter and year:
            out.add(f"quarter:{year}-Q{quarter}")
            out.add(f"year:{year}")

    for year in _FISCAL_YEAR.findall(text):
        out.add(f"fy:{year}")
        out.add(f"year:{year}")

    for quarter in _BARE_QUARTER.findall(text):
        out.add(f"quarter:Q{quarter}")

    out.update(f"year:{y}" for y in _BARE_YEAR.findall(text))
    return out


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------


def canonical_values(text: str) -> set[str]:
    """All canonical value tokens in ``text`` -- the source-side view."""
    return canonical_quantities(text) | canonical_dates(text)


def value_requirements(text: str) -> list[set[str]]:
    """Acceptable-form groups for every value asserted by ``text``."""
    return quantity_groups(text) + [{d} for d in canonical_dates(text)]


def unsupported_values(claim: str, source: str) -> set[str]:
    """Values asserted by ``claim`` that ``source`` does not contain.

    A value is supported when any of its acceptable forms appears in the
    source. Empty result means every figure and date in the claim is present,
    in some spelling, in the source.

    Currency is handled with one extra rule. A claim naming a currency is
    satisfied by an unlabelled figure of the same magnitude -- sources
    routinely establish the unit once in a heading and omit it after -- but
    *never* when the source labels that same figure with a different currency.
    "$5M" against a source saying "€5M" is a real contradiction, not a
    formatting difference.
    """
    available = canonical_values(source)
    missing: set[str] = set()

    for group in value_requirements(claim):
        if group & available:
            continue

        token = sorted(group)[0]
        if token.startswith("money:"):
            _, _currency, amount = token.split(":", 2)
            conflicting = {
                other
                for other in available
                if other.startswith("money:") and other.rsplit(":", 1)[-1] == amount
            }
            # Unlabelled figure of the same size, and no rival currency: accept.
            if f"num:{amount}" in available and not conflicting:
                continue

        missing.add(token)

    return missing
