"""Robust JSON extraction from model prose.

The obvious approach is ``re.search(r"\\{.*\\}", text, re.DOTALL)``. Because ``.*`` is
greedy, that matches from the *first* ``{`` in the response to the *last*
``}`` anywhere in it. Any of these break it:

* a preamble containing a brace ("Here's the JSON {as requested}:") -- the
  match starts at the wrong brace and the result is unparseable;
* a fenced block followed by commentary that itself contains ``}``;
* two JSON objects in one reply -- the span swallows both plus the text
  between them.

In every one of those cases ``json.loads`` raised, the helper returned ``{}``,
and the caller's ``.get("route", "local")`` quietly produced a default. A
parse failure was therefore indistinguishable from a real decision, and the
run continued at full confidence on an empty payload.

This module scans for balanced spans instead, respecting string literals and
escapes, and returns *every* candidate so the caller can pick the one matching
its schema.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _balanced_spans(text: str) -> list[str]:
    """Yield top-level ``{...}`` / ``[...]`` spans with balanced delimiters.

    Tracks string state so a brace inside a JSON string value cannot close the
    span early -- the single most common failure of naive brace counting.
    """
    spans: list[str] = []
    stack: list[str] = []
    start = -1
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
                if not stack and start >= 0:
                    spans.append(text[start : i + 1])
                    start = -1
            else:
                # Unbalanced closer: abandon the current span rather than
                # letting it run to the end of the document.
                stack.clear()
                start = -1

    return spans


def candidates(text: str) -> list[Any]:
    """All parseable JSON values in ``text``, fenced blocks preferred."""
    found: list[Any] = []
    seen: set[str] = set()

    def _try(raw: str) -> None:
        raw = raw.strip()
        if not raw or raw in seen:
            return
        seen.add(raw)
        try:
            found.append(json.loads(raw))
        except json.JSONDecodeError:
            repaired = _repair(raw)
            if repaired is not None:
                found.append(repaired)

    for block in _FENCE.findall(text):
        _try(block)
    for span in _balanced_spans(text):
        _try(span)
    _try(text)
    return found


def _repair(raw: str) -> Any | None:
    """Fix the two syntax errors models actually make.

    Trailing commas before a closer, and single-quoted keys/strings. Anything
    beyond that is not worth guessing at -- a wrong repair is more dangerous
    than a clean failure, because it produces a plausible object.
    """
    fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    if '"' not in fixed and "'" in fixed:
        try:
            return json.loads(fixed.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    return None


def extract_object(text: str) -> dict[str, Any] | None:
    """First JSON *object* in ``text``, or None.

    Returning None rather than ``{}`` is deliberate: the caller must be able
    to tell "the model failed to produce JSON" from "the model produced an
    empty object", a distinction a bare ``.get(key, default)`` collapses.
    """
    for value in candidates(text):
        if isinstance(value, dict):
            return value
        # A bare list is sometimes returned when the schema's only field is a
        # list; hand it back wrapped so the model_validate call can decide.
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return {"items": value}
    return None
