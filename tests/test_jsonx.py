"""Cases a greedy-regex JSON scrape gets wrong."""

from __future__ import annotations

# The naive implementation, kept inline as a reference so the tests below
# document what the balanced-span scanner actually buys rather than asserting
# in a vacuum.
import json
import re

from ragverify import jsonx


def _naive_extract(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def test_plain_object():
    assert jsonx.extract_object('{"route": "web"}') == {"route": "web"}


def test_brace_in_preamble_breaks_naive_regex():
    text = 'Here is the JSON {as requested}:\n{"route": "web", "confidence": 0.9}'
    # Greedy match spans from the first brace to the last, so it never parses.
    assert _naive_extract(text) == {}
    assert jsonx.extract_object(text) == {"route": "web", "confidence": 0.9}


def test_trailing_commentary_with_brace():
    text = '{"verdict": "sufficient"}\n\nNote: the {gaps} field was omitted.'
    assert _naive_extract(text) == {}
    assert jsonx.extract_object(text) == {"verdict": "sufficient"}


def test_two_objects_takes_the_first():
    text = '{"route": "local"} and then {"route": "web"}'
    assert _naive_extract(text) == {}
    assert jsonx.extract_object(text) == {"route": "local"}


def test_fenced_block_preferred():
    text = 'Reasoning about {this} first.\n```json\n{"verdict": "partial"}\n```'
    assert jsonx.extract_object(text) == {"verdict": "partial"}


def test_brace_inside_string_value():
    text = '{"rationale": "the {corpus} covers it", "route": "local"}'
    assert jsonx.extract_object(text)["route"] == "local"


def test_escaped_quote_inside_string():
    text = r'{"rationale": "they said \"yes\" clearly", "route": "web"}'
    assert jsonx.extract_object(text)["route"] == "web"


def test_nested_objects_and_arrays():
    text = '{"gaps": ["a", "b"], "meta": {"n": 2}}'
    assert jsonx.extract_object(text) == {"gaps": ["a", "b"], "meta": {"n": 2}}


def test_trailing_comma_repaired():
    assert jsonx.extract_object('{"route": "web",}') == {"route": "web"}


def test_no_json_returns_none_not_empty_dict():
    # The critical distinction: the naive version returns {} here, which is
    # indistinguishable from a valid empty decision and silently becomes a
    # default via .get(). None forces the caller to handle it.
    assert jsonx.extract_object("I cannot answer that.") is None
    assert _naive_extract("I cannot answer that.") == {}


def test_unbalanced_braces_do_not_run_away():
    assert jsonx.extract_object('{"a": 1') is None
