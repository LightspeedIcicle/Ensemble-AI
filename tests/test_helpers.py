"""Small utilities that everything else depends on.

`response_text` exists because of a latent break: every call site used
`response.content[0].text`, which was only correct while thinking was off. With
adaptive thinking on, index 0 is a ThinkingBlock and `.text` raises.
"""
import types

import pytest

from core.helpers import estimate_tokens, parse_json, response_text


def block(kind, text=""):
    return types.SimpleNamespace(type=kind, text=text)


def response(*blocks):
    return types.SimpleNamespace(content=list(blocks))


# ── response_text ─────────────────────────────────────────────────────────────

def test_reads_a_plain_text_response():
    assert response_text(response(block("text", "hello"))) == "hello"


def test_skips_thinking_blocks_that_come_first():
    # THE regression. Thinking blocks precede text; content[0] is not the answer.
    r = response(block("thinking", "reasoning..."), block("text", "the answer"))
    assert response_text(r) == "the answer"


def test_empty_content_returns_empty_string_not_an_exception():
    # A refusal returns an empty content array. Must not raise.
    assert response_text(response()) == ""


def test_thinking_only_response_returns_empty_string():
    assert response_text(response(block("thinking", "..."))) == ""


# ── parse_json ────────────────────────────────────────────────────────────────

def test_parses_bare_json():
    assert parse_json('{"decision": "local"}') == {"decision": "local"}


def test_strips_markdown_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extracts_json_from_surrounding_prose():
    assert parse_json('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1}


@pytest.mark.parametrize("raw", ["", "not json at all", "{unclosed", "```\n```"])
def test_unparseable_returns_none(raw):
    # None is the fail-safe signal callers branch on. It must never raise.
    assert parse_json(raw) is None


# ── estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_scales_with_length():
    assert estimate_tokens("one two three four") > estimate_tokens("one two")


def test_estimate_of_empty_is_zero():
    assert estimate_tokens("") == 0
