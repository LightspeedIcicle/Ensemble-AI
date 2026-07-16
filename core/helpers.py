# core/helpers.py
# Small pure utilities shared across pipeline stages. No external clients here.

import json


def response_text(response):
    """Pull the text out of an Anthropic response.

    Not `response.content[0].text`. With adaptive thinking enabled, thinking
    blocks come FIRST in `content`, so index 0 is a ThinkingBlock and `.text`
    raises. Every call in this pipeline used to index position zero, which was
    only correct because thinking happened to be off — a latent break that would
    have fired the moment anyone turned it on.

    Returns the first text block's content, or "" if the response carried none
    (a refusal returns an empty content array).
    """
    return next((b.text for b in response.content if b.type == "text"), "")


def parse_json(raw):
    """Best-effort JSON extraction from an LLM response.

    Handles the two things models commonly do wrong: wrapping output in
    ```json code fences, and surrounding the object with prose. Returns the
    parsed dict/list, or None if nothing parseable is found.
    """
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to extracting the outermost { ... } object
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


def estimate_tokens(text):
    """Rough token estimate — approximately 1 token per 0.75 words."""
    return int(len(text.split()) / 0.75)
