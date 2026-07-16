# core/helpers.py
# Small pure utilities shared across pipeline stages. No external clients here.

import json


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
