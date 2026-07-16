# core/monitor.py
# Stage 5 — Monitoring / validation (judge role).
# Takes the discrepancies and unique points the council surfaced and runs each
# through validity + relevance checks, sorting them into validated / removed /
# ambiguous. This is the fact-checking gate — the most capability-sensitive of
# the three judge roles, since it needs real world knowledge to adjudicate.
#
# The schema asks for exactly what the pipeline consumes and nothing else.
# `validated` previously carried "source" and "verdict" fields that no caller
# ever read — consolidate and knowledge.persist both do [v["item"] for v in ...] —
# so the judge was generating them at output-token prices to be parsed and
# dropped. If a field isn't read, it isn't requested.

import json

from core.clients import anthropic_client, MONITOR_MODEL
from core.helpers import parse_json


def monitor_discrepancies(original_prompt, discrepancies, unique_to_a, unique_to_b):
    """Validate and filter the council's disagreements and unique claims.

    Short-circuits to an empty result when there is nothing to check.
    """
    if not discrepancies and not unique_to_a and not unique_to_b:
        return {"validated": [], "removed": [], "ambiguous": []}

    monitor_prompt = f"""You are a monitoring agent validating information from two AI responses.

Original task: {original_prompt}

DISCREPANCIES:
{json.dumps(discrepancies, indent=2)}

UNIQUE TO CLAUDE:
{json.dumps(unique_to_a, indent=2)}

UNIQUE TO GEMINI:
{json.dumps(unique_to_b, indent=2)}

For each item apply two checks:
1. VALIDITY: Is this factually accurate?
2. RELEVANCE: Is this relevant to the original task?

Keep every string terse — this output is consumed by the pipeline, not read as prose.

Return only this JSON with no preamble or markdown:
{{
    "validated": [{{"item": "..."}}],
    "removed": [{{"item": "...", "reason": "..."}}],
    "ambiguous": [{{"item": "...", "reason": "..."}}]
}}"""

    response = anthropic_client.messages.create(
        model=MONITOR_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": monitor_prompt}],
    )
    return parse_json(response.content[0].text)
