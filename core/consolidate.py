# core/consolidate.py
# Stage 6 — Consolidation (judge role).
# Combines what both council members agreed on with the judge-validated extras
# into a single clean answer. The seams between sources are hidden — the caller
# gets one coherent response, not a committee transcript.

import json

from core.clients import anthropic_client, CONSOLIDATE_MODEL


def consolidate(original_prompt, agreement, monitor_results):
    """Produce the final single answer from agreed + validated information."""
    validated_items = [v["item"] for v in monitor_results.get("validated", [])]

    consolidation_prompt = f"""You are a consolidation agent. Produce a single clean answer combining agreed information with validated details.

Original question: {original_prompt}

Agreed information:
{json.dumps(agreement, indent=2)}

Additional validated information:
{json.dumps(validated_items, indent=2)}

Produce a clear, well-structured answer. Do not mention multiple AI sources."""

    response = anthropic_client.messages.create(
        model=CONSOLIDATE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": consolidation_prompt}],
    )
    return response.content[0].text
