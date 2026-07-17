# core/council.py
# Stage 3 & 4 — The external council: concurrent fan-out and cross-comparison.
# The two council members (Claude + Gemini) answer the same prompt in parallel.
# The judge (a higher-tier model) then referees the two answers against each
# other to surface agreement and discrepancies.

import asyncio

from core.budget import DEFAULT_LEVEL, max_tokens, sampling
from core.clients import (
    anthropic_client,
    gemini_client,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    COMPARE_MODEL,
)
from core.helpers import parse_json, response_text


# ── Council members (answer the query) ────────────────────────────────────────

def query_claude(prompt, budget=DEFAULT_LEVEL):
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens(budget, "member"),
        messages=[{"role": "user", "content": prompt}],
        **sampling(budget),
    )
    return response_text(response)


def query_gemini(prompt):
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# ── Concurrent fan-out ────────────────────────────────────────────────────────

async def fan_out(prompt, budget=DEFAULT_LEVEL):
    """Query both council members concurrently. Returns (claude_response, gemini_response).

    Both SDK calls are blocking, so they run in the default executor and are
    awaited together with asyncio.gather.

    `budget` reaches query_claude only. Gemini is a different provider with a
    different parameter surface — effort and adaptive thinking are Anthropic
    concepts. The dial deliberately stops at the API boundary rather than
    pretending to a reach it does not have. See DECISIONS.md.
    """
    loop = asyncio.get_event_loop()
    claude_task = loop.run_in_executor(None, query_claude, prompt, budget)
    gemini_task = loop.run_in_executor(None, query_gemini, prompt)
    return await asyncio.gather(claude_task, gemini_task)


# ── Cross-comparison (judge role) ─────────────────────────────────────────────

def compare_responses(prompt, claude_response, gemini_response, budget=DEFAULT_LEVEL):
    """Have the judge referee the two answers, returning structured agreement/discrepancies."""
    comparison_prompt = f"""You are a precise analytical agent. Compare two AI responses and identify meaningful discrepancies.

Original prompt: {prompt}

Response A (Claude):
{claude_response}

Response B (Gemini):
{gemini_response}

Return only this JSON with no preamble or markdown:
{{
    "agreement": ["points both responses agree on"],
    "discrepancies": [
        {{
            "topic": "what the discrepancy is about",
            "response_a": "what Claude said",
            "response_b": "what Gemini said"
        }}
    ],
    "unique_to_a": ["points only Claude mentioned"],
    "unique_to_b": ["points only Gemini mentioned"]
}}"""

    response = anthropic_client.messages.create(
        model=COMPARE_MODEL,
        max_tokens=max_tokens(budget, "compare"),
        messages=[{"role": "user", "content": comparison_prompt}],
        **sampling(budget),
    )
    return parse_json(response_text(response))
