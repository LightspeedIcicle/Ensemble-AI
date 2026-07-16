# core/router.py
# Stage 2 — Local routing.
# The local LLM decides whether a query can be answered locally (free) or must
# be escalated to the external council (paid). It also serves as the local
# answerer for the simple path. Bias is deliberately toward escalation.

from core.clients import (
    ollama_client,
    LOCAL_MODEL,
    TEMP_DETERMINISTIC,
    TEMP_GENERATIVE,
)
from core.helpers import parse_json

ROUTER_SYSTEM = """You are a routing agent. Decide whether a query should be handled locally or escalated to powerful external AI models for deep research and cross-validation.

ALWAYS escalate if the query:
- Has multiple contributing causes, factors, or dimensions
- Involves historical analysis, interpretation, or synthesis
- Could have legitimate disagreement between sources
- Requires comprehensive coverage rather than a summary
- Is an essay-type or analytical question
- Involves science, medicine, law, economics, or technical domains

Handle locally ONLY if the query:
- Has a single, universally agreed factual answer (capitals, dates, definitions)
- Is casual conversation or a greeting
- Is a simple arithmetic or logic question

Be conservative — when in doubt, escalate. The cost of under-escalating is worse than over-escalating.

Return only this JSON with no preamble or markdown:
{
    "decision": "local" or "escalate",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}"""


def query_local(prompt, system=None, temperature=TEMP_GENERATIVE):
    """Send a prompt to the local LLM.

    `system` is an optional system message — used to pass the router
    instructions, or the accumulated master prompt when answering locally.
    `temperature` is set per call because routing and answering want opposite
    values from the same model (see core/clients.py).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama_client.chat(
        model=LOCAL_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def local_router(prompt):
    """Ask the local LLM to decide: handle locally or escalate?

    Runs near-greedy: a router that samples hot returns a different verdict for
    the same query across runs, which makes the whole pipeline non-reproducible.

    Fails safe — any parse failure defaults to escalation.
    """
    result = query_local(prompt, system=ROUTER_SYSTEM, temperature=TEMP_DETERMINISTIC)
    parsed = parse_json(result)
    if not parsed:
        return {"decision": "escalate", "confidence": 0.5, "reason": "routing parse failed, defaulting to escalate"}
    return parsed
