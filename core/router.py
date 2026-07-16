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

ROUTER_SYSTEM = """You are a routing agent. Decide whether a query is answered locally (free) or escalated to a council of frontier models (paid).

Apply these rules IN ORDER. The first rule that matches decides — stop there.

1. ESCALATE if well-informed sources could legitimately disagree about the answer: multiple contributing causes, interpretation, synthesis, comparison, recommendation, or anything needing comprehensive coverage rather than a fact.

2. ESCALATE if being wrong would matter and the answer is not common knowledge: medicine, law, finance, safety, or any claim someone might act on.

3. LOCAL if the answer is one settled fact that a reference work would state the same way every time: capitals, dates, unit conversions, simple arithmetic, or the definition of an established term.

4. LOCAL if it is conversational: a greeting, small talk, or a remark needing no new information.

5. ESCALATE. This is the default. If nothing above clearly matched, you are in doubt — and a wrong answer to a hard question costs more than a few API tokens.

Rule 3 is narrow. It covers only answers that are SETTLED — where every reference work says the same thing and there is nothing for experts to argue about. A question in a technical field can still be settled: "What is a closure in programming?" is a definition every textbook states identically, so rule 3 sends it local. But settled is a property of the ANSWER, not the field, and most questions are not settled:

- "Should I use closures or classes here?" — a judgement experts differ on. Rule 1.
- "Is intermittent fasting healthy?" — contested, and about health. Rule 1, and rule 2.
- "Is this dosage safe?" — someone could be harmed by a wrong answer. Rule 2.

If you find yourself reasoning that something is "basically settled" or "mostly agreed," it is not settled. Rule 3 requires no hedging at all. When the hedge appears, you have already left rule 3 and rule 5 applies.

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
