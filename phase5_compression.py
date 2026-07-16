import asyncio
import anthropic
import ollama as ollama_client
from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None

# ── Compression Agent ─────────────────────────────────────────────────────────

def estimate_tokens(text):
    """Rough token estimate — approximately 1 token per 0.75 words."""
    return int(len(text.split()) / 0.75)

def compress_prompt(prompt):
    """Convert verbose prompt to token-efficient shorthand using local LLM."""
    system = """You are a compression agent. Rewrite the given prompt in the most token-efficient form possible while preserving complete meaning.

Rules:
- Remove filler words, pleasantries, and redundant phrasing
- Remove phrases like 'Could you please', 'I would like to know', 'can you tell me'
- Collapse verbose phrasing into direct questions
- Preserve all key terms, named entities, and specific requirements
- Never lose meaning — compression must be lossless in terms of intent

Return only the compressed prompt as a plain string. No JSON, no explanation, no preamble."""

    result = ollama_client.chat(
        model="dolphin-llama3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compress this prompt: {prompt}"}
        ]
    )
    
    compressed = result["message"]["content"].strip()
    
    # Strip any accidental quotes the model might wrap around the output
    if compressed.startswith('"') and compressed.endswith('"'):
        compressed = compressed[1:-1].strip()
    
    original_tokens = estimate_tokens(prompt)
    compressed_tokens = estimate_tokens(compressed)
    reduction = round((1 - compressed_tokens / original_tokens) * 100) if original_tokens > 0 else 0

    return {
        "compressed": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_percent": reduction
    }

# ── Local LLM ─────────────────────────────────────────────────────────────────

def query_local(prompt, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama_client.chat(model="dolphin-llama3", messages=messages)
    return response["message"]["content"]

def local_router(prompt):
    system = """You are a routing agent. Decide whether a query should be handled locally or escalated to powerful external AI models for deep research and cross-validation.

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

Be conservative — when in doubt, escalate.

Return only this JSON with no preamble or markdown:
{
    "decision": "local" or "escalate",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}"""

    result = query_local(prompt, system=system)
    parsed = parse_json(result)
    if not parsed:
        return {"decision": "escalate", "confidence": 0.5, "reason": "routing parse failed"}
    return parsed

# ── External Models ───────────────────────────────────────────────────────────

def query_claude(prompt):
    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def query_gemini(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ── Pipeline Stages ───────────────────────────────────────────────────────────

def compare_responses(prompt, claude_response, gemini_response):
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
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": comparison_prompt}]
    )
    return parse_json(response.content[0].text)

def monitor_discrepancies(original_prompt, discrepancies, unique_to_a, unique_to_b):
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

Return only this JSON with no preamble or markdown:
{{
    "validated": [{{"item": "...", "source": "...", "verdict": "..."}}],
    "removed": [{{"item": "...", "reason": "..."}}],
    "ambiguous": [{{"item": "...", "reason": "..."}}]
}}"""

    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": monitor_prompt}]
    )
    return parse_json(response.content[0].text)

def consolidate(original_prompt, agreement, monitor_results):
    validated_items = [v["item"] for v in monitor_results.get("validated", [])]
    consolidation_prompt = f"""You are a consolidation agent. Produce a single clean answer combining agreed information with validated details.

Original question: {original_prompt}

Agreed information:
{json.dumps(agreement, indent=2)}

Additional validated information:
{json.dumps(validated_items, indent=2)}

Produce a clear, well-structured answer. Do not mention multiple AI sources."""

    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": consolidation_prompt}]
    )
    return response.content[0].text

# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(prompt):
    print(f"\nOriginal prompt: {prompt}")
    print("=" * 60)

    # Step 1 - Compress
    print("\n[Compression] Compressing prompt...")
    compression = compress_prompt(prompt)
    compressed = compression["compressed"]
    print(f"[Compression] Original: ~{compression['original_tokens']} tokens")
    print(f"[Compression] Compressed: ~{compression['compressed_tokens']} tokens")
    print(f"[Compression] Reduction: {compression['reduction_percent']}%")
    print(f"[Compression] Compressed prompt: {compressed}")

    # Step 2 - Route using compressed prompt
    print("\n[Routing] Evaluating query complexity...")
    routing = local_router(compressed)
    print(f"[Routing] Decision: {routing['decision'].upper()} | Confidence: {routing['confidence']} | Reason: {routing['reason']}")

    if routing["decision"] == "local":
        print("\n[Local] Handling query locally...")
        response = query_local(compressed)
        print(f"\n--- LOCAL RESPONSE ---\n{response}")
        return

    # Step 3 - Fan out using compressed prompt
    print("\n[Escalating] Sending to Claude and Gemini...")
    loop = asyncio.get_event_loop()
    claude_task = loop.run_in_executor(None, query_claude, compressed)
    gemini_task = loop.run_in_executor(None, query_gemini, compressed)
    claude_response, gemini_response = await asyncio.gather(claude_task, gemini_task)
    print("[Phase 1 complete] Both models responded")

    # Step 4 - Compare
    comparison = compare_responses(compressed, claude_response, gemini_response)
    if not comparison:
        print("Comparison failed")
        return
    print(f"[Phase 2 complete] Found {len(comparison.get('discrepancies', []))} discrepancies")

    # Step 5 - Monitor
    monitor_results = monitor_discrepancies(
        compressed,
        comparison.get("discrepancies", []),
        comparison.get("unique_to_a", []),
        comparison.get("unique_to_b", [])
    )
    if not monitor_results:
        print("Monitoring failed")
        return
    validated = monitor_results.get("validated", [])
    removed = monitor_results.get("removed", [])
    ambiguous = monitor_results.get("ambiguous", [])
    print(f"[Phase 3 complete] Validated: {len(validated)} | Removed: {len(removed)} | Ambiguous: {len(ambiguous)}")

    # Step 6 - Consolidate using original prompt for natural output
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results)
    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = "Could you please explain to me what were the most significant and important causes that led to the outbreak of the French Revolution in the late 18th century?"
    asyncio.run(run_pipeline(test_prompt))
