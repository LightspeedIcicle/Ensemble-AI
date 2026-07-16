import asyncio
import anthropic
from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None

def compare_responses(prompt, claude_response, gemini_response):
    comparison_prompt = f"""You are a precise analytical agent. Compare two AI responses to the same prompt and identify meaningful discrepancies.

Original prompt: {prompt}

Response A (Claude):
{claude_response}

Response B (Gemini):
{gemini_response}

Return only this JSON structure with no preamble or markdown:
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

    monitor_prompt = f"""You are a monitoring agent responsible for validating and filtering information from two AI responses.

Original task: {original_prompt}

You have been given discrepancies and unique points to evaluate.

DISCREPANCIES (where responses disagreed):
{json.dumps(discrepancies, indent=2)}

UNIQUE TO RESPONSE A (Claude only):
{json.dumps(unique_to_a, indent=2)}

UNIQUE TO RESPONSE B (Gemini only):
{json.dumps(unique_to_b, indent=2)}

For each item, apply two checks:
1. VALIDITY: Is this factually accurate based on your knowledge?
2. RELEVANCE: Is this relevant to the original task?

Return only this JSON structure with no preamble or markdown:
{{
    "validated": [
        {{
            "item": "the information",
            "source": "discrepancy or unique_a or unique_b",
            "verdict": "brief explanation of why it is valid and relevant"
        }}
    ],
    "removed": [
        {{
            "item": "the information",
            "reason": "invalid or irrelevant - brief explanation"
        }}
    ],
    "ambiguous": [
        {{
            "item": "the information",
            "reason": "why this could not be clearly resolved"
        }}
    ]
}}"""

    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": monitor_prompt}]
    )
    return parse_json(response.content[0].text)

def consolidate(original_prompt, agreement, monitor_results):
    validated_items = [v["item"] for v in monitor_results.get("validated", [])]
    
    consolidation_prompt = f"""You are a consolidation agent. Your job is to produce a single clean answer to a question by combining agreed information with validated additional details.

Original question: {original_prompt}

Information both AI responses agreed on:
{json.dumps(agreement, indent=2)}

Additional validated information to incorporate:
{json.dumps(validated_items, indent=2)}

Produce a single clear, well-structured answer that incorporates all of the above. Do not mention that this came from multiple AI sources. Just answer the question directly and thoroughly."""

    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": consolidation_prompt}]
    )
    return response.content[0].text

async def run_pipeline(prompt):
    print(f"\nPrompt: {prompt}")
    print("=" * 60)

    # Phase 1 - Fan out
    loop = asyncio.get_event_loop()
    claude_task = loop.run_in_executor(None, query_claude, prompt)
    gemini_task = loop.run_in_executor(None, query_gemini, prompt)
    claude_response, gemini_response = await asyncio.gather(claude_task, gemini_task)

    print("\n[Phase 1 complete] Both models responded")

    # Phase 2 - Compare
    comparison = compare_responses(prompt, claude_response, gemini_response)
    if not comparison:
        print("Comparison failed")
        return

    print(f"[Phase 2 complete] Found {len(comparison.get('discrepancies', []))} discrepancies")

    # Phase 3 - Monitor
    monitor_results = monitor_discrepancies(
        prompt,
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

    # Consolidation
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results)

    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

if __name__ == "__main__":
    test_prompt = "What are the main causes of the French Revolution?"
    asyncio.run(run_pipeline(test_prompt))
