import asyncio
import anthropic
from google import genai
from dotenv import load_dotenv
import os
import json

# Load API keys
load_dotenv()

# Initialize clients
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

def compare_responses(prompt, claude_response, gemini_response):
    comparison_prompt = f"""You are a precise analytical agent. Your job is to compare two AI responses to the same prompt and identify meaningful discrepancies between them.

Original prompt: {prompt}

Response A (Claude):
{claude_response}

Response B (Gemini):
{gemini_response}

Analyze both responses and return a JSON object with this exact structure:
{{
    "agreement": ["list of points both responses agree on"],
    "discrepancies": [
        {{
            "topic": "what the discrepancy is about",
            "response_a": "what Claude said",
            "response_b": "what Gemini said"
        }}
    ],
    "unique_to_a": ["points only Claude mentioned"],
    "unique_to_b": ["points only Gemini mentioned"]
}}

Return only the JSON object. No preamble, no explanation, no markdown formatting."""

    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": comparison_prompt}]
    )
    
    raw = response.content[0].text.strip()
    
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object directly
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse comparison", "raw": raw}

async def query_and_compare(prompt):
    print(f"\nPrompt: {prompt}")
    print("=" * 60)
    
    # Phase 1 - Fan out to both models simultaneously
    loop = asyncio.get_event_loop()
    claude_task = loop.run_in_executor(None, query_claude, prompt)
    gemini_task = loop.run_in_executor(None, query_gemini, prompt)
    
    claude_response, gemini_response = await asyncio.gather(claude_task, gemini_task)
    
    print(f"\nCLAUDE:\n{claude_response}")
    print("-" * 60)
    print(f"\nGEMINI:\n{gemini_response}")
    print("-" * 60)
    
    # Phase 2 - Cross comparison
    print("\nRunning cross-comparison analysis...")
    comparison = compare_responses(prompt, claude_response, gemini_response)
    
    if "error" in comparison:
        print(f"Comparison error: {comparison['error']}")
        return
    
    print("\n--- COMPARISON RESULTS ---")
    
    print("\nAGREEMENT:")
    for point in comparison.get("agreement", []):
        print(f"  ✓ {point}")
    
    print("\nDISCREPANCIES:")
    discrepancies = comparison.get("discrepancies", [])
    if discrepancies:
        for d in discrepancies:
            print(f"\n  Topic: {d['topic']}")
            print(f"  Claude: {d['response_a']}")
            print(f"  Gemini: {d['response_b']}")
    else:
        print("  None found")
    
    print("\nUNIQUE TO CLAUDE:")
    for point in comparison.get("unique_to_a", []):
        print(f"  → {point}")
    
    print("\nUNIQUE TO GEMINI:")
    for point in comparison.get("unique_to_b", []):
        print(f"  → {point}")

# Entry point
if __name__ == "__main__":
    test_prompt = "What are the main causes of the French Revolution?"
    asyncio.run(query_and_compare(test_prompt))
