import asyncio
import anthropic
import ollama as ollama_client
from google import genai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from pathlib import Path

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Knowledge Log Setup ───────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_LOG = KNOWLEDGE_DIR / "log.json"
MASTER_PROMPT_FILE = KNOWLEDGE_DIR / "master_prompt.txt"

def init_knowledge_store():
    """Create knowledge directory and files if they don't exist."""
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    if not KNOWLEDGE_LOG.exists():
        KNOWLEDGE_LOG.write_text(json.dumps({"entries": []}, indent=2))
    if not MASTER_PROMPT_FILE.exists():
        MASTER_PROMPT_FILE.write_text("You are a helpful assistant with the following accumulated knowledge:\n\n")

def load_knowledge_log():
    return json.loads(KNOWLEDGE_LOG.read_text())

def save_knowledge_log(data):
    KNOWLEDGE_LOG.write_text(json.dumps(data, indent=2))

def load_master_prompt():
    return MASTER_PROMPT_FILE.read_text()

def save_master_prompt(content):
    MASTER_PROMPT_FILE.write_text(content)

def add_to_knowledge_log(prompt, validated_items, source_topic):
    """Add validated items to the knowledge log."""
    data = load_knowledge_log()
    new_entries = []
    for item in validated_items:
        entry = {
            "id": len(data["entries"]) + len(new_entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "source_topic": source_topic,
            "knowledge": item,
            "added_to_master": False
        }
        new_entries.append(entry)
        data["entries"].append(entry)
    save_knowledge_log(data)
    return new_entries

def check_duplicate(new_item, existing_entries):
    """Ask local LLM if this item is already captured in the log."""
    if not existing_entries:
        return False
    
    existing_knowledge = [e["knowledge"] for e in existing_entries[-20:]]
    
    result = ollama_client.chat(
        model="dolphin-llama3",
        messages=[
            {
                "role": "system",
                "content": """You are a deduplication agent. Determine if new information is already captured in existing knowledge.
Return only this JSON with no preamble or markdown:
{"is_duplicate": true or false, "reason": "brief explanation"}"""
            },
            {
                "role": "user",
                "content": f"New item: {new_item}\n\nExisting knowledge:\n{json.dumps(existing_knowledge, indent=2)}"
            }
        ]
    )
    parsed = parse_json(result["message"]["content"])
    if not parsed:
        return False
    return parsed.get("is_duplicate", False)

def request_master_prompt_addition(items, source_topic):
    """Ask user for confirmation before adding to master prompt."""
    print("\n--- KNOWLEDGE LOG UPDATE ---")
    print(f"Source topic: {source_topic}")
    print(f"The following {len(items)} items were validated and are new:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    
    print("\nOptions:")
    print("  [a] Add all to master prompt")
    print("  [s] Select specific items")
    print("  [l] Log only (don't add to master prompt)")
    print("  [n] Skip")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == "a":
        return items
    elif choice == "s":
        selected = []
        for i, item in enumerate(items, 1):
            confirm = input(f"Add item {i}? (y/n): ").strip().lower()
            if confirm == "y":
                selected.append(item)
        return selected
    elif choice == "l":
        return []
    else:
        return None

def add_to_master_prompt(items, source_topic):
    """Append validated knowledge to the master prompt file."""
    current = load_master_prompt()
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    addition = f"\n## {source_topic} ({timestamp})\n"
    for item in items:
        addition += f"- {item}\n"
    
    save_master_prompt(current + addition)
    
    data = load_knowledge_log()
    for entry in data["entries"]:
        if entry["knowledge"] in items:
            entry["added_to_master"] = True
    save_knowledge_log(data)
    
    print(f"[Knowledge] Added {len(items)} items to master prompt")

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

def estimate_tokens(text):
    return int(len(text.split()) / 0.75)

# ── Compression Agent ─────────────────────────────────────────────────────────

def compress_prompt(prompt):
    system = """You are a compression agent. Rewrite the given prompt in the most token-efficient form possible while preserving complete meaning.

Rules:
- Remove filler words, pleasantries, and redundant phrasing
- Remove phrases like 'Could you please', 'I would like to know'
- Collapse verbose phrasing into direct questions
- Preserve all key terms, named entities, and specific requirements
- Never lose meaning

Return only the compressed prompt as a plain string. No JSON, no explanation, no preamble."""

    result = ollama_client.chat(
        model="dolphin-llama3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compress this prompt: {prompt}"}
        ]
    )
    compressed = result["message"]["content"].strip()
    if compressed.startswith('"') and compressed.endswith('"'):
        compressed = compressed[1:-1].strip()
    
    original_tokens = estimate_tokens(prompt)
    compressed_tokens = estimate_tokens(compressed)
    reduction = round((1 - compressed_tokens / original_tokens) * 100) if original_tokens > 0 else 0
    return {"compressed": compressed, "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens, "reduction_percent": reduction}

# ── Local LLM ─────────────────────────────────────────────────────────────────

def query_local(prompt, system=None):
    messages = []
    master = load_master_prompt()
    if master.strip():
        messages.append({"role": "system", "content": master})
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama_client.chat(model="dolphin-llama3", messages=messages)
    return response["message"]["content"]

def local_router(prompt):
    system = """You are a routing agent. Decide whether a query should be handled locally or escalated to external AI models.

ALWAYS escalate if the query:
- Has multiple contributing causes, factors, or dimensions
- Involves historical analysis, interpretation, or synthesis
- Could have legitimate disagreement between sources
- Requires comprehensive coverage rather than a summary
- Is an essay-type or analytical question
- Involves science, medicine, law, economics, or technical domains

Handle locally ONLY if the query:
- Has a single, universally agreed factual answer
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
    init_knowledge_store()
    
    print(f"\nOriginal prompt: {prompt}")
    print("=" * 60)

    # Step 1 - Compress
    print("\n[Compression] Compressing prompt...")
    compression = compress_prompt(prompt)
    compressed = compression["compressed"]
    print(f"[Compression] {compression['original_tokens']} → {compression['compressed_tokens']} tokens ({compression['reduction_percent']}% reduction)")
    print(f"[Compression] Compressed: {compressed}")

    # Step 2 - Route
    print("\n[Routing] Evaluating query...")
    routing = local_router(compressed)
    print(f"[Routing] {routing['decision'].upper()} | Confidence: {routing['confidence']} | {routing['reason']}")

    if routing["decision"] == "local":
        print("\n[Local] Handling query locally...")
        response = query_local(compressed)
        print(f"\n--- LOCAL RESPONSE ---\n{response}")
        return

    # Step 3 - Fan out
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

    # Step 6 - Consolidate
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results)
    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

    # Step 7 - Knowledge log
    if validated:
        validated_knowledge = [v["item"] for v in validated]
        existing = load_knowledge_log()["entries"]
        
        # Filter duplicates
        new_items = []
        for item in validated_knowledge:
            if not check_duplicate(item, existing):
                new_items.append(item)
        
        if new_items:
            # Extract topic from compressed prompt
            topic = compressed[:50] + "..." if len(compressed) > 50 else compressed
            
            # Ask user what to do
            selected = request_master_prompt_addition(new_items, topic)
            
            if selected is None:
                print("[Knowledge] Skipped")
            else:
                # Always log everything
                add_to_knowledge_log(prompt, new_items, topic)
                print(f"[Knowledge] Logged {len(new_items)} items")
                
                # Add selected to master prompt
                if selected:
                    add_to_master_prompt(selected, topic)
        else:
            print("[Knowledge] No new information to log")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = "How has this effected modern democracy?"
    asyncio.run(run_pipeline(test_prompt))
