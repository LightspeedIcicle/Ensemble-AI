# pipeline.py
# Ensemble AI — orchestrator.
#
# Wires the core/ stages into the full request → response flow:
#
#   compress → route ──local──> local answer
#                   └─escalate─> fan-out → compare → monitor → consolidate → persist
#
# This file stays a thin conductor: every stage's real logic lives in core/.

import asyncio

from core import knowledge
from core.compress import compress_prompt
from core.router import local_router, query_local
from core.council import fan_out, compare_responses
from core.monitor import monitor_discrepancies
from core.consolidate import consolidate


async def run_pipeline(prompt):
    knowledge.init_knowledge_store()

    print(f"\nOriginal prompt: {prompt}")
    print("=" * 60)

    # ── Stage 1 — Compress ────────────────────────────────────────────────────
    print("\n[Compression] Compressing prompt...")
    compression = compress_prompt(prompt)
    compressed = compression["compressed"]
    print(f"[Compression] {compression['original_tokens']} → {compression['compressed_tokens']} tokens "
          f"({compression['reduction_percent']}% reduction)")
    print(f"[Compression] Compressed: {compressed}")

    # ── Stage 2 — Route ───────────────────────────────────────────────────────
    print("\n[Routing] Evaluating query...")
    routing = local_router(compressed)
    print(f"[Routing] {routing['decision'].upper()} | Confidence: {routing['confidence']} | {routing['reason']}")

    if routing["decision"] == "local":
        # Local path: answer directly, primed with accumulated master-prompt knowledge.
        print("\n[Local] Handling query locally...")
        response = query_local(compressed, system=knowledge.load_master_prompt())
        print(f"\n--- LOCAL RESPONSE ---\n{response}")
        return

    # ── Stage 3 — Fan out ─────────────────────────────────────────────────────
    print("\n[Escalating] Sending to Claude and Gemini...")
    claude_response, gemini_response = await fan_out(compressed)
    print("[Fan-out complete] Both council members responded")

    # ── Stage 4 — Compare ─────────────────────────────────────────────────────
    comparison = compare_responses(compressed, claude_response, gemini_response)
    if not comparison:
        print("Comparison failed")
        return
    print(f"[Comparison complete] Found {len(comparison.get('discrepancies', []))} discrepancies")

    # ── Stage 5 — Monitor / validate ──────────────────────────────────────────
    monitor_results = monitor_discrepancies(
        compressed,
        comparison.get("discrepancies", []),
        comparison.get("unique_to_a", []),
        comparison.get("unique_to_b", []),
    )
    if not monitor_results:
        print("Monitoring failed")
        return
    validated = monitor_results.get("validated", [])
    removed = monitor_results.get("removed", [])
    ambiguous = monitor_results.get("ambiguous", [])
    print(f"[Monitoring complete] Validated: {len(validated)} | Removed: {len(removed)} | Ambiguous: {len(ambiguous)}")

    # ── Stage 6 — Consolidate ─────────────────────────────────────────────────
    # Use the original (uncompressed) prompt so the final answer reads naturally.
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results)
    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

    # ── Stage 7 — Persist knowledge ───────────────────────────────────────────
    knowledge.persist(prompt, compressed, validated)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = (
        "Could you please explain to me what were the most significant and "
        "important causes that led to the outbreak of the French Revolution in "
        "the late 18th century?"
    )
    asyncio.run(run_pipeline(test_prompt))
