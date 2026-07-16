# pipeline.py
# Ensemble AI — orchestrator.
#
# Wires the core/ stages into the full request → response flow:
#
#   route ──local──> compress (if large) → local answer
#        └─escalate─> fan-out → compare → monitor → consolidate → persist
#
# Note the order: routing happens on the ORIGINAL prompt, and compression only
# ever runs on the local branch. Nothing bound for a frontier model is compressed
# — see core/compress.py for why that inverted the quality gradient.
#
# This file stays a thin conductor: every stage's real logic lives in core/.

import asyncio

from core import knowledge
from core.compress import maybe_compress
from core.router import local_router, query_local
from core.council import fan_out, compare_responses
from core.monitor import monitor_discrepancies
from core.consolidate import consolidate


async def run_pipeline(prompt):
    knowledge.init_knowledge_store()

    print(f"\nOriginal prompt: {prompt}")
    print("=" * 60)

    # ── Stage 1 — Route ───────────────────────────────────────────────────────
    # Routing reads the original prompt. The router's whole job is judging
    # nuance and difficulty, so handing it a lossily-compressed prompt would
    # degrade the exact signal it decides on.
    print("\n[Routing] Evaluating query...")
    routing = local_router(prompt)
    print(f"[Routing] {routing['decision'].upper()} | Confidence: {routing['confidence']} | {routing['reason']}")

    if routing["decision"] == "local":
        # ── Stage 2 — Compress (local → local only) ───────────────────────────
        # Both ends of this handoff are the same local model, so a lossy rewrite
        # costs nothing a frontier model would have noticed. Skipped entirely
        # unless the prompt is large enough for context pressure to be real.
        local_prompt = prompt
        compression = maybe_compress(prompt)
        if compression:
            local_prompt = compression["compressed"]
            print(f"\n[Compression] {compression['original_tokens']} → {compression['compressed_tokens']} tokens "
                  f"({compression['reduction_percent']}% reduction)")
        else:
            print("\n[Compression] Skipped — prompt is already small")

        # Answer directly, primed with accumulated master-prompt knowledge.
        print("[Local] Handling query locally...")
        response = query_local(local_prompt, system=knowledge.load_master_prompt())
        print(f"\n--- LOCAL RESPONSE ---\n{response}")
        return

    # ── Stage 3 — Fan out ─────────────────────────────────────────────────────
    # The council receives the prompt exactly as the user wrote it.
    print("\n[Escalating] Sending to Claude and Gemini...")
    claude_response, gemini_response = await fan_out(prompt)
    print("[Fan-out complete] Both council members responded")

    # ── Stage 4 — Compare ─────────────────────────────────────────────────────
    comparison = compare_responses(prompt, claude_response, gemini_response)
    if not comparison:
        print("Comparison failed")
        return
    print(f"[Comparison complete] Found {len(comparison.get('discrepancies', []))} discrepancies")

    # ── Stage 5 — Monitor / validate ──────────────────────────────────────────
    monitor_results = monitor_discrepancies(
        prompt,
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
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results)
    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

    # ── Stage 7 — Persist knowledge ───────────────────────────────────────────
    knowledge.persist(prompt, validated)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = (
        "Could you please explain to me what were the most significant and "
        "important causes that led to the outbreak of the French Revolution in "
        "the late 18th century?"
    )
    asyncio.run(run_pipeline(test_prompt))
