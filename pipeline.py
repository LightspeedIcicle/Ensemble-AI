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
from core.budget import DEFAULT_LEVEL, DESCRIPTIONS, resolve
from core.compress import maybe_compress
from core.retrieval import recall_context, build_local_system
from core.router import local_router, query_local
from core.council import fan_out, compare_responses
from core.monitor import monitor_discrepancies
from core.consolidate import consolidate


async def run_pipeline(prompt, budget=DEFAULT_LEVEL):
    effort, thinking, _ = resolve(budget)   # raises now, not three API calls in
    knowledge.init_knowledge_store()

    print(f"\nOriginal prompt: {prompt}")
    print(f"Budget: {budget} — {DESCRIPTIONS[budget]}")
    print(f"        effort={effort}, thinking={'adaptive' if thinking else 'off'}")
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

        # ── Stage 2b — Retrieve ───────────────────────────────────────────────
        # Retrieval runs on the ORIGINAL prompt for the same reason routing does:
        # the search query is what decides which passages come back, and a lossy
        # rewrite degrades exactly that.
        context = recall_context(prompt)
        if context:
            print(f"[Retrieval] {len(context)} chars of context recalled")
        else:
            print("[Retrieval] No context — answering unprimed")

        # Answer directly, primed with accumulated master-prompt knowledge and
        # anything the knowledge base had to say.
        print("[Local] Handling query locally...")
        system = build_local_system(knowledge.load_master_prompt(), context)
        response = query_local(local_prompt, system=system)
        print(f"\n--- LOCAL RESPONSE ---\n{response}")
        return

    # ── Stage 3 — Fan out ─────────────────────────────────────────────────────
    # The council receives the prompt exactly as the user wrote it.
    print("\n[Escalating] Sending to Claude and Gemini...")
    claude_response, gemini_response = await fan_out(prompt, budget)
    print("[Fan-out complete] Both council members responded")

    # ── Stage 4 — Compare ─────────────────────────────────────────────────────
    comparison = compare_responses(prompt, claude_response, gemini_response, budget)
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
        budget,
    )
    if not monitor_results:
        print("Monitoring failed")
        return
    validated = monitor_results.get("validated", [])
    removed = monitor_results.get("removed", [])
    ambiguous = monitor_results.get("ambiguous", [])
    print(f"[Monitoring complete] Validated: {len(validated)} | Removed: {len(removed)} | Ambiguous: {len(ambiguous)}")

    # ── Stage 6 — Consolidate ─────────────────────────────────────────────────
    final_answer = consolidate(prompt, comparison.get("agreement", []), monitor_results, budget)
    print("\n--- FINAL CONSOLIDATED ANSWER ---")
    print(final_answer)

    # The judge already paid to explain each rejection, and a claim one council
    # member made that the judge ruled false is the single most interesting thing
    # this pipeline produces — it's cross-validation catching a hallucination in
    # the act. It was being counted and discarded. Show it.
    if removed:
        print("\n--- REJECTED BY THE JUDGE ---")
        for item in removed:
            print(f"  ✗ {item['item']}: {item['reason']}")

    if ambiguous:
        print("\n--- FLAGGED FOR REVIEW ---")
        for item in ambiguous:
            print(f"  ? {item['item']}: {item['reason']}")

    # ── Stage 7 — Persist knowledge ───────────────────────────────────────────
    knowledge.persist(prompt, validated)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from core.budget import LEVELS

    parser = argparse.ArgumentParser(
        description="Ensemble AI — local-first multi-agent pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="budget levels:\n"
        + "\n".join(f"  {n:9} {DESCRIPTIONS[n]}" for n in LEVELS),
    )
    parser.add_argument("prompt", nargs="?", help="the question (omit for the demo prompt)")
    parser.add_argument(
        "-b", "--budget", choices=list(LEVELS), default=DEFAULT_LEVEL,
        help=f"how much to spend on this query (default: {DEFAULT_LEVEL})",
    )
    args = parser.parse_args()

    test_prompt = (
        "Could you please explain to me what were the most significant and "
        "important causes that led to the outbreak of the French Revolution in "
        "the late 18th century?"
    )
    asyncio.run(run_pipeline(args.prompt or test_prompt, args.budget))
