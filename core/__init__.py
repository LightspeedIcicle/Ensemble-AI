# core/ — Ensemble AI pipeline stages.
#
# Each module owns one stage of the request → response pipeline:
#   clients      shared model clients, model IDs, sampling temperatures
#   helpers      pure utilities (JSON parsing, token estimates)
#   router       Stage 1  — local vs. escalate routing (local)
#   compress     Stage 2  — compression, local branch only (local)
#   retrieval    Stage 2b — RAG lookup, local branch only (local)
#   council      Stage 3-4 — concurrent fan-out + cross-comparison
#   monitor      Stage 5  — validation / filtering (judge)
#   consolidate  Stage 6  — final answer synthesis (judge)
#   knowledge    Stage 7  — knowledge persistence
#
# The orchestrator that wires these together lives in ../pipeline.py.
