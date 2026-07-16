# core/ — Ensemble AI pipeline stages.
#
# Each module owns one stage of the request → response pipeline:
#   clients      shared model clients + centralized model IDs
#   helpers      pure utilities (JSON parsing, token estimates)
#   compress     Stage 1 — prompt compression (local)
#   router       Stage 2 — local vs. escalate routing (local)
#   council      Stage 3-4 — concurrent fan-out + cross-comparison
#   monitor      Stage 5 — validation / filtering (judge)
#   consolidate  Stage 6 — final answer synthesis (judge)
#   knowledge    Stage 7 — knowledge persistence
#
# The orchestrator that wires these together lives in ../pipeline.py.
