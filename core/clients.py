# core/clients.py
# Centralized client initialization for all external and local model calls.
# Every other module imports from here — never initialize clients elsewhere.

import os
import anthropic
import ollama
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── API Clients ───────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ── Model Strings ─────────────────────────────────────────────────────────────
# Change model versions here only — never hardcode strings in other modules.
#
# Two roles, deliberately separated:
#   Council members answer the query (Claude + Gemini, run concurrently).
#   The judge referees their answers, validates discrepancies, and consolidates.
#   The judge is a higher tier than the members so the referee is at least as
#   capable as those it judges.
CLAUDE_MODEL    = "claude-sonnet-5"    # Council member — Claude
GEMINI_MODEL    = "gemini-3.5-flash"   # Council member — Gemini
JUDGE_MODEL     = "claude-opus-4-8"    # Judge — comparison, monitoring, consolidation

LOCAL_MODEL     = "ensemble-local"     # Pipeline's local model — see ensemble.Modelfile

# The embedding model is deliberately NOT configured here. memory.py is a
# standalone tool outside the request path, and importing this module would force
# it to hold API keys just to index a folder — so it owns its own embedder config.

# ── Local sampling temperatures ───────────────────────────────────────────────
# The local model does two jobs with opposite requirements. Routing, dedup, and
# compression are *evaluation* tasks — the same input must yield the same verdict
# on every run, or gate decisions flip between runs and the pipeline becomes
# non-reproducible. Answering is a *generation* task and can afford some warmth.
# Ollama accepts these per request, so one model serves both roles.
TEMP_DETERMINISTIC = 0.2   # routing, dedup, compression — verdicts must be stable
TEMP_GENERATIVE    = 0.7   # local answers — variety is fine here

# ── Local Client ──────────────────────────────────────────────────────────────
# ollama is used as a module directly (ollama.chat) — no client object needed.
# Exposed here as a reference so imports stay consistent across modules.
ollama_client = ollama