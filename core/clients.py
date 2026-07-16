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

LOCAL_MODEL     = "dolphin-llama3"     # Aurora's current base — swap here when eval picks a winner
EMBED_MODEL     = "nomic-embed-text"   # Local embedding model via Ollama

# ── Local Client ──────────────────────────────────────────────────────────────
# ollama is used as a module directly (ollama.chat) — no client object needed.
# Exposed here as a reference so imports stay consistent across modules.
ollama_client = ollama