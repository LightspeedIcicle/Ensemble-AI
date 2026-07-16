# Ensemble AI

A cost-efficient, multi-agent LLM pipeline. Instead of sending every query to a
single expensive model, Ensemble AI runs a **local-first pipeline** that
compresses prompts, routes cheap queries to a free local model, and escalates
only the hard ones to a **council** of frontier models — whose answers are then
refereed, validated, and consolidated by a higher-tier **judge**.

The thesis: the next generation of AI applications won't be single-model
chatbots. They'll be pipelines — composable agents with specialized roles, cost
controls, validation layers, and persistent memory.

---

## Why it exists

A single LLM call is a black box: you get one answer, no second opinion, and you
pay full price whether the question was "what's the capital of France" or a
multi-part research question. Ensemble AI addresses three problems at once:

- **Cost** — a free local model (via Ollama) triages every query and answers the
  simple ones outright. Paid APIs are only touched when the query genuinely
  warrants them. The saving comes from *not making the call*, not from shaving
  tokens off calls we make — see [DECISIONS.md](DECISIONS.md) on why compression
  was moved off the escalation path.
- **Reliability** — hard queries are answered by *two* independent frontier
  models, then cross-examined so hallucinations and disagreements surface
  instead of shipping silently.
- **Memory** — facts that survive validation are logged and can be promoted into
  a persistent "master prompt" that makes the local model smarter over time.

---

## Architecture

```
                         ┌──────────────┐
   query ──────────────► │  1. ROUTE    │  local LLM: answer here, or escalate?
                         └──────┬───────┘  (reads the original prompt)
                    local │            │ escalate
              ┌───────────┘            └───────────┐
              ▼                                     ▼
     ┌─────────────────┐              ┌──────────────────────────┐
     │  2. COMPRESS    │              │  3. FAN-OUT (concurrent)  │
     │  local→local    │              │  Claude ‖ Gemini          │
     │  only, and only │              │                           │
     │  if large       │              │  receives the ORIGINAL    │
     └────────┬────────┘              │  prompt — never a         │
              ▼                        │  compressed one           │
     ┌─────────────────┐              └────────────┬─────────────┘
     │  local answer   │                           │
     │  (free, primed  │                           │
     │   by master     │                           │
     │   prompt)       │                           │
     └─────────────────┘                           ▼
     └─────────────────┘              ┌──────────────────────────┐
                                      │  4. COMPARE   (judge)     │  agreement /
                                      └────────────┬─────────────┘  discrepancies
                                                   ▼
                                      ┌──────────────────────────┐
                                      │  5. MONITOR   (judge)     │  validity +
                                      └────────────┬─────────────┘  relevance filter
                                                   ▼
                                      ┌──────────────────────────┐
                                      │  6. CONSOLIDATE (judge)   │  one clean answer
                                      └────────────┬─────────────┘
                                                   ▼
                                      ┌──────────────────────────┐
                                      │  7. PERSIST               │  log + promote
                                      │  validated knowledge      │  to master prompt
                                      └──────────────────────────┘
```

### The council / judge split

Two distinct roles, deliberately separated (configured in `core/clients.py`):

| Role | Model | Responsibility |
|------|-------|----------------|
| **Council member** | Claude Sonnet 5 | Answers the escalated query |
| **Council member** | Gemini 3.5 Flash | Answers the escalated query (concurrently) |
| **Judge** | Claude Opus 4.8 | Refereeing, validation, consolidation |
| **Local** | dolphin-llama3 (Ollama) | Compression, routing, simple answers, dedup |

The judge is a higher tier than the members it referees — the validator should
be at least as capable as those it judges.

---

## Project structure

```
Ensemble-AI/
├── pipeline.py           # Orchestrator — wires the stages together
├── ensemble.Modelfile    # The pipeline's local model (ollama create)
├── core/
│   ├── clients.py        # Model clients, model IDs, sampling temperatures
│   ├── helpers.py        # parse_json, estimate_tokens
│   ├── router.py         # Stage 1 — local vs. escalate    (local)
│   ├── compress.py       # Stage 2 — compression, local path only (local)
│   ├── council.py        # Stage 3-4 — fan-out + comparison (members + judge)
│   ├── monitor.py        # Stage 5 — validation / filtering (judge)
│   ├── consolidate.py    # Stage 6 — final answer synthesis (judge)
│   └── knowledge.py      # Stage 7 — persistence + master-prompt promotion
├── memory.py             # Auxiliary: ChromaDB vector store over knowledge/ (RAG)
├── harvester.py          # Auxiliary: arXiv paper harvester → knowledge/
└── requirements.txt
```

`core/` is the pipeline. `memory.py` and `harvester.py` are standalone tools that
feed the knowledge base and are not part of the request/response path — which is
why `memory.py` configures its own embedding model rather than importing
`core/clients.py` and inheriting a dependency on API keys it never uses.

---

## Setup

**Prerequisites**
- Python 3.11+
- [Ollama](https://ollama.com) running locally
- API keys for Anthropic and Google Gemini

**Install**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Build the local model**
```bash
ollama pull dolphin-llama3
ollama create ensemble-local -f ensemble.Modelfile
```

`ensemble-local` is the model the pipeline drives (`LOCAL_MODEL` in
`core/clients.py`). It's a thin, persona-free wrapper over `dolphin-llama3` that
defaults to near-greedy sampling, because routing and deduplication are
evaluation tasks that have to return the same verdict twice.

**Configure** — create a `.env` file (never commit it):
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

**Run**
```bash
python pipeline.py
```

By default this runs the built-in test prompt. Edit the `__main__` block in
`pipeline.py`, or import and call `run_pipeline(your_prompt)` from your own code:

```python
import asyncio
from pipeline import run_pipeline

asyncio.run(run_pipeline("What were the main causes of the French Revolution?"))
```

---

## Design decisions

See [DECISIONS.md](DECISIONS.md) for the key architectural and build choices and
the reasoning behind each.
