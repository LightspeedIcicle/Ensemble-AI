# Ensemble AI

A multi-agent LLM pipeline that tells you **which parts of an answer to trust**.

A single LLM call is a black box. You get one answer, fluent and confident, with
no way to tell the solid parts from the invented ones. Ensemble AI answers hard
questions with *two* independent frontier models, then has a higher-tier **judge**
cross-examine them — so disagreements and hallucinations surface instead of
shipping silently. Easy questions never reach that machinery: a free local model
triages every query first, and code goes to a tool that can actually run it.

## It is not cheaper. That is the point.

| | cost |
|---|---|
| One escalated query through Ensemble AI | **$0.1653** |
| Just asking `claude-opus-4-8` the same question | **$0.0328** |

**5× the price of the thing it improves on** — measured, not estimated ([the full
breakdown](DECISIONS.md#cost)). Two council members plus three judge calls will
always cost multiples of one call. No dial changes that.

Here is what the extra $0.13 bought on that run. The council split on a claim; the
judge ruled against the one that was overstated:

> **Royal extravagance was a direct cause of near-bankruptcy** *(Response A)* —
> **Overstated.** The dominant historical view is that war debt and debt-servicing
> costs drove the fiscal crisis, not court spending, which was proportionally
> small. Response B's framing is closer to the consensus.

It also caught two claims that were perfectly true and *weren't answers to the
question* — the Tennis Court Oath and the storming of the Bastille are events *of*
the Revolution, not causes of it. A single model gives you none of that. It gives
you an answer, and no way to know which sentences to check.

Buy this when being confidently wrong is expensive. Don't buy it to save money.

---

## Where the money doesn't go

Cost control here is about **not making the call**, not about shaving tokens off
calls that happen:

- **Free local path.** A local model (via Ollama) answers settled questions for
  $0.00 and never touches a paid API.
- **Delegation.** Code that running would settle goes to the Claude CLI, which can
  execute it. Consensus about code is worth less than one execution of it — and it
  bills your Claude subscription, not the API.
- **Memory.** Facts that survive the judge are logged and can be promoted into a
  persistent master prompt, so the free path answers more over time.

What *doesn't* save money, all measured rather than assumed: compressing prompts
(the user's prompt is 0.05% of a query), shorthand (`"Hllo hw r u?"` costs **76%
more** tokens than plain English), and prompt caching (the preambles are 48 tokens
against a 4096-token minimum). See [DECISIONS.md](DECISIONS.md#cost) — the
intuitive optimizations are all wrong here, and it's worth knowing why.

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
                                      ┌──────────────────────────┐
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

Five roles, each independently assignable in `core/clients.py`:

| Role | Constant | Default | Responsibility |
|------|----------|---------|----------------|
| **Council member** | `CLAUDE_MODEL` | Claude Sonnet 5 | Answers the escalated query |
| **Council member** | `GEMINI_MODEL` | Gemini 3.5 Flash | Answers it concurrently, independently |
| **Judge — compare** | `COMPARE_MODEL` | Claude Opus 4.8 | Structural diff of the two answers |
| **Judge — monitor** | `MONITOR_MODEL` | Claude Opus 4.8 | Validity + relevance fact-checking |
| **Judge — consolidate** | `CONSOLIDATE_MODEL` | Claude Opus 4.8 | Writes the final answer |
| **Local** | `LOCAL_MODEL` | ensemble-local | Routing, compression, simple answers, dedup |
| **Delegate** | — | `claude` CLI | Code that running settles — bills your Claude subscription, not the API |

The judge is a higher tier than the members it referees — the validator should be
at least as capable as those it judges. It is three constants rather than one
because refereeing, fact-checking, and synthesis are different jobs with different
capability requirements: `compare` reads carefully but adjudicates nothing, while
`monitor` needs real world knowledge to rule on truth. Splitting them means a role
can be re-tiered in one line without touching a stage module — see
[DECISIONS.md](DECISIONS.md) for why they weren't merged instead.

---

## Project structure

```
Ensemble-AI/
├── pipeline.py           # Orchestrator — wires the stages together
├── ensemble.Modelfile    # The pipeline's local model (ollama create)
├── core/
│   ├── clients.py        # Model clients, model IDs, sampling temperatures
│   ├── helpers.py        # parse_json, estimate_tokens
│   ├── router.py         # Stage 1 — local / delegate / escalate (local)
│   ├── delegate.py       # Stage 1b — hand coding to the Claude CLI
│   ├── compress.py       # Stage 2 — compression, local path only (local)
│   ├── retrieval.py      # Stage 2b — RAG lookup, local path only (local)
│   ├── council.py        # Stage 3-4 — fan-out + comparison (members + judge)
│   ├── monitor.py        # Stage 5 — validation / filtering (judge)
│   ├── consolidate.py    # Stage 6 — final answer synthesis (judge)
│   └── knowledge.py      # Stage 7 — persistence + master-prompt promotion
├── memory.py             # ChromaDB vector store over knowledge/ — indexer + recall
├── harvester.py          # Auxiliary: arXiv paper harvester → knowledge/
└── requirements.txt
```

`core/` is the pipeline. `harvester.py` is a standalone tool run by hand.
`memory.py` is both: `ingest_knowledge()` is run manually to index `knowledge/`,
while `recall()` is read by `core/retrieval.py` on the local branch. It still
configures its own embedding model rather than importing `core/clients.py`, so
indexing a folder doesn't require the paid APIs' keys — and `core/retrieval.py`
imports it lazily, so the pipeline runs whether or not a store has been built.

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

**Test**
```bash
pip install -r requirements-dev.txt
pytest                  # 84 tests, fully offline, ~0.1s — no keys, no services
pytest -m ollama        # the routing suite against a live local model (free, ~50s)
```

`pytest` on a fresh clone runs everything that doesn't need a network. Tests that
need Ollama are skipped unless asked for, and tests that spend money never run
unless you pass `-m paid`. An unmarked test that reaches the network fails loudly
rather than quietly depending on a service.

The suite is mostly a fence around bugs that actually shipped — a `max_tokens`
ceiling below what a stage produces, a `content[0].text` that breaks the moment
thinking is on, a router rule that poached judgement questions, a prompt reaching
a shell. Each one is a test now.

**Configure** — create a `.env` file (never commit it):
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

**Run**
```bash
python pipeline.py                                    # demo prompt, balanced budget
python pipeline.py "What caused the French Revolution?"
python pipeline.py "..." --budget full                # when being wrong is expensive
python pipeline.py --help                             # list the budget levels
```

### Budget

How much to spend on one query. The dial is **`effort` and thinking**, not
`max_tokens` — turning the ceiling down doesn't shorten an answer, it truncates
one, and you're billed for every token generated before the cut.

| Level | Effort | Thinking | Use for |
|-------|--------|----------|---------|
| `minimal` | low | off | Simple factual queries |
| `low` | medium | off | Well-specified questions |
| `balanced` *(default)* | high | adaptive | Most work — what the judge needs |
| `full` | max | adaptive | Questions where being wrong is expensive |

`max_tokens` gets headroom added (never subtracted) so reasoning doesn't crowd out
the answer. The dial applies to the Anthropic calls; the Gemini council member has
its own parameter surface — see [DECISIONS.md](DECISIONS.md).

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
