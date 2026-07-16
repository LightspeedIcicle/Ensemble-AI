# Ensemble AI — Design & Build Decisions

A record of the key architectural and engineering choices, and why each was made.

---

## Pipeline architecture

**Local-first routing over always-escalating**
Every query first hits a free local model (Ollama / dolphin-llama3) that decides
whether the query can be answered locally or must be escalated to paid frontier
models. Simple factual queries never incur API cost. The routing prompt is
deliberately biased toward escalation — the cost of under-escalating (a wrong
answer to a hard question) is worse than the cost of over-escalating (a few extra
API tokens). Any parse failure in routing defaults to escalate, so the system
fails safe.

**Prompt compression before escalation**
A verbose prompt is rewritten into token-efficient shorthand by the local model
*before* it reaches the paid APIs. Cheap local tokens buy expensive API tokens.
The original prompt is retained and used for the final consolidation step so the
answer still reads naturally — compression is an internal optimization, not
something the end user sees.

**A council of two models, not one**
Escalated queries are answered by Claude *and* Gemini concurrently
(`asyncio.gather`). Two independent frontier models give a basis for
cross-validation: where they agree, confidence is high; where they disagree or
one raises a unique point, that becomes something to verify rather than something
to blindly trust.

**Separate judge from council members**
The comparison, validation, and consolidation steps use a distinct, higher-tier
model (Opus 4.8) rather than reusing one of the answering models. The referee
should be at least as capable as those it judges. Originally all three "judge"
roles reused the answering Claude model; separating them was a deliberate
upgrade. All model IDs live in `core/clients.py` so the roles can be re-tuned in
one place.

**Human-in-the-loop knowledge promotion**
Validated facts are always logged, but promotion into the persistent master
prompt (which primes future local answers) requires explicit user confirmation.
This keeps the accumulated long-term memory curated rather than letting the
system silently rewrite its own priming context. A local dedup check prevents the
same fact from being stored twice.

---

## Model choices

| Role | Model | Why |
|------|-------|-----|
| Local | dolphin-llama3 | Runs free via Ollama; good enough for compression, routing, dedup, and simple answers |
| Council member | Claude Sonnet 5 | Fast, capable frontier answerer |
| Council member | Gemini 3.5 Flash | Independent second opinion from a different lab |
| Judge | Claude Opus 4.8 | Highest-tier reasoning for refereeing and validation |

Model version strings are centralized in `core/clients.py` — never hardcoded in
stage modules — so swapping models is a one-line change.

---

## Code structure

**Refactored from monolithic phase files into `core/` modules**
The project was built iteratively as `phase1_routing.py` … `phase6_knowledge.py`,
where each phase re-implemented the entire growing pipeline (so `query_claude`,
`parse_json`, `compare_responses`, etc. were duplicated across six files). These
were migrated into one module per stage under `core/`, with a thin `pipeline.py`
orchestrator on top. Benefits: no duplicated logic, each stage independently
testable, and the model/client configuration centralized. The phase files were
committed once to preserve history, then removed.

**Thin orchestrator, fat stages**
`pipeline.py` only sequences stages and prints progress. All real logic — prompts,
model calls, parsing — lives in the `core/` stage modules. Reading `pipeline.py`
gives the whole flow at a glance; reading a `core/` module gives one stage in
depth.

**Fail-safe JSON parsing**
LLMs frequently wrap JSON in markdown fences or surrounding prose. `parse_json`
(in `core/helpers.py`) strips fences and falls back to extracting the outermost
`{ … }` object, returning `None` only when nothing is parseable. Every stage that
expects structured output routes through it.

---

## Behavior notes

- In the original `phase6`, the local model saw the accumulated master prompt
  during *routing* as well as answering. In the refactor, the master prompt is
  injected only on the local-*answer* path; routing stays uncontaminated by
  accumulated knowledge. This was an intentional correction during the refactor.

---

## Security

**pydantic-settings bumped to 2.14.2 (GHSA-4xgf-cpjx-pc3j)**
Dependabot flagged a moderate vulnerability in the pinned `pydantic-settings==2.14.1`:
`NestedSecretsSettingsSource` followed symlinks outside `secrets_dir`, enabling
local file read. Fixed by pinning the patched `2.14.2`.

**chromadb critical (GHSA-f4j7-r4q5-qw2c / CVE-2026-45829) — dismissed as tolerable risk**
A pre-authentication code injection ("ChromaToast") affects chromadb `<= 1.5.9`.
No patched release exists yet — the latest on PyPI (`1.5.9`) is the top of the
vulnerable range. It is not exploitable in this project: chromadb is used only via
an embedded `PersistentClient` in `memory.py` (a standalone RAG tool, not imported
by the core pipeline), while the exploit requires ChromaDB's HTTP server mode
(`chroma run` / `HttpClient`), which this project never runs. The alert was
dismissed as tolerable risk. **If server mode is ever added, this no longer holds —
upgrade chromadb immediately.** When a fixed version (`> 1.5.9`) ships, bump it.

## Going forward

- Keep model IDs centralized in `core/clients.py`
- Include a brief **why** in commit messages, not just **what**
- Update this file when a non-obvious decision is made
