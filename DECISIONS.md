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

**Compression runs local-to-local only — never in front of the council** *(revised)*
The original design compressed every prompt *before* routing, on the theory that
cheap local tokens buy expensive API tokens. A review of the pipeline surfaced the
flaw and the design was inverted. Two reasons, and the second is the one that
settles it:

- *It inverted the quality gradient.* Compression put the weakest model in the
  system in charge of deciding what the strongest models were allowed to see. A
  nuanced question, aggressively rewritten by an 8B local model, reaches the
  council already damaged — and no amount of frontier reasoning recovers detail
  that was deleted before it was sent.
- *The economics never supported it.* Input tokens are the cheap direction.
  Trading answer quality for a fractional saving on the cheap side of the ledger
  is a bad trade at any volume. The real cost saving here comes from the router
  declining to make the call at all — not from shaving tokens off calls we make.

The intermediate fix on offer was "bypass compression when the router escalates."
That was rejected as a patch: it still leaves the *router* reading a lossy prompt
while judging how hard the question is, which is the one input that decision
depends on. Compression now sits behind the router, on the local branch only,
where both ends of the handoff are the same model — and is skipped entirely below
`COMPRESS_MIN_TOKENS`, under which there is no context pressure to relieve and
compression is pure downside.

The stage kept its name and lost its original justification. That is the honest
outcome: a feature can survive a review and still turn out to have been built for
a reason that doesn't hold.

**A council of two models, not one**
Escalated queries are answered by Claude *and* Gemini concurrently
(`asyncio.gather`). Two independent frontier models give a basis for
cross-validation: where they agree, confidence is high; where they disagree or
one raises a unique point, that becomes something to verify rather than something
to blindly trust.

**The judge is three roles with three constants, not one**
`compare`, `monitor`, and `consolidate` each read their own model constant. They
previously all read a single `JUDGE_MODEL`, which meant the "roles can be re-tuned
in one place" claim below was aspirational: re-tiering one role required editing a
stage module, exactly what centralizing the IDs was supposed to prevent.

They are split because they are different jobs with different requirements.
`compare` is a structural diff — careful reading, no adjudication. `monitor` is
fact-checking, which needs real world knowledge and is the most
capability-sensitive of the three. `consolidate` writes the answer a human reads.
All three default to the top tier; the split exists so that can be tested rather
than assumed.

The obvious cost optimization here is to merge `compare` and `monitor` into one
judge call, which removes an intermediate JSON round-trip the pipeline generates
at output-token prices only to read back at input-token prices. It was measured
and **rejected**: merging saves 21% of an escalated query, while cutting the dead
fields and re-tiering `compare` saves 20% — a difference of $0.0012 per query.
Collapsing two independently-tunable roles permanently, to save eight hundredths
of a cent, is a bad trade. The cheaper option is only available *because* the roles
are separate stages.

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

## Cost

**Where the money actually is: judge output tokens**
An escalated query costs roughly $0.145 at list prices, and ~82% of that is Opus
generating output at $25/M across three judge calls. The user's own prompt is
**0.08%** of the query. This is worth stating plainly because it kills a whole
family of intuitive optimizations:

- *Compressing the prompt* targets 0.4% of the query. See the compression entry
  above — it was removed from the escalation path for quality reasons, but even if
  it had been free, it was aimed at a rounding error.
- *Shorthand — dropping vowels* (`"Hello how are you?"` → `"Hllo hw r u?"`) makes
  it **worse**. Billing is per token, not per character, and tokenizers are
  compression tables built from real text: `"Hello"` is one token, `"Hllo"` is
  three (`H` + `l` + `lo`). Measured across sample sentences, vowel-dropped
  shorthand cost **+76% tokens** for 30% fewer characters. Plain English is
  already the compressed form.
- *Prompt caching* silently does nothing here. `claude-opus-4-8` has a
  **4096-token minimum cacheable prefix**; the fixed instruction preambles on the
  judge calls are 48 and 77 tokens. Adding `cache_control` would not error — it
  would just never cache, and `cache_read_input_tokens` would sit at zero forever.

The levers that do work, in order: route more queries local (a local answer costs
$0.00, and the router's escalation bias is the single largest cost decision in the
system); tighten the `max_tokens` ceilings; stop generating fields nothing reads;
re-tier `compare`.

**Don't request output nothing reads**
`monitor` asked the judge for `validated[].source` and `validated[].verdict`.
Both `consolidate` and `knowledge.persist` do `[v["item"] for v in validated]` —
only `item` was ever read, so two fields per validated claim were generated at
$25/M, parsed, and dropped. They are no longer requested. If a field isn't read,
it isn't in the schema.

`removed` was the same shape of waste with the opposite fix. The judge explains
why it rejected each claim, and the pipeline called `len()` on the list and threw
the contents away. But a claim one council member asserted and the judge ruled
false is the most interesting artifact this whole pipeline produces — it is
cross-validation catching a hallucination in the act, which is the project's entire
premise. That output was already paid for and simply wasn't shown. It is now
printed. Not a saving; a waste converted into the headline feature.

---

## Sampling

**Local temperature is set per call, not per model**
The local model does two jobs with opposite requirements. Routing, dedup, and
compression are *evaluation* tasks: the same input must produce the same verdict
on every run, or gate decisions flip between runs and the pipeline stops being
reproducible. Answering is a *generation* task, where some variety is fine and
even desirable. Ollama accepts sampling options per request, so one model serves
both roles — `TEMP_DETERMINISTIC` (0.2) for evaluation, `TEMP_GENERATIVE` (0.7)
for answers. Both live in `core/clients.py`, next to the model IDs they belong to.

This corrects a real defect rather than a stylistic one: a router sampling at 0.9
is a router that contradicts itself, sending the same query down different
branches on different runs, and that is invisible until you try to reproduce a
result.

**Two Modelfiles, not one**
`ensemble.Modelfile` defines the pipeline's local model: persona-free,
near-greedy, public, built with `ollama create ensemble-local`. Conversational or
personal models live in their own Modelfile, which `.gitignore` excludes by
default. These are different artifacts with different requirements — a model tuned
for warmth and character makes an unreliable judge — and only one of them belongs
in a repository about a pipeline.

---

## Model choices

| Role | Model | Why |
|------|-------|-----|
| Local | ensemble-local (dolphin-llama3) | Runs free via Ollama; good enough for compression, routing, dedup, and simple answers |
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

## Retrieval and ingestion

**Chunking respects boundaries — and chunks may not grow**
`memory.py` originally sliced text every 1000 characters with no regard for
content, which cuts sentences, formulas, and code blocks in half; retrieval then
returns fragments that begin mid-thought. Chunks are now split at the nearest
natural boundary (paragraph → line → sentence → clause → word) with a
200-character overlap, so a fact spanning a boundary survives intact in at least
one chunk.

The non-obvious part is the ceiling. `all-MiniLM-L6-v2` silently truncates its
input at ~256 tokens. At roughly four characters per token, the existing
1000-character chunk size already sits at ~250 — coincidentally right at the
limit. So the intuitive follow-up ("we're adding overlap, so let's use bigger
chunks") would make retrieval *worse*, and worse invisibly: the embedder drops
each chunk's tail with no error, and the resulting vector describes only the
chunk's opening. `CHUNK_SIZE` cannot be raised without changing the embedder.
The constant carries this warning in the code, because the failure mode leaves no
trace at runtime.

Re-chunking also changes the fragment count per file, so ingestion deletes a
file's existing fragments before re-adding them. `upsert` alone would strand
orphans from the previous scheme in the collection.

**Retrieval is wired to the local branch only**
`memory.py` built a vector store that nothing queried — `recall()` was defined and
never called, so the harvester and the embedding index fed a knowledge base that
never reached the request path. `core/retrieval.py` closes that edge, on the local
branch only.

Not on the escalation path, deliberately. The council members are frontier models
with their own knowledge; handing them passages retrieved by a 250-token-chunk
MiniLM index is more likely to narrow their answer than improve it. The local
model is the one that benefits from the crutch.

Retrieval degrades to nothing rather than failing. `memory.py` is imported lazily
and every failure is swallowed: it pulls in chromadb and sentence-transformers and
constructs a `PersistentClient` at import time, and a missing store is a normal
state (fresh clone, no ingest run), not an error. The pipeline answers unprimed
instead of refusing to start.

**Retrieved passages are framed as data, never as instructions**
Wiring retrieval in means text this project did not write — arXiv PDFs the
harvester downloaded — now reaches the local model's system prompt. That is a
prompt-injection surface, and it is worth being explicit about what makes it
tolerable rather than discovering it later:

- Passages are fenced in `<retrieved>` tags and explicitly labelled as reference
  data the model should consult and not obey.
- The master prompt leads; retrieved text is appended after it, never before.
- Ingestion is manual, so nothing enters the store without a human running the
  harvester.
- The local branch returns before stage 7, so a local answer never reaches
  `knowledge.persist()`. Retrieved text cannot round-trip into the master prompt.

The last two are the load-bearing ones. **If ingestion is ever automated, or the
local branch ever persists its answers, this stops being contained** — untrusted
text would then flow into the store, out through retrieval, and back into the
priming context for every future local answer. Re-evaluate before changing either.

**arXiv ingestion stays sequential and slow — deliberately**
A review recommended converting the harvester to `aiohttp` to "rip down entire
libraries in seconds." This was rejected and the opposite implemented. arXiv's API
terms ask for roughly one request every three seconds on a single connection;
parallelising would get the client rate-limited or blocked, and it abuses a free
service this project depends on. The observation behind the advice — synchronous
is slow — is true and irrelevant: this is a background batch job, and background
batch jobs are allowed to be slow. Delays were raised from 1–2s to 3s. If bulk
volume is ever genuinely needed, the answer is arXiv's real bulk channels (OAI-PMH
for metadata, S3 requester-pays for full text), not hammering the public API.

**Ingestion stays manual until an evaluation gate exists**
Automating the harvester would mean unattended arXiv text flowing into the vector
store the local model retrieves from. Combined with any self-modification
authority, that is a prompt-injection surface feeding the component that approves
its own changes. Manual ingestion *is* the gate until a real one is built. This is
the reason automation is deferred — not inertia.

**Personal keywords are gitignored, not committed**
`harvester.py`'s keyword list is public and contains only terms arXiv actually
indexes. Personal or exploratory search terms belong in `keywords.local.txt`,
which `.gitignore` excludes and the harvester merges at runtime when present. A
public repository is a permanent record; anything that shouldn't live in one
forever shouldn't go into one at all.

---

## Deferred deliberately

**Local Ollama calls stay synchronous**
A review rated the blocking `ollama.chat` calls High severity: they hold the event
loop while Claude and Gemini run under `asyncio.gather`. The diagnosis is right and
the severity is not. Blocking the loop only costs something when other work needs
the loop, and this pipeline is single-user and mostly sequential — local synthesis
runs *after* the gather resolves. That makes it a hygiene issue, not a bottleneck.
It becomes real if local calls ever need to run *while* cloud calls are in flight,
or if streaming is added. The fix (`ollama.AsyncClient`, or `run_in_executor` as
already used for the external APIs) is understood and deferred on purpose rather
than missed. Recorded here so the next reader doesn't re-derive it.

---

## Behavior notes

- In the original `phase6`, the local model saw the accumulated master prompt
  during *routing* as well as answering. In the refactor, the master prompt is
  injected only on the local-*answer* path; routing stays uncontaminated by
  accumulated knowledge. This was an intentional correction during the refactor.

- `harvester.py`'s `KEYWORDS` entries need their trailing commas. Python silently
  concatenates adjacent string literals, so a missing comma doesn't raise — it
  fuses two keywords into one term that matches nothing, and the harvester
  cheerfully reports success while searching for a string that cannot exist. Two
  such fusions were live (19 declared keywords parsed as 13); both are fixed and
  the list now carries a comment so the next edit doesn't reintroduce a third.

- `knowledge.persist()` takes the original prompt rather than a compressed one for
  its topic label. It used to borrow the compressed prompt, which the escalation
  path no longer produces; a truncation of the original serves the same purpose.

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

## The tiebreaker

**When cost and information conflict, information wins.**

This is not a platitude — it is the rule that decided most of the entries above,
and it decided them against the cheaper option every time:

- Compression was removed from the escalation path even though it "saved tokens,"
  because the tokens it saved were a rounding error and the meaning it cost was not.
- `compare` and `monitor` were not merged, because a 0.8% saving is not worth
  permanently collapsing two roles that answer different questions.
- `removed` is printed rather than trimmed to a count, because a hallucination
  caught in the act is the most valuable thing this pipeline produces.

The corollary is what makes the rule usable rather than sentimental: when a saving
costs *no* information, take it without hesitation. `validated[].source` and
`verdict` were deleted the moment it was clear nothing read them. A field nothing
reads carries no information by definition, so the tiebreaker never fires.

The cheap thing and the right thing are usually the same thing. This rule is for
the cases where they aren't.

---

## Going forward

- When cost and information conflict, information wins
- Keep model IDs and sampling temperatures centralized in `core/clients.py`
- Never send a compressed prompt to a frontier model
- Never raise `CHUNK_SIZE` without changing the embedder
- Keep personal data out of the public repo — `keywords.local.txt`, not `KEYWORDS`
- Include a brief **why** in commit messages, not just **what**
- Update this file when a non-obvious decision is made — including the ones where
  the conclusion was "don't do the thing that was recommended"
