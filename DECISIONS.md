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

**The router's rules are ordered, because two absolutes contradicted each other**
The original routing prompt had two lists. `ALWAYS escalate if the query` included
*"involves science, medicine, law, economics, or technical domains."* `Handle
locally ONLY if the query` included *"has a single, universally agreed factual
answer (capitals, dates, definitions)."*

"What is a closure in programming?" is a **definition** and it is **technical**.
Both lists matched. `ALWAYS` and `ONLY` are both absolute, so the model was asked
to choose between two rules it had been told were non-negotiable — and it chose
differently across runs, at every temperature including 0.0. This sat on the single
highest-leverage decision in the system: on a borderline query the router was a
coin flip between free and ~$0.13.

The rules now apply **in order, first match decides**, with an explicit default
(escalate) at the bottom. That dissolves the contradiction: a definition in a
technical field reaches rule 3 and goes local; a judgement call in the same field
matches rule 1 first and escalates.

The fix needed a second pass, which is worth recording too. Its first version
closed with "the SHAPE of the question decides, not its subject — a technical field
is not by itself a reason to escalate." But rule 2 escalates on *"medicine, law,
finance, safety"* — a subject. That correction contradicted rule 2 three lines
after dissolving the original contradiction, and it measured that way: *"Is
intermittent fasting healthy?"* began going local 1 run in 4. A medical question,
to an 8B model, with no cross-validation — the exact failure the router exists to
prevent, introduced while fixing a cheaper one. Rule 3 is now scoped to *settled*
answers, with "if you find yourself reasoning that something is basically settled,
it is not settled" as the tell.

Measured after: 10 labelled queries × 5 runs at temp 0.2 → **10/10 correct, 0
flips, 0 should-escalate queries going local.** Before: 7/8, with the closure query
split 2/2. Limits worth stating — one model, one session, small n, hand-labelled
expectations, and one parse failure on `"What is 17 times 23?"` (which fails safe
to escalate, so a trivial query occasionally costs $0.13). Enough to show the
contradiction is gone. Not a regression suite.

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

**The budget dial is `effort`, not `max_tokens`**
The obvious way to build "let the user spend less" is to turn `max_tokens` down.
That is not a cheaper answer — it is a **truncated** one. The model generates until
it hits the ceiling and stops mid-sentence, and every token it generated on the
way is billed. You pay full price for a broken answer, which is strictly the worst
cell in the table. `max_tokens` is a safety rail the model cannot see; it has no
idea it is about to be cut off, so it cannot budget around it.

`effort` is the API's actual mechanism for this. It asks the model to calibrate
its own depth — less exploration, a more direct route — and the *model* decides
what to leave out. That is the difference between a shorter answer and an amputated
one, and it is why the dial in `core/budget.py` moves `effort` and thinking rather
than the ceiling.

`max_tokens` does move with the dial, but only ever **upward**, and never as an
economy. It caps thinking and response *together*, so enabling thinking against a
ceiling tuned for a non-thinking call means the reasoning eats the budget and the
answer truncates. The headroom is a floor being raised out of the way.

**The base ceilings were guesses, and two were too low** *(found by measurement)*
The original per-stage ceilings — 1000/1000/2000/1500 — predate this file and were
never checked against output. Measured:

| stage | old base | actual output | now |
|-------|----------|---------------|-----|
| member | 1000 | 665–704 | 1500 |
| compare | 1000 | **1298** | 2000 |
| monitor | 2000 | **2588** | 3500 |
| consolidate | 1500 | 1305 | 2000 |

Two of four were below what the stage produces. At `balanced` a 3× multiplier hid
it. The first `minimal` run (1×) hit the real thing: compare stopped at
`max_tokens`, emitted unparseable JSON, `parse_json` returned `None`, and the
pipeline aborted — **after billing $0.0488 for no answer**. That is verbatim the
failure this section warns about, shipped by the dial that warns about it.

Headroom is now **added** rather than multiplied, and bases come from observed
output. Multiplying was wrong twice: it assumed the bases were right, and it broke
the top end too — monitor at 3500×6 is 21K, past the streaming threshold. Additive
is also the truer model: thinking costs roughly a fixed budget of reasoning, not a
proportion of the answer. Re-measure the bases if the prompts change.

**The judge wasn't thinking. The council member was.**
No call in this pipeline set `thinking`, and the defaults are not symmetric:
omitting the parameter runs **adaptive thinking on `claude-sonnet-5`** and **no
thinking at all on `claude-opus-4-8`**. So the Sonnet council member reasoned
before answering, and the Opus fact-checking gate that referees it did not. The
judge principle — the referee should be at least as capable as those it judges —
was inverted at the reasoning level, silently, by a parameter nobody wrote.

This is why `core/budget.py` sets `thinking` **explicitly at every level**,
including the levels that turn it off. A dial built on defaults inherits the
asymmetry and hides it one layer deeper. The default level, `balanced`, is the
cheapest one that lets the judge think.

Enabling thinking also broke every response parse in the codebase, latently:
`response.content[0].text` was correct only because thinking was off. With adaptive
thinking on, index 0 is a `ThinkingBlock` and `.text` raises. `core/helpers.py`
now has `response_text()`, which selects the text block instead of assuming its
position.

**The dial stops at the API boundary**
`effort` and adaptive thinking are Anthropic parameters. The Gemini council member
has its own surface and does not receive them. The dial could have been made to
*look* like it reaches Gemini by mapping levels onto Google's parameters, but the
mapping would be invented rather than equivalent, and a control that silently means
something different on one of two council members is worse than one that visibly
stops. If Gemini's knobs are wired up later, they should be their own named entry
in `LEVELS`, not a guess folded into this one.

**Where the money actually is: judge output tokens** *(measured)*
One real escalated query — "What were the main causes of the French Revolution?"
at `balanced`, Anthropic calls only (Gemini is billed by Google):

| stage | model | in | out | cost | stop |
|-------|-------|-----|------|------|------|
| member | sonnet-5 | 20 | 665 | $0.0100 | end_turn |
| compare | opus-4-8 | 2772 | 1298 | $0.0463 | end_turn |
| monitor | opus-4-8 | 1265 | 2588 | $0.0710 | end_turn |
| consolidate | opus-4-8 | 1080 | 1305 | $0.0380 | end_turn |
| | | | | **$0.1654** | |

**Opus output alone is 78% of the query.** The user's prompt was **20 tokens** —
0.05%. `minimal` on the same query measured **$0.1196 (−28%)**.

These replace an earlier estimate of $0.1453 / 82%, which was described here as a
worst case priced from the `max_tokens` ceilings. It was not a worst case: the real
query came in **14% above** it. The ratio was close; the absolute was wrong in the
direction claimed impossible. Estimates are labelled as estimates in this file for
that reason.

This kills a whole family of intuitive optimizations:

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

**What low temperature does and does not buy** *(corrected — this entry originally
claimed something that measurement disproved)*

It was first written here that a router sampling at 0.9 "contradicts itself,
sending the same query down different branches on different runs," and that 0.2
fixes it. That was asserted, not tested. When it was finally tested — same query,
12 runs per setting, against `ensemble-local`:

| temp | verdicts on "What is a closure in programming?" | deterministic |
|------|------------------------------------------------|---------------|
| 0.0  | 11 escalate / 1 local | **no** |
| 0.2  | 7 escalate / 5 local  | **no** |
| 0.5  | 8 escalate / 4 local  | **no** |
| 0.9  | 5 escalate / 6 local / 1 parse failure | **no** |

**It flips at temperature 0.0.** Temperature was never the cause. It reduces
variance — 0.0 was steadier than 0.9 — but it cannot make a router reproducible,
and no setting here did.

Low temperature is still worth having, for a smaller and different reason than
claimed: **valid JSON**. Only 0.9 produced output `parse_json` couldn't read, and
`local_router` fails safe to escalate on a parse failure — so hot sampling
silently converts a free routing decision into a paid council call. That is a real
cost, just not the one originally written down.

The actual cause of the flipping was the prompt. See below.

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

## Testing

**Run the change, then write down what it does — in that order**

A process rule, adopted because the opposite order failed three times in one
session:

- This file claimed low temperature made routing reproducible. It flips at 0.0.
  The claim was written, committed, and pushed before anyone ran it.
- `LOCAL_MODEL` was pointed at `ensemble-local` and the `ollama create` step was
  documented in the README. The model was never built. `master` was broken for
  several commits and it surfaced only when someone tried to run the thing.
- The first fix for the router's contradictory rules introduced a new
  contradiction, sending a medical question to an 8B model 1 run in 4. It was
  caught because the fix was measured before it was committed.

Every claim in this file that was measured held up. Both that were asserted were
wrong. That is not carelessness, and noticing harder would not have prevented it:
a plausible explanation for why a change *should* work is exactly as easy to write
when the change does not work. Prose has no failure mode. That is the whole problem
with it as evidence.

The standard:

- **A behavioral change ships with the run that demonstrates it.** Not a
  description of the run. The run.
- **The test must be able to fail.** The first routing test used a query that hit
  four ALWAYS-escalate triggers at once; it passed at every temperature and proved
  nothing. A test that cannot separate the hypotheses is decoration with a
  checkmark on it.
- **State the limits next to the result.** "10/10 across 5 runs, one model, one
  session, hand-labelled" is a finding. "Fixed" is a claim.
- **When a measurement contradicts something already written here, correct the
  entry and mark it corrected.** This log is worth something because it is
  trustworthy, not because it is consistent. An entry that was wrong and says so is
  worth more than one that was quietly rewritten.

The cheap version of this rule: if you cannot run it, say you did not run it.
Everything above is downstream of that.

---

## The tiebreaker

**A correct answer, carrying the context needed to trust it, beats a cheap one.**

Not "more output is better" — that is a different and worse rule. The bar is
*sufficiency*: everything the reader needs to act on the answer and to check it,
and nothing past that. A padded answer misses the bar the same way a truncated one
does; it just misses more expensively. Token cost is a real constraint and worth
respecting. It simply loses this particular argument, every time.

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

These are not close in weight. A council, a referee, and a fact-checking gate are
an expensive way to answer a question, and the only thing all that machinery buys
is a correct answer you can tell is correct. Spending it and then trimming the
result to save tokens is paying for the whole apparatus and throwing away its
output — backwards no matter how the arithmetic comes out.

**The near-miss worth recording.** Deleting the dead fields and shortening the
live ones look like the same move — both are "ask the judge for fewer tokens" —
and they were briefly made together. `monitor` was given a blanket instruction to
keep every string terse "because this output is consumed by the pipeline, not read
as prose." That was false in the same commit that wrote it: `removed[].reason` had
just been made visible, so the judge was being told to compress the hallucination
explanation that had *just* been promoted to the headline feature. They are
opposite moves. Deleting output nobody reads is free; shortening output somebody
reads is a cost disguised as a saving, and it disguises well precisely because the
token arithmetic looks identical. `monitor` now gives per-field guidance instead:
tight for `item` (code reads it), explicitly uncompressed for `reason` (a person
does).

---

## Going forward

- Run the change before writing down what it does; the test must be able to fail
- A correct answer with the context to trust it beats a cheap one — sufficiency,
  not length, is the bar
- Keep model IDs and sampling temperatures centralized in `core/clients.py`
- Never send a compressed prompt to a frontier model
- Never raise `CHUNK_SIZE` without changing the embedder
- Keep personal data out of the public repo — `keywords.local.txt`, not `KEYWORDS`
- Include a brief **why** in commit messages, not just **what**
- Update this file when a non-obvious decision is made — including the ones where
  the conclusion was "don't do the thing that was recommended"
