"""The router, against a live Ollama. Free, ~1 minute.

    pytest -m ollama

This is the highest-leverage component in the pipeline: it decides free vs ~$0.15
on every query, and it is the weakest model in the system. It has shipped two
bugs that only a live run could find:

  - The prompt contained two contradictory absolutes ("ALWAYS escalate ...
    technical domains" and "handle locally ONLY if ... definitions"), so a
    definition in a technical field was a coin flip. It flipped at temperature 0.0.
  - Imperatives ("write a function that...") made it COMPLY rather than classify,
    producing unparseable JSON. Fencing the query in <query> tags fixed it.

Neither was findable by reading. Both were obvious the first time it ran.

These assert on the MAJORITY verdict across several runs, because the router is
not deterministic at any temperature — that was measured, and the entry in
DECISIONS.md claiming otherwise was corrected.
"""
import json
import urllib.error
import urllib.request
from collections import Counter

import pytest

pytestmark = pytest.mark.ollama

RUNS = 3   # majority of 3; raise for a more confident signal, at ~2s per run

# (query, expected route, why it belongs to that branch)
CASES = [
    # delegate — running something settles it
    ("Write a function that merges two sorted linked lists.", "delegate", "the answer is code"),
    ("Why does my code throw IndexError: list index out of range?", "delegate", "needs execution"),
    ("Fix this: for i in range(len(x)): print(x[i+1])", "delegate", "needs execution"),
    # escalate — same subject, but no execution settles a judgement
    ("Should I use async or threads for a write-heavy workload?", "escalate", "experts differ"),
    ("Is it better to use inheritance or composition here?", "escalate", "experts differ"),
    # local — same subject again, but settled
    ("What does the yield keyword do in Python?", "local", "settled definition"),
    ("What is a closure in programming?", "local", "settled definition"),
    # non-coding regression set
    ("What is the capital of France?", "local", "settled fact"),
    ("Hey, how's it going?", "local", "conversational"),
    ("What were the main causes of the French Revolution?", "escalate", "sources disagree"),
    ("Is 400mg of ibuprofen safe with alcohol?", "escalate", "stakes + contested"),
]


def _ollama_up():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
            return any(n.startswith("ensemble-local") for n in names)
    except (urllib.error.URLError, OSError, ValueError):
        return False


@pytest.fixture(scope="module", autouse=True)
def require_ollama():
    if not _ollama_up():
        pytest.skip("Ollama not serving ensemble-local — see README (ollama create)")


@pytest.fixture(scope="module")
def route(stub_sdks_module):
    from core.router import local_router
    return lambda q: Counter(local_router(q)["decision"] for _ in range(RUNS))


@pytest.fixture(scope="module")
def stub_sdks_module():
    """Module-scoped SDK stubs — core.clients builds paid clients at import."""
    import sys
    import types
    saved = {k: sys.modules.get(k) for k in
             ("anthropic", "dotenv", "google", "google.genai")}
    a = types.ModuleType("anthropic"); a.Anthropic = lambda **kw: None
    d = types.ModuleType("dotenv"); d.load_dotenv = lambda *x, **k: None
    g = types.ModuleType("google"); gg = types.ModuleType("google.genai")
    gg.Client = lambda **kw: None; g.genai = gg
    sys.modules.update({"anthropic": a, "dotenv": d, "google": g, "google.genai": gg})
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.mark.parametrize("query,expected,why", CASES, ids=[c[1] + ":" + c[0][:28] for c in CASES])
def test_routes_correctly(route, query, expected, why):
    votes = route(query)
    winner = votes.most_common(1)[0][0]
    assert winner == expected, f"{why}: wanted {expected}, got {dict(votes)}"


def test_verdicts_are_always_valid_lowercase(route):
    # The model emits "LOCAL" as readily as "local", and pipeline.py compares
    # against lowercase literals — an uncased verdict silently escalated a free
    # query. local_router normalizes; this is the fence around that.
    from core.router import VALID_DECISIONS
    for query, _, _ in CASES[:4]:
        for verdict in route(query):
            assert verdict in VALID_DECISIONS
            assert verdict == verdict.lower()


def test_the_delegate_rule_does_not_poach_judgement_questions(route):
    # When the delegate branch was added, "should I use async or threads?" started
    # routing to the CLI 4/4 — the rule matched on the word "code" instead of on
    # whether execution settles the question. This is that regression.
    for query, expected, _ in CASES:
        if expected == "delegate":
            continue
        assert route(query)["delegate"] == 0, f"delegate poached: {query!r}"
