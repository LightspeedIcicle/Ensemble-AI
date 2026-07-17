"""Shared test setup.

Tests are tiered by what they need, so `pytest` on a fresh clone with no API keys
and no services still runs the majority of them:

    (default)     pure logic + mocked SDKs. No network, no keys, no Ollama.
    -m ollama     needs Ollama serving `ensemble-local`. Free, ~1 min.
    -m paid       spends real API money. Never runs unless asked for explicitly.

Run:
    pytest                      # everything free and offline
    pytest -m ollama            # the routing suite
    pytest -m "not ollama"      # explicitly offline
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: needs Ollama serving ensemble-local")
    config.addinivalue_line("markers", "paid: spends real API money — opt in explicitly")


def pytest_collection_modifyitems(config, items):
    """Keep the default run fast and offline; make spending money deliberate.

    Markers alone do not skip anything — they only enable `-m` filtering. Without
    this hook a bare `pytest` runs everything, which meant a 53-second default
    that quietly required Ollama, contradicting the promise at the top of this
    file. A test suite whose docstring lies about what it needs is worse than no
    docstring.
    """
    selected = config.getoption("-m") or ""
    for marker, reason in (
        ("ollama", "needs Ollama; run with -m ollama"),
        ("paid", "spends real money; run with -m paid"),
    ):
        if marker in selected:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def stub_sdks(monkeypatch):
    """Import core.* without the paid SDKs or an API key.

    core/clients.py constructs Anthropic and Gemini clients at import time, so any
    test touching core.router / core.council / pipeline needs these in place first.
    """
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = lambda **kw: types.SimpleNamespace(messages=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = lambda **kw: types.SimpleNamespace(models=types.SimpleNamespace())
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    ollama = types.ModuleType("ollama")
    ollama.chat = lambda **kw: {"message": {"content": ""}}
    monkeypatch.setitem(sys.modules, "ollama", ollama)

    for name in [m for m in sys.modules if m.startswith("core") or m == "pipeline"]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def block(*_a, **_kw):
    """Fail loudly if a test reaches the network."""
    raise AssertionError("this test tried to make a network call")


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Offline by default. Tests marked `ollama` or `paid` opt out."""
    if request.node.get_closest_marker("ollama") or request.node.get_closest_marker("paid"):
        return
    import socket
    monkeypatch.setattr(socket.socket, "connect", block)
