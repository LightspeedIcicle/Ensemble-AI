"""The chunker. Pure logic — no vector database required.

The constraint these tests exist to protect: all-MiniLM-L6-v2 silently truncates
at ~256 tokens (~1000 chars). A chunk over that limit is not an error — it is a
chunk whose tail is dropped before embedding, with retrieval quietly degrading.
Nothing at runtime will tell you. These tests are the only thing that will.
"""
import pytest

from core.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_text

PROSE = (
    "# Heading\n\nThe Casimir effect arises from vacuum fluctuations. It was "
    "predicted in 1948. The force scales as the inverse fourth power of separation.\n\n"
    + "Second paragraph with a formula E = hbar*c*pi^2/(240*d^4) that must not be "
      "cut mid-expression. " * 12
    + "\n\nFinal paragraph. " + "Filler sentence here. " * 40
)


def test_no_chunk_exceeds_the_embedder_ceiling():
    # The whole reason this module has a constant instead of a magic number.
    for chunk in split_text(PROSE):
        assert len(chunk) <= CHUNK_SIZE


def test_chunks_stay_under_the_256_token_truncation_limit():
    # ~4 chars/token. Over ~256 tokens the embedder drops the tail SILENTLY.
    for chunk in split_text(PROSE):
        assert len(chunk) // 4 <= 256


def test_consecutive_chunks_overlap():
    # A fact spanning a boundary must survive whole in at least one chunk.
    chunks = split_text(PROSE)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        tail = a[-CHUNK_OVERLAP:]
        assert any(b.startswith(tail[-k:]) for k in range(len(tail), 20, -1)), \
            "consecutive chunks share no overlap"


def test_chunks_do_not_end_mid_word():
    # The original hard slice cut sentences and formulas in half. This is the fix.
    for chunk in split_text(PROSE)[:-1]:
        assert not chunk[-1].isalnum(), f"chunk ends mid-word: ...{chunk[-30:]!r}"


def test_no_content_is_lost():
    import re
    joined = re.sub(r"\s+", " ", " ".join(split_text(PROSE)))
    for i in range(0, len(PROSE) - 40, 200):
        window = re.sub(r"\s+", " ", PROSE[i:i + 40]).strip()
        assert window in joined, f"lost from output: {window!r}"


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_input_yields_no_chunks(text):
    assert split_text(text) == []


def test_short_input_is_one_chunk():
    assert split_text("hi") == ["hi"]


def test_text_with_no_separators_still_terminates():
    # A pathological input must not spin forever looking for a boundary.
    chunks = split_text("x" * 3000)
    assert chunks and all(len(c) <= CHUNK_SIZE for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    # Otherwise `start` never advances and the loop hangs.
    with pytest.raises(ValueError):
        split_text(PROSE, chunk_size=100, overlap=100)


def test_chunk_shorter_than_overlap_still_advances():
    # The max(end - overlap, start + 1) guard. Without it: infinite loop.
    text = "a. " + "b" * 50 + ". " + "c" * 2000
    assert split_text(text, chunk_size=60, overlap=55)
