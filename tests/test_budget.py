"""The budget dial.

These exist because of a bug that shipped: `minimal` used a 1x multiplier against
base ceilings that were inherited guesses, two of which were below what the stage
actually produces. compare truncated, emitted unparseable JSON, and the run
aborted -- after billing $0.05 for no answer. Every assertion here is a fence
around that failure.
"""
import pytest

from core.budget import (
    BASE_MAX_TOKENS,
    DEFAULT_LEVEL,
    DESCRIPTIONS,
    LEVELS,
    max_tokens,
    resolve,
    sampling,
)

# What each stage ACTUALLY produced on a real escalated query at `balanced`.
# If a ceiling drops below one of these, that stage truncates. Re-measure and
# update these numbers if the prompts change.
MEASURED_OUTPUT = {"member": 704, "compare": 1298, "monitor": 2588, "consolidate": 1305}


@pytest.mark.parametrize("stage,produced", MEASURED_OUTPUT.items())
@pytest.mark.parametrize("level", list(LEVELS))
def test_no_level_truncates_a_stage(level, stage, produced):
    # THE regression test. A ceiling below observed output does not shorten the
    # answer -- it cuts it off and bills you for the whole thing.
    assert max_tokens(level, stage) > produced


@pytest.mark.parametrize("level", list(LEVELS))
@pytest.mark.parametrize("stage", list(BASE_MAX_TOKENS))
def test_ceilings_stay_under_the_streaming_threshold(level, stage):
    # Above ~16K output, non-streaming requests risk SDK HTTP timeouts. The old
    # multiplicative scheme would have put monitor at 21K.
    assert max_tokens(level, stage) <= 16000


@pytest.mark.parametrize("level", list(LEVELS))
def test_headroom_is_added_never_subtracted(level):
    for stage, base in BASE_MAX_TOKENS.items():
        assert max_tokens(level, stage) >= base, "the dial must never ration output"


@pytest.mark.parametrize("level", list(LEVELS))
def test_thinking_is_always_explicit(level):
    # Omitting `thinking` runs adaptive on claude-sonnet-5 and NOTHING on
    # claude-opus-4-8. Relying on the default gave this pipeline a thinking
    # council member refereed by an unthinking judge. Never leave it implicit.
    assert sampling(level)["thinking"]["type"] in ("adaptive", "disabled")


def test_default_level_lets_the_judge_think():
    assert sampling(DEFAULT_LEVEL)["thinking"]["type"] == "adaptive"


@pytest.mark.parametrize("level", list(LEVELS))
def test_thinking_levels_get_real_headroom(level):
    # max_tokens caps thinking AND response together. Thinking against a ceiling
    # tuned for a non-thinking call means reasoning eats the answer.
    _, thinking, headroom = resolve(level)
    assert (headroom > 0) == thinking


def test_effort_is_a_documented_api_value():
    valid = {"low", "medium", "high", "xhigh", "max"}
    for level in LEVELS:
        assert sampling(level)["output_config"]["effort"] in valid


def test_unknown_level_fails_before_any_api_call():
    with pytest.raises(ValueError, match="unknown budget level"):
        resolve("cheap")


def test_unknown_stage_fails_loudly():
    with pytest.raises(ValueError, match="unknown stage"):
        max_tokens(DEFAULT_LEVEL, "consolidat")   # typo, not a silent default


def test_every_level_is_described():
    # The CLI prints these. A level without one is a level nobody can choose.
    assert set(DESCRIPTIONS) == set(LEVELS)


def test_cost_ordering_is_monotonic():
    # minimal <= low <= balanced <= full, or the dial is lying about its direction.
    order = ["minimal", "low", "balanced", "full"]
    totals = [sum(max_tokens(l, s) for s in BASE_MAX_TOKENS) for l in order]
    assert totals == sorted(totals)
