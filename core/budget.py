# core/budget.py
# The token budget dial — how much the user is willing to spend on one query.
#
# WHAT THE DIAL IS NOT: max_tokens.
#
# Turning max_tokens down does not produce a shorter answer. It produces a
# truncated one — the model generates until it hits the ceiling and then stops
# mid-sentence, and you are billed for every token it generated on the way. You
# pay full price for a broken answer. That is the worst outcome available, and it
# is what "just lower the limit" actually buys.
#
# WHAT THE DIAL IS: effort, and whether the model thinks.
#
# `effort` asks the model to calibrate its own depth — fewer tool calls, less
# exploration, a more direct route to the answer. The model decides what to leave
# out, which is the difference between a shorter answer and an amputated one. It
# is the API's actual mechanism for this tradeoff; max_tokens is a safety rail
# the model cannot see.
#
# WHY max_tokens STILL MOVES: headroom, not economy.
#
# max_tokens caps thinking + response together. Enabling thinking against a
# ceiling tuned for a non-thinking call means the reasoning eats the budget and
# the answer truncates — the exact failure above, introduced by trying to improve
# quality. So the ceiling scales UP with thinking. It is a floor being raised out
# of the way, never a lever being pulled down.

# ── Levels ────────────────────────────────────────────────────────────────────
#
# `thinking` is set EXPLICITLY at every level, never left to the default, because
# the defaults disagree: omitting the parameter runs adaptive thinking on
# claude-sonnet-5 and no thinking at all on claude-opus-4-8. Relying on them gave
# this pipeline a thinking council member refereed by an unthinking judge. A dial
# built on defaults would inherit that silently.
#
# `think_headroom` is ADDED to a stage's base ceiling, never multiplied by it.
#
# The first version of this multiplied (1x/1x/3x/6x), which was wrong twice over.
# It assumed the base ceilings were correct — they were not; measured output showed
# compare needed ~1300 against a base of 1000 and monitor ~2600 against 2000, so at
# 1x the `minimal` level TRUNCATED compare, produced unparseable JSON, and aborted
# the run after billing $0.05 for no answer. That is precisely the failure this
# module's header warns about, shipped by the module itself. And multiplying a
# corrected base breaks the other end: monitor at 3500x6 is 21K, past the point
# where non-streaming requests risk SDK timeouts.
#
# Additive headroom is right because thinking costs roughly a fixed budget of
# reasoning, not a proportion of the answer. A stage's base is what it needs to
# answer; the headroom is room to think first.

LEVELS = {
    # name        effort    thinking  think_headroom
    "minimal":  ("low",     False,    0),
    "low":      ("medium",  False,    0),
    "balanced": ("high",    True,     4000),
    "full":     ("max",     True,     8000),
}

DEFAULT_LEVEL = "balanced"

# What each level costs you, in the only terms that matter here. Printed by the
# pipeline so the choice is visible rather than buried in a config file.
DESCRIPTIONS = {
    "minimal":  "cheapest. No thinking, low effort. Use for simple factual queries.",
    "low":      "no thinking, medium effort. Reasonable for well-specified questions.",
    "balanced": "adaptive thinking, high effort. The default — what the judge needs.",
    "full":     "adaptive thinking, max effort. For questions where being wrong is expensive.",
}


def resolve(level):
    """Validate a budget level and return its (effort, thinking, max_mult) tuple."""
    if level not in LEVELS:
        raise ValueError(
            f"unknown budget level {level!r} — choose one of: {', '.join(LEVELS)}"
        )
    return LEVELS[level]


def sampling(level):
    """Request kwargs for an Anthropic call at this budget level.

    Splat into messages.create(). Applies to the Anthropic calls only — the Gemini
    council member has its own API and its own parameters, so this dial does not
    reach it. That asymmetry is deliberate and documented in DECISIONS.md.
    """
    effort, think, _ = resolve(level)
    return {
        "thinking": {"type": "adaptive"} if think else {"type": "disabled"},
        "output_config": {"effort": effort},
    }


# Measured ceilings. Each is what the stage actually produced on a real escalated
# query, plus margin — NOT a guess. The originals (1000/1000/2000/1500) were
# inherited guesses, and two of the four were below what the stage needed:
#
#   stage         old base   measured output   now
#   member          1000          665–704      1500
#   compare         1000          1298         2000   <-- truncated at the old base
#   monitor         2000          2588         3500   <-- truncated at the old base
#   consolidate     1500          1305         2000
#
# Re-measure and revise these if the prompts change. A base below what a stage
# produces does not shorten its answer — it cuts the answer off and bills you for
# the whole thing.
BASE_MAX_TOKENS = {
    "member": 1500,
    "compare": 2000,
    "monitor": 3500,
    "consolidate": 2000,
}


def max_tokens(level, stage):
    """The max_tokens ceiling for a stage at this budget level.

    base (what the stage needs to answer) + headroom (room to think first).
    Additive, never multiplicative — see the note on LEVELS.
    """
    if stage not in BASE_MAX_TOKENS:
        raise ValueError(f"unknown stage {stage!r} — one of: {', '.join(BASE_MAX_TOKENS)}")
    _, _, headroom = resolve(level)
    return BASE_MAX_TOKENS[stage] + headroom
