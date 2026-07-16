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
# `max_mult` scales the per-stage max_tokens ceiling. Kept at 6× or below so the
# largest call stays under ~16K output, above which non-streaming requests risk
# SDK HTTP timeouts.

LEVELS = {
    # name        effort    thinking  max_mult
    "minimal":  ("low",     False,    1),
    "low":      ("medium",  False,    1),
    "balanced": ("high",    True,     3),
    "full":     ("max",     True,     6),
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


def max_tokens(level, base):
    """Scale a stage's max_tokens ceiling for this budget level.

    `base` is the stage's own non-thinking ceiling. The multiplier exists to keep
    thinking from crowding out the answer, not to ration output.
    """
    _, _, mult = resolve(level)
    return base * mult
