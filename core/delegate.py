# core/delegate.py
# Stage 1b — Delegation to a tool that can execute.
#
# WHY THIS EXISTS
#
# This whole pipeline is a proxy for truth. No oracle exists for "what caused the
# French Revolution", so it approximates one: two independent models answer, a
# referee cross-examines them, and agreement stands in for correctness.
#
# For code, the oracle exists. You can run it. Two frontier models agreeing that a
# function is correct is worth dramatically less than executing that function once,
# and it costs ~$0.15 to get the weaker answer. So where a real oracle is
# available, the pipeline should use the oracle instead of its proxy.
#
# The Claude CLI has one: it can write a file, run it, read the traceback, and fix
# it. That is a category the council cannot reach at any budget level.
#
# SECURITY — read before changing anything here
#
# This hands text to an agent with shell and filesystem access. Two properties
# make that safe today, and both are easy to remove by accident:
#
#   1. The subprocess is invoked with an ARGUMENT LIST and shell=False. Never
#      build a command string. `claude -p "{prompt}"` through a shell is a command
#      injection the first time a prompt contains a backtick or a semicolon.
#
#   2. The prompt originates from the user, and only the user. Today it comes from
#      argv — the person typing into this pipeline is the same person who could
#      type `claude -p` themselves, so delegation grants no authority they did not
#      already have.
#
# Property 2 is the load-bearing one, and it is a property of the CALLER, not of
# this file. It breaks the moment anything else can reach the router: a web UI, an
# HTTP endpoint, a job queue, or retrieved arXiv text finding a path in. At that
# point this becomes remote code execution with the operator's credentials. If the
# pipeline ever grows a non-interactive entry point, this stage must be gated
# behind an explicit allow-list or removed.

import shutil
import subprocess

# The CLI loads its own context before answering; ~6s is typical for a small
# request, but a debugging task may read files and run things.
TIMEOUT_SECONDS = 180


def is_available():
    """Is the Claude CLI on PATH?"""
    return shutil.which("claude") is not None


def delegate(prompt):
    """Hand `prompt` to the Claude CLI. Returns its answer, or None on failure.

    Returning None rather than raising is deliberate: the caller falls back to the
    council. A missing or broken CLI should cost money, not the answer.
    """
    if not is_available():
        print("[Delegate] claude CLI not on PATH — falling back to the council")
        return None

    try:
        # Argument list, shell=False (the default). See the security note above.
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(f"[Delegate] CLI exceeded {TIMEOUT_SECONDS}s — falling back to the council")
        return None
    except OSError as e:
        print(f"[Delegate] could not run the CLI ({e}) — falling back to the council")
        return None

    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        print(f"[Delegate] CLI exited {result.returncode}: {err[-1] if err else 'no stderr'}")
        return None

    answer = (result.stdout or "").strip()
    if not answer:
        print("[Delegate] CLI returned nothing — falling back to the council")
        return None

    return answer
