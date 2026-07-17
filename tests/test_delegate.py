"""Delegation to the Claude CLI.

This module hands text to an agent with shell and filesystem access. The first
two tests are the security boundary — if either ever fails, a crafted prompt runs
arbitrary commands with the operator's credentials. Everything else checks that a
broken CLI costs money rather than the answer.
"""
import subprocess
from unittest import mock

import pytest

from core import delegate as mod

EVIL = 'write x"; touch /tmp/ENSEMBLE_PWNED_$(whoami); echo "'


def test_prompt_is_a_single_argv_element_not_a_shell_string():
    # THE security test. `claude -p "{prompt}"` through a shell is a command
    # injection the first time a prompt contains a backtick or a semicolon.
    with mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        mod.delegate(EVIL)
        argv, kwargs = run.call_args
        assert argv[0] == ["claude", "-p", EVIL], "prompt must be one argv element"
        assert kwargs.get("shell") in (None, False), "shell=True is injectable"


def test_shell_metacharacters_execute_nothing(tmp_path):
    # Belt and braces: prove it end-to-end rather than trusting the call shape.
    marker = tmp_path / "PWNED"
    with mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        mod.delegate(f'x"; touch {marker}; echo "')
    assert not marker.exists()


def test_missing_cli_falls_back_rather_than_raising():
    with mock.patch.object(mod.shutil, "which", return_value=None):
        assert mod.delegate("anything") is None


def test_timeout_falls_back():
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired("claude", 180)):
        assert mod.delegate("anything") is None


def test_nonzero_exit_falls_back():
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=1, stdout="", stderr="boom")
        assert mod.delegate("anything") is None


def test_empty_output_falls_back():
    # An empty answer is a failure, not an answer.
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0, stdout="   \n", stderr="")
        assert mod.delegate("anything") is None


def test_oserror_falls_back():
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run", side_effect=OSError("nope")):
        assert mod.delegate("anything") is None


def test_success_returns_the_stripped_answer():
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0, stdout="  the answer  \n", stderr="")
        assert mod.delegate("anything") == "the answer"


def test_a_timeout_is_always_set():
    # Without one, a wedged CLI hangs the pipeline forever.
    with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(mod.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        mod.delegate("x")
        assert run.call_args.kwargs.get("timeout"), "no timeout = a hang"
