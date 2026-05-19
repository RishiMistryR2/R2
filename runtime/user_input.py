"""
runtime/user_input.py — terminal command handling for R2 live operation.

handle_command(raw_cmd, state) → str
    Process a user command string and return the output.
    Never raises — all exceptions return an error string.

read_command_nonblocking() → Optional[str]
    Non-blocking stdin check (uses select on Unix).

read_command_blocking(prompt) → str
    Blocking readline with prompt for interactive mode.

build_command_context(state) → dict
    Build the context dict consumed by cli/commands.dispatch().
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from runtime.runtime_state import RuntimeState

logger = logging.getLogger("runtime.input")


# ──────────────────────────────────────────────
# Main command handler
# ──────────────────────────────────────────────

def handle_command(raw_cmd: str, state: RuntimeState) -> str:
    """
    Process a raw command string and return the output string.

    Routes:
      "quit" / "exit" / "q"   → signal shutdown, return farewell
      "ask <question>"         → commentary_cycle.answer_question()
      everything else          → cli/commands.dispatch()
    """
    cmd = raw_cmd.strip().lower()
    if not cmd:
        return ""

    logger.info(f"[input] command: {cmd!r}")

    # Shutdown
    if cmd in ("quit", "exit", "q"):
        state.request_shutdown()
        return "\n  R2 shutting down. Trade the process.\n"

    # Free-form AI question
    if cmd.startswith("ask ") or cmd.startswith("ask\t"):
        question = raw_cmd.strip()[4:].strip()
        if not question:
            return "\n  Usage: ask <your question>\n"
        return _handle_ai_question(question, state)

    # Standard CLI commands
    return _dispatch_cli_command(cmd, state)


def build_command_context(state: RuntimeState) -> dict:
    """Build the context dict consumed by cli/commands.dispatch()."""
    return {
        "state":           state.market_state,
        "score":           state.setup_score,
        "context_state":   state.context_state,
        "session_memory":  state.session_memory,
        "pattern_memory":  state.pattern_memory,
        "reflection":      None,
        "trades":          None,
        "report":          None,
        "active_patterns": (
            state.startup_briefing.active_market_patterns
            if state.startup_briefing else []
        ),
        "active_trade":    state.active_trade,
    }


# ──────────────────────────────────────────────
# Non-blocking / blocking stdin
# ──────────────────────────────────────────────

def read_command_nonblocking() -> Optional[str]:
    """
    Return a line from stdin if one is ready, else None.
    Uses select() on Unix.  Windows falls back gracefully.
    """
    try:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            return line.rstrip("\n") if line else None
        return None
    except Exception:
        return None


def read_command_blocking(prompt: str = "r2> ") -> str:
    """
    Blocking readline with prompt.
    Used in interactive mode.  Returns "quit" on EOF/interrupt.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    except Exception as e:
        logger.warning(f"[input] readline error: {e}")
        return ""


# ──────────────────────────────────────────────
# Routing helpers
# ──────────────────────────────────────────────

def _handle_ai_question(question: str, state: RuntimeState) -> str:
    """Route to commentary_cycle.answer_question() safely."""
    try:
        from runtime.commentary_cycle import answer_question
        text = answer_question(state, question)
        return f"\n{text}\n"
    except Exception as e:
        logger.error(f"[input] AI question error: {e}")
        return f"\n  [Error processing question: {e}]\n"


def _dispatch_cli_command(cmd: str, state: RuntimeState) -> str:
    """Route to cli/commands.dispatch() safely."""
    try:
        from cli.commands import dispatch
        ctx    = build_command_context(state)
        result = dispatch(cmd, ctx)
        return result.output if result.success else f"\n  {result.error}\n"
    except Exception as e:
        logger.error(f"[input] dispatch error for {cmd!r}: {e}")
        return f"\n  [Error: {e}]\n"