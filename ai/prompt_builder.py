"""
ai/prompt_builder.py — structured prompt templates for R2 AI interactions.

Pure functions.  No I/O, no state.
All functions return a (system_prompt, user_message) tuple ready for
the Claude API.

Request types:
  startup_briefing   — pre-market orientation
  market_commentary  — live market check
  setup_explanation  — explain a scored setup
  trade_review       — review an active trade
  warning_note       — explain an active warning
  end_of_day         — session reflection
  user_question      — answer a specific question

Design rules:
  - Keep user messages ≤ 1500 tokens (≈6000 chars).
  - Use XML-style tags for structure: <context>, <request>.
  - Inject active risk_flags into every prompt so R2 honours guardrails.
  - Never instruct Claude to bypass its own rules.
  - System prompt comes from ai/r2_persona.py.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

from ai.r2_persona import (
    R2_SYSTEM_PROMPT,
    R2_BRIEF_SYSTEM_PROMPT,
    RISK_FLAGS,
)
from ai.state_packager import StatePackage


# Type alias
PromptPair = Tuple[str, str]   # (system_prompt, user_message)

# Max chars for the context JSON block before truncation
_MAX_CONTEXT_CHARS = 5500


# ──────────────────────────────────────────────
# Public builders
# ──────────────────────────────────────────────

def build_startup_briefing_prompt(package: StatePackage) -> PromptPair:
    """Pre-market orientation prompt."""
    ctx  = _serialise(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Provide a concise pre-market briefing.\n"
        "Cover: current market conditions, regime, any elevated risks for today, "
        "relevant patterns from memory, and what to watch for at the open.\n"
        "Be specific. Be brief. Lead with what matters most.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_market_commentary_prompt(package: StatePackage) -> PromptPair:
    """Live market check — brief, uses brief system prompt to save tokens."""
    ctx  = _serialise(package, keys=("market", "setup", "context", "warnings", "risk_flags"))
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Describe the current market state in 2-4 sentences.\n"
        "Mention regime, narrative, and any material context/risk factors.\n"
        "If a setup is forming, note its quality grade.\n"
        "</request>"
    )
    return R2_BRIEF_SYSTEM_PROMPT, user


def build_setup_explanation_prompt(
    package:   StatePackage,
    direction: Optional[str] = None,
) -> PromptPair:
    """Full setup analysis — score, FOR/AGAINST, levels, risk."""
    ctx      = _serialise(package)
    risk     = _risk_block(package.risk_flags)
    dir_note = f"Direction under consideration: {direction}\n" if direction else ""

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"{dir_note}"
        "<request>\n"
        "Explain this setup as if briefing a disciplined trader.\n"
        "Include: score rationale, key reasons FOR and AGAINST, "
        "stop logic, and R:R assessment.\n"
        "Flag any concerns clearly. Be direct — do not oversell.\n"
        "Do not recommend execution.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_trade_review_prompt(package: StatePackage) -> PromptPair:
    """Active trade review — thesis, stops, monitoring."""
    ctx  = _serialise(package, keys=("market", "trade", "context", "warnings", "risk_flags"))
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Review the active trade.\n"
        "Is the thesis still intact? Are any stops at risk? "
        "What should the trader watch?\n"
        "If stop flags are triggered, say so clearly.\n"
        "Do not recommend execution. Observations only.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_warning_explanation_prompt(
    package:     StatePackage,
    warning_msg: str,
) -> PromptPair:
    """Explain a specific warning in plain language."""
    ctx  = _serialise(package, keys=("market", "session", "warnings", "risk_flags"))
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"<warning>\n{warning_msg}\n</warning>\n"
        "<request>\n"
        "Explain this warning in plain language.\n"
        "Why is it relevant right now? What should the trader be aware of?\n"
        "Be concise. No more than 3 sentences.\n"
        "</request>"
    )
    return R2_BRIEF_SYSTEM_PROMPT, user


def build_end_of_day_prompt(package: StatePackage) -> PromptPair:
    """End-of-session reflection."""
    ctx  = _serialise(package, keys=("session", "patterns", "warnings", "risk_flags"))
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        "<request>\n"
        "Provide a brief end-of-day reflection.\n"
        "Cover: session P&L, execution quality, notable mistakes, "
        "and one actionable lesson for next session.\n"
        "Distinguish between execution quality and outcome quality — "
        "a well-executed trade that lost is different from a lucky winner.\n"
        "Keep it under 150 words.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


def build_user_question_prompt(
    package:  StatePackage,
    question: str,
) -> PromptPair:
    """Answer a specific user question with full context."""
    ctx  = _serialise(package)
    risk = _risk_block(package.risk_flags)

    user = (
        f"<context>\n{ctx}\n</context>\n"
        f"{risk}"
        f"<question>\n{question.strip()}\n</question>\n"
        "<request>\n"
        "Answer the question directly. Lead with the answer, then explain.\n"
        "Use the context data where relevant.\n"
        "Be concise. Advisory only — do not recommend execution.\n"
        "</request>"
    )
    return R2_SYSTEM_PROMPT, user


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _serialise(
    package:   StatePackage,
    keys:      Optional[tuple] = None,
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """Serialise StatePackage (or a subset) to compact JSON. Truncates if needed."""
    d = package.to_dict()
    if keys:
        d = {k: d[k] for k in keys if k in d}

    text = json.dumps(d, indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated for token budget]"
    return text


def _risk_block(risk_flags: list) -> str:
    """Render active risk flags as a compact XML block. Empty if no flags."""
    if not risk_flags:
        return ""
    lines = "\n".join(
        f"  - {flag}: {RISK_FLAGS.get(flag, flag)}"
        for flag in risk_flags
    )
    return f"<risk_flags>\n{lines}\n</risk_flags>\n"


def estimate_tokens(system: str, user: str) -> int:
    """Rough token estimate (chars / 4)."""
    return (len(system) + len(user)) // 4