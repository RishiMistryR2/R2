"""
runtime/commentary_cycle.py — AI commentary with throttling.

Commentary is generated only when:
  - A meaningful state change is detected (regime, narrative, warnings, grade)
  - Enough time has elapsed since the last commentary (cooldown)
  - The user explicitly requests it

Philosophy:
  Low noise.  A quiet session is correct behaviour when nothing is happening.
  Commentary should earn its place — not spam the terminal.

All functions are exception-safe.
AI failures NEVER halt market processing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from runtime.runtime_state import RuntimeState

try:
    from ai.state_packager import build_full_package
except Exception:  # pragma: no cover - defensive fallback for degraded runtime startup
    build_full_package = None

logger = logging.getLogger("runtime.commentary")

# Trigger reason constants
TRIGGER_STARTUP   = "startup_briefing"
TRIGGER_REGIME    = "regime_change"
TRIGGER_NARRATIVE = "narrative_change"
TRIGGER_WARNING   = "new_warning"
TRIGGER_SETUP     = "setup_formed"
TRIGGER_USER      = "user_request"
TRIGGER_PERIODIC  = "periodic_check"


# ──────────────────────────────────────────────
# Trigger logic
# ──────────────────────────────────────────────

def should_generate_commentary(
    state:          RuntimeState,
    cooldown_ready: bool,
    force:          bool = False,
) -> tuple:
    """
    Determine whether commentary should be generated now.
    Returns (should_generate: bool, reason: str).
    """
    if force:
        return True, TRIGGER_USER

    if not cooldown_ready:
        return False, ""

    if state.regime_changed():
        return True, TRIGGER_REGIME

    if state.narrative_changed():
        return True, TRIGGER_NARRATIVE

    if state.warning_severity_increased():
        return True, TRIGGER_WARNING

    if state.setup_grade_changed():
        return True, TRIGGER_SETUP

    return False, ""


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

def generate_commentary(
    state:    RuntimeState,
    trigger:  str           = TRIGGER_PERIODIC,
    question: Optional[str] = None,
) -> Optional[str]:
    """
    Generate AI commentary for the current state.
    Returns the commentary string, or None if generation failed / was a fallback.
    Never raises.
    """
    try:
        from ai.claude_client import (
            generate_startup_commentary,
            generate_market_commentary,
            generate_setup_explanation,
            answer_user_question,
        )

        package = build_full_package(
            state          = state.market_state,
            score          = state.setup_score,
            direction      = state.setup_direction,
            context_state  = state.context_state,
            session_memory = state.session_memory,
            pattern_memory = state.pattern_memory,
            warnings       = state.active_warnings,
            structure_state = state.structure_state,
            request_type   = trigger,
        )

        mock = state.mock_ai

        if trigger == TRIGGER_STARTUP:
            result = generate_startup_commentary(package, mock=mock)
        elif question:
            result = answer_user_question(package, question, mock=mock)
        elif trigger == TRIGGER_SETUP:
            direction_str = (
                state.setup_direction.value if state.setup_direction else None
            )
            result = generate_setup_explanation(package, direction=direction_str, mock=mock)
        else:
            result = generate_market_commentary(package, mock=mock)

        if result.is_fallback:
            logger.info(f"[commentary] Fallback ({trigger}): {result.text[:60]}")
            return None

        text = result.text
        _commit_commentary(state, text, trigger)

        logger.info(
            f"[commentary] Generated [{trigger}]"
            f"  conf={result.confidence_label}"
            f"  flags={result.risk_flags}"
            f"  chars={len(text)}"
        )
        return text

    except Exception as e:
        logger.error(f"[commentary] Generation error ({trigger}): {e}", exc_info=True)
        return None


def generate_startup_briefing_commentary(state: RuntimeState) -> Optional[str]:
    """Generate the pre-market / session-start commentary. Never raises."""
    return generate_commentary(state, trigger=TRIGGER_STARTUP)


def answer_question(state: RuntimeState, question: str) -> str:
    """
    Answer a user question with AI.
    Returns commentary text, or a data-driven fallback if AI is unavailable.
    Never raises.
    """
    result = generate_commentary(state, trigger=TRIGGER_USER, question=question)
    if result:
        return result
    return _build_fallback_answer(state, question)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _commit_commentary(state: RuntimeState, text: str, trigger: str) -> None:
    """
    Store commentary in RuntimeState and append an event log entry.

    The full text (whitespace collapsed) is stored without a fixed-length
    truncation so the dashboard can display the complete narrative.
    Long entries are wrapped at render time by _render_events_section.
    """
    state.latest_commentary = text
    state.commentary_type   = trigger
    state.commentary_at     = datetime.now(tz=timezone.utc)

    # Collapse internal whitespace/newlines to a single space so the
    # event string is one logical line that the dashboard can word-wrap.
    flat  = " ".join(text.split())
    state.add_event(f"AI [{trigger}]: {flat}")


def _build_fallback_answer(state: RuntimeState, question: str) -> str:
    """
    Build a data-driven answer without AI when the API is unavailable.
    """
    lines = ["[AI unavailable]  Answering from current data:"]

    if state.market_state and state.market_state.regime:
        regime = state.market_state.regime.regime.value.replace("_", " ").title()
        conf   = state.market_state.regime.confidence
        lines.append(f"  Regime: {regime}  ({conf:.0%})")

    if state.setup_score:
        eff = (
            state.setup_score.adjusted_total
            if state.setup_score.adjusted_total > 0
            else state.setup_score.total
        )
        lines.append(f"  Setup score: {eff:.0f}/100  [{state.setup_score.grade()}]")

    if state.active_warnings:
        lines.append(f"  Active warnings: {len(state.active_warnings)}")
        for w in state.active_warnings[:2]:
            lines.append(f"    · {w.message}")

    lines.append("  Review the dashboard for full detail.")
    return "\n".join(lines)