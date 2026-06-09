"""
runtime/commentary_cycle.py — AI commentary generation with throttling.

Commentary is generated only when:
  - A meaningful state change is detected (regime, narrative, warnings)
  - Enough time has elapsed since the last commentary (cooldown)
  - The user explicitly requests it

Philosophy:
  Low noise. Commentary should earn its place.
  A quiet session is correct behaviour when nothing is happening.

All functions are exception-safe.
AI failures never halt market processing.

Structure debug logging:
  Set logger "runtime.commentary" to DEBUG to see structure pipeline details.
  Lines tagged [structure-debug] are intentionally verbose — they can be
  removed once the structure pipeline is confirmed working in production.
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


# ──────────────────────────────────────────────
# Trigger reasons
# ──────────────────────────────────────────────

TRIGGER_STARTUP   = "startup_briefing"
TRIGGER_REGIME    = "regime_change"
TRIGGER_NARRATIVE = "narrative_change"
TRIGGER_WARNING   = "new_warning"
TRIGGER_SETUP     = "setup_formed"
TRIGGER_USER      = "user_request"
TRIGGER_PERIODIC  = "periodic_check"


# ──────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────

def should_generate_commentary(
    state:          RuntimeState,
    cooldown_ready: bool,
    force:          bool = False,
) -> tuple[bool, str]:
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
    if _setup_grade_changed(state):
        return True, TRIGGER_SETUP
    return False, ""


def generate_commentary(
    state:    RuntimeState,
    trigger:  str = TRIGGER_PERIODIC,
    question: Optional[str] = None,
) -> Optional[str]:
    """
    Generate AI commentary for the current state.
    Returns the commentary string, or None if generation failed / fallback.
    Never raises.
    """
    try:
        from ai.claude_client import (
            generate_startup_commentary,
            generate_market_commentary,
            generate_setup_explanation,
            answer_user_question,
        )

        # ── Debug: verify structure pipeline before packaging ──────────
        _log_structure_debug(state, trigger)

        package = build_full_package(
            state           = state.market_state,
            score           = state.setup_score,
            direction       = state.setup_direction,
            context_state   = state.context_state,
            session_memory  = state.session_memory,
            pattern_memory  = state.pattern_memory,
            warnings        = state.active_warnings,
            structure_state = state.structure_state,
            request_type    = trigger,
        )

        # ── Debug: confirm structure and session reached the package ────
        pkg_dict = package.to_dict()
        struct   = pkg_dict.get("structure")
        ms       = pkg_dict.get("market", {}).get("market_session", "unknown")
        if struct:
            logger.debug(
                f"[structure-debug] Package OK  "
                f"session={ms}  "
                f"direction={struct.get('direction')}  "
                f"state={struct.get('market_state')}  "
                f"seq={struct.get('sequence')}"
            )
        else:
            logger.warning(
                f"[structure-debug] Package MISSING structure  session={ms}  "
                "commentary will be indicator-driven"
            )

        mock = state.mock_ai

        if trigger == TRIGGER_STARTUP:
            result = generate_startup_commentary(package, mock=mock)
        elif question:
            result = answer_user_question(package, question, mock=mock)
        elif trigger in (TRIGGER_REGIME, TRIGGER_NARRATIVE,
                         TRIGGER_WARNING, TRIGGER_PERIODIC):
            result = generate_market_commentary(package, mock=mock)
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
            f"[commentary] Generated [{trigger}]  "
            f"conf={result.confidence_label}  "
            f"flags={result.risk_flags}  "
            f"chars={len(text)}"
        )
        return text

    except Exception as e:
        logger.error(f"[commentary] Generation error ({trigger}): {e}", exc_info=True)
        return None


def generate_startup_briefing_commentary(state: RuntimeState) -> Optional[str]:
    """Generate the session-start commentary (pre-market, after-hours, etc.)."""
    return generate_commentary(state, trigger=TRIGGER_STARTUP)


def answer_question(state: RuntimeState, question: str) -> str:
    """
    Answer a user question with AI.
    Returns commentary text or a data-driven fallback.
    """
    result = generate_commentary(state, trigger=TRIGGER_USER, question=question)
    if result:
        return result
    return _build_fallback_answer(state, question)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_prev_grade: str = ""


def _setup_grade_changed(state: RuntimeState) -> bool:
    global _prev_grade
    grade = state.setup_score.grade() if state.setup_score else ""
    changed    = grade != _prev_grade and grade != ""
    _prev_grade = grade
    return changed


def _log_structure_debug(state: RuntimeState, trigger: str) -> None:
    """
    Log a summary of structure_state and market_session before packaging.
    Runs at DEBUG level — silent during normal INFO operation.
    Tagged [structure-debug] for easy grepping.
    """
    # Session
    ms = getattr(getattr(state, "market_state", None), "market_session", None)
    ms_label = ms.value if ms is not None else "unknown"
    logger.debug(f"[structure-debug] trigger={trigger}  market_session={ms_label}")

    # Structure
    struct = getattr(state, "structure_state", None)
    if struct is None:
        logger.warning(
            f"[structure-debug] structure_state is None  trigger={trigger}  "
            "session={ms_label}  — commentary will lack structure context. "
            "Ensure run_update_cycle() ran before generate_commentary()."
        )
        return

    try:
        logger.debug(
            f"[structure-debug] structure present  "
            f"direction={struct.trend.direction}  "
            f"mkt_state={struct.market_state.state.value}  "
            f"seq={struct.trend.sequence_str()}  "
            f"intact={struct.trend.trend_intact}  "
            f"events={len(struct.behavioral_events)}"
        )
        if struct.structure_narrative:
            logger.debug(
                f"[structure-debug] narrative: "
                f"{struct.structure_narrative[:120]}"
            )
        if struct.buyer_context:
            logger.debug(
                f"[structure-debug] buyers:  {struct.buyer_context[:100]}"
            )
        if struct.seller_context:
            logger.debug(
                f"[structure-debug] sellers: {struct.seller_context[:100]}"
            )
        if struct.missing_confirmation:
            logger.debug(
                f"[structure-debug] missing: {struct.missing_confirmation[:100]}"
            )
        kl = struct.key_levels
        logger.debug(
            f"[structure-debug] levels: "
            f"sess_H={kl.session_high}  sess_L={kl.session_low}  "
            f"pdH={kl.prior_day_high}  pdL={kl.prior_day_low}  "
            f"vwap={kl.vwap}"
        )
    except Exception as e:
        logger.debug(f"[structure-debug] Could not log structure details: {e}")


def _commit_commentary(state: RuntimeState, text: str, trigger: str) -> None:
    """
    Store commentary in RuntimeState and append full text to event log.

    No fixed-length truncation — the dashboard wraps long lines at render
    time via _render_events_section.  Collapsing whitespace keeps the event
    as a single logical line.
    """
    state.latest_commentary = text
    state.commentary_type   = trigger
    state.commentary_at     = datetime.now(tz=timezone.utc)
    # Collapse internal whitespace so the event is one logical line
    flat = " ".join(text.split())
    state.recent_events.append(f"AI [{trigger}]: {flat}")


def _build_fallback_answer(state: RuntimeState, question: str) -> str:
    """Data-driven answer without AI when the API is unavailable."""
    lines = ["[AI unavailable]  Answering from current data:"]

    # Lead with structure if available
    struct = getattr(state, "structure_state", None)
    if struct:
        lines.append(
            f"  Structure: {struct.trend.direction} | {struct.trend.sequence_str()} "
            f"| state={struct.market_state.state.value}"
        )
        if struct.structure_narrative:
            lines.append(f"  {struct.structure_narrative}")

    if state.market_state and state.market_state.regime:
        regime = state.market_state.regime.regime.value.replace("_", " ").title()
        lines.append(f"  Regime: {regime}  ({state.market_state.regime.confidence:.0%})")

    if state.setup_score:
        eff = (state.setup_score.adjusted_total
               if state.setup_score.adjusted_total > 0
               else state.setup_score.total)
        lines.append(f"  Setup score: {eff:.0f}/100  [{state.setup_score.grade()}]")

    if state.active_warnings:
        lines.append(f"  Active warnings: {len(state.active_warnings)}")
        for w in state.active_warnings[:2]:
            lines.append(f"    · {w.message}")

    lines.append("  Review the dashboard for full detail.")
    return "\n".join(lines)