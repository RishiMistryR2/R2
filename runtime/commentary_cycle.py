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

# ── Event intelligence state (module-level) ─────────────────────
# Avoids threading through RuntimeState. Single process — safe.
_last_snapshot:      "object | None" = None      # MarketSnapshot | None
_last_informational: "object | None" = None      # datetime | None

# Minimum seconds between INFORMATIONAL-priority commentary.
# CRITICAL and IMPORTANT events always bypass this limit.
_MIN_INFORMATIONAL_GAP: float = 300.0  # 5 minutes

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

        # ── Event intelligence — runs BEFORE the package build ──────
        # Suppresses insignificant events with zero API cost.
        _current_snap = None
        _event_result = None

        if trigger not in ("startup_briefing", "user_request"):
            try:
                from engine.event_engine import build_snapshot, evaluate_event
                global _last_snapshot, _last_informational

                _current_snap = build_snapshot(state, trigger)
                _event_result = evaluate_event(_current_snap, _last_snapshot, trigger)

                if _event_result.suppress:
                    logger.debug(
                        f"[event] Suppressed [{trigger}]  "
                        f"score={_event_result.score:.0f}  "
                        f"priority={_event_result.priority}"
                    )
                    return None

                # Rate-limit informational events
                if (_event_result.priority == "informational"
                        and _last_informational is not None):
                    elapsed = (
                        datetime.now(tz=timezone.utc) - _last_informational
                    ).total_seconds()
                    if elapsed < _MIN_INFORMATIONAL_GAP:
                        logger.debug(
                            f"[event] Informational rate-limited [{trigger}]  "
                            f"elapsed={elapsed:.0f}s  gap={_MIN_INFORMATIONAL_GAP:.0f}s"
                        )
                        return None

                logger.info(
                    f"[event] Approved [{trigger}]  "
                    f"priority={_event_result.priority}  "
                    f"score={_event_result.score:.0f}"
                    + (f"  — {_event_result.change_summary[:100]}"
                       if _event_result.change_summary else "")
                )
            except Exception as _ee:
                # Event intelligence is non-fatal — proceed to commentary
                logger.warning(f"[event] Intelligence check error (non-fatal): {_ee}")

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
            confidence_state = getattr(state, "confidence_state", None),
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
        if trigger == TRIGGER_STARTUP:
            text = _stamp_briefing(text, state)
        _commit_commentary(state, text, trigger)

        # Advance event intelligence state only after successful commentary.
        if _current_snap is not None:
            _last_snapshot = _current_snap
        if (_event_result is not None
                and _event_result.priority == "informational"):
            _last_informational = datetime.now(tz=timezone.utc)

        # ── Outcome tracker: record observation on significant events ──
        # _event_result and _current_snap are set by the v2.4 event
        # intelligence block above this point.  On versions without those
        # variables a NameError is caught safely.
        try:
            _er = _event_result   # noqa: F821  — defined in event intel block
            _cs = _current_snap   # noqa: F821
            if (_er is not None
                    and _er.priority in ("critical", "important")
                    and _cs is not None):
                from memory.outcome_tracker  import get_tracker
                from engine.pattern_engine   import classify_patterns
                from engine.narrative_engine import _to_fields
                _struct   = getattr(state, "structure_state", None)
                _fields   = _to_fields(_struct) if _struct else {}
                _patterns = classify_patterns(_fields)
                if _patterns:
                    _ms_obj  = getattr(
                        getattr(state, "market_state", None),
                        "market_session", None,
                    )
                    _session = _ms_obj.value if _ms_obj else ""
                    # v2.7: history stores the structural score; the
                    # calibrated value is kept alongside for audit only.
                    _conf = getattr(state, "confidence_state", None)
                    get_tracker().record_observation(
                        _cs, _patterns, session=_session,
                        structural_confidence=getattr(
                            _conf, "structural_confidence", None),
                        calibrated_confidence=getattr(
                            _conf, "calibrated_confidence", None),
                        evidence=(getattr(_conf, "evidence", None)
                                  if _conf is not None else None),
                    )
        except Exception as _ot_err:
            logger.debug(
                f"[outcome] record_observation failed (non-fatal): {_ot_err}"
            )

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


def _stamp_briefing(text: str, state: RuntimeState) -> str:
    """
    v2.7 — the startup briefing is a session-open snapshot.  Prefix when it
    was generated and which cycle's confidence it used, so later dashboard
    values aren't read as contradictions.  Prefixed (not appended) because
    the runtime loop prints only the first 300 chars of commentary.
    Never raises: on any failure the text is returned unchanged.
    """
    try:
        from engine.market_session import to_et_string
        when = to_et_string(datetime.now(tz=timezone.utc))
        cs   = getattr(state, "confidence_state", None)
        src  = f" · confidence from cycle {cs.source_cycle}" if cs is not None else ""
        return f"[Generated {when}{src} — session-open snapshot]\n{text}"
    except Exception as e:
        logger.warning(f"[commentary] Briefing timestamp failed (non-fatal): {e}")
        return text


def _commit_commentary(state: RuntimeState, text: str, trigger: str) -> None:
    """
    Store commentary in RuntimeState and append full text to event log.

    Logged via add_event, so the entry is timestamped
    ("HH:MM:SS  AI [trigger]: text") and subject to the rolling max_events
    cap like every other event.  No fixed-length truncation — the dashboard
    decides how much to show.  Collapsing whitespace keeps the event as a
    single logical line.
    """
    state.latest_commentary = text
    state.commentary_type   = trigger
    state.commentary_at     = datetime.now(tz=timezone.utc)
    # Collapse internal whitespace so the event is one logical line
    flat = " ".join(text.split())
    state.add_event(f"AI [{trigger}]: {flat}")


def _build_fallback_answer(state: RuntimeState, question: str) -> str:
    """
    Produce trader-quality commentary without the AI API.
    Delegates to the Structure Narrative Engine for flowing prose.
    """
    from engine.narrative_engine import build_narrative

    struct   = getattr(state, "structure_state", None)
    ms       = getattr(getattr(state, "market_state", None), "market_session", None)
    regime   = ""
    entries  = True
    price    = None

    if state.market_state:
        if state.market_state.regime:
            regime = state.market_state.regime.regime.value
        if state.market_state.session:
            entries = state.market_state.session.entries_allowed
        price = getattr(state.market_state, "current_price", None)
        if price is None:
            ind_5m = (state.market_state.indicators or {}).get("5m")
            if ind_5m:
                price = getattr(ind_5m, "close", None)

    narrative = build_narrative(
        struct,
        session=ms,
        regime=regime,
        entries_allowed=entries,
        current_price=price,
    )

    suffix: list[str] = []
    if state.setup_score:
        eff = (state.setup_score.adjusted_total
               if state.setup_score.adjusted_total > 0
               else state.setup_score.total)
        suffix.append(f"[Setup: {eff:.0f}/100 {state.setup_score.grade()}]")
    if state.active_warnings:
        for w in state.active_warnings[:2]:
            suffix.append(f"[Warning: {w.message}]")

    prefix = "[AI unavailable] "
    if suffix:
        return prefix + narrative + "  " + "  ".join(suffix)
    return prefix + narrative