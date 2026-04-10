"""
Session Manager.

Computes the current SessionState from wall-clock time (ET).
Pure time-based logic: no market data required.

SessionState affects:
  - score_modifier: applied to all setup scores in that phase
  - entry_filter:   "normal" | "strict" | "no_new_entries"
  - entries_allowed: hard gate — no entries outside market hours

Phase definitions (Eastern Time):
  PREMARKET:   04:00 – 09:30  no entries, context-building only
  OPEN:        09:30 – 10:00  volatile, first 30 min — strict filter
  MIDDAY:      10:00 – 14:30  normal operations, cleanest signals
  POWER_HOUR:  14:30 – 16:00  elevated volume, decisive moves, full size
  AFTER_HOURS: 16:00 – 20:00  no entries
  OVERNIGHT:   20:00 – 04:00  no entries
"""

from __future__ import annotations
from datetime import datetime, timezone, time as dt_time
from typing import Optional

from engine.models import (
    SessionPhase,
    SessionState,
    SESSION_SCORE_MODIFIERS,
)

try:
    import pytz
    ET = pytz.timezone("US/Eastern")
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False


# ──────────────────────────────────────────────
# Phase boundaries (ET)
# ──────────────────────────────────────────────

MARKET_OPEN  = dt_time(9, 30)
OPEN_END     = dt_time(10, 0)
POWER_START  = dt_time(14, 30)
MARKET_CLOSE = dt_time(16, 0)
AH_END       = dt_time(20, 0)
PRE_START    = dt_time(4, 0)


def compute_session_state(now_utc: Optional[datetime] = None) -> SessionState:
    """
    Compute SessionState for the given UTC time (defaults to now).
    Returns a complete SessionState with all derived fields populated.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)

    now_et = _to_et(now_utc)
    t = now_et.time()

    phase = _classify_phase(t)
    market_open   = MARKET_OPEN <= t < MARKET_CLOSE
    entries_allowed, entry_filter = _entry_rules(phase)
    score_modifier = SESSION_SCORE_MODIFIERS.get(phase, 1.0)
    phase_note     = _phase_note(phase)

    mins_since_open = None
    mins_to_close   = None
    if market_open:
        open_today  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        mins_since_open = max(0, int((now_et - open_today).total_seconds() / 60))
        mins_to_close   = max(0, int((close_today - now_et).total_seconds() / 60))

    return SessionState(
        phase=phase,
        market_open=market_open,
        entries_allowed=entries_allowed,
        minutes_since_open=mins_since_open,
        minutes_to_close=mins_to_close,
        score_modifier=score_modifier,
        entry_filter=entry_filter,
        phase_note=phase_note,
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _classify_phase(t: dt_time) -> SessionPhase:
    if t < PRE_START or t >= AH_END:
        return SessionPhase.OVERNIGHT
    if PRE_START <= t < MARKET_OPEN:
        return SessionPhase.PREMARKET
    if MARKET_OPEN <= t < OPEN_END:
        return SessionPhase.OPEN
    if OPEN_END <= t < POWER_START:
        return SessionPhase.MIDDAY
    if POWER_START <= t < MARKET_CLOSE:
        return SessionPhase.POWER_HOUR
    return SessionPhase.AFTER_HOURS


def _entry_rules(phase: SessionPhase) -> tuple[bool, str]:
    """Returns (entries_allowed, entry_filter)."""
    rules = {
        SessionPhase.PREMARKET:   (False, "no_new_entries"),
        SessionPhase.OPEN:        (True,  "strict"),
        SessionPhase.MIDDAY:      (True,  "normal"),
        SessionPhase.POWER_HOUR:  (True,  "normal"),
        SessionPhase.AFTER_HOURS: (False, "no_new_entries"),
        SessionPhase.OVERNIGHT:   (False, "no_new_entries"),
    }
    return rules.get(phase, (False, "no_new_entries"))


def _phase_note(phase: SessionPhase) -> str:
    notes = {
        SessionPhase.PREMARKET:   "context building only — no entries",
        SessionPhase.OPEN:        "first 30 min: high volatility, wait for direction",
        SessionPhase.MIDDAY:      "normal operations — cleanest signals",
        SessionPhase.POWER_HOUR:  "elevated volume — decisive moves likely",
        SessionPhase.AFTER_HOURS: "no entries after close",
        SessionPhase.OVERNIGHT:   "market closed",
    }
    return notes.get(phase, "")


def _to_et(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to ET. Falls back to UTC-5 offset if pytz missing."""
    if HAS_PYTZ:
        return utc_dt.astimezone(ET)
    # Approximate: UTC-4 (EDT) in summer, UTC-5 (EST) in winter
    # For production: always use pytz or zoneinfo
    from datetime import timedelta
    return utc_dt - timedelta(hours=4)   # approximate EDT


# ──────────────────────────────────────────────
# Intent builder (convenience)
# ──────────────────────────────────────────────

def default_intent() -> "SessionIntent":
    """Return the default session intent (no user overrides)."""
    from engine.models import SessionIntent
    return SessionIntent()


def intent_from_dict(d: dict) -> "SessionIntent":
    """
    Reconstruct a SessionIntent from a dict (e.g. from a config file
    or a user command: "set intent: bearish, reduced risk, min score 75").
    """
    from engine.models import SessionIntent
    return SessionIntent(
        directional_bias=d.get("directional_bias"),
        min_score=float(d.get("min_score", 60.0)),
        risk_tolerance=d.get("risk_tolerance", "normal"),
        avoid_chop=d.get("avoid_chop", True),
        require_htf_alignment=d.get("require_htf_alignment", False),
        max_trades=int(d.get("max_trades", 3)),
        max_loss_pct=d.get("max_loss_pct"),
        target_gain_pct=d.get("target_gain_pct"),
        notes=d.get("notes", ""),
    )
