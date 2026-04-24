"""
trades/time_stop.py — time-stop evaluation for options day trades.

Pure functions. Does NOT auto-exit. Produces TimeStopResult
with a recommendation and confidence reduction.

Logic:
  Each setup type has an expected-move window (minutes after entry).
  If the setup has not moved the required distance in that window:
    - Confidence is reduced
    - A warning is issued
    - Recommendation is one of: HOLD / SCALE_DOWN / EXIT

  Phase multipliers adjust the window based on session phase
  (e.g. OPEN moves faster → 0.60× window; MIDDAY → 1.00×).

R2 advises only — user decides whether to exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from config.settings import (
    DEFAULT_TIME_STOP_WINDOWS,
    TIME_STOP_PHASE_MULTIPLIERS,
)
from engine.models import SetupType, SessionPhase


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TimeStopAction(Enum):
    HOLD       = "hold"
    SCALE_DOWN = "scale_down"
    EXIT       = "exit"


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────

@dataclass
class TimeStopResult:
    """
    Output of evaluate_time_stop().

    triggered:             True when the time window has elapsed without sufficient move
    action:                HOLD / SCALE_DOWN / EXIT recommendation
    confidence_reduction:  amount to subtract from setup confidence (0.0–1.0)
    minutes_elapsed:       time since entry
    window_minutes:        adjusted expected-move window
    progress_pct:          how far price has moved toward target as a %
    notes:                 human-readable explanation
    """
    triggered:            bool
    action:               TimeStopAction
    confidence_reduction: float
    minutes_elapsed:      float
    window_minutes:       float
    progress_pct:         float
    notes:                List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "triggered":            self.triggered,
            "action":               self.action.value,
            "confidence_reduction": round(self.confidence_reduction, 3),
            "minutes_elapsed":      round(self.minutes_elapsed, 1),
            "window_minutes":       self.window_minutes,
            "progress_pct":         round(self.progress_pct, 1),
            "notes":                self.notes,
        }


# ──────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────

def evaluate_time_stop(
    entry_time:       datetime,
    entry_price:      float,
    current_price:    float,
    target_price:     float,
    stop_price:       float,
    setup_type:       Optional[SetupType] = None,
    session_phase:    Optional[SessionPhase] = None,
    now:              Optional[datetime] = None,
    custom_window_minutes: Optional[int] = None,
) -> TimeStopResult:
    """
    Evaluate whether the time-stop threshold has been breached.

    Parameters
    ----------
    entry_time:    datetime when the trade was entered
    entry_price:   underlying price at entry
    current_price: current underlying price
    target_price:  profit target on underlying
    stop_price:    stop price on underlying
    setup_type:    SetupType (determines base window)
    session_phase: SessionPhase (adjusts window via multiplier)
    now:           override current time (for testing)
    custom_window_minutes: override window (for testing)
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    notes: List[str] = []

    # ── 1. Compute effective window ────────────
    base_window = _get_base_window(setup_type, custom_window_minutes)
    phase_mult  = _get_phase_multiplier(session_phase)
    effective_window = max(3, int(base_window * phase_mult))
    notes.append(
        f"Window: {base_window}m base × {phase_mult:.2f} phase = {effective_window}m"
    )

    # ── 2. Time elapsed ────────────────────────
    elapsed_seconds = (now - entry_time).total_seconds()
    elapsed_minutes = elapsed_seconds / 60
    notes.append(f"Elapsed: {elapsed_minutes:.1f}m of {effective_window}m window")

    # ── 3. Progress toward target ──────────────
    total_move    = abs(target_price - entry_price)
    actual_move   = current_price - entry_price  # signed
    progress_pct  = 0.0
    if total_move > 0:
        progress_pct = round((actual_move / total_move) * 100, 1)

    # ── 4. Evaluate ────────────────────────────
    if elapsed_minutes < effective_window:
        # Still within window — no action
        return TimeStopResult(
            triggered=False,
            action=TimeStopAction.HOLD,
            confidence_reduction=0.0,
            minutes_elapsed=elapsed_minutes,
            window_minutes=effective_window,
            progress_pct=progress_pct,
            notes=notes,
        )

    # Window elapsed — assess situation
    notes.append(f"Time window elapsed (progress: {progress_pct:.1f}%)")

    # Good progress: no concern even if time is up
    if progress_pct >= 50.0:
        notes.append("Good progress — hold position")
        return TimeStopResult(
            triggered=False,
            action=TimeStopAction.HOLD,
            confidence_reduction=0.05,
            minutes_elapsed=elapsed_minutes,
            window_minutes=effective_window,
            progress_pct=progress_pct,
            notes=notes,
        )

    # Negative progress: moving toward stop
    if progress_pct < 0:
        notes.append("Price moving against position — EXIT recommended")
        return TimeStopResult(
            triggered=True,
            action=TimeStopAction.EXIT,
            confidence_reduction=0.30,
            minutes_elapsed=elapsed_minutes,
            window_minutes=effective_window,
            progress_pct=progress_pct,
            notes=notes,
        )

    # No / minimal progress: SCALE_DOWN
    notes.append(f"Minimal progress ({progress_pct:.1f}%) — SCALE_DOWN recommended")
    return TimeStopResult(
        triggered=True,
        action=TimeStopAction.SCALE_DOWN,
        confidence_reduction=0.15,
        minutes_elapsed=elapsed_minutes,
        window_minutes=effective_window,
        progress_pct=progress_pct,
        notes=notes,
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_base_window(
    setup_type:   Optional[SetupType],
    custom_window: Optional[int],
) -> int:
    if custom_window is not None:
        return custom_window
    if setup_type is None:
        return DEFAULT_TIME_STOP_WINDOWS["default"]

    mapping = {
        SetupType.BREAKOUT:     DEFAULT_TIME_STOP_WINDOWS.get("breakout",  8),
        SetupType.BREAKDOWN:    DEFAULT_TIME_STOP_WINDOWS.get("breakout",  8),
        SetupType.CONTINUATION: DEFAULT_TIME_STOP_WINDOWS.get("continuation", 12),
        SetupType.REVERSAL:     DEFAULT_TIME_STOP_WINDOWS.get("reversal",  15),
    }
    return mapping.get(setup_type, DEFAULT_TIME_STOP_WINDOWS["default"])


def _get_phase_multiplier(session_phase: Optional[SessionPhase]) -> float:
    if session_phase is None:
        return 1.0
    key_map = {
        SessionPhase.OPEN:       "open",
        SessionPhase.MIDDAY:     "midday",
        SessionPhase.POWER_HOUR: "power_hour",
    }
    key = key_map.get(session_phase, "midday")
    return TIME_STOP_PHASE_MULTIPLIERS.get(key, 1.0)


def recommended_time_window(
    setup_type:    Optional[SetupType],
    session_phase: Optional[SessionPhase] = None,
) -> int:
    """
    Return the recommended time-stop window (minutes) for a given
    setup type and session phase. Used when building a TradeState.
    """
    base = _get_base_window(setup_type, None)
    mult = _get_phase_multiplier(session_phase)
    return max(3, int(base * mult))