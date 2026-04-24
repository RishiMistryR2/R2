"""
trades/stop_calculator.py — pure stop-level computation.

Pure functions only: no state, no side effects.

Shares stop model:
  stop = structural level (swing low/high or S/R) ± ATR buffer
  The underlying must close through the stop to trigger.

Options two-layer stop model:
  Layer A (primary):   underlying breaks its invalidation level
  Layer B (secondary): option premium falls by emergency threshold

Neither layer auto-executes — R2 issues a recommendation only.
User executes manually.

All functions return a StopLevels dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import (
    ATR_BUFFER_OPTIONS,
    ATR_BUFFER_SHARES,
    ATM_DELTA_APPROX,
    NEAR_OTM_DELTA_APPROX,
    OTM_DELTA_APPROX,
    PREMIUM_EMERGENCY_BUFFER,
    MAX_PREMIUM_LOSS_PCT_DEFAULT,
)
from engine.models import TradeDirection


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────

@dataclass
class StopLevels:
    """
    Complete stop level recommendation for a trade.

    underlying_stop:      price on the underlying where thesis is invalid
    primary_alert_price:  underlying price that triggers Layer A warning
    option_stop_est:      estimated option value at underlying_stop
    premium_emergency_stop_pct: % option premium loss that triggers Layer B
    stop_notes:           human-readable explanations
    """
    underlying_stop:            float
    primary_alert_price:        float    # = underlying_stop (alias for clarity)

    # Options-specific (None for shares trades)
    option_stop_est:             Optional[float] = None
    premium_emergency_stop_pct: Optional[float] = None

    # Context
    atr_used:    float = 0.0
    stop_notes:  List[str] = field(default_factory=list)

    @property
    def stop_distance(self) -> float:
        """Distance from underlying entry to stop (populated by caller)."""
        return 0.0

    def to_dict(self) -> dict:
        return {
            "underlying_stop":            round(self.underlying_stop, 3),
            "primary_alert_price":        round(self.primary_alert_price, 3),
            "option_stop_est":            round(self.option_stop_est, 4) if self.option_stop_est else None,
            "premium_emergency_stop_pct": self.premium_emergency_stop_pct,
            "atr_used":                   round(self.atr_used, 4),
            "notes":                      self.stop_notes,
        }


# ──────────────────────────────────────────────
# Shares stop
# ──────────────────────────────────────────────

def compute_shares_stop(
    direction:        TradeDirection,
    entry_price:      float,
    swing_level:      float,          # swing low (long) or swing high (short)
    atr:              float,
    atr_buffer_mult:  float = ATR_BUFFER_SHARES,
    support_levels:   Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
) -> StopLevels:
    """
    Compute a structural stop for a shares (or equivalent) position.

    For CALLS (long):
      stop = swing_low - atr * buffer
      If a support level is closer to entry than swing_low, use that instead.

    For PUTS (short):
      stop = swing_high + atr * buffer
      If a resistance level is closer to entry than swing_high, use that instead.

    Returns StopLevels with underlying_stop set.
    """
    notes: List[str] = []
    buffer = atr * atr_buffer_mult

    if direction == TradeDirection.CALLS:
        raw_stop = swing_level - buffer
        notes.append(f"Long stop: swing_low({swing_level:.2f}) - ATR_buf({buffer:.3f}) = {raw_stop:.3f}")

        # Check if a nearby support is a better anchor
        if support_levels:
            candidates = [s for s in support_levels if s < entry_price and s > raw_stop - atr]
            if candidates:
                best = max(candidates)
                raw_stop = best - buffer
                notes.append(f"Support anchor at {best:.2f} used instead")

        if raw_stop >= entry_price:
            notes.append("WARNING: stop above entry — invalid setup")

    else:  # PUTS / short
        raw_stop = swing_level + buffer
        notes.append(f"Short stop: swing_high({swing_level:.2f}) + ATR_buf({buffer:.3f}) = {raw_stop:.3f}")

        if resistance_levels:
            candidates = [r for r in resistance_levels if r > entry_price and r < raw_stop + atr]
            if candidates:
                best = min(candidates)
                raw_stop = best + buffer
                notes.append(f"Resistance anchor at {best:.2f} used instead")

        if raw_stop <= entry_price:
            notes.append("WARNING: stop below entry — invalid setup")

    return StopLevels(
        underlying_stop=round(raw_stop, 4),
        primary_alert_price=round(raw_stop, 4),
        atr_used=atr,
        stop_notes=notes,
    )


# ──────────────────────────────────────────────
# Options two-layer stop
# ──────────────────────────────────────────────

def compute_options_stop(
    direction:         TradeDirection,
    underlying_entry:  float,
    swing_level:       float,
    atr:               float,
    option_entry_price: float,
    delta_approx:       float = ATM_DELTA_APPROX,
    atr_buffer_mult:    float = ATR_BUFFER_OPTIONS,
    premium_emergency_stop_pct: Optional[float] = None,
    support_levels:     Optional[List[float]] = None,
    resistance_levels:  Optional[List[float]] = None,
) -> StopLevels:
    """
    Compute the two-layer stop model for an options position.

    Layer A (primary / thesis stop):
      Derived from shares_stop logic on the underlying.
      Fires when the underlying closes through invalidation.

    Layer B (secondary / emergency premium stop):
      Fires when the option premium has declined by
      `premium_emergency_stop_pct` from entry.
      Default = MAX_PREMIUM_LOSS_PCT_DEFAULT, capped at 50%.

    estimated_option_stop_price:
      Approximated as:
        option_entry_price - (underlying_stop_distance × delta_approx)
      This is a planning estimate only. Actual option price will differ.

    Returns StopLevels with both layers populated.
    """
    notes: List[str] = []
    buffer = atr * atr_buffer_mult

    # ── Layer A: underlying thesis stop ───────
    shares_stop = compute_shares_stop(
        direction=direction,
        entry_price=underlying_entry,
        swing_level=swing_level,
        atr=atr,
        atr_buffer_mult=atr_buffer_mult,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
    )
    underlying_stop = shares_stop.underlying_stop
    notes.extend(shares_stop.stop_notes)

    # ── Estimated option value at thesis stop ─
    underlying_stop_distance = abs(underlying_entry - underlying_stop)
    option_loss_est = underlying_stop_distance * delta_approx
    option_stop_est = max(0.01, round(option_entry_price - option_loss_est, 3))
    option_loss_pct = round((option_entry_price - option_stop_est) / option_entry_price * 100, 1)
    notes.append(
        f"Option est at thesis stop: {option_stop_est:.3f} "
        f"({option_loss_pct:.1f}% loss from entry {option_entry_price:.3f})"
    )

    # ── Layer B: emergency premium stop ───────
    if premium_emergency_stop_pct is None:
        premium_emergency_stop_pct = min(
            50.0,
            max(option_loss_pct * PREMIUM_EMERGENCY_BUFFER, MAX_PREMIUM_LOSS_PCT_DEFAULT)
        )
    premium_emergency_stop_pct = round(premium_emergency_stop_pct, 1)
    notes.append(
        f"Layer B: premium emergency stop at -{premium_emergency_stop_pct}% "
        f"(${option_entry_price * premium_emergency_stop_pct / 100:.3f} loss per contract)"
    )

    return StopLevels(
        underlying_stop=underlying_stop,
        primary_alert_price=underlying_stop,
        option_stop_est=option_stop_est,
        premium_emergency_stop_pct=premium_emergency_stop_pct,
        atr_used=atr,
        stop_notes=notes,
    )


# ──────────────────────────────────────────────
# Stop distance helpers
# ──────────────────────────────────────────────

def stop_distance_pct(entry: float, stop: float) -> float:
    """Absolute percentage distance between entry and stop."""
    if not entry:
        return 0.0
    return round(abs(entry - stop) / entry * 100, 4)


def is_stop_too_tight(
    entry:          float,
    stop:           float,
    min_distance_pct: float = 0.15,
) -> bool:
    """
    Return True if stop is closer than min_distance_pct to entry.
    Tight stops are noise-prone and should be flagged.
    """
    return stop_distance_pct(entry, stop) < min_distance_pct


def estimate_option_delta(moneyness: str = "atm") -> float:
    """
    Return approximate delta for option stop calculations.
    moneyness: "atm" | "near_otm" | "otm"
    """
    return {
        "atm":      ATM_DELTA_APPROX,
        "near_otm": NEAR_OTM_DELTA_APPROX,
        "otm":      OTM_DELTA_APPROX,
    }.get(moneyness, ATM_DELTA_APPROX)