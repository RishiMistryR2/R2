"""
trades/position_sizer.py — risk-based position sizing and R:R validation.

Pure functions only. All sizing is advisory — R2 never executes.

Position size formula (risk-based):
  risk_dollars     = account_size × risk_per_trade_pct / 100
  stop_distance    = |entry - stop| / entry   (as fraction)
  position_size    = risk_dollars / (entry × stop_distance)

For options, the "position" is the number of option contracts:
  option_risk      = account_size × risk_per_trade_pct / 100
  option_qty       = option_risk / (option_entry × premium_stop_pct / 100)
  (but max_position_pct cap also applies)

R:R validation:
  rr_ratio = |target - entry| / |entry - stop|
  FAIL if rr_ratio < min_rr_ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import (
    ACCOUNT_SIZE_DEFAULT,
    MAX_POSITION_SIZE_PCT,
    MIN_RR_RATIO_DEFAULT,
    MIN_RR_RATIO_EVENT_AGGRESSIVE,
    MIN_STOP_DISTANCE_PCT,
    RISK_PER_TRADE_PCT_DEFAULT,
)
from engine.models import InstrumentType, TradeDirection


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────

@dataclass
class SizingOutput:
    """
    Complete position sizing recommendation.

    suggested_shares / suggested_contracts: primary output
    risk_dollar:      dollar amount at risk
    rr_ratio:         risk:reward ratio (0.0 if target not provided)
    flags:            any warnings (tight stop, poor R:R, oversized)
    sizing_compliant: True when all flags pass
    """
    instrument_type:    InstrumentType
    suggested_shares:   float = 0.0       # for shares trades
    suggested_contracts: int  = 0          # for options trades
    position_notional:  float = 0.0        # total value of position
    risk_dollar:        float = 0.0
    risk_pct_of_account: float = 0.0
    rr_ratio:           float = 0.0
    stop_distance_pct:  float = 0.0
    flags:              List[str] = field(default_factory=list)
    sizing_compliant:   bool  = True

    def to_dict(self) -> dict:
        return {
            "instrument":     self.instrument_type.value,
            "shares":         round(self.suggested_shares, 2),
            "contracts":      self.suggested_contracts,
            "notional":       round(self.position_notional, 2),
            "risk_dollar":    round(self.risk_dollar, 2),
            "risk_pct":       round(self.risk_pct_of_account, 3),
            "rr_ratio":       round(self.rr_ratio, 2),
            "stop_dist_pct":  round(self.stop_distance_pct, 3),
            "flags":          self.flags,
            "compliant":      self.sizing_compliant,
        }


@dataclass
class RRResult:
    """Risk:reward validation output."""
    rr_ratio:   float
    passes:     bool
    min_rr:     float
    notes:      List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rr_ratio": round(self.rr_ratio, 2),
            "passes":   self.passes,
            "min_rr":   self.min_rr,
            "notes":    self.notes,
        }


# ──────────────────────────────────────────────
# R:R validation
# ──────────────────────────────────────────────

def validate_rr(
    entry:         float,
    stop:          float,
    target:        float,
    direction:     TradeDirection,
    event_mode:    bool = False,
    min_rr_override: Optional[float] = None,
) -> RRResult:
    """
    Compute and validate the risk:reward ratio.

    Returns RRResult with pass/fail and the ratio.
    A ratio below the threshold does not block the setup — R2
    downgrades the score and flags it. User decides.
    """
    notes: List[str] = []

    # Determine effective minimum
    if min_rr_override is not None:
        min_rr = min_rr_override
    elif event_mode:
        min_rr = MIN_RR_RATIO_EVENT_AGGRESSIVE
        notes.append(f"Event mode: reduced min R:R = {min_rr}")
    else:
        min_rr = MIN_RR_RATIO_DEFAULT

    # Compute R:R
    if direction == TradeDirection.CALLS:
        risk_side   = abs(entry - stop)
        reward_side = abs(target - entry)
    else:
        risk_side   = abs(stop - entry)
        reward_side = abs(entry - target)

    if risk_side == 0:
        notes.append("ERROR: zero stop distance — cannot compute R:R")
        return RRResult(rr_ratio=0.0, passes=False, min_rr=min_rr, notes=notes)

    rr_ratio = round(reward_side / risk_side, 3)
    passes   = rr_ratio >= min_rr

    if not passes:
        notes.append(
            f"R:R {rr_ratio:.2f} < minimum {min_rr:.2f} — "
            f"setup valid but score will be reduced"
        )
    else:
        notes.append(f"R:R {rr_ratio:.2f} ≥ {min_rr:.2f} — acceptable")

    return RRResult(rr_ratio=rr_ratio, passes=passes, min_rr=min_rr, notes=notes)


# ──────────────────────────────────────────────
# Shares position sizing
# ──────────────────────────────────────────────

def size_shares_position(
    entry_price:        float,
    stop_price:         float,
    account_size:       float = ACCOUNT_SIZE_DEFAULT,
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT_DEFAULT,
    target_price:       Optional[float] = None,
    event_mode:         bool = False,
) -> SizingOutput:
    """
    Compute recommended share count using the fixed-risk model.

      risk_dollars = account_size × risk_pct / 100
      stop_dist    = |entry - stop| per share
      shares       = risk_dollars / stop_dist

    Capped at MAX_POSITION_SIZE_PCT of account.
    """
    flags: List[str] = []
    compliant = True

    risk_dollars  = account_size * risk_per_trade_pct / 100
    stop_distance = abs(entry_price - stop_price)

    if stop_distance == 0:
        flags.append("INVALID: zero stop distance")
        return SizingOutput(
            instrument_type=InstrumentType.SHARES,
            risk_dollar=risk_dollars,
            flags=flags,
            sizing_compliant=False,
        )

    stop_dist_pct = stop_distance / entry_price * 100

    if stop_dist_pct < MIN_STOP_DISTANCE_PCT:
        flags.append(f"Tight stop: {stop_dist_pct:.3f}% < {MIN_STOP_DISTANCE_PCT}% minimum")
        compliant = False

    raw_shares   = risk_dollars / stop_distance
    max_notional = account_size * MAX_POSITION_SIZE_PCT / 100
    max_shares   = max_notional / entry_price

    if raw_shares * entry_price > max_notional:
        flags.append(
            f"Position capped at {MAX_POSITION_SIZE_PCT:.0f}% of account "
            f"({max_notional:,.0f})"
        )
        raw_shares = max_shares
        compliant  = False

    notional = raw_shares * entry_price
    rr_ratio = 0.0
    if target_price:
        rr = validate_rr(entry_price, stop_price, target_price,
                         TradeDirection.CALLS, event_mode=event_mode)
        rr_ratio = rr.rr_ratio
        if not rr.passes:
            flags.extend(rr.notes)
            compliant = False

    return SizingOutput(
        instrument_type=InstrumentType.SHARES,
        suggested_shares=round(raw_shares, 2),
        position_notional=round(notional, 2),
        risk_dollar=round(risk_dollars, 2),
        risk_pct_of_account=round(notional / account_size * 100, 3),
        rr_ratio=rr_ratio,
        stop_distance_pct=round(stop_dist_pct, 4),
        flags=flags,
        sizing_compliant=compliant,
    )


# ──────────────────────────────────────────────
# Options position sizing
# ──────────────────────────────────────────────

def size_options_position(
    option_entry_price:         float,    # premium per share (× 100 per contract)
    premium_emergency_stop_pct: float,    # Layer B stop %
    account_size:               float = ACCOUNT_SIZE_DEFAULT,
    risk_per_trade_pct:         float = RISK_PER_TRADE_PCT_DEFAULT,
    underlying_entry:           Optional[float] = None,
    underlying_stop:            Optional[float] = None,
    underlying_target:          Optional[float] = None,
    direction:                  TradeDirection = TradeDirection.CALLS,
    event_mode:                 bool = False,
) -> SizingOutput:
    """
    Compute recommended contract count for an options position.

    Risk is defined as the dollar amount lost if Layer B fires:
      risk_dollars     = account_size × risk_pct / 100
      loss_per_contract = option_entry_price × 100 × premium_stop_pct / 100
      contracts         = floor(risk_dollars / loss_per_contract)

    Capped at MAX_POSITION_SIZE_PCT of account (by notional).
    """
    flags:    List[str] = []
    compliant = True

    risk_dollars      = account_size * risk_per_trade_pct / 100
    loss_per_contract = option_entry_price * 100 * premium_emergency_stop_pct / 100

    if loss_per_contract <= 0:
        flags.append("INVALID: zero option stop distance")
        return SizingOutput(
            instrument_type=InstrumentType.OPTIONS,
            risk_dollar=risk_dollars,
            flags=flags,
            sizing_compliant=False,
        )

    raw_contracts = risk_dollars / loss_per_contract
    max_notional  = account_size * MAX_POSITION_SIZE_PCT / 100
    max_contracts = max_notional / (option_entry_price * 100)

    if raw_contracts > max_contracts:
        flags.append(
            f"Options position capped at {MAX_POSITION_SIZE_PCT:.0f}% of account"
        )
        raw_contracts = max_contracts
        compliant     = False

    contracts = max(1, int(raw_contracts))
    notional  = contracts * option_entry_price * 100

    # R:R from underlying if available
    rr_ratio = 0.0
    if underlying_entry and underlying_stop and underlying_target:
        rr = validate_rr(
            underlying_entry, underlying_stop, underlying_target,
            direction, event_mode=event_mode,
        )
        rr_ratio = rr.rr_ratio
        if not rr.passes:
            flags.extend(rr.notes)
            compliant = False

    stop_dist_pct = 0.0
    if underlying_entry and underlying_stop:
        stop_dist_pct = abs(underlying_entry - underlying_stop) / underlying_entry * 100
        if stop_dist_pct < MIN_STOP_DISTANCE_PCT:
            flags.append(f"Tight underlying stop: {stop_dist_pct:.3f}%")
            compliant = False

    return SizingOutput(
        instrument_type=InstrumentType.OPTIONS,
        suggested_contracts=contracts,
        position_notional=round(notional, 2),
        risk_dollar=round(risk_dollars, 2),
        risk_pct_of_account=round(notional / account_size * 100, 3),
        rr_ratio=rr_ratio,
        stop_distance_pct=round(stop_dist_pct, 4),
        flags=flags,
        sizing_compliant=compliant,
    )