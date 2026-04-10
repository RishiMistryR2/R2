"""
Rule Engine — all trading logic as explicit, named predicates.

Design principles:
  - Every condition is a named function returning (bool, reason_str)
  - No side effects, no state
  - Pure: (MarketState, ...) -> (bool, str)
  - Every rule is independently testable

Entry rules follow a gate hierarchy:
  HTF gate → 15m gate → momentum gate → timing gate

Exit rules are independent monitors checked on every tick.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from engine.models import (
    Direction,
    ExitReason,
    IndicatorSnapshot,
    MarketState,
    TradeDirection,
    TradeState,
    TrendState,
    TrendStrength,
)
from strategy.timeframes import is_entry_allowed
from config.settings import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)


# ──────────────────────────────────────────────
# Entry rules
# ──────────────────────────────────────────────

@dataclass
class EntrySignal:
    allowed:   bool
    direction: Optional[TradeDirection]
    reason:    str
    gates_passed: int   # how many of 4 gates passed (useful for scoring)


def check_long_entry(state: MarketState) -> EntrySignal:
    """
    Full long (calls) entry gate check.
    All gates must pass. Returns first failure with reason.

    Gate 1 — HTF permission:
      1h must be bullish (or bullish+daily agreement)
      No HTF conflict

    Gate 2 — Setup prerequisite (15m):
      15m trend = BULLISH
      15m EMA stack = bull_aligned or mixed (not bear_aligned)

    Gate 3 — Momentum (5m):
      5m RSI 35–65 (not overbought entering, not oversold)
      5m MACD histogram > 0 OR just crossed positive
      Price above VWAP

    Gate 4 — Entry timing (5m/1m):
      5m close above ema_9
      No resistance within 1 ATR above entry price
    """
    gates_passed = 0

    # ── Gate 0: Regime + Session + Intent pre-checks ─────
    if hasattr(state, "regime") and state.regime and not state.regime.entry_permitted:
        return EntrySignal(False, None,
            f"Regime gate: {state.regime.regime.value} — entries not permitted ({state.regime.caution_note})",
            gates_passed)
    if hasattr(state, "session") and state.session and not state.session.entries_allowed:
        return EntrySignal(False, None,
            f"Session gate: {state.session.phase.value} — {state.session.phase_note}",
            gates_passed)
    if hasattr(state, "intent") and state.intent:
        if state.intent.directional_bias == "bearish":
            return EntrySignal(False, None,
                "Intent gate: session bias is bearish — calls blocked",
                gates_passed)

    # ── Gate 1: HTF ───────────────────────────
    allowed, reason = is_entry_allowed(state.trends, Direction.BULLISH)
    if not allowed:
        return EntrySignal(False, None, f"Gate 1 (HTF): {reason}", gates_passed)
    gates_passed += 1

    # ── Gate 2: 15m setup ─────────────────────
    trend_15m = state.trends.get("15m")
    ind_15m   = state.indicators.get("15m")
    if not trend_15m or not ind_15m:
        return EntrySignal(False, None, "Gate 2: no 15m data", gates_passed)

    if trend_15m.direction != Direction.BULLISH:
        return EntrySignal(
            False, None,
            f"Gate 2 (15m): trend is {trend_15m.direction.value}, need bullish",
            gates_passed,
        )
    if ind_15m.ema_stack == "bear_aligned":
        return EntrySignal(False, None, "Gate 2 (15m): EMA stack bear-aligned", gates_passed)
    gates_passed += 1

    # ── Gate 3: Momentum ──────────────────────
    ind_5m = state.indicators.get("5m")
    if not ind_5m:
        return EntrySignal(False, None, "Gate 3: no 5m data", gates_passed)

    rsi_fail, rsi_reason = _check_rsi_long(ind_5m)
    if rsi_fail:
        return EntrySignal(False, None, f"Gate 3 (momentum): {rsi_reason}", gates_passed)

    if ind_5m.macd_hist is not None and ind_5m.macd_hist < 0:
        return EntrySignal(
            False, None,
            f"Gate 3 (momentum): MACD hist negative ({ind_5m.macd_hist:.4f})",
            gates_passed,
        )

    if ind_5m.price_vs_vwap == "below":
        return EntrySignal(False, None, "Gate 3 (momentum): price below VWAP", gates_passed)
    gates_passed += 1

    # ── Gate 4: Entry timing ──────────────────
    ind_1m = state.indicators.get("1m")
    timing_ok, timing_reason = _check_timing_long(ind_5m, ind_1m, state)
    if not timing_ok:
        return EntrySignal(False, None, f"Gate 4 (timing): {timing_reason}", gates_passed)
    gates_passed += 1

    return EntrySignal(
        allowed=True,
        direction=TradeDirection.CALLS,
        reason="All 4 gates passed",
        gates_passed=gates_passed,
    )


def check_short_entry(state: MarketState) -> EntrySignal:
    """
    Full short (puts) entry gate check. Mirror of long.

    Gate 1 — HTF: 1h bearish, no conflict
    Gate 2 — Setup: 15m BEARISH, EMA stack not bull_aligned
    Gate 3 — Momentum: RSI 35-65, MACD hist < 0, price below VWAP
    Gate 4 — Timing: 5m close below ema_9, no support within 1 ATR
    """
    gates_passed = 0

    # ── Gate 0: Regime + Session + Intent pre-checks ─────
    if hasattr(state, "regime") and state.regime and not state.regime.entry_permitted:
        return EntrySignal(False, None,
            f"Regime gate: {state.regime.regime.value} — entries not permitted",
            gates_passed)
    if hasattr(state, "session") and state.session and not state.session.entries_allowed:
        return EntrySignal(False, None,
            f"Session gate: {state.session.phase.value} — {state.session.phase_note}",
            gates_passed)
    if hasattr(state, "intent") and state.intent:
        if state.intent.directional_bias == "bullish":
            return EntrySignal(False, None,
                "Intent gate: session bias is bullish — puts blocked",
                gates_passed)

    # ── Gate 1 ────────────────────────────────
    allowed, reason = is_entry_allowed(state.trends, Direction.BEARISH)
    if not allowed:
        return EntrySignal(False, None, f"Gate 1 (HTF): {reason}", gates_passed)
    gates_passed += 1

    # ── Gate 2 ────────────────────────────────
    trend_15m = state.trends.get("15m")
    ind_15m   = state.indicators.get("15m")
    if not trend_15m or not ind_15m:
        return EntrySignal(False, None, "Gate 2: no 15m data", gates_passed)

    if trend_15m.direction != Direction.BEARISH:
        return EntrySignal(
            False, None,
            f"Gate 2 (15m): trend is {trend_15m.direction.value}, need bearish",
            gates_passed,
        )
    if ind_15m.ema_stack == "bull_aligned":
        return EntrySignal(False, None, "Gate 2 (15m): EMA stack bull-aligned", gates_passed)
    gates_passed += 1

    # ── Gate 3 ────────────────────────────────
    ind_5m = state.indicators.get("5m")
    if not ind_5m:
        return EntrySignal(False, None, "Gate 3: no 5m data", gates_passed)

    rsi_fail, rsi_reason = _check_rsi_short(ind_5m)
    if rsi_fail:
        return EntrySignal(False, None, f"Gate 3 (momentum): {rsi_reason}", gates_passed)

    if ind_5m.macd_hist is not None and ind_5m.macd_hist > 0:
        return EntrySignal(
            False, None,
            f"Gate 3 (momentum): MACD hist positive ({ind_5m.macd_hist:.4f})",
            gates_passed,
        )

    if ind_5m.price_vs_vwap == "above":
        return EntrySignal(False, None, "Gate 3 (momentum): price above VWAP", gates_passed)
    gates_passed += 1

    # ── Gate 4 ────────────────────────────────
    ind_1m = state.indicators.get("1m")
    timing_ok, timing_reason = _check_timing_short(ind_5m, ind_1m, state)
    if not timing_ok:
        return EntrySignal(False, None, f"Gate 4 (timing): {timing_reason}", gates_passed)
    gates_passed += 1

    return EntrySignal(
        allowed=True,
        direction=TradeDirection.PUTS,
        reason="All 4 gates passed",
        gates_passed=gates_passed,
    )


# ──────────────────────────────────────────────
# Exit rules
# ──────────────────────────────────────────────

@dataclass
class ExitSignal:
    exit_now:  bool
    reason:    Optional[ExitReason]
    detail:    str
    urgency:   str = "normal"   # "normal" | "immediate"


def check_exit_conditions(
    trade: TradeState,
    state: MarketState,
) -> ExitSignal:
    """
    Evaluate all exit conditions for an active trade.
    Returns the first triggered condition in priority order:

    Priority:
      1. Stop loss          (hard stop, immediate)
      2. Structure loss     (price closes through stop level)
      3. Trend flip         (5m or 15m reversal, immediate)
      4. Momentum exhaustion (RSI extreme + MACD diverging)
      5. Trailing stop      (activated after threshold gain)
      6. Profit target      (full exit at target)
    """
    price = trade.current_price
    if price == 0.0:
        return ExitSignal(False, None, "no price data")

    # ── 1. Stop loss ──────────────────────────
    sl_hit, sl_detail = _check_stop_loss(trade, price)
    if sl_hit:
        return ExitSignal(True, ExitReason.STOP_LOSS, sl_detail, urgency="immediate")

    # ── 2. Structure loss ─────────────────────
    struct_hit, struct_detail = _check_structure_loss(trade, state)
    if struct_hit:
        return ExitSignal(True, ExitReason.STRUCTURE_LOSS, struct_detail, urgency="immediate")

    # ── 3. Trend flip ─────────────────────────
    flip_hit, flip_detail = _check_trend_flip(trade, state)
    if flip_hit:
        return ExitSignal(True, ExitReason.TREND_FLIP, flip_detail, urgency="immediate")

    # ── 4. Momentum exhaustion ────────────────
    exhaust_hit, exhaust_detail = _check_momentum_exhaustion(trade, state)
    if exhaust_hit:
        return ExitSignal(True, ExitReason.MOMENTUM_EXHAUSTED, exhaust_detail)

    # ── 5. Trailing stop ──────────────────────
    trail_hit, trail_detail = _check_trailing_stop(trade, price)
    if trail_hit:
        return ExitSignal(True, ExitReason.TRAILING_STOP, trail_detail)

    # ── 6. Profit target ──────────────────────
    target_hit, target_detail = _check_profit_target(trade, price)
    if target_hit:
        return ExitSignal(True, ExitReason.PROFIT_TARGET, target_detail)

    return ExitSignal(False, None, "no exit conditions met")


# ──────────────────────────────────────────────
# Weakening / flip detection
# ──────────────────────────────────────────────

def detect_trend_weakening(
    trend: TrendState,
    ind:   IndicatorSnapshot,
    prev_ind: Optional[IndicatorSnapshot] = None,
) -> Tuple[bool, str]:
    """
    Trend weakening: 2 of 3 signals required.

    1. RSI overbought (calls) or oversold (puts) while MACD hist negative/positive
    2. MACD histogram shrinking (current < previous)
    3. Price within EMA-9 proximity without expanding (EMA stack went mixed)

    Returns (is_weakening, reason).
    """
    signals = []

    # Signal 1: RSI extreme + opposing MACD
    if trend.direction == Direction.BULLISH:
        if ind.rsi and ind.rsi > RSI_OVERBOUGHT and ind.macd_hist and ind.macd_hist < 0:
            signals.append("RSI overbought with negative MACD hist")
    elif trend.direction == Direction.BEARISH:
        if ind.rsi and ind.rsi < RSI_OVERSOLD and ind.macd_hist and ind.macd_hist > 0:
            signals.append("RSI oversold with positive MACD hist")

    # Signal 2: MACD histogram shrinking
    if (prev_ind
            and ind.macd_hist is not None
            and prev_ind.macd_hist is not None):
        if trend.direction == Direction.BULLISH:
            if ind.macd_hist < prev_ind.macd_hist and ind.macd_hist > 0:
                signals.append(f"MACD hist shrinking ({prev_ind.macd_hist:.4f} -> {ind.macd_hist:.4f})")
        elif trend.direction == Direction.BEARISH:
            if ind.macd_hist > prev_ind.macd_hist and ind.macd_hist < 0:
                signals.append(f"MACD hist shrinking ({prev_ind.macd_hist:.4f} -> {ind.macd_hist:.4f})")

    # Signal 3: EMA stack degrading
    if ind.ema_stack == "mixed" and trend.strength in (TrendStrength.STRONG, TrendStrength.MODERATE):
        signals.append("EMA stack mixed despite prior strong trend")

    if len(signals) >= 2:
        return True, " | ".join(signals)
    return False, ""


def detect_trend_break(
    trend:     TrendState,
    ind:       IndicatorSnapshot,
    prev_ind:  Optional[IndicatorSnapshot] = None,
) -> Tuple[bool, str]:
    """
    Trend break: the trend's structural integrity is compromised.

    Soft break (warning): price tests key level but hasn't closed through.
    Hard break (confirmed): close through last swing high/low.

    We return the hard break here (close-based, confirmed).
    """
    price = ind.close

    if trend.direction == Direction.BULLISH and trend.last_swing_low:
        if price < trend.last_swing_low:
            return True, f"Price {price:.2f} closed below swing low {trend.last_swing_low:.2f}"

    if trend.direction == Direction.BEARISH and trend.last_swing_high:
        if price > trend.last_swing_high:
            return True, f"Price {price:.2f} closed above swing high {trend.last_swing_high:.2f}"

    return False, ""


def detect_trend_flip(
    curr_trend: TrendState,
    prev_trend: Optional[TrendState],
    curr_ind:   IndicatorSnapshot,
    prev_ind:   Optional[IndicatorSnapshot] = None,
) -> Tuple[bool, str]:
    """
    Trend flip: a confirmed direction change.

    Confirmed when:
      1. Structure break on the timeframe (close through last swing)
      2. EMA 9 crossed EMA 20 in new direction
      3. (Optional) New direction held for 2+ candles

    We require conditions 1 AND 2.
    """
    if prev_trend is None:
        return False, ""

    # Direction changed
    if curr_trend.direction == prev_trend.direction:
        return False, ""
    if curr_trend.direction == Direction.NEUTRAL:
        return False, ""

    # Structure break confirms
    if not curr_trend.structure_break:
        return False, "direction changed but no structure break yet"

    # EMA 9 / 20 crossover
    ema9_cross = False
    if (curr_ind.ema_9 and curr_ind.ema_20 and prev_ind
            and prev_ind.ema_9 and prev_ind.ema_20):
        if (curr_trend.direction == Direction.BULLISH
                and curr_ind.ema_9 > curr_ind.ema_20
                and prev_ind.ema_9 <= prev_ind.ema_20):
            ema9_cross = True
        elif (curr_trend.direction == Direction.BEARISH
                and curr_ind.ema_9 < curr_ind.ema_20
                and prev_ind.ema_9 >= prev_ind.ema_20):
            ema9_cross = True

    if curr_trend.structure_break and ema9_cross:
        return True, (
            f"Trend flipped {prev_trend.direction.value} -> {curr_trend.direction.value}: "
            f"structure break + EMA 9/20 cross"
        )

    # Structure break alone is a soft flip (worth flagging)
    if curr_trend.structure_break:
        return True, (
            f"Trend flipped {prev_trend.direction.value} -> {curr_trend.direction.value}: "
            f"structure break (EMA cross pending)"
        )

    return False, ""


# ──────────────────────────────────────────────
# Exit sub-checks (private)
# ──────────────────────────────────────────────

def _check_stop_loss(trade: TradeState, price: float) -> Tuple[bool, str]:
    if trade.direction == TradeDirection.CALLS:
        if price <= trade.stop_loss:
            loss_pct = (trade.entry_price - price) / trade.entry_price * 100
            return True, f"Stop loss hit at {price:.2f} (-{loss_pct:.2f}%)"
    else:
        if price >= trade.stop_loss:
            loss_pct = (price - trade.entry_price) / trade.entry_price * 100
            return True, f"Stop loss hit at {price:.2f} (-{loss_pct:.2f}%)"
    return False, ""


def _check_structure_loss(trade: TradeState, state: MarketState) -> Tuple[bool, str]:
    """Price closed through stop level on 5m candle."""
    ind_5m = state.indicators.get("5m")
    if not ind_5m:
        return False, ""
    price = ind_5m.close

    if trade.direction == TradeDirection.CALLS and price <= trade.stop_loss:
        return True, f"5m close {price:.2f} below stop {trade.stop_loss:.2f}"
    if trade.direction == TradeDirection.PUTS and price >= trade.stop_loss:
        return True, f"5m close {price:.2f} above stop {trade.stop_loss:.2f}"
    return False, ""


def _check_trend_flip(trade: TradeState, state: MarketState) -> Tuple[bool, str]:
    """5m or 15m trend flipped against our position."""
    for tf in ("5m", "15m"):
        trend = state.trends.get(tf)
        if not trend:
            continue
        if trade.direction == TradeDirection.CALLS and trend.direction == Direction.BEARISH:
            if trend.structure_break:
                return True, f"{tf} trend flipped bearish with structure break"
        if trade.direction == TradeDirection.PUTS and trend.direction == Direction.BULLISH:
            if trend.structure_break:
                return True, f"{tf} trend flipped bullish with structure break"
    return False, ""


def _check_momentum_exhaustion(trade: TradeState, state: MarketState) -> Tuple[bool, str]:
    """RSI extreme + MACD diverging against position."""
    ind_5m = state.indicators.get("5m")
    if not ind_5m or not ind_5m.rsi or not ind_5m.macd_hist:
        return False, ""

    if trade.direction == TradeDirection.CALLS:
        if ind_5m.rsi > 78 and ind_5m.macd_hist < 0:
            return True, f"RSI exhausted ({ind_5m.rsi:.1f}) with negative MACD hist"
    else:
        if ind_5m.rsi < 22 and ind_5m.macd_hist > 0:
            return True, f"RSI exhausted ({ind_5m.rsi:.1f}) with positive MACD hist"
    return False, ""


def _check_trailing_stop(trade: TradeState, price: float) -> Tuple[bool, str]:
    if trade.trailing_stop is None:
        return False, ""
    if trade.direction == TradeDirection.CALLS:
        if price <= trade.trailing_stop:
            return True, f"Trailing stop {trade.trailing_stop:.2f} hit at {price:.2f}"
    else:
        if price >= trade.trailing_stop:
            return True, f"Trailing stop {trade.trailing_stop:.2f} hit at {price:.2f}"
    return False, ""


def _check_profit_target(trade: TradeState, price: float) -> Tuple[bool, str]:
    if trade.direction == TradeDirection.CALLS:
        if price >= trade.profit_target:
            gain_pct = (price - trade.entry_price) / trade.entry_price * 100
            return True, f"Profit target {trade.profit_target:.2f} reached (+{gain_pct:.2f}%)"
    else:
        if price <= trade.profit_target:
            gain_pct = (trade.entry_price - price) / trade.entry_price * 100
            return True, f"Profit target {trade.profit_target:.2f} reached (+{gain_pct:.2f}%)"
    return False, ""


# ──────────────────────────────────────────────
# RSI / timing helpers
# ──────────────────────────────────────────────

def _check_rsi_long(ind: IndicatorSnapshot) -> Tuple[bool, str]:
    """Returns (failed, reason). Fails if RSI is overbought at entry."""
    if ind.rsi is None:
        return False, ""
    if ind.rsi > 68:
        return True, f"RSI overbought at entry ({ind.rsi:.1f} > 68)"
    if ind.rsi < 30:
        return True, f"RSI oversold ({ind.rsi:.1f}) — likely bear capitulation, not a long setup"
    return False, ""


def _check_rsi_short(ind: IndicatorSnapshot) -> Tuple[bool, str]:
    """Returns (failed, reason). Fails if RSI is oversold at entry."""
    if ind.rsi is None:
        return False, ""
    if ind.rsi < 32:
        return True, f"RSI oversold at entry ({ind.rsi:.1f} < 32)"
    if ind.rsi > 70:
        return True, f"RSI overbought ({ind.rsi:.1f}) — likely bull exhaustion, not a short setup"
    return False, ""


def _check_timing_long(
    ind_5m:  IndicatorSnapshot,
    ind_1m:  Optional[IndicatorSnapshot],
    state:   MarketState,
) -> Tuple[bool, str]:
    """5m close above EMA-9, no resistance directly overhead."""
    price = ind_5m.close
    ema9  = ind_5m.ema_9

    if ema9 and price < ema9:
        return False, f"5m close {price:.2f} below EMA-9 {ema9:.2f}"

    # Check for resistance within 1 ATR
    if ind_5m.atr:
        immediate_resistance = [
            r for r in state.resistance_levels
            if r.price > price and (r.price - price) < ind_5m.atr
        ]
        if immediate_resistance:
            r = immediate_resistance[0]
            return False, f"Resistance at {r.price:.2f} within 1 ATR above entry"

    return True, "timing ok"


def _check_timing_short(
    ind_5m:  IndicatorSnapshot,
    ind_1m:  Optional[IndicatorSnapshot],
    state:   MarketState,
) -> Tuple[bool, str]:
    """5m close below EMA-9, no support directly below."""
    price = ind_5m.close
    ema9  = ind_5m.ema_9

    if ema9 and price > ema9:
        return False, f"5m close {price:.2f} above EMA-9 {ema9:.2f}"

    if ind_5m.atr:
        immediate_support = [
            s for s in state.support_levels
            if s.price < price and (price - s.price) < ind_5m.atr
        ]
        if immediate_support:
            s = immediate_support[0]
            return False, f"Support at {s.price:.2f} within 1 ATR below entry"

    return True, "timing ok"
