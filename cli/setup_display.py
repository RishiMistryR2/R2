"""
cli/setup_display.py — setup presentation for the R2 terminal interface.

Pure formatting functions.  No I/O.

Philosophy:
  R2 explains setups the way a disciplined trading partner would:
  factual, concise, honest about both the case FOR and AGAINST.
  No hype, no urgency, no emotional language.
"""

from __future__ import annotations

from typing import List, Optional


# ──────────────────────────────────────────────
# Top-level setup card
# ──────────────────────────────────────────────

def render_setup_card(
    state,
    score,
    direction=None,
    context_state=None,
    active_trade=None,
) -> str:
    """
    Render a full setup card: score, direction, levels, reasoning.

    Parameters
    ----------
    state         : MarketState
    score         : SetupScore
    direction     : TradeDirection | None
    context_state : ContextState | None
    active_trade  : TradeState | None
    """
    lines: List[str] = []
    lines.append(_divider("SETUP ANALYSIS"))

    if score is None or (score.total == 0 and score.adjusted_total == 0):
        lines.append("  No setup currently detected.")
        return "\n".join(lines)

    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    grade     = score.grade()
    dir_str   = direction.value.upper() if direction else "—"

    lines.append(f"  Direction  : {dir_str}")
    lines.append(f"  Score      : {effective:.0f}/100  [{grade}]")
    lines.append(f"  Components :")
    lines.append(f"    Trend alignment    {score.trend_alignment:.0f}/30")
    lines.append(f"    Momentum quality   {score.momentum_quality:.0f}/25")
    lines.append(f"    Structure context  {score.structure_context:.0f}/25")
    lines.append(f"    Entry timing       {score.entry_timing:.0f}/20")

    # Multipliers / adjustments
    lines.append(_divider("SCORE ADJUSTMENTS", width=48, char="-"))
    if score.regime_multiplier != 1.0:
        lines.append(f"  Regime mult    : {score.regime_multiplier:.2f}×")
    if score.session_multiplier != 1.0:
        lines.append(f"  Session mult   : {score.session_multiplier:.2f}×")
    if score.readiness_penalty:
        lines.append(f"  Readiness pen  : -{score.readiness_penalty:.1f} pts")
    if score.context_adjustment:
        sign = "+" if score.context_adjustment > 0 else ""
        lines.append(f"  Context adj    : {sign}{score.context_adjustment:.1f} pts")

    # Alignment and reasoning
    lines.append(_divider("REASONING", width=48, char="-"))
    lines.extend(_render_reasoning(score, state, context_state))

    # Stop levels
    lines.append(_divider("LEVELS", width=48, char="-"))
    lines.extend(_render_levels(state, direction))

    return "\n".join(lines)


def render_setup_summary_line(score, direction=None) -> str:
    """One-line setup summary for the dashboard header."""
    if score is None or score.total == 0:
        return "  No setup"
    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    grade     = score.grade()
    dir_str   = direction.value.upper() if direction else "—"
    return f"  {dir_str}  {grade}  ({effective:.0f}/100)"


def render_for_against(score, state, context_state=None) -> str:
    """
    Render FOR / AGAINST reasoning in a structured two-column style.
    This is R2's "trading partner" voice: calm, analytical, balanced.
    """
    for_points:     List[str] = []
    against_points: List[str] = []

    # FOR
    if score.trend_alignment >= 24:
        for_points.append("Strong multi-timeframe trend alignment")
    elif score.trend_alignment >= 18:
        for_points.append("Adequate trend alignment")

    if score.momentum_quality >= 20:
        for_points.append("Momentum confirming direction")

    if score.structure_context >= 20:
        for_points.append("Clean structure, clear invalidation level")

    if score.entry_timing >= 15:
        for_points.append("Good entry timing on current candle")

    ctx_adj = getattr(score, "context_adjustment", 0)
    if ctx_adj > 0:
        for_points.append("Context and news sentiment aligned")

    if context_state and getattr(context_state, "context_bias", "neutral") != "neutral":
        bias = context_state.context_bias
        for_points.append(f"Macro context: {bias}")

    # AGAINST
    if score.trend_alignment < 15:
        against_points.append("Trend alignment below standard")

    if score.momentum_quality < 12:
        against_points.append("Momentum not confirming")

    if score.entry_timing < 10:
        against_points.append("Entry timing is marginal")

    if score.readiness_penalty > 0:
        against_points.append("Score reduced by readiness penalty")

    if ctx_adj < 0:
        against_points.append("Context or macro working against")

    if context_state and getattr(context_state, "elevated_risk", False):
        against_points.append(f"Elevated volatility risk: {context_state.volatility_risk_level}")

    if context_state and context_state.upcoming_events:
        against_points.append("Active macro events in window")

    if not for_points:
        for_points.append("No strong confirming factors identified")
    if not against_points:
        against_points.append("No notable opposing factors")

    lines: List[str] = []
    lines.append("  FOR:")
    for pt in for_points[:4]:
        lines.append(f"    + {pt}")
    lines.append("  AGAINST:")
    for pt in against_points[:4]:
        lines.append(f"    - {pt}")
    return "\n".join(lines)


def render_rr_summary(
    entry:    float,
    stop:     float,
    target:   float,
    rr_ratio: float,
    min_rr:   float = 1.5,
) -> str:
    """Render a compact R:R summary line."""
    stop_dist  = abs(entry - stop)
    stop_pct   = stop_dist / entry * 100 if entry else 0.0
    reward_pts = abs(target - entry)
    ok_marker  = "✓" if rr_ratio >= min_rr else "✗"
    return (
        f"  R:R  {rr_ratio:.2f}  {ok_marker}   "
        f"Stop: {stop:.2f} ({stop_pct:.2f}%)   "
        f"Target: {target:.2f} (+{reward_pts:.2f})"
    )


def render_stop_summary(trade_state) -> str:
    """Render active stop status from a TradeState."""
    if trade_state is None:
        return "  No active trade"
    lines: List[str] = []
    entry = trade_state.underlying_entry_price or trade_state.entry_price
    stop  = trade_state.underlying_stop_price
    tgt   = trade_state.profit_target
    if entry and stop:
        dist_pct = abs(entry - stop) / entry * 100 if entry else 0
        lines.append(f"  Underlying stop : {stop:.2f}  ({dist_pct:.2f}% from entry)")
    if trade_state.option_entry_price and trade_state.premium_emergency_stop_pct:
        lines.append(
            f"  Premium stop    : -{trade_state.premium_emergency_stop_pct:.0f}%  "
            f"(Layer B emergency)"
        )
    if trade_state.time_stop_config:
        w = trade_state.time_stop_config.expected_move_window_minutes
        lines.append(f"  Time stop       : {w} min window")
    if tgt:
        lines.append(f"  Target          : {tgt:.2f}")
    # Flag any stop already triggered
    if trade_state.thesis_stop_triggered:
        lines.append("  ✖ THESIS STOP TRIGGERED")
    if trade_state.time_stop_triggered:
        lines.append("  ✖ TIME STOP TRIGGERED")
    if trade_state.premium_stop_triggered:
        lines.append("  ✖ PREMIUM STOP TRIGGERED")
    return "\n".join(lines) if lines else "  No stop data"


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _render_reasoning(score, state, context_state) -> List[str]:
    lines: List[str] = []
    if score.alignment_detail:
        lines.append(f"  Alignment  : {score.alignment_detail}")
    if score.momentum_detail:
        lines.append(f"  Momentum   : {score.momentum_detail}")
    if score.structure_detail:
        lines.append(f"  Structure  : {score.structure_detail}")
    if score.timing_detail:
        lines.append(f"  Timing     : {score.timing_detail}")
    if not any([score.alignment_detail, score.momentum_detail,
                score.structure_detail, score.timing_detail]):
        lines.append("  (Detail not available)")
    return lines


def _render_levels(state, direction) -> List[str]:
    lines: List[str] = []
    ind_5m = state.indicators.get("5m") if state.indicators else None
    if ind_5m:
        price = ind_5m.close
        vwap  = ind_5m.vwap
        atr   = ind_5m.atr
        lines.append(f"  Price (5m) : {price:.2f}")
        if vwap:
            diff = price - vwap
            lines.append(f"  VWAP       : {vwap:.2f}  ({diff:+.2f})")
        if atr:
            lines.append(f"  ATR (5m)   : {atr:.3f}")
    else:
        lines.append("  (No 5m indicator data)")
    # Swing levels from trends
    trend_15m = state.trends.get("15m") if state.trends else None
    if trend_15m:
        swing_low  = getattr(trend_15m, "last_swing_low", None)
        swing_high = getattr(trend_15m, "last_swing_high", None)
        if swing_low:
            lines.append(f"  Swing low  : {swing_low:.2f}")
        if swing_high:
            lines.append(f"  Swing high : {swing_high:.2f}")
    return lines


def _divider(label: str = "", width: int = 56, char: str = "─") -> str:
    if label:
        pad   = max(0, width - len(label) - 4)
        left  = pad // 2
        right = pad - left
        return f"  {char * left} {label} {char * right}"
    return f"  {char * width}"