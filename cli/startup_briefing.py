"""
cli/startup_briefing.py — pre-market briefing renderer for R2.

Consumes:
  - StartupBriefing   (memory.startup_context)
  - SessionMemory     (memory.session_memory)
  - PatternMemory     (memory.pattern_memory)
  - MarketState       (engine.models)

Pure formatting functions.  No I/O.

The briefing is designed to orient the trader at session start:
  - What happened yesterday
  - What patterns are active
  - What risks to watch today
  - What the market looks like right now
Concise.  No padding.  Every line earns its place.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional


WIDTH = 72


def render_startup_briefing(
    briefing,
    state         = None,
    context_state = None,
) -> str:
    """
    Render the full pre-market briefing to a string.

    Parameters
    ----------
    briefing      : StartupBriefing
    state         : MarketState | None  (current market snapshot if available)
    context_state : ContextState | None
    """
    sections: List[str] = []
    sections.append(_header(briefing))
    sections.append(_previous_session(briefing))
    sections.append(_recent_performance(briefing))
    sections.append(_top_risks(briefing))
    sections.append(_active_patterns(briefing))
    if state is not None:
        sections.append(_current_market(state, context_state))
    sections.append(_footer(briefing))

    return "\n".join(s for s in sections if s)


# ──────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────

def _header(briefing) -> str:
    today  = briefing.session_date
    gen_at = briefing.generated_at[:19].replace("T", " ")
    symbol = "SPY"
    lines  = [
        _divider("R2 PRE-MARKET BRIEFING"),
        f"  Date     : {today}   Symbol : {symbol}",
        f"  Prepared : {gen_at} UTC",
    ]
    return "\n".join(lines)


def _previous_session(briefing) -> str:
    if not briefing.prev_session_date:
        return _section("PREVIOUS SESSION") + "\n  No prior session data."

    pnl_sign = "+" if briefing.prev_session_pnl_pct >= 0 else ""
    wr       = briefing.prev_session_win_rate * 100
    lines    = [
        _section("PREVIOUS SESSION"),
        f"  Date     : {briefing.prev_session_date}",
        f"  Trades   : {briefing.prev_session_trades}  "
        f"Win rate : {wr:.0f}%  "
        f"P&L : {pnl_sign}{briefing.prev_session_pnl_pct:.2f}%",
    ]
    if briefing.prev_session_regime:
        lines.append(f"  Regime   : {briefing.prev_session_regime.replace('_', ' ').title()}")
    if briefing.prev_session_lesson:
        lines.append(f"  Lesson   : {briefing.prev_session_lesson}")
    return "\n".join(lines)


def _recent_performance(briefing) -> str:
    if briefing.recent_trades_n == 0:
        return _section("RECENT PERFORMANCE") + "\n  No trade history available."

    wr   = briefing.recent_win_rate * 100
    exp  = briefing.recent_expectancy
    sign = "+" if exp >= 0 else ""
    lines = [
        _section("RECENT PERFORMANCE"),
        f"  Last {briefing.recent_trades_n} trades  ·  "
        f"Win rate {wr:.0f}%  ·  "
        f"Expectancy {sign}{exp:.3f}%",
    ]
    if briefing.recent_best_setup:
        lines.append(f"  Best setup   : {briefing.recent_best_setup}")
    if briefing.recent_worst_setup and briefing.recent_worst_setup != briefing.recent_best_setup:
        lines.append(f"  Weakest setup: {briefing.recent_worst_setup}")
    return "\n".join(lines)


def _top_risks(briefing) -> str:
    lines = [_section("TODAY'S TOP RISKS")]
    if not briefing.top_risks:
        lines.append("  No elevated risks identified.")
    else:
        for i, risk in enumerate(briefing.top_risks, 1):
            lines.append(f"  {i}. {risk}")
    return "\n".join(lines)


def _active_patterns(briefing) -> str:
    market   = briefing.active_market_patterns
    behavior = briefing.active_patterns
    mistakes = briefing.active_mistakes

    if not market and not behavior and not mistakes:
        return _section("PATTERNS") + "\n  No surfaced patterns."

    lines = [_section("PATTERNS")]

    if market:
        lines.append("  Market patterns:")
        for p in market[:3]:
            pid = p.get("pattern_id", p.get("description", "—"))
            wr  = p.get("win_rate", 0.0)
            occ = p.get("occurrence_count", p.get("occurrences", 0))
            lines.append(f"    · {pid}  (wr {wr:.0%}, n={occ})")

    if behavior:
        lines.append("  Behavioral patterns:")
        for p in behavior[:2]:
            desc = p.get("description", "—")
            conf = p.get("confidence_score", 0.0)
            lines.append(f"    · {desc}  (conf {conf:.0%})")

    if mistakes:
        lines.append("  Recurring mistakes:")
        for p in mistakes[:2]:
            desc = p.get("description", "—")
            occ  = p.get("occurrences", 0)
            lines.append(f"    ⚠ {desc}  ({occ}×)")

    return "\n".join(lines)


def _current_market(state, context_state) -> str:
    lines = [_section("CURRENT MARKET")]

    ind_5m = state.indicators.get("5m") if state.indicators else None
    if ind_5m and ind_5m.close:
        price = ind_5m.close
        vwap  = ind_5m.vwap
        if vwap:
            diff = price - vwap
            rel  = "above" if diff > 0 else "below"
            lines.append(f"  Price : {price:.2f}  ·  VWAP {vwap:.2f} ({rel} by {abs(diff):.2f})")
        else:
            lines.append(f"  Price : {price:.2f}")

    if state.regime:
        regime = state.regime.regime.value.replace("_", " ").title()
        conf   = f"{state.regime.confidence:.0%}"
        lines.append(f"  Regime: {regime}  {conf}")

    if state.narrative:
        narr = state.narrative.narrative.value.replace("_", " ").title()
        lines.append(f"  Trend : {narr}")

    if context_state:
        ctx_bias = context_state.context_bias
        ctx_risk = context_state.volatility_risk_level
        lines.append(f"  Context: {ctx_bias}  [{ctx_risk}]")
        if context_state.elevated_risk:
            lines.append(f"  ⚠ {context_state.risk_note or 'Elevated risk environment'}")

    return "\n".join(lines)


def _footer(briefing) -> str:
    focus = briefing.user_focus
    lines = [_divider()]
    if focus:
        lines.append(f"  Focus today: {focus}")
    if briefing.user_restrictions:
        restrictions = ", ".join(briefing.user_restrictions)
        lines.append(f"  Restrictions: {restrictions}")
    lines.append("  Trade the process.  Follow the rules.")
    lines.append(_divider())
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────

def _section(label: str) -> str:
    pad = max(0, WIDTH - len(label) - 4)
    return f"\n  ── {label} {'─' * pad}"


def _divider(label: str = "") -> str:
    if label:
        pad   = max(0, WIDTH - len(label) - 4)
        left  = pad // 2
        right = pad - left
        return f"  {'═' * left} {label} {'═' * right}"
    return f"  {'═' * WIDTH}"


def render_briefing_summary_line(briefing) -> str:
    """One-line summary for embedding in the dashboard header."""
    if not briefing.prev_session_date:
        return "No prior session"
    wr   = briefing.prev_session_win_rate * 100
    pnl  = briefing.prev_session_pnl_pct
    sign = "+" if pnl >= 0 else ""
    risks = len(briefing.top_risks)
    return (
        f"Prev: {briefing.prev_session_date}  "
        f"{wr:.0f}% WR  P&L {sign}{pnl:.2f}%  "
        f"Risks: {risks}"
    )