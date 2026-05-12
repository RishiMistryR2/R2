"""
cli/dashboard.py — terminal dashboard renderer for R2.

Pure formatting functions.  No I/O, no side effects.

Layout (top to bottom):
  ┌─ HEADER: symbol, time, session phase ─────────────────┐
  │  MARKET STATUS: price, regime, narrative, VIX          │
  │  READINESS + CONTEXT BIAS                              │
  │  SETUP SCORE (if any)                                  │
  │  ACTIVE TRADE STATUS (if any)                          │
  │  WARNINGS                                              │
  │  ACTIVE PATTERNS                                       │
  │  RECENT EVENTS                                         │
  └────────────────────────────────────────────────────────┘

Design principles:
  - Clean, information-dense, no decorative noise
  - Every line has a purpose
  - Calm tone: data, not drama
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from cli.warnings import collect_warnings, render_warnings
from cli.setup_display import render_setup_summary_line, render_for_against, render_stop_summary


# Width constant — standard 80-col terminal
WIDTH = 72


# ──────────────────────────────────────────────
# Main render entry point
# ──────────────────────────────────────────────

def render_dashboard(
    state,
    score           = None,
    context_state   = None,
    session_memory  = None,
    active_patterns = None,
    recent_events:  List[str] = None,
    now:            Optional[datetime] = None,
) -> str:
    """
    Render the complete R2 dashboard to a string.
    Caller prints; this function only formats.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    sections: List[str] = []
    sections.append(_render_header(state, now))
    sections.append(_render_market_status(state, context_state))
    sections.append(_render_readiness(state, context_state))

    if score is not None:
        direction = getattr(state, "setup_direction", None)
        sections.append(_render_score_section(score, direction))

    if state.active_trade:
        sections.append(_render_active_trade(state.active_trade))

    warnings = collect_warnings(
        state          = state,
        context_state  = context_state,
        session_memory = session_memory,
        score          = score,
        active_patterns = active_patterns or [],
    )
    sections.append(_render_warnings_section(warnings))

    if active_patterns:
        sections.append(_render_patterns_section(active_patterns))

    if recent_events:
        sections.append(_render_events_section(recent_events))

    return _box("\n".join(s for s in sections if s))


# ──────────────────────────────────────────────
# Section renderers (all return str)
# ──────────────────────────────────────────────

def _render_header(state, now: datetime) -> str:
    symbol = state.symbol if state.symbol else "SPY"
    ts     = now.strftime("%H:%M:%S UTC")
    phase  = ""
    if state.session:
        phase = f"  [{state.session.phase.value.upper()}]"
    return f"  R2  ·  {symbol}{phase}  ·  {ts}"


def _render_market_status(state, context_state) -> str:
    lines: List[str] = []
    lines.append(_section_label("MARKET"))

    # Price
    ind_5m = state.indicators.get("5m") if state.indicators else None
    if ind_5m and ind_5m.close:
        price = ind_5m.close
        vwap  = ind_5m.vwap
        vwap_str = ""
        if vwap:
            diff     = price - vwap
            rel      = "above" if diff > 0 else "below"
            vwap_str = f"  VWAP {vwap:.2f} ({rel} {abs(diff):.2f})"
        lines.append(f"  Price  : {price:.2f}{vwap_str}")

    # Regime
    if state.regime:
        regime = state.regime.regime.value.replace("_", " ").title()
        conf   = f"  ({state.regime.confidence:.0%})"
        mult   = f"  {state.regime.score_multiplier:.2f}×" if state.regime.score_multiplier != 1.0 else ""
        lines.append(f"  Regime : {regime}{conf}{mult}")

    # Narrative
    if state.narrative:
        narr = state.narrative.narrative.value.replace("_", " ").title()
        htf  = getattr(state, "htf_bias", "") or ""
        htf_str = f"  · HTF {htf}" if htf and htf != "neutral" else ""
        lines.append(f"  Outlook: {narr}{htf_str}")

    # VIX / context
    if context_state:
        if context_state.vix_level:
            lines.append(f"  VIX    : {context_state.vix_level:.1f}  [{context_state.vix_regime}]")
        if context_state.news_sentiment != 0:
            sentiment = context_state.context_bias
            score_str = f"{context_state.news_sentiment:+.2f}"
            lines.append(f"  News   : {sentiment}  ({score_str})")

    return "\n".join(lines)


def _render_readiness(state, context_state) -> str:
    lines: List[str] = []
    lines.append(_section_label("STATUS"))

    if state.session:
        r  = state.session.readiness.value.replace("_", " ").title()
        me = "✓" if state.session.market_open else "✗"
        ea = "✓" if state.session.entries_allowed else "✗"
        lines.append(f"  Readiness : {r}")
        lines.append(f"  Market    : {me}  Entries: {ea}")

    ctx_bias = getattr(state, "context_bias", "neutral")
    ctx_risk = getattr(state, "context_risk_level", "normal")
    lines.append(f"  Ctx bias  : {ctx_bias}  [{ctx_risk} volatility]")

    return "\n".join(lines)


def _render_score_section(score, direction) -> str:
    lines: List[str] = []
    lines.append(_section_label("SETUP SCORE"))
    lines.append(render_setup_summary_line(score, direction))
    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    bar = _score_bar(effective)
    lines.append(f"  {bar}")

    # Component minibar
    comps = [
        ("Align",     score.trend_alignment,   30),
        ("Momentum",  score.momentum_quality,  25),
        ("Structure", score.structure_context, 25),
        ("Timing",    score.entry_timing,       20),
    ]
    for label, val, max_val in comps:
        pct      = int(val / max_val * 10) if max_val else 0
        bar_part = "█" * pct + "·" * (10 - pct)
        lines.append(f"  {label:<10} {val:>4.0f}/{max_val}  {bar_part}")

    return "\n".join(lines)


def _render_active_trade(trade) -> str:
    lines: List[str] = []
    lines.append(_section_label("ACTIVE TRADE"))
    lifecycle  = getattr(trade.lifecycle, "value", "—") if trade.lifecycle else "—"
    direction  = trade.direction.value.upper() if trade.direction else "—"
    instrument = trade.instrument_type.value if trade.instrument_type else "—"
    lines.append(f"  {direction}  ·  {instrument}  ·  [{lifecycle.upper()}]")
    entry = trade.underlying_entry_price or trade.entry_price
    if entry:
        lines.append(f"  Entry      : {entry:.2f}")
    lines.append(render_stop_summary(trade))
    pnl = getattr(trade, "pnl_pct", None)
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"  P&L        : {sign}{pnl:.2f}%")
    return "\n".join(lines)


def _render_warnings_section(warnings) -> str:
    lines: List[str] = []
    lines.append(_section_label("WARNINGS"))
    lines.append(render_warnings(warnings))
    return "\n".join(lines)


def _render_patterns_section(patterns: List[dict]) -> str:
    if not patterns:
        return ""
    lines: List[str] = []
    lines.append(_section_label("ACTIVE PATTERNS"))
    for p in patterns[:4]:
        pid  = p.get("pattern_id", p.get("description", "unknown"))
        conf = p.get("confidence_score", 0.0)
        occ  = p.get("occurrence_count", p.get("occurrences", 0))
        wr   = p.get("win_rate", 0.0)
        lines.append(
            f"  {pid[:36]:<36}  "
            f"conf {conf:.0%}  wr {wr:.0%}  n={occ}"
        )
    return "\n".join(lines)


def _render_events_section(events: List[str]) -> str:
    if not events:
        return ""
    lines: List[str] = []
    lines.append(_section_label("RECENT EVENTS"))
    for ev in events[-4:]:
        lines.append(f"  · {ev}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Compact single-line renderers (for narrow views)
# ──────────────────────────────────────────────

def render_regime_line(state) -> str:
    if not state.regime:
        return "Regime: —"
    regime = state.regime.regime.value.replace("_", " ").title()
    conf   = f"{state.regime.confidence:.0%}"
    return f"Regime: {regime}  {conf}"


def render_price_line(state) -> str:
    ind = state.indicators.get("5m") if state.indicators else None
    if not ind:
        return "Price: —"
    price = ind.close
    vwap  = ind.vwap
    if vwap:
        diff = price - vwap
        return f"Price: {price:.2f}  VWAP {vwap:.2f}  ({diff:+.2f})"
    return f"Price: {price:.2f}"


def render_session_line(state) -> str:
    if not state.session:
        return "Session: —"
    phase = state.session.phase.value.upper()
    r     = state.session.readiness.value.replace("_", " ").title()
    open_ = "Open" if state.session.market_open else "Closed"
    return f"Session: {phase}  ·  {r}  ·  Market {open_}"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _section_label(label: str) -> str:
    pad = max(0, WIDTH - len(label) - 4)
    return f"\n  ── {label} {'─' * pad}"


def _score_bar(score: float, width: int = 40) -> str:
    filled = int(score / 100 * width)
    filled = max(0, min(width, filled))
    empty  = width - filled
    bar    = "█" * filled + "·" * empty
    return f"[{bar}]  {score:.0f}"


def _box(content: str) -> str:
    border = "═" * (WIDTH + 2)
    return f"╔{border}╗\n{content}\n╚{border}╝"