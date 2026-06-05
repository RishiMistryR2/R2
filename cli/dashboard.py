"""
cli/dashboard.py — terminal dashboard renderer for R2.

Pure formatting functions.  No I/O, no side effects.

Recent Events:
  Long entries (AI commentary) are word-wrapped across multiple lines
  within the dashboard box rather than truncated.  The first line of each
  event is prefixed with "  · "; continuation lines are indented with "    "
  so they are visually associated with their parent entry.
  Up to MAX_RECENT_EVENTS events are shown, most recent last.

Price display (MARKET section):
  Uses current_price when stamped by build_live_state (trade/quote),
  falls back to 5m bar close.  Source, bid/ask, spread, and last-update
  time (ET) are shown for live-mode verification.

Session display (STATUS section):
  Uses state.market_session (MarketSession) for authoritative labels.
"""

from __future__ import annotations

import textwrap
import re
from datetime import datetime, timezone
from typing import List, Optional

from cli.warnings import collect_warnings, render_warnings
from cli.setup_display import render_setup_summary_line, render_for_against, render_stop_summary

WIDTH              = 72
MAX_RECENT_EVENTS  = 4    # how many events to show in the dashboard


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
        state=state, context_state=context_state,
        session_memory=session_memory, score=score,
        active_patterns=active_patterns or [],
    )
    sections.append(_render_warnings_section(warnings))

    if active_patterns:
        sections.append(_render_patterns_section(active_patterns))
    if recent_events:
        sections.append(_render_events_section(recent_events))

    return _box("\n".join(s for s in sections if s))


# ──────────────────────────────────────────────
# Section renderers
# ──────────────────────────────────────────────

def _render_header(state, now: datetime) -> str:
    symbol = state.symbol if state.symbol else "SPY"
    ts     = now.strftime("%H:%M:%S UTC")
    phase  = ""
    if state.session:
        phase = f"  [{state.session.phase.value.upper()}]"
    source = getattr(state, "data_source", "MOCK")
    return f"  R2  ·  {symbol}  [DATA:{source}]{phase}  ·  {ts}"


def _render_market_status(state, context_state) -> str:
    lines: List[str] = []
    lines.append(_section_label("MARKET"))

    current_price = getattr(state, "current_price",           None)
    price_source  = getattr(state, "current_price_source",    None)
    price_ts      = getattr(state, "current_price_timestamp", None)
    bid           = getattr(state, "bid",                     None)
    ask           = getattr(state, "ask",                     None)
    spread        = getattr(state, "spread",                  None)
    ind_5m        = state.indicators.get("5m") if state.indicators else None
    vwap          = ind_5m.vwap if ind_5m else None

    if current_price is not None:
        price      = current_price
        source_tag = f"  [{price_source}]" if price_source else ""
        lines.append(f"  Price  : {price:.2f}{source_tag}")

        if bid is not None and ask is not None:
            spread_str = f"  spread {spread:.4g}" if spread is not None else ""
            lines.append(f"  Bid/Ask: {bid:.2f} / {ask:.2f}{spread_str}")

        if price_ts:
            try:
                from engine.market_session import to_et_string
                lines.append(f"  Updated: {to_et_string(price_ts)}")
            except Exception:
                lines.append(f"  Updated: {price_ts.strftime('%H:%M:%S UTC')}")

        if vwap:
            diff = price - vwap
            rel  = "above" if diff > 0 else "below"
            lines.append(f"  VWAP   : {vwap:.2f}  ({rel} {abs(diff):.2f})")

    elif ind_5m and ind_5m.close:
        price = ind_5m.close
        lines.append(f"  Price  : {price:.2f}")
        if vwap:
            diff = price - vwap
            rel  = "above" if diff > 0 else "below"
            lines.append(f"  VWAP   : {vwap:.2f}  ({rel} {abs(diff):.2f})")

    if state.regime:
        regime = state.regime.regime.value.replace("_", " ").title()
        conf   = f"  ({state.regime.confidence:.0%})"
        mult   = (f"  {state.regime.score_multiplier:.2f}×"
                  if state.regime.score_multiplier != 1.0 else "")
        lines.append(f"  Regime : {regime}{conf}{mult}")

    if state.narrative:
        narr    = state.narrative.narrative.value.replace("_", " ").title()
        htf     = getattr(state, "htf_bias", "") or ""
        htf_str = f"  · HTF {htf}" if htf and htf != "neutral" else ""
        lines.append(f"  Outlook: {narr}{htf_str}")

    if context_state:
        if context_state.vix_level:
            lines.append(
                f"  VIX    : {context_state.vix_level:.1f}  [{context_state.vix_regime}]"
            )
        if context_state.news_sentiment != 0:
            lines.append(
                f"  News   : {context_state.context_bias}"
                f"  ({context_state.news_sentiment:+.2f})"
            )

    return "\n".join(lines)


def _render_readiness(state, context_state) -> str:
    lines: List[str] = []
    lines.append(_section_label("STATUS"))

    market_session = getattr(state, "market_session", None)
    if market_session is not None:
        try:
            from engine.market_session import session_label
            sess_str = session_label(market_session)
        except Exception:
            sess_str = market_session.value.upper()
        lines.append(f"  Session : {sess_str}")
    elif state.session:
        lines.append(
            f"  Session : {state.session.phase.value.replace('_', ' ').upper()}"
        )

    if state.session:
        r  = state.session.readiness.value.replace("_", " ").title()
        me = "✓" if state.session.market_open    else "✗"
        ea = "✓" if state.session.entries_allowed else "✗"
        lines.append(f"  Readiness : {r}")
        lines.append(f"  Market    : {me}  Entries: {ea}")

    ctx_bias = getattr(state, "context_bias",       "neutral")
    ctx_risk = getattr(state, "context_risk_level", "normal")
    lines.append(f"  Ctx bias  : {ctx_bias}  [{ctx_risk} volatility]")

    return "\n".join(lines)


def _render_score_section(score, direction) -> str:
    lines: List[str] = []
    lines.append(_section_label("SETUP SCORE"))
    lines.append(render_setup_summary_line(score, direction))
    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    lines.append(f"  {_score_bar(effective)}")
    for label, val, max_val in [
        ("Align",     score.trend_alignment,   30),
        ("Momentum",  score.momentum_quality,  25),
        ("Structure", score.structure_context, 25),
        ("Timing",    score.entry_timing,      20),
    ]:
        pct      = int(val / max_val * 10) if max_val else 0
        bar_part = "█" * pct + "·" * (10 - pct)
        lines.append(f"  {label:<10} {val:>4.0f}/{max_val}  {bar_part}")
    return "\n".join(lines)


def _render_active_trade(trade) -> str:
    lines: List[str] = []
    lines.append(_section_label("ACTIVE TRADE"))
    lifecycle  = getattr(trade.lifecycle, "value", "—") if trade.lifecycle else "—"
    direction  = trade.direction.value.upper()           if trade.direction  else "—"
    instrument = trade.instrument_type.value             if trade.instrument_type else "—"
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
            f"  {pid[:36]:<36}  conf {conf:.0%}  wr {wr:.0%}  n={occ}"
        )
    return "\n".join(lines)


def _render_events_section(events: List[str]) -> str:
    """
    Render the most recent events, word-wrapping long entries.

    Each event occupies one or more terminal lines:
      First line:        "  · <text>"
      Continuation lines:"    <text>"    (4-space indent, no bullet)

    Up to MAX_RECENT_EVENTS events are shown (most recent).
    No fixed-length truncation — the full event text is always displayed.
    """
    if not events:
        return ""

    lines: List[str] = []
    lines.append(_section_label("RECENT EVENTS"))

    wrap_width = WIDTH + 2

    for ev in events[-MAX_RECENT_EVENTS:]:
        # Prefer sentence-aware wrapping so compact risk/context phrases like
        # "No elevated macro risk" stay intact instead of being split across
        # line breaks. If no sentence boundaries exist, fall back to normal
        # word wrapping.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev) if s.strip()]
        chunks = sentences if len(sentences) > 1 else [ev]

        first_line = True
        for chunk in chunks:
            wrapped = textwrap.wrap(
                chunk,
                width             = wrap_width,
                initial_indent    = "  · " if first_line else "    ",
                subsequent_indent = "    ",
                break_long_words  = False,
                break_on_hyphens  = False,
            )
            if wrapped:
                lines.extend(wrapped)
            else:
                lines.append(("  · " if first_line else "    ") + chunk)
            first_line = False

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Compact single-line renderers
# ──────────────────────────────────────────────

def render_regime_line(state) -> str:
    if not state.regime:
        return "Regime: —"
    return (f"Regime: {state.regime.regime.value.replace('_',' ').title()}"
            f"  {state.regime.confidence:.0%}")


def render_price_line(state) -> str:
    current_price = getattr(state, "current_price",        None)
    price_source  = getattr(state, "current_price_source", None)
    bid           = getattr(state, "bid",                  None)
    ask           = getattr(state, "ask",                  None)
    ind           = state.indicators.get("5m") if state.indicators else None

    if current_price is not None:
        price = current_price
        if bid is not None and ask is not None:
            price_str = f"{price:.2f}  [{bid:.2f}/{ask:.2f}]"
        elif price_source:
            price_str = f"{price:.2f}  [{price_source}]"
        else:
            price_str = f"{price:.2f}"
    elif ind:
        price     = ind.close
        price_str = f"{price:.2f}"
    else:
        return "Price: —"

    vwap = ind.vwap if ind else None
    if vwap:
        diff = price - vwap
        return f"Price: {price_str}  VWAP {vwap:.2f}  ({diff:+.2f})"
    return f"Price: {price_str}"


def render_session_line(state) -> str:
    market_session = getattr(state, "market_session", None)
    if market_session is not None:
        try:
            from engine.market_session import session_label
            sess = session_label(market_session)
        except Exception:
            sess = market_session.value.upper()
    elif state.session:
        sess = state.session.phase.value.upper()
    else:
        return "Session: —"
    open_ = "Open" if (state.session and state.session.market_open) else "Closed"
    return f"Session: {sess}  ·  Market {open_}"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _section_label(label: str) -> str:
    pad = max(0, WIDTH - len(label) - 4)
    return f"\n  ── {label} {'─' * pad}"


def _score_bar(score: float, width: int = 40) -> str:
    filled = max(0, min(width, int(score / 100 * width)))
    return f"[{'█' * filled}{'·' * (width - filled)}]  {score:.0f}"


def _box(content: str) -> str:
    border = "═" * (WIDTH + 2)
    return f"╔{border}╗\n{content}\n╚{border}╝"