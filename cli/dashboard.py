"""
cli/dashboard.py — terminal dashboard renderer for R2 (Rich).

Pure formatting functions.  No I/O, no side effects.

Public API
----------
  render_dashboard_rich(...)  → Rich renderable (used by runtime_loop)
  render_dashboard(...)       → plain string: the same layout exported as text
                                (Console(record=True, color_system=None, width=100))

Layout (full terminal width)
----------------------------
  Top bar     R2 · SPY · [DATA:LIVE] · [SESSION] · HH:MM:SS ET
  Left  60%   MARKET · STATUS · WARNINGS · ACTIVE TRADE (if any)
  Right 40%   CONVICTION · SETUP SCORE (if any) · ACTIVE PATTERNS (if any)
  Bottom      RECENT EVENTS — newest first, max 6, two lines each

The layout is sized to its content: each region is measured at the width
the Layout gives it, so nothing is clipped and no blank rows are padded to
the terminal height.

Color rules
-----------
  Direction / regime   green bull/uptrend · red bear/downtrend · yellow chop/ranging
  Confidence           red < 40 · yellow 40–59 · green ≥ 60
  Entries              green ✓ allowed · red ✗ not allowed
  Price vs VWAP        green above · red below
  Session              green open/midday/power_hour · yellow premarket/after_hours · dim closed

All text is built from Text objects (never markup strings), so values like
"[DATA:LIVE]" or "[regime_change]" are never parsed as style tags.

The plain-text helpers at the bottom of this file (_render_header,
_render_market_status, _render_readiness, _render_events_section,
render_*_line, _score_bar) predate the Rich layout and are kept with their
original output for callers and tests that use them directly.
"""

from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from cli.warnings import collect_warnings
from cli.setup_display import render_setup_summary_line, render_stop_summary

WIDTH              = 72   # width of the legacy plain-text sections
MAX_RECENT_EVENTS  = 4    # legacy _render_events_section only
MAX_PANEL_EVENTS   = 6    # RECENT EVENTS panel
EVENT_MAX_LINES    = 2    # rows per event in the panel before "…"
EXPORT_WIDTH       = 100  # render_dashboard() plain-text export width


# ──────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────

def render_dashboard(
    state,
    score           = None,
    context_state   = None,
    session_memory  = None,
    active_patterns = None,
    recent_events:  List[str] = None,
    now:            Optional[datetime] = None,
    confidence_state = None,   # v2.7: engine.confidence_engine.ConfidenceState
) -> str:
    """The Rich dashboard exported as plain text (no color, 100 columns)."""
    console = Console(record=True, color_system=None, width=EXPORT_WIDTH,
                      file=io.StringIO())
    console.print(render_dashboard_rich(
        state, score=score, context_state=context_state,
        session_memory=session_memory, active_patterns=active_patterns,
        recent_events=recent_events, now=now, confidence_state=confidence_state,
    ))
    return console.export_text()


def render_dashboard_rich(
    state,
    score           = None,
    context_state   = None,
    session_memory  = None,
    active_patterns = None,
    recent_events:  List[str] = None,
    now:            Optional[datetime] = None,
    confidence_state = None,
):
    """Rich renderable for the full dashboard.  Same signature as render_dashboard."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    warnings = collect_warnings(
        state=state, context_state=context_state,
        session_memory=session_memory, score=score,
        active_patterns=active_patterns or [],
    )

    left: list = [
        _market_panel(state, context_state),
        _status_panel(state),
        _warnings_panel(warnings),
    ]
    if getattr(state, "active_trade", None):
        left.append(_trade_panel(state.active_trade))

    right: list = [_conviction_panel(confidence_state)]
    setup = _setup_panel(score, getattr(state, "setup_direction", None))
    if setup is not None:
        right.append(setup)
    patterns = _patterns_panel(active_patterns)
    if patterns is not None:
        right.append(patterns)

    return _Dashboard(
        top    = _top_bar(state, now),
        left   = Group(*left),
        right  = Group(*right),
        events = _events_panel(recent_events or []),
    )


# ──────────────────────────────────────────────
# Content-sized layout
# ──────────────────────────────────────────────

class _Dashboard:
    """
    Top bar / two columns (3:2) / events, in a Rich Layout whose region
    heights are measured from content at the widths the Layout assigns.
    """

    def __init__(self, top, left, right, events) -> None:
        self.top, self.left, self.right, self.events = top, left, right, events

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width

        # Ask a Layout for the column widths so measurement matches rendering.
        probe = Layout()
        probe.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
        regions = probe.render(console, options.update(width=width, height=1))
        left_w  = regions[probe["left"]].region.width
        right_w = regions[probe["right"]].region.width

        body_h = max(_height(console, options, self.left, left_w),
                     _height(console, options, self.right, right_w))
        top_h  = _height(console, options, self.top, width)
        ev_h   = _height(console, options, self.events, width) if self.events else 0

        layout = Layout(name="root")
        parts  = [Layout(self.top, name="top", size=top_h),
                  Layout(name="body", size=body_h)]
        if self.events is not None:
            parts.append(Layout(self.events, name="events", size=ev_h))
        layout.split_column(*parts)
        layout["body"].split_row(
            Layout(self.left,  name="left",  ratio=3),
            Layout(self.right, name="right", ratio=2),
        )
        total = top_h + body_h + ev_h
        yield from console.render(layout, options.update(width=width, height=total))


def _height(console: Console, options: ConsoleOptions, renderable, width: int) -> int:
    lines = console.render_lines(renderable, options.update(width=width, height=None),
                                 pad=False)
    return max(1, len(lines))


# ──────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────

_SESSION_STYLE = {
    "open": "green", "midday": "green", "power_hour": "green",
    "premarket": "yellow", "after_hours": "yellow",
    "closed": "dim",
}

_TRIGGER_STYLE = {
    "critical": "red",     "regime_change": "red",    "new_warning": "red",
    "important": "yellow", "narrative_change": "yellow", "setup_formed": "yellow",
    "informational": "dim", "periodic": "dim", "periodic_check": "dim",
}

_SEVERITY_STYLE = {"CRITICAL": "bold red", "WARNING": "yellow",
                   "CAUTION": "yellow", "INFO": "dim"}


def confidence_style(pct: float) -> str:
    """Red < 40, yellow 40–59, green ≥ 60.  pct is on a 0–100 scale."""
    if pct < 40:
        return "red"
    if pct < 60:
        return "yellow"
    return "green"


def direction_style(text: str) -> str:
    """Green bull/uptrend, red bear/downtrend, yellow chop/ranging, else none."""
    t = (text or "").lower()
    if any(w in t for w in ("bull", "uptrend", "long")):
        return "green"
    if any(w in t for w in ("bear", "downtrend", "short")):
        return "red"
    if any(w in t for w in ("chop", "rang", "neutral")):
        return "yellow"
    return ""


def session_style(session_key: str) -> str:
    return _SESSION_STYLE.get((session_key or "").lower(), "")


def trigger_style(trigger: str) -> str:
    return _TRIGGER_STYLE.get((trigger or "").lower(), "cyan")


def _panel(body, title: str) -> Panel:
    return Panel(body, title=Text(title, style="bold"), title_align="left",
                 border_style="bright_black", padding=(0, 1))


def _row(label: str, *value) -> Text:
    """'Label    value' with a fixed 9-col dim label."""
    return Text.assemble((f"{label:<9}", "dim"), *value)


def _enum_value(v) -> str:
    return str(getattr(v, "value", v) or "")


# ──────────────────────────────────────────────
# Top bar
# ──────────────────────────────────────────────

def _top_bar(state, now: datetime) -> Text:
    symbol = getattr(state, "symbol", None) or "SPY"
    source = getattr(state, "data_source", "MOCK")
    ms     = getattr(state, "market_session", None)
    if ms is not None:
        key   = _enum_value(ms)
        label = key.replace("_", " ").upper()
    elif getattr(state, "session", None):
        key   = state.session.phase.value
        label = key.replace("_", " ").upper()
    else:
        key, label = "", "—"
    try:
        from engine.market_session import to_et_string
        clock = to_et_string(now)
    except Exception:
        clock = now.strftime("%H:%M:%S UTC")

    sep = (" · ", "dim")
    return Text.assemble(
        ("R2", "bold"), sep, (symbol, "bold"), sep,
        (f"[DATA:{source}]", "green" if source == "LIVE" else "yellow"), sep,
        (f"[{label}]", session_style(key)), sep,
        (clock, "dim"),
    )


# ──────────────────────────────────────────────
# Left column
# ──────────────────────────────────────────────

def _market_panel(state, context_state) -> Panel:
    rows: list = []
    ind_5m = state.indicators.get("5m") if getattr(state, "indicators", None) else None
    vwap   = ind_5m.vwap if ind_5m else None

    price  = getattr(state, "current_price", None)
    source = getattr(state, "current_price_source", None)
    if price is None and ind_5m and ind_5m.close:
        price, source = ind_5m.close, None

    if price is not None:
        p_style = "bold"
        if vwap:
            p_style += " green" if price > vwap else " red" if price < vwap else ""
        rows.append(_row("Price", (f"{price:.2f}", p_style),
                         (f"  [{source}]" if source else "", "dim")))
        bid, ask = getattr(state, "bid", None), getattr(state, "ask", None)
        if bid is not None and ask is not None:
            spread = getattr(state, "spread", None)
            rows.append(_row("Bid/Ask", f"{bid:.2f} / {ask:.2f}",
                             (f"  spread {spread:.4g}" if spread is not None else "", "dim")))
        ts = getattr(state, "current_price_timestamp", None)
        if ts:
            try:
                from engine.market_session import to_et_string
                rows.append(_row("Updated", (to_et_string(ts), "dim")))
            except Exception:
                pass
        if vwap:
            diff = price - vwap
            rel  = "above" if diff > 0 else "below"
            rows.append(_row("VWAP", f"{vwap:.2f}  ",
                             (f"{rel} {abs(diff):.2f}", "green" if diff > 0 else "red")))

    regime = getattr(state, "regime", None)
    if regime:
        name = regime.regime.value.replace("_", " ").title()
        pct  = regime.confidence * 100
        mult = (f"  {regime.score_multiplier:.2f}×"
                if regime.score_multiplier != 1.0 else "")
        rows.append(_row("Regime", (name, direction_style(name)), "  ",
                         (f"{pct:.0f}%", confidence_style(pct)), (mult, "dim")))

    narrative = getattr(state, "narrative", None)
    if narrative:
        outlook = narrative.narrative.value.replace("_", " ").title()
        htf     = _enum_value(getattr(state, "htf_bias", ""))
        parts   = [(outlook, direction_style(outlook))]
        if htf and htf != "neutral":
            parts += [("  · HTF ", "dim"), (htf, direction_style(htf))]
        rows.append(_row("Outlook", *parts))

    if context_state:
        if getattr(context_state, "vix_level", None):
            rows.append(_row("VIX", f"{context_state.vix_level:.1f}  ",
                             (f"[{context_state.vix_regime}]", "dim")))
        if getattr(context_state, "news_sentiment", 0):
            bias = context_state.context_bias
            rows.append(_row("News", (bias, direction_style(bias)),
                             (f"  ({context_state.news_sentiment:+.2f})", "dim")))

    return _panel(Group(*rows) if rows else Text("No market data", style="dim"), "MARKET")


def _status_panel(state) -> Panel:
    rows: list = []
    ms = getattr(state, "market_session", None)
    if ms is not None:
        key = _enum_value(ms)
        rows.append(_row("Session", (key.replace("_", " ").upper(), session_style(key))))
    elif getattr(state, "session", None):
        key = state.session.phase.value
        rows.append(_row("Session", (key.replace("_", " ").upper(), session_style(key))))

    sess = getattr(state, "session", None)
    if sess:
        rows.append(_row("Ready", sess.readiness.value.replace("_", " ").title()))
        rows.append(_row("Market", _tick(sess.market_open, "open", "closed")))
        rows.append(_row("Entries", _tick(sess.entries_allowed, "allowed", "disabled")))

    bias = getattr(state, "context_bias", "neutral")
    risk = getattr(state, "context_risk_level", "normal")
    rows.append(_row("Ctx bias", (bias, direction_style(bias)),
                     (f"  [{risk} volatility]", "dim")))
    return _panel(Group(*rows), "STATUS")


def _tick(ok: bool, yes: str, no: str) -> Text:
    return Text(f"✓ {yes}", style="green") if ok else Text(f"✗ {no}", style="red")


def _warnings_panel(warnings) -> Panel:
    if not warnings:
        return _panel(Text("No active warnings", style="dim"), "WARNINGS")
    rows = []
    for w in warnings:
        sev   = w.severity.name if hasattr(w.severity, "name") else str(w.severity)
        style = _SEVERITY_STYLE.get(sev, "")
        line  = Text.assemble((f"{sev:<8} ", style), (w.message, style))
        if getattr(w, "detail", ""):
            line.append(f"\n         {w.detail}", style="dim")
        rows.append(line)
    return _panel(Group(*rows), "WARNINGS")


def _trade_panel(trade) -> Panel:
    lifecycle  = getattr(trade.lifecycle, "value", "—") if trade.lifecycle else "—"
    direction  = trade.direction.value.upper()          if trade.direction  else "—"
    instrument = trade.instrument_type.value            if trade.instrument_type else "—"
    dir_style  = ("green" if direction in ("CALLS", "LONG")
                  else "red" if direction in ("PUTS", "SHORT") else "")
    rows = [Text.assemble((direction, dir_style), f"  ·  {instrument}  ·  ",
                          (f"[{lifecycle.upper()}]", "dim"))]
    entry = trade.underlying_entry_price or trade.entry_price
    if entry:
        rows.append(_row("Entry", f"{entry:.2f}"))
    stop = render_stop_summary(trade).strip()
    if stop:
        rows.append(Text(stop, style="dim"))
    pnl = getattr(trade, "pnl_pct", None)
    if pnl is not None:
        rows.append(_row("P&L", (f"{pnl:+.2f}%", "green" if pnl >= 0 else "red")))
    return _panel(Group(*rows), "ACTIVE TRADE")


# ──────────────────────────────────────────────
# Right column
# ──────────────────────────────────────────────

# 3-row block digits so Final Conviction is the largest number on screen.
_BIG = {
    "0": ("█▀█", "█ █", "▀▀▀"), "1": ("▀█ ", " █ ", "▀▀▀"),
    "2": ("▀▀█", "█▀▀", "▀▀▀"), "3": ("▀▀█", " ▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"), "5": ("█▀▀", "▀▀█", "▀▀▀"),
    "6": ("█▀▀", "█▀█", "▀▀▀"), "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"), "9": ("█▀█", "▀▀█", "▀▀▀"),
    "%": ("▀  █", " ▄▀ ", "█  ▄"),
}


def big_number(text: str, style: str = "") -> Text:
    """Render digits and '%' as 3-row block characters."""
    rows = ["", "", ""]
    for ch in text:
        glyph = _BIG.get(ch, ("   ", "   ", "   "))
        for i in range(3):
            rows[i] += glyph[i] + " "
    return Text("\n".join(r.rstrip() for r in rows), style=f"bold {style}".strip())


def _conviction_panel(confidence_state) -> Panel:
    """
    Final Conviction (calibrated) as the dominant number, the Historical
    Adjustment that produced it, and the structural score in small text.
    Regime confidence stays in MARKET: different concepts, never merged.
    """
    if confidence_state is None:
        return _panel(Text("No confidence computed yet", style="dim"), "CONVICTION")
    try:
        from engine.confidence_engine import _label_for
        from memory.statistical_intelligence import MIN_SAMPLE_SIZE
        final      = float(confidence_state.calibrated_confidence)
        structural = float(confidence_state.structural_confidence)
        label, _   = _label_for(final)
        delta      = final - structural
        cal        = confidence_state.calibration_result
        style      = confidence_style(final)

        rows: list = [
            big_number(f"{final:.0f}%", style),
            Text.assemble(("Final Conviction ", "dim"), (f"{final:.0f}%", f"bold {style}"),
                          "  ", (label, style)),
        ]
        insufficient = cal is not None and not cal.sufficient_history
        if insufficient or abs(delta) < 0.5:
            adj = Text("none", style="dim")
        else:
            adj = Text(f"{delta:+.0f} pts", style="green" if delta > 0 else "red")
        rows.append(Text.assemble(("Historical Adjustment ", "dim"), adj))

        small = Text(f"structural {structural:.0f}%", style="dim")
        if insufficient:
            small.append(f" · {cal.sample_size} obs, need {MIN_SAMPLE_SIZE}", style="dim")
        if cal is not None and getattr(cal, "conflict_dampened", False):
            small.append("  ")
            small.append(" dampened ", style="black on yellow")
        rows.append(small)
        return _panel(Group(*rows), "CONVICTION")
    except Exception:
        return _panel(Text("Confidence unavailable", style="dim"), "CONVICTION")


def _setup_panel(score, direction) -> Optional[Panel]:
    if score is None or not getattr(score, "total", 0):
        return None
    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    style     = confidence_style(effective)
    bar_w     = 20
    filled    = max(0, min(bar_w, int(effective / 100 * bar_w)))
    rows = [
        Text(render_setup_summary_line(score, direction).strip(), style="bold"),
        Text.assemble(("█" * filled, style), ("·" * (bar_w - filled), "dim"),
                      (f"  {effective:.0f}", f"bold {style}"), ("  grade ", "dim"),
                      (score.grade(), "bold")),
    ]
    return _panel(Group(*rows), "SETUP SCORE")


def _patterns_panel(patterns) -> Optional[Panel]:
    if not patterns:
        return None
    rows = []
    for p in patterns[:4]:
        pid  = p.get("pattern_id", p.get("description", "unknown"))
        conf = p.get("confidence_score", 0.0) * 100
        rows.append(Text.assemble((pid[:28], direction_style(pid)), "  ",
                                  (f"{conf:.0f}%", confidence_style(conf))))
    return _panel(Group(*rows), "ACTIVE PATTERNS")


# ──────────────────────────────────────────────
# Recent events
# ──────────────────────────────────────────────

_TS_RE      = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", re.S)
_TRIGGER_RE = re.compile(r"^AI \[([\w-]+)\]:\s*(.*)$", re.S)


def parse_event(event: str) -> tuple:
    """Split an event-log entry into (timestamp, trigger, text); missing parts are ''."""
    ts, rest = "", event.strip()
    m = _TS_RE.match(rest)
    if m:
        ts, rest = m.group(1), m.group(2)
    trigger = ""
    m = _TRIGGER_RE.match(rest)
    if m:
        trigger, rest = m.group(1), m.group(2)
    return ts, trigger, rest


class _Clamped:
    """Wrap text to the available width; keep at most max_lines rows, ending in '…'."""

    def __init__(self, text: Text, max_lines: int) -> None:
        self.text, self.max_lines = text, max_lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = self.text.wrap(console, options.max_width)
        if len(lines) > self.max_lines:
            lines = lines[: self.max_lines]
            last  = lines[-1]
            last.rstrip()
            last.truncate(options.max_width - 1)
            last.append("…", style="dim")
        yield from lines


def _events_panel(events: List[str]) -> Optional[Panel]:
    """Newest first, at most MAX_PANEL_EVENTS, each clamped to EVENT_MAX_LINES rows."""
    if not events:
        return None
    rows = []
    for ev in list(reversed(events))[:MAX_PANEL_EVENTS]:
        ts, trigger, text = parse_event(ev)
        line = Text()
        if ts:
            line.append(f"{ts} ", style="dim")
        if trigger:
            line.append(f"[{trigger}] ", style=trigger_style(trigger))
        line.append(" ".join(text.split()))
        rows.append(_Clamped(line, EVENT_MAX_LINES))
    return _panel(Group(*rows), "RECENT EVENTS")


# ══════════════════════════════════════════════════════════════
# Plain-text helpers (pre-Rich).  Output unchanged; kept for callers and
# tests that use them directly.  Not used by the Rich layout.
# ══════════════════════════════════════════════════════════════

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


def _render_events_section(events: List[str]) -> str:
    """
    Legacy plain-text events block: the most recent MAX_RECENT_EVENTS,
    oldest first, word-wrapped without truncation.
    """
    if not events:
        return ""

    lines: List[str] = []
    lines.append(_section_label("RECENT EVENTS"))

    wrap_width = WIDTH + 2

    for ev in events[-MAX_RECENT_EVENTS:]:
        # Sentence-aware wrapping keeps short phrases on one line.
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


def _section_label(label: str) -> str:
    pad = max(0, WIDTH - len(label) - 4)
    return f"\n  ── {label} {'─' * pad}"


def _score_bar(score: float, width: int = 40) -> str:
    filled = max(0, min(width, int(score / 100 * width)))
    return f"[{'█' * filled}{'·' * (width - filled)}]  {score:.0f}"
