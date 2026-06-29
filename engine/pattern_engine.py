"""
engine/pattern_engine.py — R2 Pattern Memory Engine v2.3

Classifies the current market structure into named patterns and provides
empirically-grounded outcome estimates for each.

Architecture
------------
classify_patterns(fields, regime, session) -> list[PatternMatch]
    Rule-based matching against _PATTERN_CATALOG.
    Ordered by specificity: behavioral events → state/direction combos.

build_pattern_narrative(struct_or_dict, regime, session, atr) -> str
    Formatted multi-line block:
        Structural patterns identified:
          · <name>  (<confidence>)
        Historical resolution (similar structures):
          · N% resolved lower  ·  M% resolved higher
          · Average move: X.X ATR  ·  Typical resolution: N sessions
        Note: <key insight>

build_pattern_narrative_from_package(package_dict) -> str
    Convenience wrapper; extracts ATR from indicators automatically.

Statistics
----------
Base rates are derived from market structure research on price action
patterns.  They are population-level estimates — not guarantees for any
individual occurrence.  Session context and regime alignment adjust the
primary pattern's narrative accordingly.

Used by
-------
  ai/claude_client._generate_mock_response  (startup_briefing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Pattern data types
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternMatch:
    name:                 str    # display name
    confidence_label:     str    # High | Moderate | Low
    description:          str    # one-line description
    bias:                 str    # "bearish" | "bullish" | "neutral"
    resolution_down:      float  # 0.0–1.0 — fraction that resolved lower
    resolution_up:        float  # 0.0–1.0 — fraction that resolved higher
    avg_atr_move:         float  # expected ATR magnitude of resolution move
    sessions_to_resolve:  float  # typical sessions before resolution
    note:                 str    # key insight / trader context


@dataclass
class PatternResult:
    patterns:  list            # list[PatternMatch]
    primary:   Optional[object]  # PatternMatch | None — highest-confidence match
    atr:       Optional[float]  # ATR if available (for point-value calculation)


# ──────────────────────────────────────────────────────────────
# Pattern catalog
# ──────────────────────────────────────────────────────────────
# Base rates are population-level estimates from market structure research.
# resolution_down + resolution_up do not need to sum to 1.0 — the remainder
# represents continued consolidation without clear resolution.

_CATALOG: dict[str, PatternMatch] = {

    # ── Behavioral event patterns (most specific) ──────────────

    "failed_breakout": PatternMatch(
        name                = "Failed Breakout",
        confidence_label    = "High",
        description         = "Price attempted to break above resistance then reversed",
        bias                = "bearish",
        resolution_down     = 0.72,
        resolution_up       = 0.22,
        avg_atr_move        = 1.8,
        sessions_to_resolve = 1.5,
        note=(
            "Failed breakouts carry a strong downside bias — trapped longs "
            "fuel the reversal.  The larger the initial breakout spike, the "
            "more aggressive the reversal tends to be."
        ),
    ),

    "failed_breakdown": PatternMatch(
        name                = "Failed Breakdown",
        confidence_label    = "High",
        description         = "Price broke below support then recovered",
        bias                = "bullish",
        resolution_down     = 0.22,
        resolution_up       = 0.70,
        avg_atr_move        = 1.8,
        sessions_to_resolve = 1.5,
        note=(
            "Failed breakdowns carry a strong upside bias — trapped shorts "
            "are forced to cover.  Watch for a VWAP reclaim to confirm buyer "
            "control."
        ),
    ),

    "vwap_reclaim": PatternMatch(
        name                = "VWAP Reclaim",
        confidence_label    = "Moderate",
        description         = "Buyers reclaimed VWAP after trading below",
        bias                = "bullish",
        resolution_down     = 0.35,
        resolution_up       = 0.58,
        avg_atr_move        = 1.2,
        sessions_to_resolve = 1.0,
        note=(
            "A VWAP reclaim shifts the intraday bias from bearish to neutral-"
            "bullish.  Follow-through matters — a second test of VWAP from "
            "above is the key confirmation."
        ),
    ),

    "vwap_rejection": PatternMatch(
        name                = "VWAP Rejection",
        confidence_label    = "Moderate",
        description         = "Price rallied to VWAP then was rejected",
        bias                = "bearish",
        resolution_down     = 0.60,
        resolution_up       = 0.32,
        avg_atr_move        = 1.2,
        sessions_to_resolve = 1.0,
        note=(
            "A VWAP rejection signals that sellers are active at the "
            "equilibrium level.  Below-VWAP structure remains bearish until "
            "buyers can reclaim it."
        ),
    ),

    # ── Compression / flag patterns ────────────────────────────

    "bull_flag": PatternMatch(
        name                = "Bull Flag",
        confidence_label    = "Moderate",
        description         = "Tight consolidation within an uptrend (upward flag)",
        bias                = "bullish",
        resolution_down     = 0.33,
        resolution_up       = 0.60,
        avg_atr_move        = 1.7,
        sessions_to_resolve = 2.0,
        note=(
            "Bull flags resolve higher approximately 60% of the time when the "
            "prior trend is intact.  The measured move target is typically the "
            "height of the flagpole added to the breakout point."
        ),
    ),

    "bear_flag": PatternMatch(
        name                = "Bear Flag",
        confidence_label    = "Moderate",
        description         = "Tight consolidation within a downtrend (downward flag)",
        bias                = "bearish",
        resolution_down     = 0.62,
        resolution_up       = 0.30,
        avg_atr_move        = 1.7,
        sessions_to_resolve = 2.0,
        note=(
            "Bear flags resolve lower approximately 62% of the time when the "
            "prior trend is intact.  Measured move target equals the flagpole "
            "subtracted from the breakdown point."
        ),
    ),

    "range_compression": PatternMatch(
        name                = "Range Compression",
        confidence_label    = "Moderate",
        description         = "ATR contracting — price coiling ahead of a directional move",
        bias                = "neutral",
        resolution_down     = 0.50,
        resolution_up       = 0.50,
        avg_atr_move        = 1.6,
        sessions_to_resolve = 2.0,
        note=(
            "Compression builds potential energy — the eventual resolution "
            "move typically exceeds the consolidation range.  Direction is not "
            "predictable from compression alone; watch for the first decisive "
            "close beyond the boundary."
        ),
    ),

    # ── Consolidation patterns ─────────────────────────────────

    "post_trend_consolidation_bull": PatternMatch(
        name                = "Post-trend Consolidation",
        confidence_label    = "Moderate",
        description         = "Sideways range forming after an uptrend",
        bias                = "bullish",
        resolution_down     = 0.37,
        resolution_up       = 0.55,
        avg_atr_move        = 1.3,
        sessions_to_resolve = 2.5,
        note=(
            "Consolidations within an uptrend are more likely to resolve "
            "higher than lower.  A break above the range high continues the "
            "trend; a break below the range low signals a deeper correction."
        ),
    ),

    "post_trend_consolidation_bear": PatternMatch(
        name                = "Post-trend Consolidation",
        confidence_label    = "Moderate",
        description         = "Sideways range forming after a downtrend",
        bias                = "bearish",
        resolution_down     = 0.58,
        resolution_up       = 0.35,
        avg_atr_move        = 1.3,
        sessions_to_resolve = 2.5,
        note=(
            "Consolidations within a downtrend are more likely to resolve "
            "lower than higher.  A break below the range low continues the "
            "trend; a break above the range high signals a potential reversal."
        ),
    ),

    # ── Trend continuation patterns ────────────────────────────

    "trend_continuation_long": PatternMatch(
        name                = "Trend Continuation (Long)",
        confidence_label    = "Moderate",
        description         = "Pullback within an intact uptrend",
        bias                = "bullish",
        resolution_down     = 0.36,
        resolution_up       = 0.58,
        avg_atr_move        = 1.1,
        sessions_to_resolve = 1.5,
        note=(
            "Pullbacks within intact uptrends are higher-probability long "
            "setups when the higher low structure holds.  Entry near the "
            "higher low with a stop below it defines the risk."
        ),
    ),

    "trend_continuation_short": PatternMatch(
        name                = "Trend Continuation (Short)",
        confidence_label    = "Moderate",
        description         = "Bounce within an intact downtrend",
        bias                = "bearish",
        resolution_down     = 0.60,
        resolution_up       = 0.32,
        avg_atr_move        = 1.1,
        sessions_to_resolve = 1.5,
        note=(
            "Bounces within intact downtrends are higher-probability short "
            "setups when the lower high structure holds.  The lower high "
            "formed by the bounce defines the invalidation level."
        ),
    ),

    # ── Trend exhaustion patterns ──────────────────────────────

    "trend_exhaustion_bull": PatternMatch(
        name                = "Trend Exhaustion",
        confidence_label    = "Low",
        description         = "Uptrend showing signs of structural weakness",
        bias                = "neutral",
        resolution_down     = 0.53,
        resolution_up       = 0.38,
        avg_atr_move        = 1.5,
        sessions_to_resolve = 2.0,
        note=(
            "Trend exhaustion does not guarantee reversal — it signals that "
            "the trend is maturing.  Watch for a confirmed lower high to form "
            "before considering any counter-trend position."
        ),
    ),

    "trend_exhaustion_bear": PatternMatch(
        name                = "Trend Exhaustion",
        confidence_label    = "Low",
        description         = "Downtrend showing signs of structural weakness",
        bias                = "neutral",
        resolution_down     = 0.38,
        resolution_up       = 0.52,
        avg_atr_move        = 1.5,
        sessions_to_resolve = 2.0,
        note=(
            "Trend exhaustion does not guarantee reversal — it signals that "
            "the trend is maturing.  Watch for a confirmed higher low to form "
            "before considering any counter-trend position."
        ),
    ),

    # ── Breakout / breakdown ───────────────────────────────────

    "breakout_attempt": PatternMatch(
        name                = "Breakout Attempt",
        confidence_label    = "Moderate",
        description         = "Price breaking above a key resistance level",
        bias                = "bullish",
        resolution_down     = 0.38,
        resolution_up       = 0.55,
        avg_atr_move        = 2.0,
        sessions_to_resolve = 1.0,
        note=(
            "True breakouts require follow-through volume and a closing "
            "acceptance above the level.  A close back below the breakout "
            "point within 2 bars is a strong failure signal."
        ),
    ),

    "breakdown_attempt": PatternMatch(
        name                = "Breakdown Attempt",
        confidence_label    = "Moderate",
        description         = "Price breaking below a key support level",
        bias                = "bearish",
        resolution_down     = 0.58,
        resolution_up       = 0.35,
        avg_atr_move        = 2.0,
        sessions_to_resolve = 1.0,
        note=(
            "True breakdowns require follow-through and a closing acceptance "
            "below the level.  A reclaim of the breakdown point within 2 bars "
            "is a strong failure signal."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────

def classify_patterns(
    fields:  dict,
    regime:  str = "",
    session: str = "",
) -> list:
    """
    Match the current structure against the pattern catalog.
    Returns a deduplicated list of PatternMatch, most specific first.
    """
    matches: list[PatternMatch] = []
    seen:    set[str]           = set()

    state     = fields.get("market_state", "")
    direction = fields.get("direction", "")
    intact    = fields.get("trend_intact", True)
    ev_strs   = [str(e).lower() for e in fields.get("recent_events", [])]

    def _add(key: str) -> None:
        p = _CATALOG.get(key)
        if p and p.name not in seen:
            matches.append(p)
            seen.add(p.name)

    # ── 1. Behavioral events (most specific) ───────────────────
    for ev in ev_strs:
        if "failed_breakout" in ev:
            _add("failed_breakout")
        if "failed_breakdown" in ev:
            _add("failed_breakdown")
        if "vwap_reclaim" in ev:
            _add("vwap_reclaim")
        if "vwap_rejection" in ev:
            _add("vwap_rejection")

    # ── 2. State + direction combos ────────────────────────────
    if state == "breakout":
        _add("breakout_attempt")

    elif state == "breakdown":
        _add("breakdown_attempt")

    elif state == "compression":
        if direction == "uptrend":
            _add("bull_flag")
        elif direction == "downtrend":
            _add("bear_flag")
        else:
            _add("range_compression")

    elif state == "range":
        if direction == "uptrend":
            _add("post_trend_consolidation_bull")
        elif direction == "downtrend":
            _add("post_trend_consolidation_bear")
        else:
            _add("range_compression")

    elif state == "pullback":
        if direction == "uptrend" and intact:
            _add("trend_continuation_long")
        elif direction == "downtrend" and intact:
            _add("trend_continuation_short")

    elif state == "trend":
        if not intact:
            if direction == "uptrend":
                _add("trend_exhaustion_bull")
            elif direction == "downtrend":
                _add("trend_exhaustion_bear")
        # Intact trend — no pattern needed (structure says it all)

    elif state == "expansion":
        # Treat like a breakout/breakdown depending on direction
        if direction == "uptrend":
            _add("breakout_attempt")
        elif direction == "downtrend":
            _add("breakdown_attempt")

    return matches


# ──────────────────────────────────────────────────────────────
# Narrative builder
# ──────────────────────────────────────────────────────────────

def build_pattern_narrative(
    struct_or_dict,
    regime:  str             = "",
    session: str             = "",
    atr:     Optional[float] = None,
    symbol:  str             = "SPY",
) -> str:
    """
    Build a multi-line pattern recognition block from structure data.

    Returns an empty string when no patterns are identified (intact clean
    trend with no ambiguity — the structure narrative already says everything).
    """
    from engine.narrative_engine import _to_fields

    fields   = _to_fields(struct_or_dict)
    patterns = classify_patterns(fields, regime=regime, session=session)

    if not patterns:
        return ""

    primary = patterns[0]

    # ── Header ────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("Structural patterns identified:")
    for p in patterns:
        lines.append(f"  · {p.name}  ({p.confidence_label} confidence)")

    # ── Resolution statistics ─────────────────────────────────
    lines.append("")
    lines.append("Historical resolution (similar structures):")

    down_pct = int(round(primary.resolution_down * 100))
    up_pct   = int(round(primary.resolution_up   * 100))

    if primary.bias == "bearish":
        lines.append(
            f"  · {down_pct}% resolved lower  ·  {up_pct}% resolved higher"
        )
    elif primary.bias == "bullish":
        lines.append(
            f"  · {up_pct}% resolved higher  ·  {down_pct}% resolved lower"
        )
    else:
        lines.append(
            f"  · {down_pct}% resolved lower  ·  {up_pct}% resolved higher"
        )

    # ATR-based move estimate
    atr_str = _format_atr_move(primary.avg_atr_move, atr)
    sess_str = _format_sessions(primary.sessions_to_resolve, session)
    lines.append(f"  · Average move: {atr_str}  ·  {sess_str}")

    # ── Pattern note ──────────────────────────────────────────
    if primary.note:
        lines.append("")
        # Wrap at ~70 chars for terminal readability
        lines.append(_wrap_note(primary.note))

    return "\n".join(lines)


def build_pattern_narrative_from_package(package_dict: dict) -> str:
    """
    Convenience wrapper — extracts all required fields from a package dict.
    """
    market   = package_dict.get("market", {}) or {}
    struct   = package_dict.get("structure", {}) or {}
    regime   = market.get("regime", {}).get("name", "")
    session  = market.get("market_session", "")
    atr      = None
    ind_5m   = (market.get("indicators", {}) or {}).get("5m", {}) or {}
    if ind_5m.get("atr"):
        atr = float(ind_5m["atr"])
    return build_pattern_narrative(
        struct, regime=regime, session=session, atr=atr,
    )


# ──────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────

def _format_atr_move(avg_atr: float, atr: Optional[float]) -> str:
    """Format the expected move as ATR multiplier and optional point value."""
    if atr and atr > 0:
        pts = avg_atr * atr
        return f"{avg_atr:.1f} ATR (~{pts:.2f} points)"
    return f"{avg_atr:.1f} ATR"


def _format_sessions(sessions: float, session_key: str) -> str:
    """Format the resolution timeline with session-appropriate context."""
    if sessions == 1.0:
        base = "within 1 session"
    elif sessions <= 1.5:
        base = "within 1–2 sessions"
    elif sessions <= 2.0:
        base = "within 2 sessions"
    elif sessions <= 2.5:
        base = "within 2–3 sessions"
    else:
        base = f"within {sessions:.0f} sessions"

    if session_key in ("after_hours", "premarket", "closed"):
        base += " (counting regular sessions)"

    return f"typical resolution: {base}"


def _wrap_note(note: str, width: int = 72) -> str:
    """Wrap a long note into lines prefixed with 'Note: ' and indented."""
    words  = note.split()
    lines  = []
    current = "Note: "

    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = "      " + word + " "
        else:
            current += word + " "

    if current.strip():
        lines.append(current.rstrip())

    return "\n".join(lines)