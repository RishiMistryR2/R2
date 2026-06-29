"""
engine/event_engine.py — R2 Event Intelligence Engine v2.4

Answers the question "Did anything materially change?" before R2 spends
tokens generating commentary.  All work is pure computation — no API
calls, no Alpaca access, no ML.

Public API
----------
build_snapshot(state, trigger) -> MarketSnapshot
    Lightweight structural snapshot from RuntimeState. Safe; never raises.

evaluate_event(current, previous, trigger) -> EventIntelligenceResult
    Compare two snapshots.  .suppress == True → caller returns None.

snapshot_key(snap) -> str
    String fingerprint for exact-duplicate detection.

Priority levels
---------------
  critical        score >= 40   always emit
  important       score 20–39   always emit
  informational   score  5–19   rate-limited (min gap configurable)
  suppressed      score  < 5    caller returns None; no AI call

Materiality dimensions scored
-----------------------------
  1.  Structure direction reversed             +35
  2a. Trend integrity broken                   +25
  2b. Trend integrity restored                 +20
  3a. State transition → breakout/breakdown    +25
  3b. State transition → trend                 +20
  3c. Other state transition                   +15
  4.  Structural level (HL/LH) updated         +12
  5.  Confirmation achieved                    +25
  6.  Confidence delta >= 10 pts               +10
      Confidence delta  5–9 pts                 +4
  7.  Regime name changed                      +15
      Regime confidence delta >= 0.20           +5
  8.  Pattern set changed                      +10
      Pattern added                             +8
      Pattern removed                           +6
  9.  ATR price move >= 1.0× ATR              +20  (important)
      ATR price move >= 0.5× ATR               +8  (informational)
  —.  Exact duplicate                         −100 (suppressor)

Triggers that bypass scoring
----------------------------
  startup_briefing, user_request → always allow, score = 100
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Lightweight structural view of the market at one point in time."""

    # Structure
    direction:             str
    market_state:          str
    trend_intact:          bool
    sequence:              str

    # Key levels
    session_high:          Optional[float]
    session_low:           Optional[float]
    last_hl:               Optional[float]
    last_lh:               Optional[float]

    # Price + ATR (for move-size materiality)
    current_price:         Optional[float]
    atr_5m:                Optional[float]

    # Confirmation / thesis
    missing_confirmation:  str

    # Regime
    regime_name:           str
    regime_conf:           float

    # Confidence (pre-computed from confidence engine)
    confidence_score:      float

    # Patterns
    pattern_names:         tuple         # hashable; tuple[str, ...]

    # Meta
    timestamp:             datetime
    trigger:               str


@dataclass
class ChangeItem:
    key:         str     # e.g. "direction_change", "price_move_large"
    priority:    str     # "critical" | "important" | "informational" | "suppressed"
    description: str     # human-readable sentence
    score:       float   # significance points contributed by this item


@dataclass
class EventIntelligenceResult:
    priority:       str     # "critical" | "important" | "informational" | "suppressed"
    score:          float   # total (clamped 0–100)
    changes:        list    # list[ChangeItem]
    suppress:       bool    # True → caller returns None immediately
    change_summary: str     # "Direction: uptrend → downtrend.  Confidence: 57% → 74%."
    trigger:        str


# ──────────────────────────────────────────────────────────────
# Triggers that bypass significance scoring
# ──────────────────────────────────────────────────────────────

_ALWAYS_ALLOW: frozenset = frozenset({"startup_briefing", "user_request"})


# ──────────────────────────────────────────────────────────────
# Significance score table
# ──────────────────────────────────────────────────────────────

_SCORES: dict[str, float] = {
    # Structure
    "direction_change":          35.0,
    "trend_broken":              25.0,
    "trend_restored":            20.0,
    "state_to_breakout":         25.0,
    "state_to_breakdown":        25.0,
    "state_to_trend":            20.0,
    "state_change":              15.0,
    "sequence_extended":         10.0,

    # Levels
    "hl_level_change":           12.0,
    "lh_level_change":           12.0,

    # Confirmation
    "confirmation_achieved":     25.0,
    "confirmation_changed":       8.0,

    # Confidence
    "confidence_major":          10.0,   # abs delta >= 10
    "confidence_minor":           4.0,   # abs delta 5–9

    # Regime
    "regime_name_change":        15.0,
    "regime_conf_major":          5.0,

    # Patterns
    "pattern_change":            10.0,
    "pattern_added":              8.0,
    "pattern_removed":            6.0,

    # ATR price move
    "price_move_large":          20.0,   # >= 1.0× ATR
    "price_move_meaningful":      8.0,   # >= 0.5× ATR

    # Suppressor
    "exact_duplicate":         -100.0,
}


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def build_snapshot(state, trigger: str) -> MarketSnapshot:
    """
    Build a MarketSnapshot from a RuntimeState.
    Always safe — returns a blank snapshot if anything fails.
    """
    try:
        return _build(state, trigger)
    except Exception as exc:
        import logging
        logging.getLogger("engine.event").warning(
            f"[event] build_snapshot failed (non-fatal): {exc}"
        )
        return _blank(trigger)


def evaluate_event(
    current:  MarketSnapshot,
    previous: Optional[MarketSnapshot],
    trigger:  str,
) -> EventIntelligenceResult:
    """
    Compare current snapshot to previous and decide significance.

    Returns EventIntelligenceResult where:
      .suppress == True  → caller should return None (no AI call)
      .suppress == False → caller should proceed with commentary
    """
    if trigger in _ALWAYS_ALLOW or previous is None:
        return EventIntelligenceResult(
            priority="important", score=100.0, changes=[],
            suppress=False, change_summary="", trigger=trigger,
        )

    changes = _detect_changes(current, previous)
    raw     = sum(c.score for c in changes)
    score   = float(max(0.0, min(100.0, raw)))

    # If any change item is explicitly critical, the event is critical
    # regardless of the total score.  Exact-duplicate items carry priority
    # "suppressed" and cause an early return from _detect_changes, so they
    # can never coexist with critical items here.
    if any(c.priority == "critical" for c in changes):
        priority = "critical"
    else:
        priority = _to_priority(score)

    suppress = priority == "suppressed"
    summary  = _build_summary(changes, current, previous)

    return EventIntelligenceResult(
        priority=priority, score=score, changes=changes,
        suppress=suppress, change_summary=summary, trigger=trigger,
    )


def snapshot_key(snap: MarketSnapshot) -> str:
    """
    Compact fingerprint for exact-duplicate detection.
    Confidence is bucketed to ±5 points so a 1-point drift doesn't
    produce a different key.
    """
    conf_bucket = round(snap.confidence_score / 5.0) * 5
    return (
        f"{snap.direction}|{snap.market_state}|{int(snap.trend_intact)}"
        f"|{snap.regime_name}|{conf_bucket}"
        f"|{','.join(sorted(snap.pattern_names))}"
        f"|{snap.missing_confirmation[:40]}"
    )


# ──────────────────────────────────────────────────────────────
# Snapshot builder (internal)
# ──────────────────────────────────────────────────────────────

def _build(state, trigger: str) -> MarketSnapshot:
    from engine.narrative_engine  import _to_fields, _session_key
    from engine.pattern_engine    import classify_patterns
    from engine.confidence_engine import compute_confidence

    struct   = getattr(state, "structure_state", None)
    fields   = _to_fields(struct) if struct else {}
    ms_obj   = getattr(getattr(state, "market_state", None), "market_session", None)
    sess_key = _session_key(ms_obj)

    # Regime
    regime_name = ""
    regime_conf = 0.5
    entries     = True
    price       = None
    atr_5m      = None

    if state.market_state:
        reg = state.market_state.regime
        if reg:
            try:
                regime_name = reg.regime.value
            except AttributeError:
                regime_name = str(getattr(reg, "regime", ""))
            regime_conf = float(getattr(reg, "confidence", 0.5))

        sess = state.market_state.session
        if sess:
            entries = bool(getattr(sess, "entries_allowed", True))

        # Current price
        price = getattr(state.market_state, "current_price", None)
        if price is None:
            indicators = getattr(state.market_state, "indicators", {}) or {}
            ind_5m = indicators.get("5m")
            if ind_5m:
                price  = getattr(ind_5m, "close", None)
                atr_5m = getattr(ind_5m, "atr",   None)
        else:
            # Price was set; still try to get ATR
            indicators = getattr(state.market_state, "indicators", {}) or {}
            ind_5m = indicators.get("5m")
            if ind_5m:
                atr_5m = getattr(ind_5m, "atr", None)

    # Confidence
    cr = compute_confidence(fields, regime_name, regime_conf, sess_key, entries)

    # Patterns
    patterns      = classify_patterns(fields, regime=regime_name, session=sess_key)
    pattern_names = tuple(p.name for p in patterns)

    kl = fields.get("key_levels", {}) or {}

    return MarketSnapshot(
        direction            = fields.get("direction", ""),
        market_state         = fields.get("market_state", ""),
        trend_intact         = bool(fields.get("trend_intact", True)),
        sequence             = fields.get("sequence", ""),
        session_high         = kl.get("session_high"),
        session_low          = kl.get("session_low"),
        last_hl              = fields.get("last_hl"),
        last_lh              = fields.get("last_lh"),
        current_price        = float(price) if price is not None else None,
        atr_5m               = float(atr_5m) if atr_5m is not None else None,
        missing_confirmation = (fields.get("missing_confirmation") or "").strip(),
        regime_name          = regime_name,
        regime_conf          = float(regime_conf),
        confidence_score     = float(cr.score),
        pattern_names        = pattern_names,
        timestamp            = datetime.now(tz=timezone.utc),
        trigger              = trigger,
    )


def _blank(trigger: str) -> MarketSnapshot:
    return MarketSnapshot(
        direction="", market_state="", trend_intact=True, sequence="",
        session_high=None, session_low=None, last_hl=None, last_lh=None,
        current_price=None, atr_5m=None, missing_confirmation="",
        regime_name="", regime_conf=0.5, confidence_score=35.0,
        pattern_names=(), timestamp=datetime.now(tz=timezone.utc), trigger=trigger,
    )


# ──────────────────────────────────────────────────────────────
# Change detection (9 dimensions + ATR)
# ──────────────────────────────────────────────────────────────

def _detect_changes(curr: MarketSnapshot, prev: MarketSnapshot) -> list:
    changes: list[ChangeItem] = []

    def _add(key: str, priority: str, desc: str) -> None:
        changes.append(ChangeItem(
            key=key, priority=priority, description=desc,
            score=_SCORES.get(key, 0.0),
        ))

        # ── Exact duplicate (suppressor) ─────────────────────────
    if snapshot_key(curr) == snapshot_key(prev):
        # Do not suppress if price moved materially relative to ATR.
        # The structural fingerprint can be unchanged while price expands
        # enough to matter to a trader.
        atr = curr.atr_5m if (curr.atr_5m and curr.atr_5m > 0.0) else prev.atr_5m
        material_price_move = False

        if (
            curr.current_price is not None
            and prev.current_price is not None
            and atr
            and atr > 0.0
        ):
            material_price_move = (
                abs(curr.current_price - prev.current_price) / atr
            ) >= 0.5

        if not material_price_move:
            _add("exact_duplicate", "suppressed",
                 "State fingerprint unchanged — duplicate event suppressed.")
            return changes   # no further evaluation needed

    # ── 1. Direction ──────────────────────────────────────────
    if curr.direction != prev.direction and prev.direction and curr.direction:
        _add("direction_change", "critical",
             f"Direction reversed: {prev.direction} → {curr.direction}.")

    # ── 2. Trend integrity ────────────────────────────────────
    if prev.trend_intact and not curr.trend_intact:
        _add("trend_broken", "critical",
             "Trend structure violated — integrity lost.")
    elif not prev.trend_intact and curr.trend_intact:
        _add("trend_restored", "important",
             "Trend structure restored.")

    # ── 3. Market state ───────────────────────────────────────
    if curr.market_state != prev.market_state and curr.market_state and prev.market_state:
        if curr.market_state == "breakout":
            level_note = (f" (above {curr.session_high:.2f})"
                          if curr.session_high else "")
            _add("state_to_breakout", "critical",
                 f"State → breakout{level_note}.")
        elif curr.market_state == "breakdown":
            level_note = (f" (below {curr.session_low:.2f})"
                          if curr.session_low else "")
            _add("state_to_breakdown", "critical",
                 f"State → breakdown{level_note}.")
        elif curr.market_state == "trend":
            _add("state_to_trend", "important",
                 f"State transition: {prev.market_state} → trend.")
        else:
            _add("state_change", "important",
                 f"Market state: {prev.market_state} → {curr.market_state}.")

    # ── 4. Structural levels ──────────────────────────────────
    if _level_changed(curr.last_hl, prev.last_hl):
        tag = "higher low" if curr.direction == "uptrend" else "support level"
        _add("hl_level_change", "important",
             f"{tag.capitalize()} updated: {_f(prev.last_hl)} → {_f(curr.last_hl)}.")

    if _level_changed(curr.last_lh, prev.last_lh):
        tag = "lower high" if curr.direction == "downtrend" else "resistance level"
        _add("lh_level_change", "important",
             f"{tag.capitalize()} updated: {_f(prev.last_lh)} → {_f(curr.last_lh)}.")

    # ── 5. Confirmation ───────────────────────────────────────
    if prev.missing_confirmation and not curr.missing_confirmation:
        _add("confirmation_achieved", "critical",
             "Confirmation achieved — structural criteria met.")
    elif (prev.missing_confirmation and curr.missing_confirmation
          and prev.missing_confirmation != curr.missing_confirmation):
        _add("confirmation_changed", "informational",
             "Confirmation criteria updated.")

    # ── 6. Confidence ─────────────────────────────────────────
    conf_delta = curr.confidence_score - prev.confidence_score
    abs_delta  = abs(conf_delta)
    if abs_delta >= 10.0:
        direction_word = "increased" if conf_delta > 0 else "decreased"
        _add("confidence_major", "important",
             f"Confidence {direction_word}: "
             f"{prev.confidence_score:.0f}% → {curr.confidence_score:.0f}%.")
    elif abs_delta >= 5.0:
        _add("confidence_minor", "informational",
             f"Confidence: {prev.confidence_score:.0f}% → {curr.confidence_score:.0f}%.")

    # ── 7. Regime ─────────────────────────────────────────────
    if curr.regime_name != prev.regime_name and prev.regime_name and curr.regime_name:
        prev_r = prev.regime_name.replace("_", " ")
        curr_r = curr.regime_name.replace("_", " ")
        _add("regime_name_change", "important",
             f"Regime: {prev_r} → {curr_r}.")
    elif abs(curr.regime_conf - prev.regime_conf) >= 0.20:
        word = "strengthened" if curr.regime_conf > prev.regime_conf else "weakened"
        _add("regime_conf_major", "informational",
             f"Regime confidence {word}: "
             f"{prev.regime_conf:.0%} → {curr.regime_conf:.0%}.")

    # ── 8. Patterns ───────────────────────────────────────────
    prev_set = set(prev.pattern_names)
    curr_set = set(curr.pattern_names)
    added    = curr_set - prev_set
    removed  = prev_set - curr_set

    if added and removed:
        _add("pattern_change", "important",
             f"Pattern changed: "
             f"[{', '.join(sorted(removed))}] → [{', '.join(sorted(added))}].")
    elif added:
        for name in sorted(added):
            _add("pattern_added", "informational",
                 f"New pattern identified: {name}.")
    elif removed:
        for name in sorted(removed):
            _add("pattern_removed", "informational",
                 f"Pattern no longer active: {name}.")

    # ── ATR-based price move ──────────────────────────────────
    # Use current ATR first, fall back to previous if current has none.
    atr = curr.atr_5m if (curr.atr_5m and curr.atr_5m > 0.0) else prev.atr_5m
    if (curr.current_price is not None
            and prev.current_price is not None
            and atr and atr > 0.0):
        price_delta = curr.current_price - prev.current_price
        abs_move    = abs(price_delta)
        atr_ratio   = abs_move / atr
        direction   = "higher" if price_delta > 0 else "lower"

        if atr_ratio >= 1.0:
            _add("price_move_large", "important",
                 f"Price moved {direction} {abs_move:.2f} pts "
                 f"({atr_ratio:.1f}× ATR) — significant expansion.")
        elif atr_ratio >= 0.5:
            _add("price_move_meaningful", "informational",
                 f"Price moved {direction} {abs_move:.2f} pts "
                 f"({atr_ratio:.1f}× ATR).")
        # sub-0.5× ATR → no item; zero score contribution

    return changes

# ──────────────────────────────────────────────────────────────
# Change summary
# ──────────────────────────────────────────────────────────────

def _build_summary(
    changes: list,
    curr:    MarketSnapshot,
    prev:    MarketSnapshot,
) -> str:
    visible = [c for c in changes if c.key != "exact_duplicate"]
    if not visible:
        return ""

    _priority_rank = {
        "critical": 0, "important": 1, "informational": 2, "suppressed": 3,
    }
    ranked = sorted(visible,
                    key=lambda c: (_priority_rank.get(c.priority, 4), -c.score))

    # Top 3 descriptions
    parts = [c.description for c in ranked[:3]]
    return "  ".join(parts)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _to_priority(score: float) -> str:
    if score >= 40.0:
        return "critical"
    if score >= 20.0:
        return "important"
    if score >= 5.0:
        return "informational"
    return "suppressed"


def _level_changed(curr_val: Optional[float], prev_val: Optional[float]) -> bool:
    if curr_val is None and prev_val is None:
        return False
    if curr_val is None or prev_val is None:
        return True
    return abs(curr_val - prev_val) > 0.01


def _f(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "—"