"""
ai/state_packager.py — package R2's internal state for AI consumption.

Transforms internal dataclass objects into compact, token-efficient dicts
suitable for inclusion in Claude prompts.

Design principles:
  - Summarise, don't dump.  The AI needs signal, not noise.
  - Every field is optional — missing data is handled gracefully.
  - Priority TFs only: 1d, 1h, 15m, 5m.  Skip 1m for AI context.
  - Keep numeric precision reasonable (2 dp for prices, 1 dp for indicators).
  - Total context target: ≤ 1500 tokens for market + state package.

Public functions:
    package_market_state(state)        → dict
    package_setup(score, direction)    → dict
    package_context(ctx)               → dict
    package_trade(trade)               → dict
    package_session(session)           → dict
    package_patterns(pm)               → dict
    package_warnings(warnings)         → dict
    build_full_package(...)            → StatePackage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Priority timeframes for AI context (exclude 1m — too noisy)
_PRIORITY_TFS = ("1d", "1h", "15m", "5m")


# ──────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────

@dataclass
class StatePackage:
    """
    A compact, serialisable snapshot of R2's current state.
    Passed to prompt_builder to construct AI messages.
    """
    assembled_at:    str = ""
    market:          Optional[Dict] = None
    structure:       Optional[Dict] = None  # StructureState.to_dict()
    setup:           Optional[Dict] = None
    context:         Optional[Dict] = None
    trade:           Optional[Dict] = None
    session:         Optional[Dict] = None
    patterns:        Optional[Dict] = None
    warnings:        Optional[Dict] = None
    risk_flags:      List[str]      = field(default_factory=list)
    request_type:    str            = "analysis"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"assembled_at": self.assembled_at,
                              "request_type": self.request_type}
        for key in ("market", "structure", "setup", "context", "trade",
            "session", "patterns", "warnings"):
            val = getattr(self, key)
            if val:
                d[key] = val
        if self.risk_flags:
            d["risk_flags"] = self.risk_flags
        return d

    def estimated_tokens(self) -> int:
        import json
        return len(json.dumps(self.to_dict(), default=str)) // 4


# ──────────────────────────────────────────────
# Individual packagers
# ──────────────────────────────────────────────

def package_market_state(state) -> Dict:
    """
    Compact MarketState → dict for AI context.
    Includes regime, narrative, and key indicators for priority TFs only.
    """
    result: Dict[str, Any] = {}

    result["symbol"]    = getattr(state, "symbol", "SPY")
    result["timestamp"] = _iso(getattr(state, "timestamp", None))

    # Authoritative market session (premarket/open/midday/power_hour/after_hours/closed).
    # This is more precise than the lossy SessionState.phase mapping.
    market_session = getattr(state, "market_session", None)
    if market_session is not None:
        market_session = getattr(market_session, "value", str(market_session))
    result["market_session"] = market_session

    # HTF bias
    htf = getattr(state, "htf_bias", None)
    if htf:
        result["htf_bias"] = htf

    # Regime
    regime = getattr(state, "regime", None)
    if regime:
        result["regime"] = {
            "name":            regime.regime.value,
            "confidence":      round(regime.confidence, 2),
            "entry_permitted": regime.entry_permitted,
            "score_mult":      round(regime.score_multiplier, 2),
        }
        if regime.caution_note:
            result["regime"]["caution"] = regime.caution_note

    # Narrative
    narrative = getattr(state, "narrative", None)
    if narrative:
        result["narrative"] = {
            "condition":     narrative.narrative.value,
            "confidence":    round(narrative.confidence, 2),
            "trade_posture": narrative.trade_posture,
        }

    # Session
    session = getattr(state, "session", None)
    if session:
        result["session"] = {
            "phase":           session.phase.value,
            "readiness":       session.readiness.value,
            "market_open":     session.market_open,
            "entries_allowed": session.entries_allowed,
        }

    # Indicators — priority TFs only, compact fields
    indicators: Dict[str, Any] = {}
    for tf, ind in (getattr(state, "indicators", None) or {}).items():
        if tf not in _PRIORITY_TFS:
            continue
        entry: Dict[str, Any] = {}
        if getattr(ind, "close", None):
            entry["close"] = round(ind.close, 2)
        if getattr(ind, "vwap", None):
            entry["vwap"]  = round(ind.vwap, 2)
        if getattr(ind, "rsi", None):
            entry["rsi"]   = round(ind.rsi, 1)
        if getattr(ind, "atr", None):
            entry["atr"]   = round(ind.atr, 3)
        if getattr(ind, "ema_stack", None):
            entry["ema_stack"] = ind.ema_stack
        if getattr(ind, "macd_bias", None):
            entry["macd_bias"] = ind.macd_bias
        if getattr(ind, "price_vs_vwap", None):
            entry["vs_vwap"] = ind.price_vs_vwap
        if entry:
            indicators[tf] = entry
    if indicators:
        result["indicators"] = indicators

    # Trends — compact
    trends: Dict[str, Any] = {}
    for tf, tr in (getattr(state, "trends", None) or {}).items():
        if tf not in _PRIORITY_TFS:
            continue
        t_entry: Dict[str, Any] = {
            "direction": tr.direction.value if hasattr(tr.direction, "value") else str(tr.direction),
            "strength":  tr.strength.value  if hasattr(tr.strength,  "value") else str(tr.strength),
        }
        if getattr(tr, "weakening", False):
            t_entry["weakening"] = True
        sl = getattr(tr, "last_swing_low",  None)
        sh = getattr(tr, "last_swing_high", None)
        if sl:
            t_entry["swing_low"]  = round(sl, 2)
        if sh:
            t_entry["swing_high"] = round(sh, 2)
        trends[tf] = t_entry
    if trends:
        result["trends"] = trends

    # Key levels (top 3 only)
    supports    = getattr(state, "support_levels",    None) or []
    resistances = getattr(state, "resistance_levels", None) or []
    if supports:
        result["support"]    = [round(s.price, 2) for s in supports[:3]]
    if resistances:
        result["resistance"] = [round(r.price, 2) for r in resistances[:3]]

    return result


def package_setup(score, direction=None) -> Dict:
    """Compact SetupScore + direction → dict."""
    if score is None:
        return {"available": False}

    effective = score.adjusted_total if score.adjusted_total > 0 else score.total
    result: Dict[str, Any] = {
        "score":     round(effective, 1),
        "grade":     score.grade(),
        "direction": direction.value if direction else None,
    }

    result["components"] = {
        "trend_alignment":   round(score.trend_alignment, 1),
        "momentum_quality":  round(score.momentum_quality, 1),
        "structure_context": round(score.structure_context, 1),
        "entry_timing":      round(score.entry_timing, 1),
    }

    if score.readiness_penalty:
        result["readiness_penalty"] = round(score.readiness_penalty, 1)
    if score.context_adjustment:
        result["context_adjustment"] = round(score.context_adjustment, 1)
    if score.regime_multiplier != 1.0:
        result["regime_mult"] = round(score.regime_multiplier, 2)

    for attr, key in [
        ("alignment_detail",  "alignment_note"),
        ("momentum_detail",   "momentum_note"),
        ("structure_detail",  "structure_note"),
        ("timing_detail",     "timing_note"),
    ]:
        val = getattr(score, attr, "")
        if val:
            result[key] = val[:80]

    return result


def package_context(ctx) -> Dict:
    """Compact ContextState → dict."""
    if ctx is None or not getattr(ctx, "context_valid", False):
        return {"available": False}

    result: Dict[str, Any] = {
        "bias":           getattr(ctx, "context_bias",         "neutral"),
        "risk_level":     getattr(ctx, "volatility_risk_level", "normal"),
        "elevated_risk":  getattr(ctx, "elevated_risk",         False),
        "news_sentiment": round(getattr(ctx, "news_sentiment", 0.0), 3),
    }

    if ctx.vix_level:
        result["vix"] = {
            "level":  round(ctx.vix_level, 1),
            "regime": ctx.vix_regime,
        }

    if ctx.risk_note:
        result["risk_note"] = ctx.risk_note

    events = getattr(ctx, "upcoming_events", None) or []
    if events:
        result["upcoming_events"] = [
            {
                "name":   getattr(e, "name", ""),
                "impact": getattr(e, "market_impact", {}).value
                          if hasattr(getattr(e, "market_impact", None), "value")
                          else str(getattr(e, "market_impact", "")),
            }
            for e in events[:3]
        ]

    return result


def package_trade(trade) -> Dict:
    """Compact TradeState → dict."""
    if trade is None:
        return {"active": False}

    result: Dict[str, Any] = {"active": True}

    lifecycle  = getattr(trade, "lifecycle",      None)
    direction  = getattr(trade, "direction",      None)
    instrument = getattr(trade, "instrument_type", None)
    if lifecycle:
        result["lifecycle"]  = lifecycle.value if hasattr(lifecycle, "value") else str(lifecycle)
    if direction:
        result["direction"]  = direction.value
    if instrument:
        result["instrument"] = instrument.value

    entry = getattr(trade, "underlying_entry_price", None) or getattr(trade, "entry_price", None)
    stop  = getattr(trade, "underlying_stop_price",  None)
    tgt   = getattr(trade, "profit_target",          None)
    pnl   = getattr(trade, "pnl_pct",                None)

    if entry:            result["entry"]  = round(entry, 2)
    if stop:             result["stop"]   = round(stop,  2)
    if tgt:              result["target"] = round(tgt,   2)
    if pnl is not None:  result["pnl_pct"] = round(pnl, 2)

    opt_entry = getattr(trade, "option_entry_price",          None)
    prem_stop = getattr(trade, "premium_emergency_stop_pct",  None)
    if opt_entry: result["option_entry"]     = round(opt_entry, 3)
    if prem_stop: result["premium_stop_pct"] = round(prem_stop, 1)

    flags = {}
    if getattr(trade, "thesis_stop_triggered",  False): flags["thesis"]  = True
    if getattr(trade, "time_stop_triggered",    False): flags["time"]    = True
    if getattr(trade, "premium_stop_triggered", False): flags["premium"] = True
    if flags:
        result["stop_flags"] = flags

    thesis = getattr(trade, "thesis", "")
    if thesis:
        result["thesis"] = thesis[:120]

    return result


def package_session(session) -> Dict:
    """Compact SessionMemory → dict."""
    if session is None:
        return {"available": False}

    result: Dict[str, Any] = {
        "date":         session.session_date,
        "trades":       session.total_trades,
        "win_rate_pct": round(session.win_rate * 100, 1),
        "daily_pnl":    round(session.daily_pnl_pct, 2),
    }

    if session.opening_regime:
        result["opening_regime"] = session.opening_regime
    if session.dominant_narrative:
        result["narrative"] = session.dominant_narrative
    if session.max_intraday_drawdown_pct:
        result["max_drawdown_pct"] = round(session.max_intraday_drawdown_pct, 2)
    if session.behavioral_flags:
        result["behavioral_flags"] = session.behavioral_flags[:3]
    if session.regime_changes:
        result["regime_changes"] = len(session.regime_changes)
    if session.lesson:
        result["session_lesson"] = session.lesson

    return result


def package_patterns(pm) -> Dict:
    """
    Compact PatternMemory → dict.
    Surfaces only patterns that have met the confidence threshold.
    """
    if pm is None:
        return {"available": False}

    result: Dict[str, Any] = {}

    market = pm.get_surfaced_market()
    if market:
        result["market_patterns"] = [
            {
                "id":         p.pattern_id,
                "win_rate":   round(p.win_rate, 2),
                "avg_return": round(p.avg_return, 3),
                "n":          p.occurrence_count,
            }
            for p in market[:3]
        ]

    behavior = pm.get_surfaced_behavior()
    if behavior:
        result["behavior_patterns"] = [
            {
                "desc":       p.description[:60],
                "confidence": round(p.confidence_score, 2),
                "n":          p.occurrences,
            }
            for p in behavior[:3]
        ]

    mistakes = [m for m in pm.get_mistakes() if m.is_surfaced]
    if mistakes:
        result["recurring_mistakes"] = [
            {"desc": m.description[:60], "n": m.occurrences}
            for m in mistakes[:3]
        ]

    return result if result else {"available": False, "no_surfaced_patterns": True}


def package_warnings(warnings: List) -> Dict:
    """Compact List[R2Warning] → dict, grouped by severity."""
    if not warnings:
        return {"count": 0}

    from cli.warnings import WarningSeverity
    groups: Dict[str, List[str]] = {}
    for w in warnings:
        sev = w.severity.label()
        groups.setdefault(sev, []).append(w.message)

    return {
        "count":  len(warnings),
        "groups": groups,
    }



# ──────────────────────────────────────────────
# Structure packager
# ──────────────────────────────────────────────

def package_structure(structure_state) -> Optional[Dict]:
    """Serialise a StructureState into a compact dict for AI context."""
    if structure_state is None:
        return None
    try:
        return structure_state.to_dict()
    except Exception:
        return None


# ──────────────────────────────────────────────
# Full package builder
# ──────────────────────────────────────────────

def build_full_package(
    state,
    score           = None,
    direction       = None,
    context_state   = None,
    active_trade    = None,
    session_memory  = None,
    pattern_memory  = None,
    warnings        = None,
    structure_state = None,  # engine.structure_engine.StructureState
    request_type:    str = "analysis",
) -> StatePackage:
    """
    Assemble a complete StatePackage from all available inputs.
    Any argument may be None — packager handles missing data gracefully.
    """
    from ai.r2_persona import extract_risk_flags

    now = datetime.now(tz=timezone.utc).isoformat()

    readiness      = "fully_ready"
    setup_grade    = "B"
    setup_score_v  = 80.0
    elevated_risk  = False
    thesis_stopped = False
    trade_count    = 0

    if state and getattr(state, "session", None):
        readiness = state.session.readiness.value

    if score:
        setup_grade   = score.grade()
        setup_score_v = score.adjusted_total if score.adjusted_total > 0 else score.total

    if context_state and getattr(context_state, "context_valid", False):
        elevated_risk = context_state.elevated_risk

    if active_trade:
        thesis_stopped = (
            getattr(active_trade, "thesis_stop_triggered",  False) or
            getattr(active_trade, "time_stop_triggered",    False) or
            getattr(active_trade, "premium_stop_triggered", False)
        )

    if session_memory:
        trade_count = session_memory.total_trades

    risk_flags = extract_risk_flags(
        readiness      = readiness,
        setup_grade    = setup_grade,
        setup_score    = setup_score_v,
        elevated_risk  = elevated_risk,
        thesis_stopped = thesis_stopped,
        trade_count    = trade_count,
    )

    return StatePackage(
        assembled_at = now,
        market       = package_market_state(state)     if state          else None,
        structure    = package_structure(structure_state),
        setup        = package_setup(score, direction)  if score          else None,
        context      = package_context(context_state)   if context_state  else None,
        trade        = package_trade(active_trade)      if active_trade   else None,
        session      = package_session(session_memory)  if session_memory else None,
        patterns     = package_patterns(pattern_memory) if pattern_memory else None,
        warnings     = package_warnings(warnings or []),
        risk_flags   = risk_flags,
        request_type = request_type,
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _iso(dt) -> str:
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)