"""
runtime/update_cycle.py — periodic market + strategy update cycle.

Data source labeling:
  - LIVE session_mode → Alpaca REST; falls back to MOCK on failure.
  - DEMO session_mode → synthetic state always labeled MOCK.
  - Both MarketState.data_source and RuntimeState.data_source are always
    stamped explicitly — never silent fake prices.

Session classification uses engine.market_session (single authoritative source).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from runtime.runtime_state import RuntimeState, SessionMode

logger = logging.getLogger("runtime.update")

# Module-level import so mock.patch("runtime.update_cycle.build_live_state") works.
try:
    from engine.alpaca_feed import build_live_state
except Exception:
    build_live_state = None  # type: ignore[assignment]


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def run_update_cycle(state: RuntimeState) -> bool:
    """
    Execute one full update cycle.
    Returns True on success, False if an exception was caught.
    """
    try:
        market = _get_market_state(state)
        if market is None:
            logger.warning("[cycle] No market state — skipping")
            return False

        market = _run_strategy(market)

        # Structure engine (non-fatal — never halts market processing)
        structure = None
        try:
            structure = _run_structure_engine(market)
        except Exception as _se:
            logger.warning(f"[cycle] Structure engine exception: {_se}")

        ctx      = _get_context(state)
        warnings = _build_warnings(market, ctx, state)

        state.market_state         = market
        state.structure_state      = structure
        state.context_state        = ctx
        state.active_warnings      = warnings
        state.max_warning_severity = _max_severity(warnings)

        if market.setup_score and market.setup_score.total > 0:
            state.setup_score     = market.setup_score
            state.setup_direction = getattr(market, "setup_direction", None)

        state.mark_update()
        state.health.record_success()

        # ── Outcome tracker: resolve pending observations ───────────
        # Runs after every successful update cycle.  Always non-fatal.
        try:
            from memory.outcome_tracker import get_tracker
            _px  = getattr(state.market_state, "current_price", None)
            _atr = None
            if state.market_state:
                _ind_map = getattr(state.market_state, "indicators", None) or {}
                _ind_5m  = _ind_map.get("5m")
                if _ind_5m:
                    _atr = getattr(_ind_5m, "atr", None)
            if _px is not None:
                get_tracker().update_pending(
                    float(_px),
                    atr=float(_atr) if _atr is not None else None,
                )
        except Exception:
            pass   # outcome tracker is always non-fatal

        logger.debug(
            f"[cycle #{state.update_count}]"
            f"  regime={_regime_label(market)}"
            f"  score={_score_label(state.setup_score)}"
            f"  warnings={len(warnings)}"
            f"  source={state.data_source}"
        )
        return True

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(f"[cycle] error: {msg}", exc_info=True)
        state.health.record_error(msg)
        return False


# ──────────────────────────────────────────────
# Step implementations
# ──────────────────────────────────────────────

def _get_market_state(state: RuntimeState):
    """
    LIVE → Alpaca REST via build_live_state (patchable at module level).
    On failure → demo with MOCK label + WARNING.
    DEMO → demo always.
    """
    if state.session_mode == SessionMode.LIVE:
        live_fn = build_live_state
        if live_fn is None:
            try:
                from engine.alpaca_feed import build_live_state as _d
                live_fn = _d
            except Exception as ie:
                logger.warning(f"[cycle] build_live_state unavailable: {ie}")

        if live_fn is not None:
            try:
                market             = live_fn(state.symbol)
                market.data_source = "LIVE"
                state.data_source  = "LIVE"
                return market
            except Exception as e:
                logger.warning(f"[cycle] Alpaca live failed — MOCK fallback: {e}")

    market             = build_demo_state(state.symbol)
    market.data_source = "MOCK"
    state.data_source  = "MOCK"
    return market


def _run_strategy(market):
    try:
        return _get_or_create_engine().process(market)
    except Exception as e:
        logger.warning(f"[cycle] StrategyEngine error: {e}")
        return market


def _get_context(state: RuntimeState):
    try:
        from context.context_state import ContextState
        return state.context_state if state.context_state else ContextState.empty()
    except Exception as e:
        logger.warning(f"[cycle] Context error: {e}")
        return None


def _build_warnings(market, ctx, state: RuntimeState) -> list:
    try:
        from cli.warnings import collect_warnings
        return collect_warnings(
            state=market, context_state=ctx,
            session_memory=state.session_memory, score=state.setup_score,
            active_patterns=[],
        )
    except Exception as e:
        logger.warning(f"[cycle] Warning collection error: {e}")
        return []


def _run_structure_engine(market) -> "Optional[Any]":
    """
    Run Market Structure Engine against current MarketState.
    Returns StructureState, or None on failure.
    Never raises — structure failure must not halt market processing.
    """
    try:
        from engine.structure_engine import analyze
        return analyze(market)
    except Exception as e:
        logger.warning(f"[cycle] StructureEngine error: {e}")
        return None

# ──────────────────────────────────────────────
# Demo / synthetic state builder
# ──────────────────────────────────────────────

def build_demo_state(symbol: str = "SPY"):
    """
    Build a plausible synthetic MarketState (fixed 545.00 price).

    Uses the authoritative session classifier so the session phase and
    market_session attribute are correct even in demo mode.
    Always stamps data_source="MOCK".
    """
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState,
        Direction, TrendStrength, SessionState, Readiness,
        RegimeState, MarketRegime, NarrativeState, MarketNarrative,
        SupportResistanceLevel, SessionIntent,
    )
    from engine.market_session import get_market_session, to_session_phase, is_regular_hours

    now     = datetime.now(tz=timezone.utc)
    session = get_market_session(now)
    phase   = to_session_phase(session)
    close   = 545.0
    atr     = 1.45
    vwap    = 543.5

    state = MarketState(symbol=symbol, timestamp=now)

    for tf, rsi, macd_hist in [
        ("1d", 58.0, 0.08), ("1h", 55.0, 0.05), ("15m", 54.0, 0.03),
        ("5m", 53.0, 0.02), ("1m", 52.0, 0.01),
    ]:
        ind               = IndicatorSnapshot(tf, now, close)
        ind.atr           = atr
        ind.ema_9         = close - 0.8
        ind.ema_20        = close - 1.5
        ind.rsi           = rsi
        ind.macd_hist     = macd_hist
        ind.ema_stack     = "bull_aligned"
        ind.vwap          = vwap
        ind.price_vs_vwap = "above"
        ind.macd_bias     = "bullish"
        state.indicators[tf] = ind

    for tf in ("1d", "1h", "15m", "5m", "1m"):
        tr                 = TrendState(tf, Direction.BULLISH, TrendStrength.MODERATE)
        tr.higher_highs    = True
        tr.last_swing_low  = close - 4.0
        tr.last_swing_high = close + 4.0
        state.trends[tf]   = tr

    state.support_levels    = [SupportResistanceLevel(541.0, "support",    3)]
    state.resistance_levels = [SupportResistanceLevel(549.0, "resistance", 3)]

    market_open = is_regular_hours(session)
    state.session = SessionState(
        phase=phase, market_open=market_open,
        entries_allowed=market_open, score_modifier=1.0,
        entry_filter="normal", phase_note="demo mode",
        readiness=Readiness.FULLY_READY,
    )
    state.regime = RegimeState(MarketRegime.TREND_BULL, 0.72, 1.0, True, "bull")
    state.narrative = NarrativeState(
        narrative=MarketNarrative.TRENDING_BULL, confidence=0.78,
        description="Bullish continuation", trade_posture="long_bias",
    )
    state.intent             = SessionIntent(min_score=60)
    state.context_bias       = "neutral"
    state.context_risk_level = "normal"
    state.bars               = _build_demo_bars(close=close, atr=atr)
    state.data_source        = "MOCK"
    state.market_session     = session   # real session, synthetic price

    return state


# ──────────────────────────────────────────────
# StrategyEngine singleton
# ──────────────────────────────────────────────

_engine_instance = None


def _get_or_create_engine():
    global _engine_instance
    if _engine_instance is None:
        from strategy.strategy_engine import StrategyEngine
        _engine_instance = StrategyEngine()
    return _engine_instance



def reset_engine() -> None:
    global _engine_instance
    _engine_instance = None


def _build_demo_bars(close: float = 545.0, atr: float = 1.45, count: int = 60) -> dict:
    """
    Generate synthetic OHLCV bars for all timeframes (demo mode).
    Uses sine-wave + linear trend so structure engine finds HH/HL swings.
    """
    import math
    from datetime import timedelta

    def _make_bars(base_close, base_atr, n, interval_minutes):
        bars = []
        now = datetime.now(tz=timezone.utc)
        for i in range(n):
            trend = (i - n // 2) * base_atr * 0.05
            wave = math.sin(i * 0.45) * base_atr * 1.1
            mid = base_close + trend + wave
            h = round(mid + base_atr * 0.55, 2)
            lo = round(mid - base_atr * 0.55, 2)
            bars.append({
                "timestamp": now - timedelta(minutes=(n - i) * interval_minutes),
                "open": round(mid - base_atr * 0.08, 2),
                "high": h,
                "low": lo,
                "close": round(mid + base_atr * 0.08, 2),
                "volume": 500_000.0,
                "vwap": round(mid, 2),
            })
        return bars

    return {
        "1d": _make_bars(close, atr * 8, 30, 1440),
        "1h": _make_bars(close, atr * 3, 40, 60),
        "15m": _make_bars(close, atr * 1.5, 50, 15),
        "5m": _make_bars(close, atr, count, 5),
        "1m": _make_bars(close, atr * 0.4, 60, 1),
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _current_phase(now: datetime):
    """SessionPhase from the authoritative market session classifier."""
    from engine.market_session import get_market_session, to_session_phase
    return to_session_phase(get_market_session(now))


def _regime_label(market) -> str:
    if market is None or market.regime is None:
        return "—"
    return market.regime.regime.value


def _score_label(score) -> str:
    if score is None:
        return "—"
    eff = score.adjusted_total if score.adjusted_total > 0 else score.total
    return f"{eff:.0f} ({score.grade()})"


def _max_severity(warnings: list) -> Optional[str]:
    if not warnings:
        return None
    from cli.warnings import WarningSeverity
    return WarningSeverity(max(w.severity.value for w in warnings)).label()