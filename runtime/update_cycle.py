"""
runtime/update_cycle.py — periodic market + strategy update cycle.

Responsibilities per cycle:
  1. Build or refresh MarketState (live or demo/synthetic)
  2. Run StrategyEngine.process() to enrich state
  3. Refresh ContextState
  4. Rebuild warnings via collect_warnings()
  5. Commit everything to RuntimeState
  6. Log cycle summary

Safety:
  Every step is exception-safe.  Errors are caught, logged, and recorded
  in LoopHealth.  The runtime loop always continues.

Data source labeling:
  - LIVE session_mode attempts Alpaca REST; on failure falls back to MOCK.
  - DEMO session_mode always produces MOCK.
  - Both MarketState.data_source and RuntimeState.data_source are always
    stamped explicitly so the dashboard and operator never see silent fakes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from runtime.runtime_state import RuntimeState, SessionMode

logger = logging.getLogger("runtime.update")

# Module-level import so mock.patch("runtime.update_cycle.build_live_state") works.
# Guarded because alpaca_feed depends on optional packages (aiohttp, websockets).
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
    RuntimeState is mutated in place.
    """
    try:
        # 1. Build/refresh MarketState
        market = _get_market_state(state)
        if market is None:
            logger.warning("[cycle] No market state available — skipping")
            return False

        # 2. Enrich via StrategyEngine
        market = _run_strategy(market)

        # 3. Context
        ctx = _get_context(state)

        # 4. Warnings
        warnings = _build_warnings(market, ctx, state)

        # 5. Commit
        state.market_state    = market
        state.context_state   = ctx
        state.active_warnings = warnings
        state.max_warning_severity = _max_severity(warnings)

        if market.setup_score and market.setup_score.total > 0:
            state.setup_score     = market.setup_score
            state.setup_direction = getattr(market, "setup_direction", None)

        state.mark_update()
        state.health.record_success()

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
        logger.error(f"[cycle] update_cycle error: {msg}", exc_info=True)
        state.health.record_error(msg)
        return False


# ──────────────────────────────────────────────
# Step implementations
# ──────────────────────────────────────────────

def _get_market_state(state: RuntimeState):
    """
    Return a fresh MarketState — live (Alpaca REST) or synthetic (demo).

    LIVE session_mode:
      - Attempt to fetch real bars from Alpaca via build_live_state.
      - On success  -> MarketState + RuntimeState tagged data_source="LIVE"
      - On failure  -> falls back to demo, tags data_source="MOCK",
                       logs a WARNING so the operator always knows.
    DEMO session_mode:
      - Always returns demo state tagged data_source="MOCK".
    """
    if state.session_mode == SessionMode.LIVE:
        # Use the module-level name so mock.patch can intercept it in tests
        live_fn = build_live_state
        if live_fn is not None:
            try:
                market = live_fn(state.symbol)
                market.data_source = "LIVE"
                state.data_source  = "LIVE"
                return market
            except Exception as e:
                logger.warning(
                    f"[cycle] Alpaca live data failed — falling back to MOCK: {e}"
                )
        else:
            logger.warning(
                "[cycle] build_live_state unavailable (import failed) — using MOCK"
            )

    # DEMO mode or live fallback
    market = build_demo_state(state.symbol)
    market.data_source = "MOCK"
    state.data_source  = "MOCK"
    return market


def _run_strategy(market):
    """Run StrategyEngine.process() safely."""
    try:
        engine = _get_or_create_engine()
        return engine.process(market)
    except Exception as e:
        logger.warning(f"[cycle] StrategyEngine error (returning raw state): {e}")
        return market


def _get_context(state: RuntimeState):
    """Return a ContextState — use cached or empty."""
    try:
        from context.context_state import ContextState
        if state.context_state is not None:
            return state.context_state
        return ContextState.empty()
    except Exception as e:
        logger.warning(f"[cycle] Context error: {e}")
        try:
            from context.context_state import ContextState
            return ContextState.empty()
        except Exception:
            return None


def _build_warnings(market, ctx, state: RuntimeState) -> list:
    """Collect current warnings from cli.warnings."""
    try:
        from cli.warnings import collect_warnings
        return collect_warnings(
            state           = market,
            context_state   = ctx,
            session_memory  = state.session_memory,
            score           = state.setup_score,
            active_patterns = [],
        )
    except Exception as e:
        logger.warning(f"[cycle] Warning collection error: {e}")
        return []


# ──────────────────────────────────────────────
# Demo / synthetic state builder
# ──────────────────────────────────────────────

def build_demo_state(symbol: str = "SPY"):
    """
    Build a plausible synthetic MarketState for demo / test mode.

    Uses stable, deterministic values — no randomness — so the dashboard
    and all commands render correctly without a live data feed.

    Always stamps data_source="MOCK" on the returned state.
    """
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState,
        Direction, TrendStrength, SessionState, SessionPhase, Readiness,
        RegimeState, MarketRegime, NarrativeState, MarketNarrative,
        SupportResistanceLevel, SessionIntent,
    )

    now   = datetime.now(tz=timezone.utc)
    close = 545.0
    atr   = 1.45
    vwap  = 543.5

    state = MarketState(symbol=symbol, timestamp=now)

    # Indicators for each timeframe
    for tf, rsi, macd_hist in [
        ("1d",  58.0, 0.08),
        ("1h",  55.0, 0.05),
        ("15m", 54.0, 0.03),
        ("5m",  53.0, 0.02),
        ("1m",  52.0, 0.01),
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

    # Trends
    for tf in ("1d", "1h", "15m", "5m", "1m"):
        tr                 = TrendState(tf, Direction.BULLISH, TrendStrength.MODERATE)
        tr.higher_highs    = True
        tr.last_swing_low  = close - 4.0
        tr.last_swing_high = close + 4.0
        state.trends[tf]   = tr

    # Support / resistance
    state.support_levels    = [SupportResistanceLevel(541.0, "support",    3)]
    state.resistance_levels = [SupportResistanceLevel(549.0, "resistance", 3)]

    # Session
    state.session = SessionState(
        phase           = _current_phase(now),
        market_open     = True,
        entries_allowed = True,
        score_modifier  = 1.0,
        entry_filter    = "normal",
        phase_note      = "demo mode",
        readiness       = Readiness.FULLY_READY,
    )

    # Regime
    state.regime = RegimeState(MarketRegime.TREND_BULL, 0.72, 1.0, True, "bull")

    # Narrative
    state.narrative = NarrativeState(
        narrative     = MarketNarrative.TRENDING_BULL,
        confidence    = 0.78,
        description   = "Bullish continuation",
        trade_posture = "long_bias",
    )

    # Intent
    state.intent = SessionIntent(min_score=60)

    # Context hints for warnings / dashboard
    state.context_bias       = "neutral"
    state.context_risk_level = "normal"

    # Always mark synthetic data explicitly — never silent fake prices
    state.data_source = "MOCK"

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
    """Reset the singleton.  Used in tests to get a fresh engine."""
    global _engine_instance
    _engine_instance = None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _current_phase(now: datetime):
    """Simple session-phase approximation based on UTC hour."""
    from engine.models import SessionPhase
    h = now.hour
    if h < 14:
        return SessionPhase.OPEN
    if h < 18:
        return SessionPhase.MIDDAY
    return SessionPhase.POWER_HOUR


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
    max_sev = max(w.severity.value for w in warnings)
    from cli.warnings import WarningSeverity
    return WarningSeverity(max_sev).label()