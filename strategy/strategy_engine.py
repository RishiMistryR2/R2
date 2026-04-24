"""
StrategyEngine: orchestrates all strategy components.

Takes a MarketState from the indicator engine and returns
an enriched MarketState with:
  - Narrative classified
  - HTF bias computed
  - Entry signals evaluated
  - Setup scored
  - Active trade thesis and exit conditions updated

The engine is stateless between calls EXCEPT for:
  - Previous trend states (for flip detection)
  - Previous indicator snapshots (for weakening detection)
  - Active trade tracking (delegated to TradeManager)

Called by MarketEngine after each candle close.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from engine.models import (
    Direction,
    IndicatorSnapshot,
    MarketState,
    NarrativeState,
    SetupScore,
    SetupType,
    TradeDirection,
    TradeState,
    TrendState,
    MarketEvent,
)
from strategy.narrative    import classify_narrative
from strategy.regime_engine import MarketRegimeEngine
from strategy.session_state import compute_session_state
from strategy.priority_engine import PriorityEngine
from strategy.rules      import (
    check_long_entry,
    check_short_entry,
    detect_trend_flip,
    detect_trend_weakening,
)
from strategy.scorer     import score_setup
from strategy.timeframes import compute_tf_consensus, get_htf_bias
from config.settings     import MIN_CANDLES_FOR_INDICATORS

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Processes a raw MarketState and returns a fully enriched one.

    Usage:
        engine = StrategyEngine()
        enriched_state = engine.process(raw_state)
    """

    def __init__(self) -> None:
        # Rolling history for delta detection
        self._prev_trends: Dict[str, TrendState]        = {}
        self._prev_indicators: Dict[str, IndicatorSnapshot] = {}
        self._prev_narrative: Optional[NarrativeState]  = None

        # New engines (Phase 3)
        self._regime_engine   = MarketRegimeEngine()
        self._priority_engine = PriorityEngine()

        # Pending events generated during processing
        self._pending_events: list[MarketEvent] = []
        self._prioritized_events: list = []

    def process(self, state: MarketState) -> MarketState:
        """
        Enrich a MarketState with strategy outputs.
        Mutates the state in-place and returns it.
        """
        self._pending_events = []

        # Skip if not enough data yet
        if not state.indicators:
            return state

        # ── Step 0: Session + Regime ─────────────────────
        # Preserve readiness from indicator engine before overwriting session
        existing_readiness = (
            state.session.readiness if state.session else None
        )
        # Respect pre-injected session (tests or user override); otherwise compute
        if state.session is None or not state.session.market_open:
            state.session = compute_session_state()
        # Carry readiness from indicator engine into the new session state
        if existing_readiness is not None:
            state.session.readiness = existing_readiness

        if state.regime is None:
            state.regime = self._regime_engine.classify(state)

        # ── Step 1: Weakening detection ──────────────────
        self._update_weakening(state)

        # ── Step 2: HTF bias ─────────────────────────────
        state.htf_bias = get_htf_bias(state.trends)

        # ── Step 3: Narrative classification ─────────────
        state.narrative = classify_narrative(state)
        self._check_narrative_change(state.narrative)

        # ── Step 4b: Context evaluation ──────────────────
        self._evaluate_context(state)

        # ── Step 4: Trend flip detection ─────────────────
        self._detect_flips(state)

        # ── Step 5: Entry signal + scoring ───────────────
        self._evaluate_entries(state)

        # ── Step 6: Active trade exit check ──────────────
        if state.active_trade:
            self._check_trade_exits(state)

        # ── Step 6b: Priority scoring ────────────────────
        self._prioritized_events = self._priority_engine.process(self._pending_events)

        # ── Step 7: Save state for next tick ─────────────
        self._prev_trends     = dict(state.trends)
        self._prev_indicators = dict(state.indicators)
        self._prev_narrative  = state.narrative

        return state

    def pop_events(self) -> list[MarketEvent]:
        """Return and clear pending events."""
        events = list(self._pending_events)
        self._pending_events = []
        return events

    def pop_prioritized_events(self) -> list:
        """Return and clear prioritized events (PrioritizedEvent list)."""
        events = list(self._prioritized_events)
        self._prioritized_events = []
        return events

    # ──────────────────────────────────────────
    # Step implementations
    # ──────────────────────────────────────────

    def _update_weakening(self, state: MarketState) -> None:
        """
        Re-evaluate trend weakening for each timeframe.
        Requires previous indicator snapshot for delta (hist shrinking).
        """
        for tf, trend in state.trends.items():
            ind      = state.indicators.get(tf)
            prev_ind = self._prev_indicators.get(tf)
            if not ind:
                continue
            weakening, reason = detect_trend_weakening(trend, ind, prev_ind)
            trend.is_weakening = weakening
            if weakening and not self._prev_trends.get(tf, TrendState(tf)).is_weakening:
                logger.debug(f"Weakening on {tf}: {reason}")
                self._pending_events.append(MarketEvent(
                    event_type="trend_weakening",
                    timestamp=datetime.now(tz=timezone.utc),
                    payload={"timeframe": tf, "reason": reason},
                    priority=1,
                ))

    def _evaluate_context(self, state: MarketState) -> None:
        """
        Step 4b — attach external context to the MarketState.

        Context evaluation is non-fatal. If context fails, the strategy
        pipeline continues without blocking trade logic.
        """
        try:
            from context.context_engine import build_context

            ctx = build_context(symbol=state.symbol)
            state.context = ctx
            state.context_bias = ctx.context_bias
            state.context_risk_level = ctx.volatility_risk_level
        except Exception as e:
            logger.debug(f"Context evaluation skipped: {e}")

    def _detect_flips(self, state: MarketState) -> None:
        """Detect trend flips on key timeframes vs previous state."""
        for tf in ("5m", "15m", "1h"):
            curr_trend = state.trends.get(tf)
            prev_trend = self._prev_trends.get(tf)
            curr_ind   = state.indicators.get(tf)
            prev_ind   = self._prev_indicators.get(tf)

            if not curr_trend or not curr_ind:
                continue

            flipped, reason = detect_trend_flip(curr_trend, prev_trend, curr_ind, prev_ind)
            if flipped:
                logger.info(f"Trend flip on {tf}: {reason}")
                self._pending_events.append(MarketEvent(
                    event_type="trend_flip",
                    timestamp=datetime.now(tz=timezone.utc),
                    payload={"timeframe": tf, "reason": reason,
                             "new_direction": curr_trend.direction.value},
                    priority=2 if tf in ("15m", "1h") else 1,
                ))

    def _evaluate_entries(self, state: MarketState) -> None:
        """
        Check both long and short entry conditions.
        If an entry qualifies, score it and attach to the state.
        """
        # Don't generate new signals when already in a trade
        if state.active_trade:
            return

        # Try long
        long_signal = check_long_entry(state)
        if long_signal.allowed:
            score = score_setup(state, TradeDirection.CALLS)
            state.setup_score     = score
            state.setup_direction = TradeDirection.CALLS
            state.setup_type      = _infer_setup_type(state, TradeDirection.CALLS)
            if score.total >= 60:
                logger.info(f"LONG signal — score {score.total:.0f} ({score.grade()}): {long_signal.reason}")
                self._pending_events.append(MarketEvent(
                    event_type="setup_formed",
                    timestamp=datetime.now(tz=timezone.utc),
                    payload={
                        "direction": "calls",
                        "score":     score.total,
                        "grade":     score.grade(),
                        "reason":    long_signal.reason,
                        "setup_type": state.setup_type.value,
                    },
                    priority=2,
                ))
            return

        # Try short
        short_signal = check_short_entry(state)
        if short_signal.allowed:
            score = score_setup(state, TradeDirection.PUTS)
            state.setup_score     = score
            state.setup_direction = TradeDirection.PUTS
            state.setup_type      = _infer_setup_type(state, TradeDirection.PUTS)
            if score.total >= 60:
                logger.info(f"SHORT signal — score {score.total:.0f} ({score.grade()}): {short_signal.reason}")
                self._pending_events.append(MarketEvent(
                    event_type="setup_formed",
                    timestamp=datetime.now(tz=timezone.utc),
                    payload={
                        "direction": "puts",
                        "score":     score.total,
                        "grade":     score.grade(),
                        "reason":    short_signal.reason,
                        "setup_type": state.setup_type.value,
                    },
                    priority=2,
                ))
            return

        # No signal — clear previous setup
        state.setup_score     = None
        state.setup_direction = None
        state.setup_type      = SetupType.NONE

    def _check_trade_exits(self, state: MarketState) -> None:
        """
        Evaluate exit conditions for the active trade.
        Fires a 'trade_exit_triggered' event if conditions are met.
        The actual exit execution is handled by TradeManager.
        """
        from strategy.rules import check_exit_conditions
        trade = state.active_trade
        if not trade:
            return

        ind_5m = state.indicators.get("5m")
        if ind_5m:
            trade.update_pnl(ind_5m.close)

        exit_signal = check_exit_conditions(trade, state)
        if exit_signal.exit_now:
            logger.info(f"Exit triggered: {exit_signal.reason} — {exit_signal.detail}")
            self._pending_events.append(MarketEvent(
                event_type="exit_triggered",
                timestamp=datetime.now(tz=timezone.utc),
                payload={
                    "trade_id":  trade.trade_id,
                    "reason":    exit_signal.reason.value if exit_signal.reason else "unknown",
                    "detail":    exit_signal.detail,
                    "urgency":   exit_signal.urgency,
                    "pnl_pct":   trade.pnl_pct,
                },
                priority=3 if exit_signal.urgency == "immediate" else 2,
            ))

    def _check_narrative_change(self, new_narrative: NarrativeState) -> None:
        """Fire an event when the market narrative changes."""
        if (self._prev_narrative
                and self._prev_narrative.narrative != new_narrative.narrative):
            logger.info(
                f"Narrative changed: {self._prev_narrative.narrative.value} → "
                f"{new_narrative.narrative.value}"
            )
            self._pending_events.append(MarketEvent(
                event_type="narrative_change",
                timestamp=datetime.now(tz=timezone.utc),
                payload={
                    "from":       self._prev_narrative.narrative.value,
                    "to":         new_narrative.narrative.value,
                    "confidence": new_narrative.confidence,
                    "description": new_narrative.description,
                },
                priority=2,
            ))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _infer_setup_type(state: MarketState, direction: TradeDirection) -> SetupType:
    """
    Classify the type of setup from the narrative and structure.
    """
    from engine.models import MarketNarrative
    if not state.narrative:
        return SetupType.NONE

    n = state.narrative.narrative

    if n in (MarketNarrative.BREAKOUT, MarketNarrative.BREAKDOWN):
        return SetupType.BREAKOUT if direction == TradeDirection.CALLS else SetupType.BREAKDOWN

    if n in (MarketNarrative.REVERSAL_BULL, MarketNarrative.REVERSAL_BEAR):
        return SetupType.REVERSAL

    if n in (MarketNarrative.TRENDING_BULL, MarketNarrative.TRENDING_BEAR):
        # In a trend, we're looking for continuations
        return SetupType.CONTINUATION

    return SetupType.NONE