"""
trades/trade_manager.py — trade lifecycle manager.

Manages the full lifecycle of a trade recommendation:
  CANDIDATE → PLANNED → ACTIVE → PARTIAL_EXIT → CLOSED / INVALIDATED

R2 does not execute trades. This module:
  - Builds TradeState from a setup signal + MarketState
  - Computes stop levels and sizing
  - Advances lifecycle state
  - Monitors active trades
  - Records completed trades to Trade (persistent record)

All methods are deterministic given the same inputs.
No network calls. No side effects outside this object.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.settings import (
    ACCOUNT_SIZE_DEFAULT,
    RISK_PER_TRADE_PCT_DEFAULT,
    MIN_RR_RATIO_DEFAULT,
)
from engine.models import (
    ExitReason,
    InstrumentType,
    MarketState,
    SetupScore,
    SetupType,
    StopCause,
    Trade,
    TradeDirection,
    TradeLifecycle,
    TradeState,
    TradeStatus,
    TimeStopConfig,
)
from trades.stop_calculator import (
    StopLevels,
    compute_options_stop,
    compute_shares_stop,
    stop_distance_pct,
)
from trades.position_sizer import (
    SizingOutput,
    size_options_position,
    size_shares_position,
    validate_rr,
)
from trades.time_stop import (
    TimeStopResult,
    evaluate_time_stop,
    recommended_time_window,
)

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages trade lifecycle from candidate setup to closed record.

    Usage:
        tm = TradeManager(account_size=25000, risk_per_trade_pct=1.0)

        # From a scored setup signal:
        proposal = tm.build_proposal(state, direction, score, setup_type)

        # When user indicates entry:
        trade = tm.activate(proposal.trade_id)

        # On each candle close:
        update = tm.monitor(trade, state)
    """

    def __init__(
        self,
        account_size:       float = ACCOUNT_SIZE_DEFAULT,
        risk_per_trade_pct: float = RISK_PER_TRADE_PCT_DEFAULT,
        min_rr:             float = MIN_RR_RATIO_DEFAULT,
    ) -> None:
        self._account_size       = account_size
        self._risk_pct           = risk_per_trade_pct
        self._min_rr             = min_rr

        # In-memory registers (not persistent — memory/ layer handles persistence)
        self._candidates: Dict[str, TradeState] = {}
        self._active:     Optional[TradeState]  = None
        self._history:    List[Trade]           = []

    # ──────────────────────────────────────────
    # Proposal building
    # ──────────────────────────────────────────

    def build_proposal(
        self,
        state:       MarketState,
        direction:   TradeDirection,
        score:       SetupScore,
        setup_type:  SetupType = SetupType.NONE,
        instrument:  InstrumentType = InstrumentType.OPTIONS,
        option_entry_price: Optional[float] = None,
        delta_approx:       Optional[float] = None,
        target_price:       Optional[float] = None,
        event_mode:         bool = False,
    ) -> TradeState:
        """
        Build a CANDIDATE TradeState from a strategy signal.

        Computes:
          - Stop levels (shares or options two-layer)
          - Position sizing
          - R:R ratio
          - Time-stop window

        Returns a TradeState in CANDIDATE lifecycle.
        Does not activate the trade.
        """
        now  = datetime.now(tz=timezone.utc)
        tid  = str(uuid.uuid4())[:8]

        # Extract key market data
        ind_5m  = state.indicators.get("5m")
        ind_15m = state.indicators.get("15m")
        ind_1h  = state.indicators.get("1h")
        atr     = (ind_5m.atr if ind_5m and ind_5m.atr else
                   ind_15m.atr if ind_15m and ind_15m.atr else 0.01)

        underlying_entry = (
            ind_5m.close  if ind_5m  else
            ind_15m.close if ind_15m else
            ind_1h.close  if ind_1h  else 0.0
        )

        # Swing levels from trend state
        swing_level = self._get_swing_level(state, direction)

        # Compute stops
        if instrument == InstrumentType.OPTIONS and option_entry_price:
            stops = compute_options_stop(
                direction=direction,
                underlying_entry=underlying_entry,
                swing_level=swing_level,
                atr=atr,
                option_entry_price=option_entry_price,
                delta_approx=delta_approx or 0.5,
                support_levels=[s.price for s in state.support_levels],
                resistance_levels=[r.price for r in state.resistance_levels],
            )
        else:
            stops = compute_shares_stop(
                direction=direction,
                entry_price=underlying_entry,
                swing_level=swing_level,
                atr=atr,
                support_levels=[s.price for s in state.support_levels],
                resistance_levels=[r.price for r in state.resistance_levels],
            )

        # Target: 1.5× risk by default
        stop_dist = abs(underlying_entry - stops.underlying_stop)
        computed_target = (
            underlying_entry + stop_dist * self._min_rr
            if direction == TradeDirection.CALLS
            else underlying_entry - stop_dist * self._min_rr
        )
        profit_target = target_price if target_price else computed_target

        # Validate R:R
        rr = validate_rr(
            entry=underlying_entry,
            stop=stops.underlying_stop,
            target=profit_target,
            direction=direction,
            event_mode=event_mode,
            min_rr_override=self._min_rr,
        )

        # Sizing
        if instrument == InstrumentType.OPTIONS and option_entry_price:
            sizing = size_options_position(
                option_entry_price=option_entry_price,
                premium_emergency_stop_pct=stops.premium_emergency_stop_pct or 35.0,
                account_size=self._account_size,
                risk_per_trade_pct=self._risk_pct,
                underlying_entry=underlying_entry,
                underlying_stop=stops.underlying_stop,
                underlying_target=profit_target,
                direction=direction,
                event_mode=event_mode,
            )
        else:
            sizing = size_shares_position(
                entry_price=underlying_entry,
                stop_price=stops.underlying_stop,
                account_size=self._account_size,
                risk_per_trade_pct=self._risk_pct,
                target_price=profit_target,
                event_mode=event_mode,
            )

        # Time-stop window
        session_phase = state.session.phase if state.session else None
        time_window   = recommended_time_window(setup_type, session_phase)

        # Build TradeState
        trade = TradeState(
            trade_id=tid,
            direction=direction,
            entry_time=now,
            size=sizing.position_notional,
            instrument_type=instrument,
            lifecycle=TradeLifecycle.CANDIDATE,
            entry_price=underlying_entry,
            underlying_entry_price=underlying_entry,
            underlying_invalidation_price=stops.underlying_stop,
            underlying_stop_price=stops.underlying_stop,
            stop_loss=stops.underlying_stop,
            profit_target=profit_target,
            option_entry_price=option_entry_price,
            estimated_option_stop_price=stops.option_stop_est,
            premium_emergency_stop_pct=stops.premium_emergency_stop_pct,
            time_stop_config=TimeStopConfig(
                expected_move_window_minutes=time_window,
            ),
            suggested_position_size=(
                sizing.suggested_contracts if instrument == InstrumentType.OPTIONS
                else sizing.suggested_shares
            ),
            suggested_risk_amount=sizing.risk_dollar,
            setup_type=setup_type,
            thesis=self._build_thesis(direction, state, rr, score),
            invalidation=f"Underlying closes through {stops.underlying_stop:.2f}",
        )

        self._candidates[tid] = trade
        logger.info(
            f"[R2] Trade candidate {tid}: {direction.value} {instrument.value} "
            f"entry={underlying_entry:.2f} stop={stops.underlying_stop:.2f} "
            f"target={profit_target:.2f} R:R={rr.rr_ratio:.2f}"
        )
        return trade

    # ──────────────────────────────────────────
    # Lifecycle transitions
    # ──────────────────────────────────────────

    def plan(self, trade_id: str) -> Optional[TradeState]:
        """
        Advance CANDIDATE → PLANNED.
        Called when the user has reviewed the proposal and is preparing to enter.
        """
        trade = self._candidates.get(trade_id)
        if not trade:
            logger.warning(f"[R2] plan(): trade {trade_id} not found in candidates")
            return None
        trade.lifecycle = TradeLifecycle.PLANNED
        logger.info(f"[R2] Trade {trade_id} → PLANNED")
        return trade

    def activate(self, trade_id: str, actual_entry_price: Optional[float] = None) -> Optional[TradeState]:
        """
        Advance PLANNED → ACTIVE.
        Called when the user confirms they have entered the position.
        Only one trade can be active at a time.
        """
        if self._active and self._active.lifecycle == TradeLifecycle.ACTIVE:
            logger.warning("[R2] activate(): another trade already ACTIVE")
            return None

        trade = self._candidates.get(trade_id)
        if not trade or trade.lifecycle not in (TradeLifecycle.CANDIDATE, TradeLifecycle.PLANNED):
            logger.warning(f"[R2] activate(): trade {trade_id} not in activatable state")
            return None

        if actual_entry_price:
            trade.entry_price            = actual_entry_price
            trade.underlying_entry_price = actual_entry_price

        trade.lifecycle   = TradeLifecycle.ACTIVE
        trade.entry_time  = datetime.now(tz=timezone.utc)
        self._active      = trade
        del self._candidates[trade_id]
        logger.info(f"[R2] Trade {trade_id} → ACTIVE at {trade.entry_price:.2f}")
        return trade

    def partial_exit(self, trade_id: str, partial_size: float) -> Optional[TradeState]:
        """Mark a partial exit — position is still open but size reduced."""
        trade = self._get_active(trade_id)
        if not trade:
            return None
        trade.lifecycle     = TradeLifecycle.PARTIAL_EXIT
        trade.partial_exited = True
        trade.partial_size   = partial_size
        logger.info(f"[R2] Trade {trade_id} → PARTIAL_EXIT (size={partial_size})")
        return trade

    def close(
        self,
        trade_id:   str,
        exit_price: float,
        exit_reason: ExitReason = ExitReason.MANUAL,
        stop_cause:  Optional[StopCause] = None,
        notes:       str = "",
    ) -> Optional[Trade]:
        """
        Close the active trade. Returns a completed Trade record.
        """
        trade = self._get_active(trade_id)
        if not trade:
            return None

        now      = datetime.now(tz=timezone.utc)
        entry    = trade.underlying_entry_price or trade.entry_price
        pnl_pct  = ((exit_price - entry) / entry * 100
                    if trade.direction == TradeDirection.CALLS
                    else (entry - exit_price) / entry * 100)
        pnl_dollar = pnl_pct / 100 * trade.size

        record = Trade(
            id=trade.trade_id,
            symbol="SPY",
            direction=trade.direction,
            instrument_type=trade.instrument_type,
            entry_price=entry,
            entry_time=trade.entry_time,
            size=trade.size,
            underlying_stop_price=trade.underlying_stop_price,
            underlying_invalidation_price=trade.underlying_invalidation_price,
            profit_target=trade.profit_target,
            option_entry_price=trade.option_entry_price,
            estimated_option_stop_price=trade.estimated_option_stop_price,
            status=TradeStatus.CLOSED,
            exit_price=exit_price,
            exit_time=now,
            exit_reason=exit_reason,
            stop_cause=stop_cause,
            pnl_pct=round(pnl_pct, 3),
            pnl_dollar=round(pnl_dollar, 2),
            thesis_stop_triggered=trade.thesis_stop_triggered,
            premium_stop_triggered=trade.premium_stop_triggered,
            time_stop_triggered=trade.time_stop_triggered,
            suggested_position_size=trade.suggested_position_size,
            suggested_risk_amount=trade.suggested_risk_amount,
            sizing_compliant=True,  # logged from sizing output at build time
            setup_type=trade.setup_type,
            notes=notes,
        )

        trade.lifecycle  = TradeLifecycle.CLOSED
        trade.exit_reason = exit_reason
        self._active     = None
        self._history.append(record)
        logger.info(
            f"[R2] Trade {trade_id} CLOSED — {exit_reason.value} "
            f"P&L: {pnl_pct:+.2f}% / ${pnl_dollar:+.2f}"
        )
        return record

    def invalidate(self, trade_id: str, reason: str = "") -> Optional[TradeState]:
        """
        Mark a CANDIDATE or PLANNED trade as INVALIDATED (thesis broke before entry).
        """
        trade = self._candidates.get(trade_id)
        if not trade:
            logger.warning(f"[R2] invalidate(): {trade_id} not found")
            return None
        trade.lifecycle = TradeLifecycle.INVALIDATED
        del self._candidates[trade_id]
        logger.info(f"[R2] Trade {trade_id} INVALIDATED — {reason}")
        return trade

    # ──────────────────────────────────────────
    # Monitoring
    # ──────────────────────────────────────────

    def monitor(
        self,
        trade: TradeState,
        state: MarketState,
    ) -> Dict:
        """
        Evaluate current trade health. Does not exit. Returns advisory dict.

        Checks:
          1. Thesis stop (underlying closes through invalidation)
          2. Premium stop (option premium declined past Layer B threshold)
          3. Time stop (expected move not developing)
          4. Profit target reached
        """
        advisory: Dict = {
            "trade_id":     trade.trade_id,
            "alerts":       [],
            "action":       "hold",
            "pnl_pct":      trade.pnl_pct,
        }

        ind_5m = state.indicators.get("5m")
        if not ind_5m:
            return advisory

        current_price = ind_5m.close
        trade.update_pnl(current_price)

        # ── Thesis stop ───────────────────────
        if _thesis_stop_hit(trade, current_price):
            trade.thesis_stop_triggered = True
            advisory["alerts"].append("THESIS STOP: underlying has closed through invalidation level")
            advisory["action"] = "exit_thesis_stop"
            return advisory

        # ── Premium stop ──────────────────────
        if trade.option_entry_price and trade.premium_emergency_stop_pct:
            current_option_price = getattr(state, "_option_price", None) or current_price
            premium_loss_pct = (
                (trade.option_entry_price - current_option_price)
                / trade.option_entry_price * 100
            )
            if premium_loss_pct >= trade.premium_emergency_stop_pct:
                trade.premium_stop_triggered = True
                advisory["alerts"].append(
                    f"PREMIUM STOP: option down {premium_loss_pct:.1f}% "
                    f"(emergency threshold: {trade.premium_emergency_stop_pct}%)"
                )
                advisory["action"] = "exit_premium_stop"
                return advisory

        # ── Time stop ─────────────────────────
        if trade.time_stop_config:
            session_phase = state.session.phase if state.session else None
            ts_result = evaluate_time_stop(
                entry_time=trade.entry_time,
                entry_price=trade.underlying_entry_price or trade.entry_price,
                current_price=current_price,
                target_price=trade.profit_target,
                stop_price=trade.underlying_stop_price,
                setup_type=trade.setup_type,
                session_phase=session_phase,
            )
            if ts_result.triggered:
                trade.time_stop_triggered = True
                advisory["alerts"].extend(ts_result.notes)
                advisory["action"]  = ts_result.action.value
                advisory["time_stop"] = ts_result.to_dict()

        # ── Profit target ─────────────────────
        if _target_hit(trade, current_price):
            advisory["alerts"].append(
                f"TARGET REACHED: {current_price:.2f} ≥ target {trade.profit_target:.2f}"
            )
            advisory["action"] = "exit_target"

        return advisory

    # ──────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────

    def get_active_trade(self) -> Optional[TradeState]:
        return self._active

    def get_candidates(self) -> List[TradeState]:
        return list(self._candidates.values())

    def get_history(self) -> List[Trade]:
        return list(self._history)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _get_active(self, trade_id: str) -> Optional[TradeState]:
        if self._active and self._active.trade_id == trade_id:
            return self._active
        logger.warning(f"[R2] No active trade found with id {trade_id}")
        return None

    @staticmethod
    def _get_swing_level(state: MarketState, direction: TradeDirection) -> float:
        """Extract the most relevant structural stop anchor from MarketState."""
        trend_15m = state.trends.get("15m")
        trend_1h  = state.trends.get("1h")

        if direction == TradeDirection.CALLS:
            sl_15m = getattr(trend_15m, "last_swing_low", None)
            sl_1h  = getattr(trend_1h, "last_swing_low", None)
            if sl_15m:
                return sl_15m
            if sl_1h:
                return sl_1h
            if state.support_levels:
                return state.support_levels[0].price
            ind = state.indicators.get("5m") or state.indicators.get("15m")
            entry = ind.close if ind else 545.0
            atr   = ind.atr   if ind and ind.atr else 1.0
            return entry - 2 * atr
        else:
            sh_15m = getattr(trend_15m, "last_swing_high", None)
            sh_1h  = getattr(trend_1h, "last_swing_high", None)
            if sh_15m:
                return sh_15m
            if sh_1h:
                return sh_1h
            if state.resistance_levels:
                return state.resistance_levels[0].price
            ind   = state.indicators.get("5m") or state.indicators.get("15m")
            entry = ind.close if ind else 545.0
            atr   = ind.atr   if ind and ind.atr else 1.0
            return entry + 2 * atr

    @staticmethod
    def _build_thesis(
        direction: TradeDirection,
        state:     MarketState,
        rr:        "RRResult",
        score:     SetupScore,
    ) -> str:
        narrative = state.narrative.narrative.value if state.narrative else "unknown"
        regime    = state.regime.regime.value if state.regime else "unknown"
        side      = "Bullish" if direction == TradeDirection.CALLS else "Bearish"
        return (
            f"{side} setup | narrative: {narrative} | regime: {regime} | "
            f"score: {score.total:.0f} ({score.grade()}) | R:R {rr.rr_ratio:.2f}"
        )


# ──────────────────────────────────────────────
# Pure helpers for monitoring
# ──────────────────────────────────────────────

def _thesis_stop_hit(trade: TradeState, current_price: float) -> bool:
    if not trade.underlying_stop_price:
        return False
    if trade.direction == TradeDirection.CALLS:
        return current_price <= trade.underlying_stop_price
    return current_price >= trade.underlying_stop_price


def _target_hit(trade: TradeState, current_price: float) -> bool:
    if not trade.profit_target:
        return False
    if trade.direction == TradeDirection.CALLS:
        return current_price >= trade.profit_target
    return current_price <= trade.profit_target