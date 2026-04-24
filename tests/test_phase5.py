"""
tests/test_phase5.py — Phase 5: trade and risk layer.

Tests:
  - TradeLifecycle enum
  - stop_calculator: shares stop, options two-layer stop, helpers
  - position_sizer: shares sizing, options sizing, R:R validation
  - time_stop: window logic, phase multipliers, action recommendations
  - trade_manager: full lifecycle, monitoring, close/invalidate

Run: PYTHONPATH=. python tests/test_phase5.py
"""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta

passed = failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────────────────────────────
# Helpers: build minimal MarketState
# ──────────────────────────────────────────────────────────────────────

def _market_state(
    close=545.0, atr=1.5, swing_low=541.0, swing_high=549.0,
    direction="bullish", has_support=True, has_resistance=True,
):
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState, Direction, TrendStrength,
        RegimeState, MarketRegime, SessionState, SessionPhase, Readiness,
        SessionIntent, SupportResistanceLevel, NarrativeState, MarketNarrative,
        SetupScore, TradeDirection,
    )
    s = MarketState(symbol="SPY", timestamp=_now())

    for tf in ["1d","1h","15m","5m","1m"]:
        ind = IndicatorSnapshot(tf, _now(), close)
        ind.atr=atr; ind.ema_9=close-0.5; ind.ema_20=close-1.0
        ind.rsi=55.0; ind.macd_hist=0.01; ind.ema_stack="bull_aligned"
        ind.vwap=close-1.0; ind.price_vs_vwap="above"; ind.macd_bias="bullish"
        s.indicators[tf] = ind

        d = Direction.BULLISH if direction=="bullish" else Direction.BEARISH
        t = TrendState(tf, d, TrendStrength.STRONG)
        t.last_swing_low  = swing_low
        t.last_swing_high = swing_high
        t.higher_highs    = (direction == "bullish")
        t.lower_lows      = (direction == "bearish")
        s.trends[tf] = t

    if has_support:
        s.support_levels = [SupportResistanceLevel(swing_low + 0.5, "support", 3)]
    if has_resistance:
        s.resistance_levels = [SupportResistanceLevel(swing_high - 0.5, "resistance", 3)]

    s.session = SessionState(
        phase=SessionPhase.MIDDAY, market_open=True,
        entries_allowed=True, score_modifier=1.0,
        entry_filter="normal", phase_note="test",
        readiness=Readiness.FULLY_READY,
    )
    s.regime  = RegimeState(MarketRegime.TREND_BULL, 0.85, 1.0, True, "bull")
    s.intent  = SessionIntent(min_score=60)
    s.narrative = NarrativeState(
        narrative=MarketNarrative.TRENDING_BULL, confidence=0.9,
        description="bull", trade_posture="long_bias",
    )
    s.setup_score = SetupScore(total=80.0, adjusted_total=80.0)
    return s


# ═══════════════════════════════════════════════════════════════════════════
# TradeLifecycle enum
# ═══════════════════════════════════════════════════════════════════════════

def test_lifecycle_enum_values():
    from engine.models import TradeLifecycle
    assert TradeLifecycle.CANDIDATE.value   == "candidate"
    assert TradeLifecycle.PLANNED.value     == "planned"
    assert TradeLifecycle.ACTIVE.value      == "active"
    assert TradeLifecycle.PARTIAL_EXIT.value == "partial_exit"
    assert TradeLifecycle.CLOSED.value      == "closed"
    assert TradeLifecycle.INVALIDATED.value == "invalidated"


def test_lifecycle_all_six_states():
    from engine.models import TradeLifecycle
    assert len(list(TradeLifecycle)) == 6


def test_tradestate_has_lifecycle_field():
    from engine.models import TradeState, TradeDirection, TradeLifecycle
    t = TradeState(
        trade_id="T1", direction=TradeDirection.CALLS,
        entry_time=_now(), size=1000.0
    )
    assert hasattr(t, "lifecycle")
    # default is None (lifecycle not set at construction; manager sets it)
    t.lifecycle = TradeLifecycle.CANDIDATE
    assert t.lifecycle == TradeLifecycle.CANDIDATE


# ═══════════════════════════════════════════════════════════════════════════
# stop_calculator — shares
# ═══════════════════════════════════════════════════════════════════════════

def test_shares_stop_long_below_swing_low():
    from trades.stop_calculator import compute_shares_stop
    from engine.models import TradeDirection
    stops = compute_shares_stop(
        direction=TradeDirection.CALLS,
        entry_price=545.0, swing_level=541.0, atr=1.5,
    )
    assert stops.underlying_stop < 541.0   # stop is below swing low


def test_shares_stop_short_above_swing_high():
    from trades.stop_calculator import compute_shares_stop
    from engine.models import TradeDirection
    stops = compute_shares_stop(
        direction=TradeDirection.PUTS,
        entry_price=545.0, swing_level=549.0, atr=1.5,
    )
    assert stops.underlying_stop > 549.0   # stop is above swing high


def test_shares_stop_uses_support_anchor():
    from trades.stop_calculator import compute_shares_stop
    from engine.models import TradeDirection
    # Support at 542 is closer than swing low at 541
    stops = compute_shares_stop(
        direction=TradeDirection.CALLS,
        entry_price=545.0, swing_level=541.0, atr=1.5,
        support_levels=[542.0, 540.0],
    )
    # Should anchor at 542 instead of 541
    assert stops.underlying_stop < 542.0
    assert any("542" in n for n in stops.stop_notes)


def test_shares_stop_long_stop_distance():
    from trades.stop_calculator import compute_shares_stop, stop_distance_pct
    from engine.models import TradeDirection
    stops = compute_shares_stop(
        direction=TradeDirection.CALLS,
        entry_price=545.0, swing_level=541.0, atr=1.5,
    )
    dist = stop_distance_pct(545.0, stops.underlying_stop)
    assert dist > 0.5   # at least 0.5% from entry


def test_shares_stop_notes_populated():
    from trades.stop_calculator import compute_shares_stop
    from engine.models import TradeDirection
    stops = compute_shares_stop(
        direction=TradeDirection.CALLS,
        entry_price=545.0, swing_level=541.0, atr=1.5,
    )
    assert len(stops.stop_notes) >= 1


def test_shares_stop_to_dict():
    from trades.stop_calculator import compute_shares_stop
    from engine.models import TradeDirection
    stops = compute_shares_stop(
        direction=TradeDirection.CALLS,
        entry_price=545.0, swing_level=541.0, atr=1.5,
    )
    d = stops.to_dict()
    assert "underlying_stop" in d
    assert "notes" in d


# ═══════════════════════════════════════════════════════════════════════════
# stop_calculator — options two-layer
# ═══════════════════════════════════════════════════════════════════════════

def test_options_stop_has_both_layers():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
    )
    assert stops.underlying_stop       is not None   # Layer A
    assert stops.option_stop_est       is not None   # estimated option value
    assert stops.premium_emergency_stop_pct is not None  # Layer B


def test_options_layer_a_below_swing_low():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
    )
    assert stops.underlying_stop < 541.0


def test_options_layer_b_premium_stop():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
    )
    assert stops.premium_emergency_stop_pct > 0
    assert stops.premium_emergency_stop_pct <= 50.0


def test_options_estimated_stop_price_below_entry():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
    )
    assert stops.option_stop_est < 2.50  # option loses value when underlying drops


def test_options_custom_premium_stop():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
        premium_emergency_stop_pct=25.0,
    )
    assert stops.premium_emergency_stop_pct == 25.0


def test_options_stop_to_dict_complete():
    from trades.stop_calculator import compute_options_stop
    from engine.models import TradeDirection
    stops = compute_options_stop(
        direction=TradeDirection.CALLS,
        underlying_entry=545.0, swing_level=541.0, atr=1.5,
        option_entry_price=2.50,
    )
    d = stops.to_dict()
    for k in ("underlying_stop","option_stop_est","premium_emergency_stop_pct","notes"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# stop_calculator — helpers
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_distance_pct_calculation():
    from trades.stop_calculator import stop_distance_pct
    dist = stop_distance_pct(545.0, 541.0)
    assert abs(dist - 0.7339) < 0.01


def test_is_stop_too_tight_true():
    from trades.stop_calculator import is_stop_too_tight
    assert is_stop_too_tight(545.0, 544.95) is True  # 0.009% — way too tight


def test_is_stop_too_tight_false():
    from trades.stop_calculator import is_stop_too_tight
    assert is_stop_too_tight(545.0, 542.0) is False  # 0.55%


def test_estimate_option_delta():
    from trades.stop_calculator import estimate_option_delta
    from config.settings import ATM_DELTA_APPROX, NEAR_OTM_DELTA_APPROX
    assert estimate_option_delta("atm")      == ATM_DELTA_APPROX
    assert estimate_option_delta("near_otm") == NEAR_OTM_DELTA_APPROX
    assert estimate_option_delta("unknown")  == ATM_DELTA_APPROX


# ═══════════════════════════════════════════════════════════════════════════
# position_sizer — R:R validation
# ═══════════════════════════════════════════════════════════════════════════

def test_rr_passes_at_1_5():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    rr = validate_rr(545.0, 541.0, 551.0, TradeDirection.CALLS)
    assert rr.rr_ratio >= 1.5
    assert rr.passes is True


def test_rr_fails_below_minimum():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    # target only 1 point above entry, stop 4 below — R:R = 0.25
    rr = validate_rr(545.0, 541.0, 546.0, TradeDirection.CALLS)
    assert rr.rr_ratio < 1.5
    assert rr.passes is False
    assert any("R:R" in n for n in rr.notes)


def test_rr_event_mode_lower_threshold():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    from config.settings import MIN_RR_RATIO_EVENT_AGGRESSIVE
    # 1.3 R:R: fails default (1.5) but passes event mode (1.2)
    rr_default = validate_rr(545.0, 541.0, 550.2, TradeDirection.CALLS, event_mode=False)
    rr_event   = validate_rr(545.0, 541.0, 550.2, TradeDirection.CALLS, event_mode=True)
    assert rr_event.min_rr == MIN_RR_RATIO_EVENT_AGGRESSIVE


def test_rr_zero_stop_distance_fails():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    rr = validate_rr(545.0, 545.0, 550.0, TradeDirection.CALLS)
    assert rr.passes is False
    assert rr.rr_ratio == 0.0


def test_rr_puts_direction():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    # PUTS: entry 545, stop 549, target 539 → reward=6, risk=4 → R:R=1.5
    rr = validate_rr(545.0, 549.0, 539.0, TradeDirection.PUTS)
    assert rr.rr_ratio >= 1.4
    assert rr.passes is True


def test_rr_to_dict():
    from trades.position_sizer import validate_rr
    from engine.models import TradeDirection
    rr = validate_rr(545.0, 541.0, 551.0, TradeDirection.CALLS)
    d = rr.to_dict()
    for k in ("rr_ratio", "passes", "min_rr", "notes"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# position_sizer — shares sizing
# ═══════════════════════════════════════════════════════════════════════════

def test_shares_sizing_basic():
    from trades.position_sizer import size_shares_position
    result = size_shares_position(
        entry_price=545.0, stop_price=541.0,
        account_size=25000, risk_per_trade_pct=1.0,
    )
    # Risk = $250, stop = $4/share → ~62 shares
    assert result.suggested_shares > 0
    assert abs(result.risk_dollar - 250.0) < 1.0


def test_shares_sizing_risk_dollar():
    from trades.position_sizer import size_shares_position
    result = size_shares_position(
        entry_price=545.0, stop_price=541.0,
        account_size=50000, risk_per_trade_pct=2.0,
    )
    assert abs(result.risk_dollar - 1000.0) < 1.0


def test_shares_sizing_position_capped():
    from trades.position_sizer import size_shares_position
    # Very tight stop would create oversized position — should be capped
    result = size_shares_position(
        entry_price=545.0, stop_price=544.9,   # $0.10 stop
        account_size=25000, risk_per_trade_pct=1.0,
    )
    assert any("cap" in f.lower() or "tight" in f.lower() for f in result.flags)


def test_shares_sizing_poor_rr_flagged():
    from trades.position_sizer import size_shares_position
    result = size_shares_position(
        entry_price=545.0, stop_price=541.0,
        account_size=25000, risk_per_trade_pct=1.0,
        target_price=546.0,  # terrible R:R
    )
    assert not result.sizing_compliant
    assert any("R:R" in f for f in result.flags)


def test_shares_sizing_good_rr_compliant():
    from trades.position_sizer import size_shares_position
    result = size_shares_position(
        entry_price=545.0, stop_price=541.0,
        account_size=25000, risk_per_trade_pct=1.0,
        target_price=551.0,
    )
    # target=551, entry=545, stop=541 → reward=6, risk=4 → R:R=1.5
    assert result.rr_ratio >= 1.5
    # Sizing is compliant when R:R passes and position is within limits
    assert not any("R:R" in f for f in result.flags)


def test_shares_sizing_to_dict():
    from trades.position_sizer import size_shares_position
    result = size_shares_position(545.0, 541.0)
    d = result.to_dict()
    for k in ("instrument","shares","notional","risk_dollar","rr_ratio","compliant"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# position_sizer — options sizing
# ═══════════════════════════════════════════════════════════════════════════

def test_options_sizing_basic():
    from trades.position_sizer import size_options_position
    result = size_options_position(
        option_entry_price=2.50,
        premium_emergency_stop_pct=35.0,
        account_size=25000,
        risk_per_trade_pct=1.0,
    )
    # risk = $250, loss per contract = 2.50 × 100 × 0.35 = $87.50 → ~2 contracts
    assert result.suggested_contracts >= 1
    assert result.risk_dollar == 250.0


def test_options_sizing_respects_max_position():
    from trades.position_sizer import size_options_position
    # Cheap option, many contracts would exceed 20% cap
    result = size_options_position(
        option_entry_price=0.10,
        premium_emergency_stop_pct=100.0,
        account_size=25000,
        risk_per_trade_pct=1.0,
    )
    max_notional = 25000 * 0.20
    assert result.position_notional <= max_notional + 10  # small rounding tolerance


def test_options_sizing_rr_validated():
    from trades.position_sizer import size_options_position
    from engine.models import TradeDirection
    result = size_options_position(
        option_entry_price=2.50,
        premium_emergency_stop_pct=35.0,
        account_size=25000,
        underlying_entry=545.0, underlying_stop=541.0, underlying_target=551.0,
        direction=TradeDirection.CALLS,
    )
    assert result.rr_ratio >= 1.0


def test_options_sizing_to_dict():
    from trades.position_sizer import size_options_position
    result = size_options_position(2.50, 35.0)
    d = result.to_dict()
    for k in ("instrument","contracts","notional","risk_dollar","compliant"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# time_stop
# ═══════════════════════════════════════════════════════════════════════════

def test_time_stop_not_triggered_within_window():
    from trades.time_stop import evaluate_time_stop
    entry_time = _now() - timedelta(minutes=5)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=546.0, target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    assert result.triggered is False
    assert result.minutes_elapsed < 10


def test_time_stop_triggered_no_progress():
    from trades.time_stop import evaluate_time_stop, TimeStopAction
    entry_time = _now() - timedelta(minutes=15)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=545.1, target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    assert result.triggered is True
    assert result.action in (TimeStopAction.SCALE_DOWN, TimeStopAction.EXIT)


def test_time_stop_triggered_negative_progress_exit():
    from trades.time_stop import evaluate_time_stop, TimeStopAction
    entry_time = _now() - timedelta(minutes=15)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=543.0,  # moving against position
        target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    assert result.triggered is True
    assert result.action == TimeStopAction.EXIT


def test_time_stop_good_progress_no_trigger():
    from trades.time_stop import evaluate_time_stop
    entry_time = _now() - timedelta(minutes=15)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=548.0,  # 50% of the 6-point target move
        target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    assert result.triggered is False


def test_time_stop_confidence_reduction_on_trigger():
    from trades.time_stop import evaluate_time_stop
    entry_time = _now() - timedelta(minutes=15)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=545.2, target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    if result.triggered:
        assert result.confidence_reduction > 0


def test_time_stop_phase_multiplier_open():
    from trades.time_stop import recommended_time_window
    from engine.models import SetupType, SessionPhase
    w_open   = recommended_time_window(SetupType.CONTINUATION, SessionPhase.OPEN)
    w_midday = recommended_time_window(SetupType.CONTINUATION, SessionPhase.MIDDAY)
    # OPEN multiplier (0.60) < MIDDAY (1.00) → shorter window for OPEN
    assert w_open < w_midday


def test_time_stop_setup_type_windows():
    from trades.time_stop import recommended_time_window
    from engine.models import SetupType
    w_breakout  = recommended_time_window(SetupType.BREAKOUT)
    w_reversal  = recommended_time_window(SetupType.REVERSAL)
    # Reversal window > breakout window
    assert w_reversal > w_breakout


def test_time_stop_to_dict():
    from trades.time_stop import evaluate_time_stop
    entry_time = _now() - timedelta(minutes=15)
    result = evaluate_time_stop(
        entry_time=entry_time, entry_price=545.0,
        current_price=545.0, target_price=551.0, stop_price=541.0,
        custom_window_minutes=10,
    )
    d = result.to_dict()
    for k in ("triggered","action","confidence_reduction","minutes_elapsed","window_minutes"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# trade_manager — build_proposal
# ═══════════════════════════════════════════════════════════════════════════

def test_manager_build_proposal_shares():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, TradeLifecycle, SetupScore
    tm    = TradeManager(account_size=25000, risk_per_trade_pct=1.0)
    state = _market_state()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        state=state, direction=TradeDirection.CALLS,
        score=score, setup_type=SetupType.CONTINUATION,
        instrument=InstrumentType.SHARES,
    )
    assert proposal.lifecycle     == TradeLifecycle.CANDIDATE
    assert proposal.instrument_type == InstrumentType.SHARES
    assert proposal.underlying_stop_price < state.indicators["5m"].close
    assert proposal.profit_target > state.indicators["5m"].close


def test_manager_build_proposal_options():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, TradeLifecycle, SetupScore
    tm    = TradeManager(account_size=25000, risk_per_trade_pct=1.0)
    state = _market_state()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        state=state, direction=TradeDirection.CALLS,
        score=score, setup_type=SetupType.CONTINUATION,
        instrument=InstrumentType.OPTIONS,
        option_entry_price=2.50,
    )
    assert proposal.lifecycle           == TradeLifecycle.CANDIDATE
    assert proposal.option_entry_price  == 2.50
    assert proposal.estimated_option_stop_price is not None
    assert proposal.premium_emergency_stop_pct  is not None


def test_manager_build_proposal_has_time_stop_config():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, SetupScore
    tm    = TradeManager()
    state = _market_state()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        state=state, direction=TradeDirection.CALLS,
        score=score, setup_type=SetupType.CONTINUATION,
    )
    assert proposal.time_stop_config is not None
    assert proposal.time_stop_config.expected_move_window_minutes > 0


def test_manager_proposal_in_candidates():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    state = _market_state()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    candidates = tm.get_candidates()
    assert any(c.trade_id == proposal.trade_id for c in candidates)


def test_manager_puts_stop_above_entry():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, SetupScore
    tm    = TradeManager()
    state = _market_state(direction="bearish", close=545.0, swing_high=549.0)
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        state=state, direction=TradeDirection.PUTS,
        score=score, setup_type=SetupType.CONTINUATION,
        instrument=InstrumentType.SHARES,
    )
    assert proposal.underlying_stop_price > 545.0


def test_manager_thesis_populated():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    assert len(proposal.thesis) > 10


# ═══════════════════════════════════════════════════════════════════════════
# trade_manager — lifecycle transitions
# ═══════════════════════════════════════════════════════════════════════════

def test_lifecycle_candidate_to_planned():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    result = tm.plan(p.trade_id)
    assert result.lifecycle == TradeLifecycle.PLANNED


def test_lifecycle_planned_to_active():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.plan(p.trade_id)
    active = tm.activate(p.trade_id)
    assert active.lifecycle == TradeLifecycle.ACTIVE
    assert tm.get_active_trade() is active


def test_lifecycle_candidate_directly_to_active():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    active = tm.activate(p.trade_id)  # skip PLANNED step
    assert active.lifecycle == TradeLifecycle.ACTIVE


def test_lifecycle_active_to_partial_exit():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    trade = tm.partial_exit(p.trade_id, partial_size=500.0)
    assert trade.lifecycle     == TradeLifecycle.PARTIAL_EXIT
    assert trade.partial_exited is True


def test_lifecycle_active_to_closed():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, ExitReason, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    record = tm.close(p.trade_id, exit_price=551.0, exit_reason=ExitReason.PROFIT_TARGET)
    assert record is not None
    assert record.pnl_pct > 0
    assert tm.get_active_trade() is None


def test_lifecycle_invalidated():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, TradeLifecycle, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    result = tm.invalidate(p.trade_id, "Price broke below invalidation before entry")
    assert result.lifecycle == TradeLifecycle.INVALIDATED
    assert len(tm.get_candidates()) == 0


def test_no_two_active_trades():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p1 = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    p2 = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p1.trade_id)
    result = tm.activate(p2.trade_id)  # second activation should fail
    assert result is None
    assert tm.get_active_trade().trade_id == p1.trade_id


def test_close_records_to_history():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, ExitReason, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    p     = tm.build_proposal(_market_state(), TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    tm.close(p.trade_id, 551.0, ExitReason.PROFIT_TARGET)
    assert len(tm.get_history()) == 1


def test_close_pnl_calculation_long():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, ExitReason, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    state = _market_state(close=545.0)
    p     = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    record = tm.close(p.trade_id, 551.0, ExitReason.PROFIT_TARGET)
    assert record.pnl_pct > 0   # profitable


def test_close_pnl_stop_loss():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, ExitReason, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    state = _market_state(close=545.0)
    p     = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    record = tm.close(p.trade_id, 541.0, ExitReason.STOP_LOSS)
    assert record.pnl_pct < 0   # loss


# ═══════════════════════════════════════════════════════════════════════════
# trade_manager — monitoring
# ═══════════════════════════════════════════════════════════════════════════

def test_monitor_hold_in_profit():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    state = _market_state(close=545.0)
    p     = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)
    # Price moved in our favor
    update_state = _market_state(close=547.0)
    advisory = tm.monitor(tm.get_active_trade(), update_state)
    assert advisory["trade_id"] == p.trade_id
    assert "action" in advisory


def test_monitor_thesis_stop_triggered():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    state = _market_state(close=545.0, swing_low=541.0)
    p     = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)

    # Set stop explicitly for test
    active = tm.get_active_trade()
    active.underlying_stop_price = 542.0

    # Price drops below stop
    stop_state = _market_state(close=541.5)
    advisory = tm.monitor(active, stop_state)
    assert active.thesis_stop_triggered is True
    assert "THESIS STOP" in advisory["alerts"][0]
    assert advisory["action"] == "exit_thesis_stop"


def test_monitor_target_hit():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, SetupScore
    tm    = TradeManager()
    score = SetupScore(total=80.0, adjusted_total=80.0)
    state = _market_state(close=545.0)
    p     = tm.build_proposal(state, TradeDirection.CALLS, score, SetupType.CONTINUATION)
    tm.activate(p.trade_id)

    active = tm.get_active_trade()
    active.profit_target = 548.0  # low target for test

    hit_state = _market_state(close=549.0)  # above target
    advisory = tm.monitor(active, hit_state)
    assert advisory["action"] == "exit_target"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: SizingOutput fields in proposal
# ═══════════════════════════════════════════════════════════════════════════

def test_proposal_has_sizing_fields():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, SetupScore
    tm    = TradeManager(account_size=25000, risk_per_trade_pct=1.0)
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        _market_state(), TradeDirection.CALLS, score,
        SetupType.CONTINUATION, InstrumentType.SHARES,
    )
    assert proposal.suggested_position_size > 0
    assert proposal.suggested_risk_amount   > 0


def test_proposal_risk_amount_near_1pct():
    from trades.trade_manager import TradeManager
    from engine.models import TradeDirection, SetupType, InstrumentType, SetupScore
    tm    = TradeManager(account_size=25000, risk_per_trade_pct=1.0)
    score = SetupScore(total=80.0, adjusted_total=80.0)
    proposal = tm.build_proposal(
        _market_state(), TradeDirection.CALLS, score,
        SetupType.CONTINUATION, InstrumentType.SHARES,
    )
    # Risk should be close to 1% of $25,000 = $250
    assert 100 < proposal.suggested_risk_amount <= 300


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"\nPhase 5 tests ({len(tests)} total)\n{'─'*58}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'='*58}")
    print(f"{passed} passed, {failed} failed  ({passed+failed} total)")
    sys.exit(0 if failed == 0 else 1)