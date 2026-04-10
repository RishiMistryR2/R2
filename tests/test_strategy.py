"""
Tests for the strategy layer.
Run: PYTHONPATH=. python tests/test_strategy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from engine.models import (
    Direction, IndicatorSnapshot, MarketState, MarketNarrative,
    NarrativeState, SetupType, TradeDirection, TradeState, TradeStatus,
    TrendState, TrendStrength, ExitReason,
    RegimeState, MarketRegime, SessionState, SessionPhase, SessionIntent,
)
from strategy.timeframes import (
    compute_tf_consensus, get_htf_bias, is_entry_allowed,
    get_position_size_modifier,
)
from strategy.narrative  import classify_narrative
from strategy.rules      import (
    check_long_entry, check_short_entry,
    detect_trend_weakening, detect_trend_flip,
    check_exit_conditions, ExitSignal,
)
from strategy.scorer     import score_setup
from strategy.strategy_engine import StrategyEngine


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _now():
    return datetime.now(tz=timezone.utc)

def _ind(tf, close=545.0, rsi=55.0, macd_hist=0.01, ema_stack="bull_aligned",
         vwap=542.0, atr=1.2, ema9=543.0, ema20=541.0, ema50=538.0, ema200=530.0,
         price_vs_vwap="above", macd_bias="bullish", rsi_zone="bullish_neutral"):
    s = IndicatorSnapshot(timeframe=tf, timestamp=_now(), close=close)
    s.rsi=rsi; s.macd_hist=macd_hist; s.ema_stack=ema_stack; s.vwap=vwap
    s.atr=atr; s.ema_9=ema9; s.ema_20=ema20; s.ema_50=ema50; s.ema_200=ema200
    s.price_vs_vwap=price_vs_vwap; s.macd_bias=macd_bias; s.rsi_zone=rsi_zone
    return s

def _trend(tf, direction=Direction.BULLISH, strength=TrendStrength.MODERATE,
           is_weakening=False, structure_break=False, last_swing_low=None,
           last_swing_high=None, higher_highs=False, lower_lows=False):
    t = TrendState(timeframe=tf, direction=direction, strength=strength)
    t.is_weakening=is_weakening; t.structure_break=structure_break
    t.last_swing_low=last_swing_low; t.last_swing_high=last_swing_high
    t.higher_highs=higher_highs; t.lower_lows=lower_lows
    return t

def _bull_state(score_threshold=False):
    """Fully aligned bull state with midday session and trend regime."""
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d","1h","15m","5m","1m"]:
        t = _trend(tf)
        t.higher_highs = True
        s.trends[tf]    = t
        s.indicators[tf] = _ind(tf)
    s.session = SessionState(
        phase=SessionPhase.MIDDAY,
        market_open=True,
        entries_allowed=True,
        score_modifier=1.0,
        entry_filter="normal",
        phase_note="normal operations",
    )
    s.regime = RegimeState(
        regime=MarketRegime.TREND_BULL,
        confidence=0.85,
        score_multiplier=1.0,
        entry_permitted=True,
        description="bull trend",
    )
    s.intent = SessionIntent(min_score=60, risk_tolerance="normal")
    return s

def _bear_state():
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d","1h","15m","5m","1m"]:
        t = _trend(tf, direction=Direction.BEARISH)
        t.lower_lows = True
        s.trends[tf] = t
        s.indicators[tf] = _ind(tf, close=540.0, rsi=48.0, macd_hist=-0.01,
                                 ema_stack="bear_aligned", vwap=543.0,
                                 ema9=542.0, ema20=544.0,
                                 price_vs_vwap="below", macd_bias="bearish",
                                 rsi_zone="bearish_neutral")
    s.session = SessionState(
        phase=SessionPhase.MIDDAY,
        market_open=True,
        entries_allowed=True,
        score_modifier=1.0,
        entry_filter="normal",
        phase_note="normal operations",
    )
    s.regime = RegimeState(
        regime=MarketRegime.TREND_BEAR,
        confidence=0.85,
        score_multiplier=1.0,
        entry_permitted=True,
        description="bear trend",
    )
    s.intent = SessionIntent(min_score=60, risk_tolerance="normal")
    return s


# ──────────────────────────────────────────────
# Timeframe consensus
# ──────────────────────────────────────────────

def test_consensus_full_bull_alignment():
    trends = {tf: _trend(tf, strength=TrendStrength.STRONG) for tf in ["1d","1h","15m","5m","1m"]}
    result = compute_tf_consensus(trends)
    assert result.direction == Direction.BULLISH
    assert result.strength in (TrendStrength.STRONG, TrendStrength.MODERATE)
    assert result.htf_agreement == True
    assert result.tradeable == True

def test_consensus_htf_conflict():
    trends = {
        "1d":  _trend("1d",  direction=Direction.BULLISH),
        "1h":  _trend("1h",  direction=Direction.BEARISH),
        "15m": _trend("15m", direction=Direction.BEARISH),
    }
    result = compute_tf_consensus(trends)
    assert result.htf_conflict == True
    assert result.tradeable == False

def test_consensus_neutral_daily_bull_1h():
    trends = {
        "1d":  _trend("1d",  direction=Direction.NEUTRAL),
        "1h":  _trend("1h",  direction=Direction.BULLISH),
        "15m": _trend("15m", direction=Direction.BULLISH),
    }
    result = compute_tf_consensus(trends)
    assert result.direction == Direction.BULLISH
    assert result.htf_conflict == False

def test_htf_bias_conflict_returns_neutral():
    trends = {
        "1d": _trend("1d", direction=Direction.BULLISH),
        "1h": _trend("1h", direction=Direction.BEARISH),
    }
    assert get_htf_bias(trends) == Direction.NEUTRAL

def test_htf_bias_both_bull():
    trends = {
        "1d": _trend("1d", direction=Direction.BULLISH),
        "1h": _trend("1h", direction=Direction.BULLISH),
    }
    assert get_htf_bias(trends) == Direction.BULLISH

def test_is_entry_allowed_blocks_htf_conflict():
    trends = {
        "1d":  _trend("1d",  direction=Direction.BULLISH),
        "1h":  _trend("1h",  direction=Direction.BEARISH),
        "15m": _trend("15m", direction=Direction.BULLISH),
    }
    allowed, reason = is_entry_allowed(trends, Direction.BULLISH)
    assert not allowed
    assert "conflict" in reason.lower()

def test_is_entry_allowed_blocks_missing_15m():
    trends = {
        "1d": _trend("1d", direction=Direction.BULLISH),
        "1h": _trend("1h", direction=Direction.BULLISH),
        "15m": _trend("15m", direction=Direction.NEUTRAL),
    }
    allowed, _ = is_entry_allowed(trends, Direction.BULLISH)
    assert not allowed

def test_position_size_full_alignment():
    trends = {
        "1d": _trend("1d", direction=Direction.BULLISH, strength=TrendStrength.STRONG),
        "1h": _trend("1h", direction=Direction.BULLISH, strength=TrendStrength.STRONG),
    }
    assert get_position_size_modifier(trends) == 1.0

def test_position_size_neutral_daily():
    trends = {
        "1d": _trend("1d", direction=Direction.NEUTRAL),
        "1h": _trend("1h", direction=Direction.BULLISH, strength=TrendStrength.STRONG),
    }
    assert get_position_size_modifier(trends) == 0.75


# ──────────────────────────────────────────────
# Narrative classification
# ──────────────────────────────────────────────

def test_narrative_trending_bull():
    state = _bull_state()
    result = classify_narrative(state)
    assert result.narrative in (MarketNarrative.TRENDING_BULL, MarketNarrative.BREAKOUT)
    assert result.trade_posture == "long_bias"

def test_narrative_trending_bear():
    state = _bear_state()
    result = classify_narrative(state)
    assert result.narrative in (MarketNarrative.TRENDING_BEAR, MarketNarrative.BREAKDOWN)
    assert result.trade_posture == "short_bias"

def test_narrative_chop_when_mixed():
    s = MarketState(symbol="SPY", timestamp=_now())
    s.trends["1h"]  = _trend("1h",  direction=Direction.NEUTRAL, strength=TrendStrength.NONE)
    s.trends["15m"] = _trend("15m", direction=Direction.NEUTRAL, strength=TrendStrength.NONE)
    s.indicators["1h"]  = _ind("1h",  ema_stack="mixed")
    s.indicators["15m"] = _ind("15m", ema_stack="mixed")
    result = classify_narrative(s)
    # Should be chop or transition
    assert result.narrative in (MarketNarrative.CHOP, MarketNarrative.TRANSITION)

def test_narrative_transition_when_weakening():
    s = _bull_state()
    s.trends["1h"].is_weakening  = True
    s.trends["15m"].is_weakening = True
    s.indicators["1h"].ema_stack = "mixed"
    result = classify_narrative(s)
    assert result.narrative in (MarketNarrative.TRANSITION, MarketNarrative.CHOP)


# ──────────────────────────────────────────────
# Entry rules
# ──────────────────────────────────────────────

def test_long_entry_passes_full_bull_state():
    state = _bull_state()
    signal = check_long_entry(state)
    assert signal.allowed == True
    assert signal.direction == TradeDirection.CALLS
    assert signal.gates_passed == 4

def test_long_entry_blocked_by_htf_conflict():
    state = _bull_state()
    state.trends["1d"].direction = Direction.BEARISH
    signal = check_long_entry(state)
    assert signal.allowed == False
    assert signal.gates_passed == 0

def test_long_entry_blocked_by_15m_not_confirmed():
    state = _bull_state()
    state.trends["15m"].direction = Direction.BEARISH
    signal = check_long_entry(state)
    assert signal.allowed == False
    assert signal.gates_passed == 0  # 15m check is inside Gate 1 (is_entry_allowed)

def test_long_entry_blocked_by_overbought_rsi():
    state = _bull_state()
    state.indicators["5m"].rsi = 74.0
    signal = check_long_entry(state)
    assert signal.allowed == False

def test_long_entry_blocked_below_vwap():
    state = _bull_state()
    state.indicators["5m"].price_vs_vwap = "below"
    signal = check_long_entry(state)
    assert signal.allowed == False

def test_short_entry_passes_full_bear_state():
    state = _bear_state()
    signal = check_short_entry(state)
    assert signal.allowed == True
    assert signal.direction == TradeDirection.PUTS
    assert signal.gates_passed == 4

def test_short_entry_blocked_above_vwap():
    state = _bear_state()
    state.indicators["5m"].price_vs_vwap = "above"
    signal = check_short_entry(state)
    assert signal.allowed == False

def test_short_entry_blocked_positive_macd():
    state = _bear_state()
    state.indicators["5m"].macd_hist = 0.015
    signal = check_short_entry(state)
    assert signal.allowed == False


# ──────────────────────────────────────────────
# Trend weakening detection
# ──────────────────────────────────────────────

def test_weakening_rsi_extreme_plus_opposing_macd():
    trend = _trend("5m", direction=Direction.BULLISH, strength=TrendStrength.STRONG)
    # Signal 1: RSI overbought + negative MACD hist
    # Signal 3: EMA stack mixed on strong trend
    ind = _ind("5m", rsi=75.0, macd_hist=-0.005, ema_stack="mixed")
    weak, reason = detect_trend_weakening(trend, ind)
    assert weak == True
    assert "RSI overbought" in reason or "EMA stack" in reason

def test_weakening_ema_stack_degraded():
    trend = _trend("5m", direction=Direction.BULLISH, strength=TrendStrength.STRONG)
    # Signal 3: EMA stack mixed on strong trend
    # Signal 2: MACD histogram shrinking (provide prev_ind)
    ind      = _ind("5m", ema_stack="mixed", macd_hist=0.003)
    prev_ind = _ind("5m", ema_stack="mixed", macd_hist=0.009)
    weak, reason = detect_trend_weakening(trend, ind, prev_ind)
    assert weak == True

def test_weakening_macd_shrinking():
    trend    = _trend("5m", direction=Direction.BULLISH, strength=TrendStrength.STRONG)
    # Signal 2: MACD shrinking; Signal 3: EMA stack mixed on strong trend
    ind      = _ind("5m", macd_hist=0.003, ema_stack="mixed")
    prev_ind = _ind("5m", macd_hist=0.008, ema_stack="bull_aligned")
    weak, reason = detect_trend_weakening(trend, ind, prev_ind)
    assert weak == True
    assert "shrinking" in reason or "EMA stack" in reason

def test_no_weakening_healthy_bull():
    trend = _trend("5m", direction=Direction.BULLISH, strength=TrendStrength.MODERATE)
    ind   = _ind("5m", rsi=55.0, macd_hist=0.012, ema_stack="bull_aligned")
    weak, _ = detect_trend_weakening(trend, ind)
    assert weak == False


# ──────────────────────────────────────────────
# Trend flip detection
# ──────────────────────────────────────────────

def test_flip_detected_with_structure_break_and_ema_cross():
    prev_trend = _trend("15m", direction=Direction.BULLISH)
    curr_trend = _trend("15m", direction=Direction.BEARISH, structure_break=True)
    curr_ind = _ind("15m", ema9=542.0, ema20=544.0)  # 9 < 20 = bear cross
    prev_ind = _ind("15m", ema9=545.0, ema20=543.0)  # 9 > 20 = was bull
    curr_ind.ema_9 = 542.0; curr_ind.ema_20 = 544.0
    prev_ind.ema_9 = 545.0; prev_ind.ema_20 = 543.0
    flipped, reason = detect_trend_flip(curr_trend, prev_trend, curr_ind, prev_ind)
    assert flipped == True

def test_no_flip_same_direction():
    curr = _trend("15m", direction=Direction.BULLISH)
    prev = _trend("15m", direction=Direction.BULLISH)
    ind  = _ind("15m")
    flipped, _ = detect_trend_flip(curr, prev, ind)
    assert flipped == False

def test_no_flip_without_structure_break():
    curr = _trend("15m", direction=Direction.BEARISH, structure_break=False)
    prev = _trend("15m", direction=Direction.BULLISH)
    ind  = _ind("15m")
    flipped, reason = detect_trend_flip(curr, prev, ind)
    # Without structure break, no flip
    assert flipped == False


# ──────────────────────────────────────────────
# Exit conditions
# ──────────────────────────────────────────────

def _active_trade(direction=TradeDirection.CALLS, entry=545.0,
                   stop=540.85, target=553.18, current=547.0):
    from engine.models import TradeState, TradeStatus
    t = TradeState(
        trade_id="T001", direction=direction,
        entry_price=entry, entry_time=_now(), size=10000.0,
        stop_loss=stop, profit_target=target,
    )
    t.current_price = current
    t.high_water = current
    t.update_pnl(current)
    return t

def test_stop_loss_triggers_calls():
    trade = _active_trade(current=540.0)  # below stop 540.85
    state = _bull_state()
    signal = check_exit_conditions(trade, state)
    assert signal.exit_now == True
    assert signal.reason == ExitReason.STOP_LOSS

def test_stop_loss_triggers_puts():
    trade = _active_trade(
        direction=TradeDirection.PUTS,
        entry=545.0, stop=549.36, target=536.82, current=550.0
    )
    state = _bear_state()
    signal = check_exit_conditions(trade, state)
    assert signal.exit_now == True
    assert signal.reason == ExitReason.STOP_LOSS

def test_profit_target_calls():
    trade = _active_trade(current=554.0)  # above target 553.18
    state = _bull_state()
    signal = check_exit_conditions(trade, state)
    assert signal.exit_now == True
    assert signal.reason == ExitReason.PROFIT_TARGET

def test_trailing_stop_triggers():
    trade = _active_trade(current=542.0)
    trade.trailing_stop = 543.0  # price dropped below trailing stop
    state = _bull_state()
    signal = check_exit_conditions(trade, state)
    assert signal.exit_now == True
    assert signal.reason == ExitReason.TRAILING_STOP

def test_no_exit_in_profit():
    trade = _active_trade(current=547.0)  # profitable, no conditions met
    state = _bull_state()
    signal = check_exit_conditions(trade, state)
    assert signal.exit_now == False


# ──────────────────────────────────────────────
# Setup scoring
# ──────────────────────────────────────────────

def test_score_perfect_bull_setup():
    state  = _bull_state()
    score  = score_setup(state, TradeDirection.CALLS)
    assert score.total >= 75
    assert score.trend_alignment >= 20
    assert score.momentum_quality >= 15
    assert score.grade() in ("A+", "B")

def test_score_components_sum_to_total():
    state = _bull_state()
    score = score_setup(state, TradeDirection.CALLS)
    component_sum = (score.trend_alignment + score.momentum_quality
                     + score.structure_context + score.entry_timing)
    assert abs(score.total - component_sum) < 0.01

def test_score_penalizes_htf_conflict():
    state = _bull_state()
    state.trends["1d"].direction = Direction.BEARISH
    state.trends["1h"].direction = Direction.BULLISH
    score = score_setup(state, TradeDirection.CALLS)
    assert score.trend_alignment == 0.0

def test_score_penalizes_wrong_vwap_side():
    state = _bull_state()
    for tf in state.indicators:
        state.indicators[tf].price_vs_vwap = "below"
    score = score_setup(state, TradeDirection.CALLS)
    assert score.structure_context < 20

def test_score_grade_below_40_is_skip():
    state = MarketState(symbol="SPY", timestamp=_now())
    score = score_setup(state, TradeDirection.CALLS)
    assert score.grade() == "skip"


# ──────────────────────────────────────────────
# Full strategy engine pipeline
# ──────────────────────────────────────────────

def test_strategy_engine_bull_generates_setup_formed_event():
    engine = StrategyEngine()
    state  = _bull_state()
    result = engine.process(state)
    events = engine.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert len(setup_events) >= 1
    assert setup_events[0].payload["direction"] == "calls"
    assert setup_events[0].payload["score"] >= 60

def test_strategy_engine_bear_generates_short_setup():
    engine = StrategyEngine()
    state  = _bear_state()
    result = engine.process(state)
    events = engine.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert any(e.payload["direction"] == "puts" for e in setup_events)

def test_strategy_engine_no_setup_in_conflict():
    engine = StrategyEngine()
    state  = _bull_state()
    state.trends["1d"].direction = Direction.BEARISH
    result = engine.process(state)
    events = engine.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert len(setup_events) == 0

def test_strategy_engine_detects_narrative_change():
    engine = StrategyEngine()
    # First call: strong bull trend
    state1 = _bull_state()
    engine.process(state1)
    engine.pop_events()
    # Second call: transition state — RSI overbought + negative MACD + mixed EMA stack
    # These produce actual weakening via detect_trend_weakening (not just flags)
    state2 = _bull_state()
    for tf in ["1h", "15m", "5m"]:
        state2.indicators[tf].rsi = 76.0       # overbought
        state2.indicators[tf].macd_hist = -0.008  # opposing MACD
        state2.indicators[tf].ema_stack = "mixed"
    # Make 1h and 15m diverge so transition fires
    state2.trends["15m"].direction = Direction.NEUTRAL
    result = engine.process(state2)
    events = engine.pop_events()
    # Should have narrative_change event (trending_bull -> transition or chop)
    narrative_events = [e for e in events if e.event_type == "narrative_change"]
    assert len(narrative_events) >= 1, f"Expected narrative_change, got: {[e.event_type for e in events]}"

def test_to_dict_contains_all_keys():
    engine = StrategyEngine()
    state  = _bull_state()
    result = engine.process(state)
    d = result.to_dict()
    for key in ["symbol","timestamp","htf_bias","narrative","setup","indicators",
                "trends","support","resistance","active_trade"]:
        assert key in d, f"Missing key: {key}"


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [(n, fn) for n, fn in list(globals().items()) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
