"""
tests/test_phase3.py — Phase 3: strategy engine.

Tests:
  - MarketRegimeEngine: classification, hysteresis, raw_scores
  - SessionState: phase computation, intent filtering
  - Scorer: component scores, regime/session multipliers, readiness penalty
  - Rules / entry gates: all gate conditions with explicit inputs
  - Priority engine: event scoring, coalescing, AI trigger decisions
  - StrategyEngine: full process() pipeline integration
  - Regime → score → entry chain end-to-end

Run: PYTHONPATH=. python tests/test_phase3.py
"""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta
from typing import Optional

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


# ─────────────────────────────────────────────────────────────
# State builder helpers
# ─────────────────────────────────────────────────────────────

def _ind(tf, close=545.0, rsi=55.0, macd_hist=0.01,
         ema_stack="bull_aligned", vwap=542.0, atr=1.2,
         ema9=543.0, ema20=541.0, ema50=538.0, ema200=530.0,
         price_vs_vwap="above", macd_bias="bullish",
         rsi_zone="bullish_neutral", macd_line=None, macd_signal=None):
    from engine.models import IndicatorSnapshot
    s = IndicatorSnapshot(timeframe=tf, timestamp=_now(), close=close)
    s.rsi=rsi; s.macd_hist=macd_hist; s.ema_stack=ema_stack; s.vwap=vwap
    s.atr=atr; s.ema_9=ema9; s.ema_20=ema20; s.ema_50=ema50; s.ema_200=ema200
    s.price_vs_vwap=price_vs_vwap; s.macd_bias=macd_bias; s.rsi_zone=rsi_zone
    s.macd_line = macd_line or (macd_hist + 0.005 if macd_hist else 0.01)
    s.macd_signal = macd_signal or 0.005
    return s


def _trend(tf, direction="bullish", strength="moderate",
           is_weakening=False, higher_highs=False, lower_lows=False,
           structure_break=False, last_swing_low=None, last_swing_high=None):
    from engine.models import TrendState, Direction, TrendStrength
    d = {"bullish": Direction.BULLISH, "bearish": Direction.BEARISH,
         "neutral": Direction.NEUTRAL}[direction]
    st = {"strong": TrendStrength.STRONG, "moderate": TrendStrength.MODERATE,
          "weak": TrendStrength.WEAK, "none": TrendStrength.NONE}[strength]
    t = TrendState(timeframe=tf, direction=d, strength=st)
    t.is_weakening=is_weakening; t.higher_highs=higher_highs
    t.lower_lows=lower_lows; t.structure_break=structure_break
    t.last_swing_low=last_swing_low; t.last_swing_high=last_swing_high
    return t


def _session(phase="midday", entries_allowed=True, readiness="fully_ready",
             score_modifier=1.0, entry_filter="normal"):
    from engine.models import SessionState, SessionPhase, Readiness
    phases = {"premarket": SessionPhase.PREMARKET, "open": SessionPhase.OPEN,
              "midday": SessionPhase.MIDDAY, "power_hour": SessionPhase.POWER_HOUR,
              "after_hours": SessionPhase.AFTER_HOURS}
    rdns = {"not_ready": Readiness.NOT_READY,
            "partially_ready": Readiness.PARTIALLY_READY,
            "fully_ready": Readiness.FULLY_READY}
    return SessionState(
        phase=phases[phase], market_open=entries_allowed,
        entries_allowed=entries_allowed, score_modifier=score_modifier,
        entry_filter=entry_filter, phase_note="test",
        readiness=rdns[readiness],
    )


def _regime(regime_name="trend_bull", entry_permitted=True, multiplier=1.0, confidence=0.8):
    from engine.models import RegimeState, MarketRegime
    regimes = {r.value: r for r in MarketRegime}
    return RegimeState(
        regime=regimes[regime_name],
        confidence=confidence,
        score_multiplier=multiplier,
        entry_permitted=entry_permitted,
        description="test regime",
    )


def _bull_state():
    """Fully aligned bull state — all gates should pass."""
    from engine.models import MarketState, SessionIntent
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d","1h","15m","5m","1m"]:
        t = _trend(tf, "bullish", "strong", higher_highs=True, last_swing_low=542.0)
        s.trends[tf] = t
        s.indicators[tf] = _ind(tf)
    s.session = _session("midday", entries_allowed=True, readiness="fully_ready")
    s.regime  = _regime("trend_bull", entry_permitted=True, multiplier=1.0)
    s.intent  = SessionIntent(min_score=60)
    return s


def _bear_state():
    """Fully aligned bear state."""
    from engine.models import MarketState, SessionIntent
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d","1h","15m","5m","1m"]:
        t = _trend(tf, "bearish", "strong", lower_lows=True, last_swing_high=546.0)
        s.trends[tf] = t
        s.indicators[tf] = _ind(tf, close=540.0, rsi=48.0, macd_hist=-0.01,
                                 ema_stack="bear_aligned", vwap=543.0, ema9=542.0,
                                 ema20=544.0, price_vs_vwap="below",
                                 macd_bias="bearish", rsi_zone="bearish_neutral")
    s.session = _session("midday", entries_allowed=True, readiness="fully_ready")
    s.regime  = _regime("trend_bear", entry_permitted=True, multiplier=1.0)
    s.intent  = SessionIntent(min_score=60)
    return s


# ═══════════════════════════════════════════════════════════════════════════
# MarketRegimeEngine
# ═══════════════════════════════════════════════════════════════════════════

def test_regime_engine_instantiates():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    assert eng._current is None


def test_regime_classifies_trend_bull():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketRegime
    eng = MarketRegimeEngine()
    state = _bull_state()
    result = eng.classify(state)
    assert result.regime == MarketRegime.TREND_BULL


def test_regime_classifies_trend_bear():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketRegime
    eng = MarketRegimeEngine()
    state = _bear_state()
    result = eng.classify(state)
    assert result.regime == MarketRegime.TREND_BEAR


def test_regime_classifies_chop_when_neutral_and_weak():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketState, MarketRegime
    eng = MarketRegimeEngine()
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1h","15m"]:
        s.trends[tf]    = _trend(tf, "neutral", "none")
        s.indicators[tf] = _ind(tf, ema_stack="mixed", macd_hist=0.0001)
    result = eng.classify(s)
    assert result.regime == MarketRegime.CHOP


def test_regime_classifies_range():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketState, MarketRegime, SupportResistanceLevel
    eng = MarketRegimeEngine()
    s = MarketState(symbol="SPY", timestamp=_now())
    # Neutral direction with defined S/R
    for tf in ["1h","15m"]:
        s.trends[tf]    = _trend(tf, "neutral", "weak")
        s.indicators[tf] = _ind(tf, ema_stack="mixed", macd_hist=0.0)
    s.support_levels    = [SupportResistanceLevel(540.0, "support", 3),
                           SupportResistanceLevel(538.0, "support", 3)]
    s.resistance_levels = [SupportResistanceLevel(548.0, "resistance", 3),
                           SupportResistanceLevel(550.0, "resistance", 3)]
    result = eng.classify(s)
    # With defined S/R and neutral direction, should classify as RANGE or CHOP
    assert result.regime in (MarketRegime.RANGE, MarketRegime.CHOP)


def test_regime_has_score_multiplier():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketRegime
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    assert result.score_multiplier == 1.0  # TREND_BULL


def test_regime_chop_multiplier_is_lowest():
    from engine.models import REGIME_SCORE_MULTIPLIERS, MarketRegime
    chop_mult = REGIME_SCORE_MULTIPLIERS[MarketRegime.CHOP]
    for regime, mult in REGIME_SCORE_MULTIPLIERS.items():
        assert mult >= chop_mult


def test_regime_entry_permitted_for_trends():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.TREND_BULL] is True
    assert REGIME_ENTRY_PERMITTED[MarketRegime.TREND_BEAR] is True


def test_regime_entry_blocked_for_chop():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.CHOP] is False


def test_regime_entry_blocked_for_transition():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.TRANSITION] is False


def test_regime_entry_blocked_for_breakout_attempt():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.BREAKOUT_ATTEMPT] is False


def test_regime_hysteresis_does_not_flip_on_one_signal():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketRegime
    eng = MarketRegimeEngine()
    bull = _bull_state()
    r1 = eng.classify(bull)
    assert r1.regime == MarketRegime.TREND_BULL

    # One contradictory reading should not flip
    bear = _bear_state()
    r2 = eng.classify(bear)
    # After just one contradictory reading, still TREND_BULL due to hysteresis
    assert r2.regime == MarketRegime.TREND_BULL


def test_regime_switches_after_persistence():
    from strategy.regime_engine import MarketRegimeEngine
    from engine.models import MarketRegime
    eng = MarketRegimeEngine()
    # Establish bull
    for _ in range(4):
        eng.classify(_bull_state())
    # Now push REGIME_PERSISTENCE=3 consecutive bear readings
    for _ in range(4):
        r = eng.classify(_bear_state())
    assert r.regime == MarketRegime.TREND_BEAR


def test_regime_raw_scores_populated():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    assert isinstance(result.raw_scores, dict)
    assert len(result.raw_scores) > 0


def test_regime_raw_scores_trend_bull_highest_in_bull_state():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    scores = result.raw_scores
    assert scores.get("trend_bull", 0) >= scores.get("chop", 0)


def test_regime_description_populated():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    assert len(result.description) > 0


def test_regime_preferred_setups_for_trend_bull():
    from engine.models import REGIME_PREFERRED_SETUPS, MarketRegime
    setups = REGIME_PREFERRED_SETUPS[MarketRegime.TREND_BULL]
    assert "continuation" in setups
    assert "pullback"     in setups


def test_regime_preferred_setups_empty_for_chop():
    from engine.models import REGIME_PREFERRED_SETUPS, MarketRegime
    assert REGIME_PREFERRED_SETUPS[MarketRegime.CHOP] == []


def test_regime_confidence_between_0_and_1():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    assert 0.0 <= result.confidence <= 1.0


def test_regime_to_dict_complete():
    from strategy.regime_engine import MarketRegimeEngine
    eng = MarketRegimeEngine()
    result = eng.classify(_bull_state())
    d = result.to_dict()
    for key in ("regime", "confidence", "score_multiplier", "entry_permitted",
                "description", "preferred_setups", "caution_note", "raw_scores"):
        assert key in d, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# Session state + intent
# ═══════════════════════════════════════════════════════════════════════════

def test_session_midday_allows_entries():
    from strategy.session_state import compute_session_state
    # Simulate a midday UTC time (14:00 UTC = 10:00 EDT)
    t = datetime(2025, 4, 9, 14, 0, 0, tzinfo=timezone.utc)
    s = compute_session_state(t)
    assert s.entries_allowed is True
    from engine.models import SessionPhase
    assert s.phase == SessionPhase.MIDDAY


def test_session_premarket_blocks_entries():
    from strategy.session_state import compute_session_state
    # 09:00 UTC = 05:00 EDT — premarket
    t = datetime(2025, 4, 9, 9, 0, 0, tzinfo=timezone.utc)
    s = compute_session_state(t)
    assert s.entries_allowed is False


def test_session_after_hours_blocks_entries():
    from strategy.session_state import compute_session_state
    # 21:00 UTC = 17:00 EDT — after close
    t = datetime(2025, 4, 9, 21, 0, 0, tzinfo=timezone.utc)
    s = compute_session_state(t)
    assert s.entries_allowed is False


def test_session_open_phase_strict_filter():
    from strategy.session_state import compute_session_state
    from engine.models import SessionPhase
    # 13:35 UTC = 09:35 EDT — first 30 min
    t = datetime(2025, 4, 9, 13, 35, 0, tzinfo=timezone.utc)
    s = compute_session_state(t)
    assert s.phase == SessionPhase.OPEN
    assert s.entry_filter == "strict"


def test_session_power_hour_score_elevated():
    from strategy.session_state import compute_session_state
    # 18:45 UTC = 14:45 EDT — power hour
    t = datetime(2025, 4, 9, 18, 45, 0, tzinfo=timezone.utc)
    s = compute_session_state(t)
    from engine.models import SessionPhase
    assert s.phase == SessionPhase.POWER_HOUR
    assert s.score_modifier >= 1.0


def test_session_to_dict_has_readiness():
    from engine.models import SessionState, SessionPhase, Readiness
    s = SessionState(phase=SessionPhase.MIDDAY, readiness=Readiness.FULLY_READY)
    d = s.to_dict()
    assert "readiness" in d
    assert d["readiness"] == "fully_ready"


def test_intent_from_dict_builds_correctly():
    from strategy.session_state import intent_from_dict
    d = {"directional_bias": "bullish", "min_score": 70.0,
         "risk_tolerance": "reduced", "max_trades": 2}
    intent = intent_from_dict(d)
    assert intent.directional_bias == "bullish"
    assert intent.min_score == 70.0
    assert intent.max_trades == 2


def test_intent_score_passes_reduced_risk():
    from engine.models import SessionIntent
    intent = SessionIntent(min_score=60.0, risk_tolerance="reduced")
    # Reduced adds 10 → effective min = 70
    assert intent.score_passes(75.0) is True
    assert intent.score_passes(65.0) is False


def test_intent_size_modifier_by_risk():
    from engine.models import SessionIntent
    assert SessionIntent(risk_tolerance="reduced").size_modifier()    == 0.5
    assert SessionIntent(risk_tolerance="normal").size_modifier()     == 1.0
    assert SessionIntent(risk_tolerance="aggressive").size_modifier() == 1.25


def test_intent_event_mode_default_off():
    from engine.models import SessionIntent, EventMode
    assert SessionIntent().event_mode == EventMode.OFF


# ═══════════════════════════════════════════════════════════════════════════
# Scorer
# ═══════════════════════════════════════════════════════════════════════════

def test_scorer_bull_total_in_range():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    assert 0 <= score.total <= 100


def test_scorer_bull_grade_b_or_better():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    assert score.grade() in ("A+", "B")


def test_scorer_components_sum_to_total():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    expected = (score.trend_alignment + score.momentum_quality
                + score.structure_context + score.entry_timing)
    assert abs(score.total - expected) < 0.01


def test_scorer_components_within_max():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    assert score.trend_alignment   <= 30
    assert score.momentum_quality  <= 25
    assert score.structure_context <= 25
    assert score.entry_timing      <= 20


def test_scorer_applies_regime_multiplier():
    from strategy.scorer import score_setup
    from engine.models import MarketState, TradeDirection, SessionIntent
    s = _bull_state()
    s.regime = _regime("chop", entry_permitted=False, multiplier=0.60)
    score = score_setup(s, TradeDirection.CALLS)
    assert score.regime_multiplier == 0.60
    assert score.adjusted_total < score.total


def test_scorer_applies_session_multiplier():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    s = _bull_state()
    s.session = _session("open", score_modifier=0.80, readiness="fully_ready")
    score = score_setup(s, TradeDirection.CALLS)
    assert score.session_multiplier == 0.80


def test_scorer_applies_readiness_penalty():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    from config.settings import PARTIALLY_READY_PENALTY
    s = _bull_state()
    s.session = _session("midday", readiness="partially_ready")
    score = score_setup(s, TradeDirection.CALLS)
    assert score.readiness_penalty == PARTIALLY_READY_PENALTY
    # adjusted_total should be less than total * multipliers without penalty
    expected_without_penalty = score.total * score.regime_multiplier * score.session_multiplier
    assert abs(score.adjusted_total - (expected_without_penalty - PARTIALLY_READY_PENALTY)) < 0.5


def test_scorer_no_readiness_penalty_when_fully_ready():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    assert score.readiness_penalty == 0.0


def test_scorer_htf_conflict_reduces_alignment():
    from strategy.scorer import score_setup
    from engine.models import MarketState, TradeDirection, SessionIntent
    s = _bull_state()
    # Make daily bearish — HTF conflict
    s.trends["1d"] = _trend("1d", "bearish", "strong")
    score_conflict = score_setup(s, TradeDirection.CALLS)
    score_clean    = score_setup(_bull_state(), TradeDirection.CALLS)
    assert score_conflict.trend_alignment < score_clean.trend_alignment


def test_scorer_below_vwap_reduces_structure():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    s = _bull_state()
    s.indicators["5m"] = _ind("5m", price_vs_vwap="below")
    score_below = score_setup(s, TradeDirection.CALLS)
    score_above = score_setup(_bull_state(), TradeDirection.CALLS)
    assert score_below.structure_context <= score_above.structure_context


def test_scorer_to_dict_has_all_keys():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    score = score_setup(_bull_state(), TradeDirection.CALLS)
    d = score.to_dict()
    for k in ("total", "adjusted_total", "grade", "regime_mult", "session_mult",
              "readiness_penalty", "components", "detail", "sizing_flags"):
        assert k in d, f"Missing key in score.to_dict(): {k}"


def test_scorer_grade_uses_adjusted_total():
    from engine.models import SetupScore
    # total = 75 (B) but adjusted = 85 (A+)
    s = SetupScore(total=75.0, adjusted_total=85.0)
    assert s.grade() == "A+"


def test_scorer_grade_falls_back_to_total():
    from engine.models import SetupScore
    # adjusted_total not set (0.0) → use total
    s = SetupScore(total=65.0, adjusted_total=0.0)
    assert s.grade() == "B"


# ═══════════════════════════════════════════════════════════════════════════
# Entry gates (rules.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_long_entry_passes_bull_state():
    from strategy.rules import check_long_entry
    assert check_long_entry(_bull_state()).allowed is True


def test_short_entry_passes_bear_state():
    from strategy.rules import check_short_entry
    assert check_short_entry(_bear_state()).allowed is True


def test_long_blocked_by_not_ready():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.session = _session("midday", readiness="not_ready")
    result = check_long_entry(s)
    assert result.allowed is False
    assert "Readiness" in result.reason


def test_short_blocked_by_not_ready():
    from strategy.rules import check_short_entry
    s = _bear_state()
    s.session = _session("midday", readiness="not_ready")
    result = check_short_entry(s)
    assert result.allowed is False
    assert "Readiness" in result.reason


def test_long_blocked_by_chop_regime():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.regime = _regime("chop", entry_permitted=False, multiplier=0.60)
    result = check_long_entry(s)
    assert result.allowed is False
    assert "chop" in result.reason.lower()


def test_short_blocked_by_transition_regime():
    from strategy.rules import check_short_entry
    s = _bear_state()
    s.regime = _regime("transition", entry_permitted=False, multiplier=0.70)
    result = check_short_entry(s)
    assert result.allowed is False


def test_long_blocked_outside_market_hours():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.session = _session("after_hours", entries_allowed=False, readiness="fully_ready")
    result = check_long_entry(s)
    assert result.allowed is False
    assert "after_hours" in result.reason.lower()


def test_long_blocked_by_bearish_intent():
    from strategy.rules import check_long_entry
    from engine.models import SessionIntent
    s = _bull_state()
    s.intent = SessionIntent(directional_bias="bearish")
    result = check_long_entry(s)
    assert result.allowed is False
    assert "bearish" in result.reason.lower()


def test_short_blocked_by_bullish_intent():
    from strategy.rules import check_short_entry
    from engine.models import SessionIntent
    s = _bear_state()
    s.intent = SessionIntent(directional_bias="bullish")
    result = check_short_entry(s)
    assert result.allowed is False
    assert "bullish" in result.reason.lower()


def test_long_blocked_by_htf_conflict():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.trends["1h"] = _trend("1h", "bearish", "strong")  # 1h conflict
    result = check_long_entry(s)
    assert result.allowed is False
    assert "Gate 1" in result.reason


def test_long_blocked_by_15m_not_bullish():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.trends["15m"] = _trend("15m", "neutral", "none")
    result = check_long_entry(s)
    assert result.allowed is False
    # 15m not confirmed fires from Gate 1 (HTF check includes 15m confirmation)
    assert "15m" in result.reason.lower() or "gate" in result.reason.lower()


def test_long_blocked_by_overbought_rsi():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.indicators["5m"] = _ind("5m", rsi=75.0)  # overbought
    result = check_long_entry(s)
    assert result.allowed is False
    assert "Gate 3" in result.reason


def test_long_blocked_below_vwap():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.indicators["5m"] = _ind("5m", price_vs_vwap="below", macd_hist=0.01)
    result = check_long_entry(s)
    assert result.allowed is False


def test_short_blocked_by_positive_macd():
    from strategy.rules import check_short_entry
    s = _bear_state()
    s.indicators["5m"] = _ind("5m", close=540.0, macd_hist=0.05,
                               price_vs_vwap="below", ema_stack="bear_aligned",
                               macd_bias="bullish", rsi=48.0)
    result = check_short_entry(s)
    assert result.allowed is False


def test_entry_signal_gates_passed_count():
    from strategy.rules import check_long_entry
    result = check_long_entry(_bull_state())
    assert result.gates_passed >= 3   # all 4 gates pass


def test_entry_signal_gates_passed_on_first_failure():
    from strategy.rules import check_long_entry
    s = _bull_state()
    s.trends["1h"] = _trend("1h", "bearish", "strong")  # fails Gate 1
    result = check_long_entry(s)
    assert result.gates_passed == 0   # failed before Gate 1


# ═══════════════════════════════════════════════════════════════════════════
# Priority engine
# ═══════════════════════════════════════════════════════════════════════════

def test_priority_engine_instantiates():
    from strategy.priority_engine import PriorityEngine
    eng = PriorityEngine()
    assert eng is not None


def test_priority_engine_empty_events():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent
    eng = PriorityEngine()
    result = eng.process([])
    assert result == []


def test_priority_engine_setup_formed_high():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent, EventPriority
    eng = PriorityEngine()
    event = MarketEvent(
        event_type="setup_formed",
        timestamp=_now(),
        payload={"direction": "calls", "score": 80, "grade": "A+",
                 "setup_type": "continuation", "reason": "test"},
    )
    results = eng.process([event])
    assert len(results) == 1
    assert results[0].priority in (EventPriority.CRITICAL, EventPriority.HIGH)


def test_priority_engine_exit_triggered_critical():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent, EventPriority
    eng = PriorityEngine()
    event = MarketEvent(
        event_type="exit_triggered",
        timestamp=_now(),
        payload={"trade_id": "T1", "reason": "stop_loss",
                 "detail": "thesis broken", "urgency": "immediate", "pnl_pct": -0.8},
    )
    results = eng.process([event])
    assert results[0].priority == EventPriority.CRITICAL


def test_priority_engine_candle_close_low():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent, EventPriority
    eng = PriorityEngine()
    event = MarketEvent(
        event_type="candle_close",
        timestamp=_now(),
        payload={"timeframe": "1m"},
    )
    results = eng.process([event])
    assert results[0].priority in (EventPriority.LOW, EventPriority.INFO)
    assert results[0].should_trigger_ai is False


def test_priority_engine_ai_trigger_on_high():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent, EventPriority
    eng = PriorityEngine()
    event = MarketEvent(
        event_type="setup_formed",
        timestamp=_now(),
        payload={"direction": "calls", "score": 82, "grade": "A+",
                 "setup_type": "continuation", "reason": "test"},
    )
    results = eng.process([event])
    # High/Critical setup events should trigger AI
    assert results[0].should_trigger_ai is True


def test_priority_engine_headline_populated():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent
    eng = PriorityEngine()
    event = MarketEvent(
        event_type="trend_flip",
        timestamp=_now(),
        payload={"timeframe": "15m", "from": "bullish", "to": "bearish", "reason": "flip"},
    )
    results = eng.process([event])
    assert len(results[0].headline) > 0


def test_priority_engine_results_sorted_highest_first():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent
    eng = PriorityEngine()
    events = [
        MarketEvent("candle_close", _now(), {"timeframe": "5m"}),
        MarketEvent("exit_triggered", _now(),
                    {"trade_id": "T1", "reason": "stop_loss",
                     "detail": "x", "urgency": "immediate", "pnl_pct": -1.0}),
    ]
    results = eng.process(events)
    assert results[0].score >= results[-1].score


def test_priority_engine_to_dict():
    from strategy.priority_engine import PriorityEngine
    from engine.models import MarketEvent
    eng = PriorityEngine()
    event = MarketEvent("trend_flip", _now(),
                        {"timeframe": "1h", "from": "neutral", "to": "bullish", "reason": "x"})
    results = eng.process([event])
    d = results[0].to_dict()
    for k in ("priority", "score", "event_type", "headline", "detail", "trigger_ai"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# StrategyEngine integration
# ═══════════════════════════════════════════════════════════════════════════

def test_strategy_engine_instantiates():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    assert eng is not None


def test_strategy_engine_returns_state():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    result = eng.process(_bull_state())
    assert result is not None


def test_strategy_engine_empty_indicators_skips():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import MarketState
    eng = StrategyEngine()
    empty = MarketState(symbol="SPY", timestamp=_now())
    result = eng.process(empty)
    assert result.setup_direction is None


def test_strategy_engine_bull_sets_setup_direction():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import TradeDirection
    eng = StrategyEngine()
    result = eng.process(_bull_state())
    assert result.setup_direction == TradeDirection.CALLS


def test_strategy_engine_bear_sets_setup_direction():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import TradeDirection
    eng = StrategyEngine()
    result = eng.process(_bear_state())
    assert result.setup_direction == TradeDirection.PUTS


def test_strategy_engine_populates_regime():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import MarketState
    eng = StrategyEngine()
    s = MarketState(symbol="SPY", timestamp=_now())
    s.session = _session("midday", readiness="fully_ready")
    for tf in ["1h","15m","5m"]:
        s.trends[tf]    = _trend(tf, "bullish", "strong", higher_highs=True)
        s.indicators[tf] = _ind(tf)
    result = eng.process(s)
    assert result.regime is not None


def test_strategy_engine_populates_narrative():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    result = eng.process(_bull_state())
    assert result.narrative is not None


def test_strategy_engine_bull_fires_setup_formed_event():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    eng.process(_bull_state())
    events = eng.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert len(setup_events) >= 1
    assert setup_events[0].payload["direction"] == "calls"


def test_strategy_engine_bear_fires_setup_formed_event():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    eng.process(_bear_state())
    events = eng.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert any(e.payload["direction"] == "puts" for e in setup_events)


def test_strategy_engine_chop_no_setup():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import MarketState, SetupType
    eng = StrategyEngine()
    s = _bull_state()
    s.regime = _regime("chop", entry_permitted=False, multiplier=0.60)
    result = eng.process(s)
    assert result.setup_direction is None


def test_strategy_engine_preserves_readiness():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import Readiness
    eng = StrategyEngine()
    s = _bull_state()
    s.session = _session("midday", readiness="partially_ready")
    result = eng.process(s)
    assert result.session.readiness == Readiness.PARTIALLY_READY


def test_strategy_engine_setup_score_attached():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    result = eng.process(_bull_state())
    assert result.setup_score is not None
    assert result.setup_score.total > 0


def test_strategy_engine_prioritized_events_produced():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    eng.process(_bull_state())
    pe = eng.pop_prioritized_events()
    assert isinstance(pe, list)


def test_strategy_engine_narrative_change_event():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import NarrativeState, MarketNarrative
    eng = StrategyEngine()
    # Establish first process (will produce TRENDING_BULL narrative)
    eng.process(_bull_state())

    # Force prev_narrative to CHOP so next process (TRENDING_BULL) triggers a change
    eng._prev_narrative = NarrativeState(narrative=MarketNarrative.CHOP, confidence=0.5)
    eng.process(_bull_state())
    events = eng.pop_events()
    narrative_events = [e for e in events if e.event_type == "narrative_change"]
    assert len(narrative_events) >= 1


def test_strategy_engine_pop_events_clears_queue():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    eng.process(_bull_state())
    eng.pop_events()
    assert eng.pop_events() == []


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end regime → score → entry chain
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_bull_trend_setup_scores_above_60():
    from strategy.strategy_engine import StrategyEngine
    eng = StrategyEngine()
    result = eng.process(_bull_state())
    assert result.setup_score is not None
    # With TREND_BULL multiplier=1.0, full bull state should score B or better
    assert result.setup_score.adjusted_total >= 60


def test_e2e_chop_blocks_setup_and_no_event():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import MarketState
    eng = StrategyEngine()
    s = _bull_state()
    s.regime = _regime("chop", entry_permitted=False, multiplier=0.60)
    eng.process(s)
    events = eng.pop_events()
    setup_events = [e for e in events if e.event_type == "setup_formed"]
    assert len(setup_events) == 0


def test_e2e_partial_readiness_reduces_adjusted_total():
    from strategy.strategy_engine import StrategyEngine
    from config.settings import PARTIALLY_READY_PENALTY
    eng1 = StrategyEngine()
    eng2 = StrategyEngine()

    full  = _bull_state()
    full.session = _session("midday", readiness="fully_ready")

    partial = _bull_state()
    partial.session = _session("midday", readiness="partially_ready")

    r1 = eng1.process(full)
    r2 = eng2.process(partial)

    if r1.setup_score and r2.setup_score:
        diff = r1.setup_score.adjusted_total - r2.setup_score.adjusted_total
        assert abs(diff - PARTIALLY_READY_PENALTY) < 1.0


def test_e2e_breakout_confirmed_multiplier_bonus():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import REGIME_SCORE_MULTIPLIERS, MarketRegime
    eng = StrategyEngine()
    s = _bull_state()
    s.regime = _regime("breakout_confirmed", entry_permitted=True, multiplier=1.10)
    result = eng.process(s)
    if result.setup_score:
        assert result.setup_score.regime_multiplier == 1.10


def test_e2e_intent_bearish_blocks_calls():
    from strategy.strategy_engine import StrategyEngine
    from engine.models import SessionIntent
    eng = StrategyEngine()
    s = _bull_state()
    s.intent = SessionIntent(directional_bias="bearish")
    result = eng.process(s)
    assert result.setup_direction is None


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"\nPhase 3 tests ({len(tests)} total)\n{'─'*54}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'='*54}")
    print(f"{passed} passed, {failed} failed  ({passed+failed} total)")
    sys.exit(0 if failed == 0 else 1)
