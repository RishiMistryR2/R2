"""
tests/test_phase1.py — Phase 1 verification.

Tests that every enum, dataclass, lookup table, and setting
defined in Phase 1 is correct and consistent.

Run: PYTHONPATH=. python tests/test_phase1.py
"""

import sys, os, traceback, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)

passed = failed = 0

def run(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1


# ─────────────────────────────────────────────
# Settings tests
# ─────────────────────────────────────────────

def test_settings_identity():
    from config.settings import ASSISTANT_NAME, ASSISTANT_VERSION, ASSISTANT_DESCRIPTION
    assert ASSISTANT_NAME == "R2"
    assert ASSISTANT_VERSION == "0.1.0"
    assert "trading" in ASSISTANT_DESCRIPTION.lower()

def test_settings_paths_exist():
    from config.settings import DATA_DIR, SESSIONS_DIR, MEMORY_DIR, JOURNAL_DIR, LOGS_DIR
    for p in (DATA_DIR, SESSIONS_DIR, MEMORY_DIR, JOURNAL_DIR, LOGS_DIR):
        assert p.exists(), f"Directory does not exist: {p}"

def test_settings_candle_history_keys():
    from config.settings import CANDLE_HISTORY, TIMEFRAMES
    assert set(CANDLE_HISTORY.keys()) == set(TIMEFRAMES)

def test_settings_timeframe_seconds_keys():
    from config.settings import TIMEFRAME_SECONDS, TIMEFRAMES
    assert set(TIMEFRAME_SECONDS.keys()) == set(TIMEFRAMES)

def test_settings_seed_bar_counts_keys():
    from config.settings import SEED_BAR_COUNTS, TIMEFRAMES
    assert set(SEED_BAR_COUNTS.keys()) == set(TIMEFRAMES)

def test_settings_seed_order_complete():
    from config.settings import SEED_ORDER, TIMEFRAMES
    assert set(SEED_ORDER) == set(TIMEFRAMES)

def test_settings_readiness_thresholds_keys():
    from config.settings import READINESS_THRESHOLDS
    assert set(READINESS_THRESHOLDS.keys()) == {"not_ready", "partially_ready", "fully_ready"}

def test_settings_readiness_thresholds_ordering():
    from config.settings import READINESS_THRESHOLDS
    # fully_ready must require more candles than partially_ready
    for tf in ("1h", "15m"):
        assert (READINESS_THRESHOLDS["fully_ready"][tf] >
                READINESS_THRESHOLDS["partially_ready"][tf] >
                READINESS_THRESHOLDS["not_ready"][tf])

def test_settings_scoring_range():
    from config.settings import MIN_SCORE_DEFAULT, MIN_RR_RATIO_DEFAULT
    assert 0 < MIN_SCORE_DEFAULT < 100
    assert MIN_RR_RATIO_DEFAULT > 1.0

def test_settings_options_params_sensible():
    from config.settings import (
        MAX_PREMIUM_LOSS_PCT_DEFAULT,
        ATM_DELTA_APPROX,
        ATR_BUFFER_OPTIONS,
        ATR_BUFFER_SHARES,
    )
    assert 0 < ATM_DELTA_APPROX < 1
    assert 0 < ATR_BUFFER_OPTIONS < ATR_BUFFER_SHARES   # options buffer tighter
    assert 10 < MAX_PREMIUM_LOSS_PCT_DEFAULT < 60

def test_settings_time_stop_windows_all_positive():
    from config.settings import DEFAULT_TIME_STOP_WINDOWS
    for k, v in DEFAULT_TIME_STOP_WINDOWS.items():
        assert v > 0, f"Time stop window for {k} must be positive"

def test_settings_psychology_cap_positive():
    from config.settings import PSYCHOLOGY_MAX_MODIFIER
    assert PSYCHOLOGY_MAX_MODIFIER > 0

def test_settings_claude_model_set():
    from config.settings import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
    assert CLAUDE_MODEL.startswith("claude-")
    assert CLAUDE_MAX_TOKENS > 0

def test_settings_memory_schema_version_format():
    from config.settings import MEMORY_SCHEMA_VERSION
    parts = MEMORY_SCHEMA_VERSION.split(".")
    assert len(parts) == 2
    assert all(p.isdigit() for p in parts)

def test_settings_env_file_exists():
    root = Path(__file__).parent.parent
    assert (root / ".env.example").exists()

def test_settings_requirements_exists():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists()

def test_settings_pyproject_exists():
    root = Path(__file__).parent.parent
    assert (root / "pyproject.toml").exists()


# ─────────────────────────────────────────────
# Enum tests
# ─────────────────────────────────────────────

def test_enum_direction_values():
    from engine.models import Direction
    assert Direction.BULLISH.value == "bullish"
    assert Direction.BEARISH.value == "bearish"
    assert Direction.NEUTRAL.value == "neutral"

def test_enum_market_regime_complete():
    from engine.models import MarketRegime
    expected = {"trend_bull", "trend_bear", "range", "chop",
                "breakout_attempt", "breakout_confirmed", "transition"}
    assert {m.value for m in MarketRegime} == expected

def test_enum_session_phase_complete():
    from engine.models import SessionPhase
    expected = {"premarket", "open", "midday", "power_hour", "after_hours", "overnight"}
    assert {p.value for p in SessionPhase} == expected

def test_enum_readiness_complete():
    from engine.models import Readiness
    assert {r.value for r in Readiness} == {"not_ready", "partially_ready", "fully_ready"}

def test_enum_event_mode_complete():
    from engine.models import EventMode
    assert {e.value for e in EventMode} == {"off", "conservative", "aggressive"}

def test_enum_event_status_complete():
    from engine.models import EventStatus
    assert {e.value for e in EventStatus} == {
        "pending", "continuation", "absorption", "inconclusive", "expired"
    }

def test_enum_exit_reason_includes_new():
    from engine.models import ExitReason
    values = {e.value for e in ExitReason}
    assert "time_stop"    in values
    assert "premium_stop" in values
    assert "manual"       in values

def test_enum_stop_cause_complete():
    from engine.models import StopCause
    assert {s.value for s in StopCause} == {
        "thesis_break", "premium_deterioration", "time_decay",
        "manual_exit", "target_reached", "partial_exit"
    }

def test_enum_instrument_type():
    from engine.models import InstrumentType
    assert {i.value for i in InstrumentType} == {"options", "shares"}

def test_enum_setup_type_includes_new():
    from engine.models import SetupType
    values = {s.value for s in SetupType}
    assert "pullback"         in values
    assert "bounce_rejection" in values
    assert "event_play"       in values

def test_enum_psych_condition():
    from engine.models import PsychCondition
    assert {p.value for p in PsychCondition} == {
        "neutral", "panic", "euphoria", "indecision", "trap"
    }

def test_enum_breadth_trend():
    from engine.models import BreadthTrend
    assert {b.value for b in BreadthTrend} == {"rising", "falling", "flat"}

def test_enum_session_shape():
    from engine.models import SessionShape
    values = {s.value for s in SessionShape}
    assert "gave_back" in values
    assert "churned"   in values
    assert "unknown"   in values


# ─────────────────────────────────────────────
# Lookup table tests
# ─────────────────────────────────────────────

def test_regime_score_multipliers_coverage():
    from engine.models import REGIME_SCORE_MULTIPLIERS, MarketRegime
    assert set(REGIME_SCORE_MULTIPLIERS.keys()) == set(MarketRegime)

def test_regime_entry_permitted_coverage():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert set(REGIME_ENTRY_PERMITTED.keys()) == set(MarketRegime)

def test_regime_preferred_setups_coverage():
    from engine.models import REGIME_PREFERRED_SETUPS, MarketRegime
    assert set(REGIME_PREFERRED_SETUPS.keys()) == set(MarketRegime)

def test_session_score_modifiers_coverage():
    from engine.models import SESSION_SCORE_MODIFIERS, SessionPhase
    assert set(SESSION_SCORE_MODIFIERS.keys()) == set(SessionPhase)

def test_chop_entry_blocked():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.CHOP] is False

def test_breakout_attempt_entry_blocked():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.BREAKOUT_ATTEMPT] is False

def test_transition_entry_blocked():
    from engine.models import REGIME_ENTRY_PERMITTED, MarketRegime
    assert REGIME_ENTRY_PERMITTED[MarketRegime.TRANSITION] is False

def test_chop_score_multiplier_lowest():
    from engine.models import REGIME_SCORE_MULTIPLIERS, MarketRegime
    chop = REGIME_SCORE_MULTIPLIERS[MarketRegime.CHOP]
    for regime, mult in REGIME_SCORE_MULTIPLIERS.items():
        assert mult >= chop, f"{regime} multiplier {mult} < chop {chop}"

def test_breakout_confirmed_multiplier_highest():
    from engine.models import REGIME_SCORE_MULTIPLIERS, MarketRegime
    bc = REGIME_SCORE_MULTIPLIERS[MarketRegime.BREAKOUT_CONFIRMED]
    for regime, mult in REGIME_SCORE_MULTIPLIERS.items():
        assert mult <= bc, f"{regime} multiplier {mult} > breakout_confirmed {bc}"

def test_open_phase_score_reduced():
    from engine.models import SESSION_SCORE_MODIFIERS, SessionPhase
    assert SESSION_SCORE_MODIFIERS[SessionPhase.OPEN] < 1.0

def test_power_hour_score_elevated():
    from engine.models import SESSION_SCORE_MODIFIERS, SessionPhase
    assert SESSION_SCORE_MODIFIERS[SessionPhase.POWER_HOUR] >= 1.0


# ─────────────────────────────────────────────
# Dataclass tests
# ─────────────────────────────────────────────

def test_candle_properties():
    from engine.models import Candle
    c = Candle(timestamp=_now(), open=100.0, high=105.0, low=98.0,
               close=103.0, volume=1000.0, timeframe="5m")
    assert c.is_bullish is True
    assert c.range == 7.0
    assert c.body_size == 3.0
    assert c.upper_wick == 2.0
    assert c.lower_wick == 2.0

def test_candle_bearish():
    from engine.models import Candle
    c = Candle(timestamp=_now(), open=105.0, high=106.0, low=98.0,
               close=100.0, volume=500.0, timeframe="1m")
    assert c.is_bullish is False

def test_candle_session_type_default():
    from engine.models import Candle
    c = Candle(timestamp=_now(), open=100.0, high=101.0, low=99.0,
               close=100.5, volume=100.0, timeframe="1m")
    assert c.session_type == "regular"

def test_setup_score_grade_A_plus():
    from engine.models import SetupScore
    s = SetupScore(adjusted_total=85.0)
    assert s.grade() == "A+"

def test_setup_score_grade_B():
    from engine.models import SetupScore
    s = SetupScore(adjusted_total=65.0)
    assert s.grade() == "B"

def test_setup_score_grade_skip():
    from engine.models import SetupScore
    s = SetupScore(adjusted_total=35.0)
    assert s.grade() == "skip"

def test_setup_score_grade_uses_adjusted_when_set():
    from engine.models import SetupScore
    # total = 90 (A+) but adjusted = 55 (C) — grade should use adjusted
    s = SetupScore(total=90.0, adjusted_total=55.0)
    assert s.grade() == "C"

def test_setup_score_grade_falls_back_to_total():
    from engine.models import SetupScore
    # adjusted_total = 0 (not set) → fall back to total
    s = SetupScore(total=75.0, adjusted_total=0.0)
    assert s.grade() == "B"

def test_setup_score_to_dict_has_sizing_flags():
    from engine.models import SetupScore
    s = SetupScore(total=70.0, adjusted_total=70.0, sizing_flags=["stop_too_tight"])
    d = s.to_dict()
    assert "sizing_flags" in d
    assert "stop_too_tight" in d["sizing_flags"]

def test_regime_state_to_dict():
    from engine.models import RegimeState, MarketRegime
    r = RegimeState(regime=MarketRegime.TREND_BULL, confidence=0.8,
                    raw_scores={"trend_bull": 7, "chop": 2})
    d = r.to_dict()
    assert d["regime"] == "trend_bull"
    assert d["confidence"] == 0.8
    assert "raw_scores" in d

def test_session_state_to_dict():
    from engine.models import SessionState, SessionPhase, Readiness
    s = SessionState(phase=SessionPhase.MIDDAY, market_open=True,
                     entries_allowed=True, readiness=Readiness.FULLY_READY)
    d = s.to_dict()
    assert d["phase"] == "midday"
    assert d["readiness"] == "fully_ready"

def test_session_intent_score_passes():
    from engine.models import SessionIntent
    intent = SessionIntent(min_score=65.0, risk_tolerance="normal")
    assert intent.score_passes(70.0) is True
    assert intent.score_passes(60.0) is False

def test_session_intent_score_passes_reduced_tolerance():
    from engine.models import SessionIntent
    intent = SessionIntent(min_score=60.0, risk_tolerance="reduced")
    # reduced adds 10 → effective min = 70
    assert intent.score_passes(75.0) is True
    assert intent.score_passes(65.0) is False

def test_session_intent_size_modifier():
    from engine.models import SessionIntent
    assert SessionIntent(risk_tolerance="reduced").size_modifier()    == 0.5
    assert SessionIntent(risk_tolerance="normal").size_modifier()     == 1.0
    assert SessionIntent(risk_tolerance="aggressive").size_modifier() == 1.25

def test_session_intent_event_mode_default():
    from engine.models import SessionIntent, EventMode
    assert SessionIntent().event_mode == EventMode.OFF

def test_trade_state_update_pnl_calls():
    from engine.models import TradeState, TradeDirection, InstrumentType
    t = TradeState(
        trade_id="T1", direction=TradeDirection.CALLS,
        entry_time=_now(), size=10000.0,
        underlying_entry_price=545.0,
    )
    t.update_pnl(550.0)
    assert t.pnl_pct > 0
    assert round(t.pnl_pct, 2) == round((550.0 - 545.0) / 545.0 * 100, 2)

def test_trade_state_update_pnl_puts():
    from engine.models import TradeState, TradeDirection, InstrumentType
    t = TradeState(
        trade_id="T2", direction=TradeDirection.PUTS,
        entry_time=_now(), size=10000.0,
        underlying_entry_price=545.0,
    )
    t.update_pnl(540.0)
    assert t.pnl_pct > 0

def test_trade_state_high_water_tracked():
    from engine.models import TradeState, TradeDirection
    t = TradeState(
        trade_id="T3", direction=TradeDirection.CALLS,
        entry_time=_now(), size=10000.0, underlying_entry_price=545.0,
    )
    t.update_pnl(550.0)
    t.update_pnl(547.0)
    assert t.high_water == 550.0

def test_psychological_state_modifier_capped_positive():
    from engine.models import PsychologicalMarketState
    p = PsychologicalMarketState()
    p.set_modifier(100.0)   # would be huge without cap
    assert p.confidence_modifier == 5.0

def test_psychological_state_modifier_capped_negative():
    from engine.models import PsychologicalMarketState
    p = PsychologicalMarketState()
    p.set_modifier(-100.0)
    assert p.confidence_modifier == -5.0

def test_event_state_to_dict():
    from engine.models import EventState, EventStatus
    e = EventState(
        principal_alert_active=True,
        event_confirmed=True,
        event_direction="bullish",
        event_status=EventStatus.CONTINUATION,
        event_strength="strong",
    )
    d = e.to_dict()
    assert d["principal_alert_active"] is True
    assert d["event_status"] == "continuation"

def test_behavioral_flag_to_dict():
    from engine.models import BehavioralFlag
    b = BehavioralFlag(
        flag_type="chop_trading",
        triggered_at=_now(),
        context="Entered during chop regime",
        severity="warning",
        session_count=2,
    )
    d = b.to_dict()
    assert d["severity"] == "warning"
    assert d["session_count"] == 2


# ─────────────────────────────────────────────
# MarketState integration tests
# ─────────────────────────────────────────────

def test_market_state_to_dict_has_all_top_level_keys():
    from engine.models import MarketState
    ms = MarketState(symbol="SPY", timestamp=_now())
    d = ms.to_dict()
    required = {"symbol", "timestamp", "htf_bias", "narrative", "setup",
                "indicators", "trends", "support", "resistance",
                "regime", "session", "intent", "event_state",
                "psychology", "internals", "active_trade", "behavioral_flags"}
    missing = required - set(d.keys())
    assert not missing, f"Missing keys: {missing}"

def test_market_state_to_dict_serializable():
    from engine.models import MarketState
    import json
    ms = MarketState(symbol="SPY", timestamp=_now())
    d = ms.to_dict()
    # Should be JSON-serializable (no datetime objects, only None/str/float/list/dict)
    json.dumps(d)

def test_market_state_to_dict_with_all_fields():
    from engine.models import (
        MarketState, RegimeState, MarketRegime, SessionState, SessionPhase,
        Readiness, SessionIntent, NarrativeState, MarketNarrative,
        SetupScore, EventState, PsychologicalMarketState, MarketInternalsState,
        TradeState, TradeDirection, BehavioralFlag, Direction
    )
    import json
    now = _now()
    ms = MarketState(symbol="SPY", timestamp=now)
    ms.regime   = RegimeState(regime=MarketRegime.TREND_BULL, confidence=0.85)
    ms.session  = SessionState(phase=SessionPhase.MIDDAY, market_open=True,
                               entries_allowed=True, readiness=Readiness.FULLY_READY)
    ms.intent   = SessionIntent(min_score=65.0)
    ms.narrative = NarrativeState(narrative=MarketNarrative.TRENDING_BULL, confidence=0.8)
    ms.setup_score = SetupScore(total=75.0, adjusted_total=75.0)
    ms.event_state = EventState(principal_alert_active=False)
    ms.psychology = PsychologicalMarketState(confidence_modifier=2.0)
    ms.internals  = MarketInternalsState(breadth_score=0.67)
    ms.active_trade = TradeState(
        trade_id="T1", direction=TradeDirection.CALLS,
        entry_time=now, size=10000.0, underlying_entry_price=545.0,
    )
    ms.behavioral_flags = [
        BehavioralFlag(flag_type="overtrading", triggered_at=now,
                       context="3rd trade below threshold", severity="warning")
    ]
    d = ms.to_dict()
    json_str = json.dumps(d)   # must not raise
    assert len(json_str) > 100

def test_market_state_behavioral_flags_filtered_to_warning_and_critical():
    from engine.models import MarketState, BehavioralFlag
    now = _now()
    ms = MarketState(symbol="SPY", timestamp=now)
    ms.behavioral_flags = [
        BehavioralFlag(flag_type="a", triggered_at=now, context="", severity="info"),
        BehavioralFlag(flag_type="b", triggered_at=now, context="", severity="warning"),
        BehavioralFlag(flag_type="c", triggered_at=now, context="", severity="critical"),
    ]
    d = ms.to_dict()
    # Only warning and critical should appear
    assert len(d["behavioral_flags"]) == 2
    severities = {f["severity"] for f in d["behavioral_flags"]}
    assert "info" not in severities


# ─────────────────────────────────────────────
# Logging tests
# ─────────────────────────────────────────────

def test_logging_setup_runs():
    import logging
    from config.logging_setup import setup_logging, get_logger
    setup_logging()   # should not raise
    logger = get_logger("test.phase1")
    logger.debug("Phase 1 logging test — debug")
    logger.info("Phase 1 logging test — info")

def test_get_logger_returns_logger():
    import logging
    from config.logging_setup import get_logger
    logger = get_logger(__name__)
    assert isinstance(logger, logging.Logger)


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    print(f"\nPhase 1 tests ({len(tests)} total)\n{'─'*50}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'='*50}")
    print(f"{passed} passed, {failed} failed out of {passed+failed} total")
    sys.exit(0 if failed == 0 else 1)
