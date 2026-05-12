"""
tests/test_phase7.py — Phase 7: CLI / operational interface.

Coverage:
  cli/warnings.py       : all warning generators, aggregator, severity sort
  cli/setup_display.py  : setup card, for/against, R:R, stop summary
  cli/dashboard.py      : section renders, score bar, full dashboard
  cli/startup_briefing.py : briefing sections, summary line
  cli/session_summary.py : session overview, trade list, lessons
  cli/commands.py       : all handlers, dispatcher, unknown command

Run: PYTHONPATH=. python tests/test_phase7.py
"""

import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def _now():
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────

def _market_state(close=545.0, atr=1.5, swing_low=541.0, swing_high=549.0,
                  regime="trend_bull", readiness="fully_ready",
                  market_open=True, entries_allowed=True):
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
        ind.vwap=close-1.5; ind.price_vs_vwap="above"; ind.macd_bias="bullish"
        s.indicators[tf] = ind
        t = TrendState(tf, Direction.BULLISH, TrendStrength.STRONG)
        t.last_swing_low  = swing_low
        t.last_swing_high = swing_high
        t.higher_highs    = True
        s.trends[tf] = t

    s.support_levels    = [SupportResistanceLevel(swing_low + 0.5, "support", 3)]
    s.resistance_levels = [SupportResistanceLevel(swing_high - 0.5, "resistance", 3)]

    readiness_map = {
        "not_ready":       Readiness.NOT_READY,
        "partially_ready": Readiness.PARTIALLY_READY,
        "fully_ready":     Readiness.FULLY_READY,
    }
    s.session = SessionState(
        phase=SessionPhase.MIDDAY, market_open=market_open,
        entries_allowed=entries_allowed, score_modifier=1.0,
        entry_filter="normal", phase_note="test",
        readiness=readiness_map.get(readiness, Readiness.FULLY_READY),
    )
    regime_map = {
        "trend_bull": MarketRegime.TREND_BULL,
        "chop":       MarketRegime.CHOP,
        "range":      MarketRegime.RANGE,
    }
    mr = regime_map.get(regime, MarketRegime.TREND_BULL)
    s.regime  = RegimeState(mr, 0.85, 1.0, True, "bull")
    s.intent  = SessionIntent(min_score=60)
    s.narrative = NarrativeState(
        narrative=MarketNarrative.TRENDING_BULL, confidence=0.9,
        description="bull", trade_posture="long_bias",
    )
    s.setup_score = SetupScore(
        total=80.0, adjusted_total=80.0,
        trend_alignment=25.0, momentum_quality=22.0,
        structure_context=20.0, entry_timing=13.0,
        alignment_detail="All TFs aligned bullish",
        momentum_detail="RSI 55, MACD positive",
        structure_detail="Above VWAP, clear swing low",
        timing_detail="5m candle quality good",
    )
    s.context_bias       = "neutral"
    s.context_risk_level = "normal"
    return s


def _setup_score(total=78.0, adjusted=78.0):
    from engine.models import SetupScore
    return SetupScore(
        total=total, adjusted_total=adjusted,
        trend_alignment=24.0, momentum_quality=20.0,
        structure_context=20.0, entry_timing=14.0,
        regime_multiplier=1.0, session_multiplier=1.0,
        readiness_penalty=0.0, context_adjustment=0.0,
        alignment_detail="Strong alignment",
        momentum_detail="Confirming",
        structure_detail="Clean levels",
        timing_detail="Good",
    )


def _context_state(bias="neutral", risk="normal", elevated=False,
                   vix=15.0, sentiment=0.0):
    from context.context_state import ContextState
    ctx = ContextState(
        timestamp=_now(),
        context_bias=bias,
        volatility_risk_level=risk,
        elevated_risk=elevated,
        news_sentiment=sentiment,
        vix_level=vix,
        vix_regime="normal",
        risk_note="",
        context_valid=True,
    )
    return ctx


def _session_memory(trades=3, wins=2, pnl=1.5):
    from memory.session_memory import SessionMemory
    s = SessionMemory(session_date="2025-06-01")
    s.opening_regime     = "trend_bull"
    s.dominant_narrative = "trending_bull"
    s.total_trades       = trades
    s.winning_trades     = wins
    s.losing_trades      = trades - wins
    s.daily_pnl_pct      = pnl
    return s


def _trade_state(entry=545.0, stop=541.0, target=551.0):
    from engine.models import TradeState, TradeDirection, InstrumentType, TradeLifecycle, TimeStopConfig
    t = TradeState(
        trade_id="TEST01", direction=TradeDirection.CALLS,
        entry_time=_now(), size=1000.0,
    )
    t.instrument_type            = InstrumentType.OPTIONS
    t.lifecycle                  = TradeLifecycle.ACTIVE
    t.entry_price                = entry
    t.underlying_entry_price     = entry
    t.underlying_stop_price      = stop
    t.profit_target              = target
    t.option_entry_price         = 2.50
    t.premium_emergency_stop_pct = 35.0
    t.time_stop_config           = TimeStopConfig(expected_move_window_minutes=12)
    t.pnl_pct                    = 0.8
    return t


def _briefing(prev_date="2025-05-31", pnl=-0.5, wr=0.4, trades=3, lesson="wait"):
    from memory.startup_context import StartupBriefing
    b = StartupBriefing(
        generated_at = _now().isoformat(),
        session_date = "2025-06-01",
    )
    b.prev_session_date     = prev_date
    b.prev_session_pnl_pct  = pnl
    b.prev_session_win_rate = wr
    b.prev_session_trades   = trades
    b.prev_session_regime   = "trend_bull"
    b.prev_session_lesson   = lesson
    b.recent_trades_n       = 10
    b.recent_win_rate       = 0.55
    b.recent_expectancy     = 0.12
    b.recent_best_setup     = "continuation"
    b.recent_worst_setup    = "reversal"
    b.top_risks             = ["Previous loss — reduce size", "Recurring mistake: chasing"]
    b.summary               = "Previous session: 3 trades, 40% WR, P&L -0.50%."
    return b


# ═══════════════════════════════════════════════════════════
# cli/warnings.py
# ═══════════════════════════════════════════════════════════

def test_warnings_readiness_not_ready():
    from cli.warnings import check_readiness, WarningSeverity
    from engine.models import Readiness
    state = _market_state(readiness="not_ready")
    w = check_readiness(state)
    assert len(w) == 1
    assert w[0].severity == WarningSeverity.CRITICAL


def test_warnings_readiness_partially_ready():
    from cli.warnings import check_readiness, WarningSeverity
    state = _market_state(readiness="partially_ready")
    w = check_readiness(state)
    assert len(w) == 1
    assert w[0].severity == WarningSeverity.WARNING


def test_warnings_readiness_fully_ready_clean():
    from cli.warnings import check_readiness
    state = _market_state(readiness="fully_ready")
    assert check_readiness(state) == []


def test_warnings_overtrading_below_limit():
    from cli.warnings import check_overtrading
    session = _session_memory(trades=2)
    assert check_overtrading(session, max_trades=3) == []


def test_warnings_overtrading_at_limit():
    from cli.warnings import check_overtrading, WarningSeverity
    session = _session_memory(trades=3)
    w = check_overtrading(session, max_trades=3)
    assert len(w) == 1
    assert w[0].severity == WarningSeverity.CAUTION


def test_warnings_overtrading_over_limit():
    from cli.warnings import check_overtrading, WarningSeverity
    session = _session_memory(trades=6)
    w = check_overtrading(session, max_trades=3)
    assert w[0].severity == WarningSeverity.WARNING


def test_warnings_macro_risk_elevated():
    from cli.warnings import check_macro_risk, WarningSeverity
    ctx = _context_state(elevated=True, risk="elevated")
    ctx.risk_note = "CPI in 30 min"
    w = check_macro_risk(ctx)
    assert any(ww.severity == WarningSeverity.WARNING for ww in w)


def test_warnings_macro_risk_none_when_normal():
    from cli.warnings import check_macro_risk
    ctx = _context_state(elevated=False, risk="normal")
    assert check_macro_risk(ctx) == []


def test_warnings_setup_confidence_critical():
    from cli.warnings import check_setup_confidence, WarningSeverity
    score = _setup_score(total=35.0, adjusted=35.0)
    w = check_setup_confidence(score)
    assert w[0].severity == WarningSeverity.CRITICAL


def test_warnings_setup_confidence_warning():
    from cli.warnings import check_setup_confidence, WarningSeverity
    score = _setup_score(total=50.0, adjusted=50.0)
    w = check_setup_confidence(score, min_score=60.0)
    assert w[0].severity == WarningSeverity.WARNING


def test_warnings_setup_confidence_none_when_above_threshold():
    from cli.warnings import check_setup_confidence
    score = _setup_score(total=75.0, adjusted=75.0)
    assert check_setup_confidence(score) == []


def test_warnings_context_conflict_long_bearish():
    from cli.warnings import check_context_conflict, WarningSeverity
    from engine.models import TradeDirection
    state = _market_state()
    state.context_bias    = "bearish"
    state.setup_direction = TradeDirection.CALLS
    w = check_context_conflict(state)
    assert len(w) == 1
    assert w[0].severity == WarningSeverity.CAUTION


def test_warnings_context_no_conflict():
    from cli.warnings import check_context_conflict
    from engine.models import TradeDirection
    state = _market_state()
    state.context_bias    = "bullish"
    state.setup_direction = TradeDirection.CALLS
    assert check_context_conflict(state) == []


def test_warnings_context_neutral_no_conflict():
    from cli.warnings import check_context_conflict
    from engine.models import TradeDirection
    state = _market_state()
    state.context_bias    = "neutral"
    state.setup_direction = TradeDirection.CALLS
    assert check_context_conflict(state) == []


def test_warnings_behavioral_flags():
    from cli.warnings import check_behavioral_flags, WarningSeverity
    session = _session_memory()
    session.behavioral_flags = ["over-traded", "revenge trading"]
    w = check_behavioral_flags(session)
    assert len(w) == 2
    assert all(ww.severity == WarningSeverity.CAUTION for ww in w)


def test_warnings_collect_sorted_by_severity():
    from cli.warnings import collect_warnings, WarningSeverity
    state   = _market_state(readiness="not_ready")
    session = _session_memory(trades=6)
    ctx     = _context_state(elevated=True, risk="elevated")
    score   = _setup_score(total=30.0, adjusted=30.0)
    ws = collect_warnings(state=state, context_state=ctx,
                          session_memory=session, score=score)
    assert len(ws) >= 2
    for i in range(len(ws) - 1):
        assert ws[i].severity.value >= ws[i+1].severity.value


def test_warnings_render_empty():
    from cli.warnings import render_warnings
    result = render_warnings([])
    assert "No active warnings" in result


def test_warnings_render_with_warnings():
    from cli.warnings import render_warnings, R2Warning, WarningSeverity, WarningCategory
    w = R2Warning(WarningCategory.READINESS, WarningSeverity.CRITICAL, "Test warning")
    result = render_warnings([w])
    assert "Test warning" in result
    assert "CRITICAL" in result


def test_warning_to_dict():
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory
    w = R2Warning(WarningCategory.MACRO, WarningSeverity.WARNING, "macro event")
    d = w.to_dict()
    for k in ("category", "severity", "message"):
        assert k in d


def test_warning_render_line():
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory
    w = R2Warning(WarningCategory.SETUP, WarningSeverity.WARNING, "Low confidence")
    line = w.render()
    assert "Low confidence" in line
    assert "WARNING" in line


# ═══════════════════════════════════════════════════════════
# cli/setup_display.py
# ═══════════════════════════════════════════════════════════

def test_setup_card_renders():
    from cli.setup_display import render_setup_card
    from engine.models import TradeDirection
    state  = _market_state()
    score  = _setup_score()
    result = render_setup_card(state, score, TradeDirection.CALLS)
    assert "SETUP ANALYSIS" in result
    assert "CALLS" in result
    assert "78" in result


def test_setup_card_no_score():
    from cli.setup_display import render_setup_card
    state  = _market_state()
    result = render_setup_card(state, None)
    assert "No setup" in result


def test_setup_summary_line():
    from cli.setup_display import render_setup_summary_line
    from engine.models import TradeDirection, SetupScore
    score  = SetupScore(total=80.0, adjusted_total=80.0)
    result = render_setup_summary_line(score, TradeDirection.CALLS)
    assert "CALLS" in result
    assert "A+" in result


def test_setup_summary_line_no_score():
    from cli.setup_display import render_setup_summary_line
    result = render_setup_summary_line(None)
    assert "No setup" in result


def test_for_against_has_both_sections():
    from cli.setup_display import render_for_against
    state  = _market_state()
    score  = _setup_score()
    result = render_for_against(score, state)
    assert "FOR:" in result
    assert "AGAINST:" in result


def test_for_against_low_score_has_against():
    from cli.setup_display import render_for_against
    state = _market_state()
    score = _setup_score(total=30.0, adjusted=30.0)
    score.trend_alignment  = 5.0
    score.momentum_quality = 5.0
    score.entry_timing     = 3.0
    result = render_for_against(score, state)
    assert "-" in result


def test_rr_summary_renders():
    from cli.setup_display import render_rr_summary
    result = render_rr_summary(545.0, 541.0, 551.0, 1.5)
    assert "1.50" in result
    assert "545" in result or "541" in result


def test_rr_summary_fail_marker():
    from cli.setup_display import render_rr_summary
    result = render_rr_summary(545.0, 541.0, 547.0, 0.5)
    assert "✗" in result


def test_rr_summary_pass_marker():
    from cli.setup_display import render_rr_summary
    result = render_rr_summary(545.0, 541.0, 551.0, 1.5)
    assert "✓" in result


def test_stop_summary_no_trade():
    from cli.setup_display import render_stop_summary
    result = render_stop_summary(None)
    assert "No active trade" in result


def test_stop_summary_with_trade():
    from cli.setup_display import render_stop_summary
    trade  = _trade_state()
    result = render_stop_summary(trade)
    assert "541" in result
    assert "Target" in result


def test_stop_summary_triggered_flag():
    from cli.setup_display import render_stop_summary
    trade = _trade_state()
    trade.thesis_stop_triggered = True
    result = render_stop_summary(trade)
    assert "THESIS STOP" in result


# ═══════════════════════════════════════════════════════════
# cli/dashboard.py
# ═══════════════════════════════════════════════════════════

def test_dashboard_renders_without_error():
    from cli.dashboard import render_dashboard
    state  = _market_state()
    result = render_dashboard(state)
    assert len(result) > 100
    assert "R2" in result


def test_dashboard_contains_market_section():
    from cli.dashboard import render_dashboard
    state  = _market_state()
    result = render_dashboard(state)
    assert "MARKET" in result
    assert "545" in result


def test_dashboard_contains_score_section():
    from cli.dashboard import render_dashboard
    from engine.models import TradeDirection
    state = _market_state()
    state.setup_direction = TradeDirection.CALLS
    score  = _setup_score()
    result = render_dashboard(state, score=score)
    assert "SETUP SCORE" in result
    assert "78" in result


def test_dashboard_contains_warnings_section():
    from cli.dashboard import render_dashboard
    state  = _market_state()
    result = render_dashboard(state)
    assert "WARNINGS" in result


def test_dashboard_active_trade_section():
    from cli.dashboard import render_dashboard
    state = _market_state()
    state.active_trade = _trade_state()
    result = render_dashboard(state)
    assert "ACTIVE TRADE" in result


def test_dashboard_with_context():
    from cli.dashboard import render_dashboard
    state  = _market_state()
    ctx    = _context_state(bias="bullish", vix=14.5)
    result = render_dashboard(state, context_state=ctx)
    assert "14.5" in result


def test_dashboard_patterns_section():
    from cli.dashboard import render_dashboard
    state    = _market_state()
    patterns = [
        {"pattern_id": "bull_flag__bullish__5m", "confidence_score": 0.80,
         "win_rate": 0.75, "occurrence_count": 12},
    ]
    result = render_dashboard(state, active_patterns=patterns)
    assert "ACTIVE PATTERNS" in result


def test_dashboard_events_section():
    from cli.dashboard import render_dashboard
    state  = _market_state()
    result = render_dashboard(state, recent_events=["SPY breaks above 545", "VWAP reclaimed"])
    assert "RECENT EVENTS" in result


def test_render_regime_line():
    from cli.dashboard import render_regime_line
    state  = _market_state(regime="trend_bull")
    result = render_regime_line(state)
    assert "Trend Bull" in result or "trend_bull" in result.lower()


def test_render_price_line():
    from cli.dashboard import render_price_line
    state  = _market_state(close=547.25)
    result = render_price_line(state)
    assert "547.25" in result


def test_render_session_line():
    from cli.dashboard import render_session_line
    state  = _market_state()
    result = render_session_line(state)
    assert "MIDDAY" in result


def test_score_bar_renders():
    from cli.dashboard import _score_bar
    bar = _score_bar(78.0)
    assert "78" in bar
    assert "█" in bar


# ═══════════════════════════════════════════════════════════
# cli/startup_briefing.py
# ═══════════════════════════════════════════════════════════

def test_briefing_renders():
    from cli.startup_briefing import render_startup_briefing
    b      = _briefing()
    result = render_startup_briefing(b)
    assert "PRE-MARKET BRIEFING" in result
    assert "2025-06-01" in result


def test_briefing_previous_session_shown():
    from cli.startup_briefing import render_startup_briefing
    b      = _briefing(prev_date="2025-05-31", pnl=-0.5, wr=0.40, trades=3, lesson="be patient")
    result = render_startup_briefing(b)
    assert "2025-05-31" in result
    assert "be patient" in result


def test_briefing_top_risks_shown():
    from cli.startup_briefing import render_startup_briefing
    b      = _briefing()
    result = render_startup_briefing(b)
    assert "TOP RISKS" in result
    assert "Previous loss" in result


def test_briefing_no_previous_session():
    from cli.startup_briefing import render_startup_briefing
    from memory.startup_context import StartupBriefing
    b      = StartupBriefing(generated_at=_now().isoformat(), session_date="2025-06-01")
    result = render_startup_briefing(b)
    assert "No prior session data" in result


def test_briefing_with_market_state():
    from cli.startup_briefing import render_startup_briefing
    b      = _briefing()
    state  = _market_state()
    result = render_startup_briefing(b, state=state)
    assert "CURRENT MARKET" in result
    assert "545" in result


def test_briefing_patterns_market():
    from cli.startup_briefing import render_startup_briefing
    from memory.startup_context import StartupBriefing
    b = StartupBriefing(generated_at=_now().isoformat(), session_date="2025-06-01")
    b.active_market_patterns = [
        {"pattern_id": "bull_flag__bullish__5m", "win_rate": 0.80,
         "confidence_score": 0.80, "occurrence_count": 12}
    ]
    result = render_startup_briefing(b)
    assert "Market patterns" in result
    assert "bull_flag" in result


def test_briefing_summary_line():
    from cli.startup_briefing import render_briefing_summary_line
    b      = _briefing(prev_date="2025-05-31", pnl=1.2, wr=0.60, trades=4)
    result = render_briefing_summary_line(b)
    assert "2025-05-31" in result
    assert "60%" in result


def test_briefing_summary_line_no_prev():
    from cli.startup_briefing import render_briefing_summary_line
    from memory.startup_context import StartupBriefing
    b      = StartupBriefing(generated_at=_now().isoformat(), session_date="2025-06-01")
    result = render_briefing_summary_line(b)
    assert "No prior" in result


def test_briefing_footer_trade_the_process():
    from cli.startup_briefing import render_startup_briefing
    b      = _briefing()
    result = render_startup_briefing(b)
    assert "Trade the process" in result


# ═══════════════════════════════════════════════════════════
# cli/session_summary.py
# ═══════════════════════════════════════════════════════════

def test_session_summary_renders():
    from cli.session_summary import render_session_summary
    session = _session_memory(trades=3, wins=2, pnl=1.2)
    result  = render_session_summary(session)
    assert "SESSION SUMMARY" in result
    assert "2025-06-01" in result


def test_session_summary_overview():
    from cli.session_summary import render_session_summary
    session = _session_memory(trades=4, wins=3, pnl=2.1)
    result  = render_session_summary(session)
    assert "3W" in result or "3" in result
    assert "2.10" in result or "2.1" in result


def test_session_summary_trade_list():
    from cli.session_summary import render_session_summary
    from memory.trade_memory import TradeRecord
    session = _session_memory()
    now     = _now().isoformat()
    t = TradeRecord.from_trade(
        trade_id="T001", session_date="2025-06-01",
        direction="calls", instrument_type="options",
        entry_price=545.0, exit_price=551.0,
        entry_time=now, exit_time=now,
        stop_price=541.0, target_price=551.0,
        pnl_pct=1.2, pnl_dollar=60.0, exit_reason="target",
        setup_type="continuation",
    )
    result = render_session_summary(session, trades=[t])
    assert "TRADES" in result
    assert "T001" in result


def test_session_summary_with_reflection():
    from cli.session_summary import render_session_summary
    from memory.reflection_engine import ReflectionEngine
    from memory.trade_memory import TradeRecord
    session = _session_memory(trades=2, wins=1, pnl=0.5)
    now     = _now().isoformat()
    trades  = [
        TradeRecord.from_trade(
            trade_id=f"T00{i}", session_date="2025-06-01",
            direction="calls", instrument_type="options",
            entry_price=545.0, exit_price=548.0 + i,
            entry_time=now, exit_time=now,
            stop_price=541.0, target_price=551.0,
            pnl_pct=0.5 * (1 if i == 0 else -1),
            pnl_dollar=25.0, exit_reason="target",
            setup_type="continuation",
        ) for i in range(2)
    ]
    trades[0].rule_compliant = True
    trades[0].rr_planned     = 1.6
    trades[1].rule_compliant = False
    trades[1].mistake_tags   = ["chased entry"]
    reflection = ReflectionEngine().generate(session, trades)
    result = render_session_summary(session, reflection=reflection, trades=trades)
    assert "EXECUTION QUALITY" in result
    assert "ANALYSIS" in result


def test_session_summary_lessons():
    from cli.session_summary import render_session_summary
    session = _session_memory(trades=1, wins=1, pnl=1.5)
    session.lesson = "Wait for VWAP reclaim"
    result = render_session_summary(session)
    assert "LESSONS" in result or "Wait for VWAP" in result


def test_session_one_liner():
    from cli.session_summary import render_session_one_liner
    session = _session_memory(trades=3, wins=2, pnl=1.5)
    result  = render_session_one_liner(session)
    assert "3T" in result
    assert "1.50%" in result


def test_session_summary_performance_breakdown():
    from cli.session_summary import render_session_summary
    from memory.trade_memory import TradeRecord
    from memory.performance_stats import build_report
    session = _session_memory()
    now     = _now().isoformat()
    trades  = []
    for i, (pnl, stype) in enumerate([
        (1.2,"continuation"),(0.8,"continuation"),(1.5,"continuation"),(-0.5,"reversal")
    ]):
        t = TradeRecord.from_trade(
            trade_id=f"T{i}", session_date="2025-06-01",
            direction="calls", instrument_type="options",
            entry_price=545.0, exit_price=548.0,
            entry_time=now, exit_time=now,
            stop_price=541.0, target_price=551.0,
            pnl_pct=pnl, pnl_dollar=60.0, exit_reason="target",
            setup_type=stype,
        )
        trades.append(t)
    report = build_report(trades)
    result = render_session_summary(session, trades=trades, report=report)
    assert "PERFORMANCE BY SETUP" in result
    assert "continuation" in result


# ═══════════════════════════════════════════════════════════
# cli/commands.py
# ═══════════════════════════════════════════════════════════

def _ctx(**kwargs):
    """Build a minimal command context dict."""
    base = {
        "state":           _market_state(),
        "score":           _setup_score(),
        "context_state":   _context_state(),
        "session_memory":  _session_memory(),
        "pattern_memory":  None,
        "reflection":      None,
        "trades":          None,
        "report":          None,
        "active_patterns": [],
    }
    base.update(kwargs)
    return base


def test_dispatch_help():
    from cli.commands import dispatch
    result = dispatch("help", _ctx())
    assert result.success
    assert "summary" in result.output
    assert "warnings" in result.output


def test_dispatch_summary():
    from cli.commands import dispatch
    result = dispatch("summary", _ctx())
    assert result.success
    assert "2025-06-01" in result.output or "1.50" in result.output


def test_dispatch_summary_no_session():
    from cli.commands import dispatch
    result = dispatch("summary", _ctx(session_memory=None))
    assert not result.success
    assert result.error


def test_dispatch_active_no_trade():
    from cli.commands import dispatch
    state = _market_state()
    state.active_trade = None
    result = dispatch("active", _ctx(state=state))
    assert result.success
    assert "No active trade" in result.output


def test_dispatch_active_with_trade():
    from cli.commands import dispatch
    from engine.models import TradeDirection
    state = _market_state()
    state.active_trade    = _trade_state()
    state.setup_direction = TradeDirection.CALLS
    result = dispatch("active", _ctx(state=state))
    assert result.success
    assert "541" in result.output


def test_dispatch_patterns_no_pm():
    from cli.commands import dispatch
    result = dispatch("patterns", _ctx(pattern_memory=None))
    assert result.success


def test_dispatch_patterns_with_pm():
    from cli.commands import dispatch
    from memory.pattern_memory import PatternMemory, MarketPatternType
    d  = tempfile.mkdtemp()
    pm = PatternMemory(f"{d}/p.json")
    pm.record_market_pattern(MarketPatternType.BULL_FLAG, "bullish", "5m", trade_id="T1")
    result = dispatch("patterns", _ctx(pattern_memory=pm))
    assert result.success


def test_dispatch_warnings_returns_output():
    from cli.commands import dispatch
    result = dispatch("warnings", _ctx())
    assert result.success
    assert len(result.output) > 0


def test_dispatch_warnings_no_state():
    from cli.commands import dispatch
    result = dispatch("warnings", _ctx(state=None))
    assert not result.success


def test_dispatch_trade_renders_score():
    from cli.commands import dispatch
    result = dispatch("trade", _ctx())
    assert result.success
    assert "SETUP ANALYSIS" in result.output or "78" in result.output


def test_dispatch_context_with_data():
    from cli.commands import dispatch
    ctx    = _context_state(bias="bullish", vix=14.5, sentiment=0.35)
    result = dispatch("context", _ctx(context_state=ctx))
    assert result.success
    assert "bullish" in result.output
    assert "14.5" in result.output


def test_dispatch_context_invalid():
    from cli.commands import dispatch
    from context.context_state import ContextState
    ctx    = ContextState.empty()
    result = dispatch("context", _ctx(context_state=ctx))
    assert result.success
    assert "unavailable" in result.output


def test_dispatch_reflection_no_data():
    from cli.commands import dispatch
    result = dispatch("reflection", _ctx(reflection=None, session_memory=None))
    assert result.success
    assert "No reflection" in result.output


def test_dispatch_reflection_with_session():
    from cli.commands import dispatch
    result = dispatch("reflection", _ctx())
    assert result.success


def test_dispatch_unknown_command():
    from cli.commands import dispatch
    result = dispatch("xyzzy", _ctx())
    assert not result.success
    assert "Unknown command" in result.error
    assert "xyzzy" in result.error


def test_dispatch_empty_command():
    from cli.commands import dispatch
    result = dispatch("", _ctx())
    assert not result.success


def test_dispatch_case_insensitive():
    from cli.commands import dispatch
    result = dispatch("HELP", _ctx())
    assert result.success


def test_dispatch_command_with_extra_spaces():
    from cli.commands import dispatch
    result = dispatch("  summary  ", _ctx())
    assert result.success


def test_available_commands():
    from cli.commands import available_commands
    cmds = available_commands()
    for expected in ["help","summary","active","patterns","warnings","trade","context","reflection"]:
        assert expected in cmds


def test_command_result_str_success():
    from cli.commands import CommandResult
    r = CommandResult(command="test", success=True, output="hello")
    assert str(r) == "hello"


def test_command_result_str_failure():
    from cli.commands import CommandResult
    r = CommandResult(command="test", success=False, output="", error="bad input")
    assert "bad input" in str(r)


# ═══════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════

def test_integration_full_dashboard_cycle():
    """Dashboard → warnings → commands all consume same state."""
    from cli.dashboard import render_dashboard
    from cli.warnings import collect_warnings
    from cli.commands import dispatch

    state   = _market_state()
    score   = _setup_score()
    ctx     = _context_state(bias="bullish")
    session = _session_memory(trades=5, wins=3, pnl=1.0)

    dash = render_dashboard(state, score=score, context_state=ctx, session_memory=session)
    assert "545" in dash

    ws = collect_warnings(state=state, context_state=ctx, session_memory=session, score=score)
    assert isinstance(ws, list)

    cmd_ctx = {
        "state": state, "score": score, "context_state": ctx,
        "session_memory": session, "pattern_memory": None,
        "reflection": None, "trades": None, "report": None,
        "active_patterns": [],
    }
    for cmd in ["help", "summary", "warnings", "trade", "context"]:
        result = dispatch(cmd, cmd_ctx)
        assert result.success, f"Command {cmd!r} failed: {result.error}"


def test_integration_startup_to_dashboard():
    """Startup briefing feeds into dashboard without errors."""
    from cli.startup_briefing import render_startup_briefing
    from cli.dashboard import render_dashboard

    b     = _briefing()
    state = _market_state()
    ctx   = _context_state()

    briefing_output  = render_startup_briefing(b, state=state, context_state=ctx)
    dashboard_output = render_dashboard(state, context_state=ctx)

    assert len(briefing_output)  > 50
    assert len(dashboard_output) > 50


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nPhase 7 tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)