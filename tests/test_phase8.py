"""
tests/test_phase8.py — Phase 8: AI / Claude integration layer.

Coverage:
  ai/r2_persona.py      : persona rules, forbidden phrases, risk flags
  ai/state_packager.py  : each packager function, full package build
  ai/prompt_builder.py  : each prompt type, token estimation, risk blocks
  ai/response_parser.py : parse_response, guardrail checks, fallback
  ai/claude_client.py   : mock mode, all public functions, history, fallback

Design: ALL tests use mock=True or pure functions — no real API calls.

Run: PYTHONPATH=. python tests/test_phase8.py
"""

import os
import sys
import traceback
from datetime import datetime, timezone

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
# Shared builders
# ─────────────────────────────────────────────

def _market_state(close=545.0, atr=1.5, regime="trend_bull",
                  readiness="fully_ready", market_open=True):
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState, Direction, TrendStrength,
        RegimeState, MarketRegime, SessionState, SessionPhase, Readiness,
        SessionIntent, SupportResistanceLevel, NarrativeState, MarketNarrative,
        SetupScore, TradeDirection,
    )
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d", "1h", "15m", "5m", "1m"]:
        ind = IndicatorSnapshot(tf, _now(), close)
        ind.atr=atr; ind.ema_9=close-0.5; ind.ema_20=close-1.0
        ind.rsi=55.0; ind.macd_hist=0.01; ind.ema_stack="bull_aligned"
        ind.vwap=close-1.5; ind.price_vs_vwap="above"; ind.macd_bias="bullish"
        s.indicators[tf] = ind
        t = TrendState(tf, Direction.BULLISH, TrendStrength.STRONG)
        t.last_swing_low=close-4.0; t.last_swing_high=close+4.0
        t.higher_highs=True
        s.trends[tf] = t
    s.support_levels    = [SupportResistanceLevel(close-3.0, "support", 3)]
    s.resistance_levels = [SupportResistanceLevel(close+3.0, "resistance", 3)]
    rm = {"trend_bull": MarketRegime.TREND_BULL, "chop": MarketRegime.CHOP}.get(
        regime, MarketRegime.TREND_BULL)
    rm_map = {"not_ready":       Readiness.NOT_READY,
              "partially_ready": Readiness.PARTIALLY_READY,
              "fully_ready":     Readiness.FULLY_READY}
    s.session = SessionState(
        phase=SessionPhase.MIDDAY, market_open=market_open,
        entries_allowed=True, score_modifier=1.0,
        entry_filter="normal", phase_note="test",
        readiness=rm_map.get(readiness, Readiness.FULLY_READY),
    )
    s.regime = RegimeState(rm, 0.85, 1.0, True, "bull")
    s.intent = SessionIntent(min_score=60)
    s.narrative = NarrativeState(
        narrative=MarketNarrative.TRENDING_BULL, confidence=0.9,
        description="bull", trade_posture="long_bias",
    )
    s.setup_score = SetupScore(total=80.0, adjusted_total=80.0,
        trend_alignment=25.0, momentum_quality=22.0,
        structure_context=20.0, entry_timing=13.0,
        alignment_detail="All TFs aligned bullish",
        momentum_detail="RSI 55, MACD positive",
        structure_detail="Above VWAP, clear swing low",
        timing_detail="5m candle quality good",
    )
    s.context_bias = "neutral"
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
        alignment_detail="Strong", momentum_detail="Good",
        structure_detail="Clean", timing_detail="Fine",
    )


def _context_state(bias="neutral", risk="normal", elevated=False,
                   vix=15.0, sentiment=0.0):
    from context.context_state import ContextState
    return ContextState(
        timestamp=_now(),
        context_bias=bias,
        volatility_risk_level=risk,
        elevated_risk=elevated,
        news_sentiment=sentiment,
        vix_level=vix,
        vix_regime="normal",
        risk_note="" if not elevated else "Test risk note",
        context_valid=True,
    )


def _session_memory(trades=3, wins=2, pnl=1.5):
    from memory.session_memory import SessionMemory
    s = SessionMemory(session_date="2025-06-01")
    s.opening_regime="trend_bull"; s.dominant_narrative="trending_bull"
    s.total_trades=trades; s.winning_trades=wins
    s.losing_trades=trades-wins; s.daily_pnl_pct=pnl
    return s


def _trade_state(entry=545.0, stop=541.0, target=551.0):
    from engine.models import (TradeState, TradeDirection, InstrumentType,
                               TradeLifecycle, TimeStopConfig)
    t = TradeState(trade_id="T01", direction=TradeDirection.CALLS,
                   entry_time=_now(), size=1000.0)
    t.instrument_type=InstrumentType.OPTIONS
    t.lifecycle=TradeLifecycle.ACTIVE
    t.entry_price=entry; t.underlying_entry_price=entry
    t.underlying_stop_price=stop; t.profit_target=target
    t.option_entry_price=2.50; t.premium_emergency_stop_pct=35.0
    t.time_stop_config=TimeStopConfig(expected_move_window_minutes=12)
    t.pnl_pct=0.8
    t.thesis="Bullish setup | regime: trend_bull | score: 78 (B) | R:R 1.50"
    return t


def _package(state=None, score=None, direction=None, context=None,
             trade=None, session=None):
    from ai.state_packager import build_full_package
    from engine.models import TradeDirection
    return build_full_package(
        state          = state   or _market_state(),
        score          = score   or _setup_score(),
        direction      = direction or TradeDirection.CALLS,
        context_state  = context or _context_state(),
        active_trade   = trade,
        session_memory = session or _session_memory(),
        pattern_memory = None,
        warnings       = [],
    )


# ═══════════════════════════════════════════════════════════
# ai/r2_persona.py
# ═══════════════════════════════════════════════════════════

def test_persona_system_prompt_not_empty():
    from ai.r2_persona import R2_SYSTEM_PROMPT
    assert len(R2_SYSTEM_PROMPT) > 200
    assert "R2" in R2_SYSTEM_PROMPT


def test_persona_brief_system_prompt_shorter():
    from ai.r2_persona import R2_SYSTEM_PROMPT, R2_BRIEF_SYSTEM_PROMPT
    assert len(R2_BRIEF_SYSTEM_PROMPT) < len(R2_SYSTEM_PROMPT)


def test_persona_forbidden_phrases_not_empty():
    from ai.r2_persona import FORBIDDEN_PHRASES
    assert len(FORBIDDEN_PHRASES) >= 5
    assert "buy here" in FORBIDDEN_PHRASES
    assert "take the trade" in FORBIDDEN_PHRASES


def test_persona_is_advisory_clean_text():
    from ai.r2_persona import is_advisory_response
    clean = "The setup grade is B. Consider the R:R before entry. Stop is at 541."
    assert is_advisory_response(clean) is True


def test_persona_is_advisory_detects_forbidden():
    from ai.r2_persona import is_advisory_response
    hype = "This is incredible setup — buy here now!"
    assert is_advisory_response(hype) is False


def test_persona_extract_risk_flags_not_ready():
    from ai.r2_persona import extract_risk_flags
    assert "not_ready" in extract_risk_flags(readiness="not_ready")


def test_persona_extract_risk_flags_partial_ready():
    from ai.r2_persona import extract_risk_flags
    assert "partial_readiness" in extract_risk_flags(readiness="partially_ready")


def test_persona_extract_risk_flags_low_score():
    from ai.r2_persona import extract_risk_flags
    assert "low_score" in extract_risk_flags(setup_score=30.0, setup_grade="skip")


def test_persona_extract_risk_flags_elevated():
    from ai.r2_persona import extract_risk_flags
    assert "elevated_macro_risk" in extract_risk_flags(elevated_risk=True)


def test_persona_extract_risk_flags_thesis_stopped():
    from ai.r2_persona import extract_risk_flags
    assert "thesis_stopped" in extract_risk_flags(thesis_stopped=True)


def test_persona_extract_risk_flags_overtrading():
    from ai.r2_persona import extract_risk_flags
    assert "overtrading" in extract_risk_flags(trade_count=4, max_trades=3)


def test_persona_extract_no_flags_clean():
    from ai.r2_persona import extract_risk_flags
    flags = extract_risk_flags(
        readiness="fully_ready", setup_grade="B", setup_score=75.0,
        elevated_risk=False, thesis_stopped=False, trade_count=1,
    )
    assert flags == []


def test_persona_risk_flags_dict_complete():
    from ai.r2_persona import RISK_FLAGS
    for key in {"not_ready","low_score","elevated_macro_risk",
                "context_conflict","thesis_stopped","overtrading","partial_readiness"}:
        assert key in RISK_FLAGS


# ═══════════════════════════════════════════════════════════
# ai/state_packager.py
# ═══════════════════════════════════════════════════════════

def test_packager_market_state_basic():
    from ai.state_packager import package_market_state
    d = package_market_state(_market_state())
    assert d["symbol"] == "SPY"
    assert "regime" in d
    assert "narrative" in d
    assert "session" in d
    assert "indicators" in d


def test_packager_market_only_priority_tfs():
    from ai.state_packager import package_market_state
    d = package_market_state(_market_state())
    for tf in d.get("indicators", {}):
        assert tf in ("1d", "1h", "15m", "5m")
    assert "1m" not in d.get("indicators", {})


def test_packager_market_has_regime_fields():
    from ai.state_packager import package_market_state
    r = package_market_state(_market_state())["regime"]
    assert "name" in r and "confidence" in r and "entry_permitted" in r


def test_packager_setup_basic():
    from ai.state_packager import package_setup
    from engine.models import TradeDirection
    d = package_setup(_setup_score(), TradeDirection.CALLS)
    assert d["score"] == 78.0
    assert d["grade"] == "B"
    assert d["direction"] == "calls"
    assert "components" in d


def test_packager_setup_none():
    from ai.state_packager import package_setup
    assert package_setup(None) == {"available": False}


def test_packager_context_basic():
    from ai.state_packager import package_context
    d = package_context(_context_state(bias="bullish", vix=16.5))
    assert d["bias"] == "bullish"
    assert d["vix"]["level"] == 16.5


def test_packager_context_none():
    from ai.state_packager import package_context
    assert package_context(None) == {"available": False}


def test_packager_context_invalid():
    from ai.state_packager import package_context
    from context.context_state import ContextState
    assert package_context(ContextState.empty()) == {"available": False}


def test_packager_trade_basic():
    from ai.state_packager import package_trade
    d = package_trade(_trade_state())
    assert d["active"] is True
    assert d["entry"] == 545.0
    assert d["stop"]  == 541.0
    assert d["direction"] == "calls"


def test_packager_trade_none():
    from ai.state_packager import package_trade
    assert package_trade(None) == {"active": False}


def test_packager_trade_stop_flags():
    from ai.state_packager import package_trade
    trade = _trade_state()
    trade.thesis_stop_triggered = True
    assert package_trade(trade)["stop_flags"]["thesis"] is True


def test_packager_session_basic():
    from ai.state_packager import package_session
    d = package_session(_session_memory(trades=3, wins=2, pnl=1.5))
    assert d["trades"] == 3
    assert d["daily_pnl"] == 1.5
    assert d["win_rate_pct"] > 0


def test_packager_session_none():
    from ai.state_packager import package_session
    assert package_session(None) == {"available": False}


def test_packager_warnings_empty():
    from ai.state_packager import package_warnings
    assert package_warnings([])["count"] == 0


def test_packager_warnings_with_items():
    from ai.state_packager import package_warnings
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory
    ws = [
        R2Warning(WarningCategory.READINESS, WarningSeverity.CRITICAL, "Not ready"),
        R2Warning(WarningCategory.MACRO,     WarningSeverity.WARNING,  "Macro risk"),
    ]
    d = package_warnings(ws)
    assert d["count"] == 2
    assert "CRITICAL" in d["groups"]
    assert "WARNING"  in d["groups"]


def test_packager_build_full_package():
    from ai.state_packager import build_full_package
    from engine.models import TradeDirection
    pkg = build_full_package(
        state=_market_state(), score=_setup_score(),
        direction=TradeDirection.CALLS, context_state=_context_state(),
        session_memory=_session_memory(),
    )
    assert pkg.market  is not None
    assert pkg.setup   is not None
    assert pkg.context is not None
    assert pkg.session is not None
    assert isinstance(pkg.risk_flags, list)
    assert isinstance(pkg.assembled_at, str)


def test_packager_full_package_not_ready_flags():
    from ai.state_packager import build_full_package
    pkg = build_full_package(state=_market_state(readiness="not_ready"))
    assert "not_ready" in pkg.risk_flags


def test_packager_full_package_to_dict():
    d = _package().to_dict()
    for k in ("assembled_at", "request_type", "market", "setup", "session"):
        assert k in d


def test_packager_token_estimate_reasonable():
    tokens = _package().estimated_tokens()
    assert 50 < tokens < 2000


# ═══════════════════════════════════════════════════════════
# ai/prompt_builder.py
# ═══════════════════════════════════════════════════════════

def test_prompt_startup_briefing():
    from ai.prompt_builder import build_startup_briefing_prompt
    system, user = build_startup_briefing_prompt(_package())
    assert len(system) > 100
    assert "<context>" in user and "<request>" in user
    assert "pre-market" in user.lower() or "briefing" in user.lower()


def test_prompt_market_commentary():
    from ai.prompt_builder import build_market_commentary_prompt
    _, user = build_market_commentary_prompt(_package())
    assert "<context>" in user and "<request>" in user


def test_prompt_setup_explanation():
    from ai.prompt_builder import build_setup_explanation_prompt
    _, user = build_setup_explanation_prompt(_package(), direction="calls")
    assert "<context>" in user and "calls" in user.lower()


def test_prompt_trade_review():
    from ai.prompt_builder import build_trade_review_prompt
    _, user = build_trade_review_prompt(_package(trade=_trade_state()))
    assert "<request>" in user and "trade" in user.lower()


def test_prompt_end_of_day():
    from ai.prompt_builder import build_end_of_day_prompt
    _, user = build_end_of_day_prompt(_package())
    assert "reflection" in user.lower() or "session" in user.lower()


def test_prompt_user_question():
    from ai.prompt_builder import build_user_question_prompt
    _, user = build_user_question_prompt(
        _package(), "Is this setup still valid after the pullback?"
    )
    assert "<question>" in user and "still valid" in user


def test_prompt_warning_explanation():
    from ai.prompt_builder import build_warning_explanation_prompt
    _, user = build_warning_explanation_prompt(_package(), "Readiness below threshold")
    assert "Readiness below threshold" in user and "<warning>" in user


def test_prompt_uses_r2_system_prompt():
    from ai.prompt_builder import build_startup_briefing_prompt
    from ai.r2_persona import R2_SYSTEM_PROMPT
    system, _ = build_startup_briefing_prompt(_package())
    assert system == R2_SYSTEM_PROMPT


def test_prompt_brief_uses_brief_system():
    from ai.prompt_builder import build_market_commentary_prompt
    from ai.r2_persona import R2_BRIEF_SYSTEM_PROMPT
    system, _ = build_market_commentary_prompt(_package())
    assert system == R2_BRIEF_SYSTEM_PROMPT


def test_prompt_risk_block_injected_when_flags():
    from ai.prompt_builder import build_startup_briefing_prompt
    from ai.state_packager import build_full_package
    pkg = build_full_package(state=_market_state(readiness="not_ready"))
    _, user = build_startup_briefing_prompt(pkg)
    assert "<risk_flags>" in user and "not_ready" in user


def test_prompt_no_risk_block_when_clean():
    from ai.prompt_builder import build_startup_briefing_prompt
    pkg = _package()
    pkg.risk_flags = []
    _, user = build_startup_briefing_prompt(pkg)
    assert "<risk_flags>" not in user


def test_prompt_token_estimate():
    from ai.prompt_builder import estimate_tokens, build_startup_briefing_prompt
    system, user = build_startup_briefing_prompt(_package())
    tokens = estimate_tokens(system, user)
    assert 300 < tokens < 3000


def test_prompt_context_truncated_for_large_input():
    from ai.prompt_builder import _serialise
    result = _serialise(_package(), max_chars=200)
    assert len(result) <= 250


# ═══════════════════════════════════════════════════════════
# ai/response_parser.py
# ═══════════════════════════════════════════════════════════

def test_parser_basic_parse():
    from ai.response_parser import parse_response
    text = "The structure looks solid. Stop is at 541. Worth watching."
    r = parse_response(text, "market_commentary")
    assert r.text == text
    assert r.recommendation_type == "market_commentary"
    assert r.guardrail_passed is True
    assert r.is_fallback is False


def test_parser_strips_whitespace():
    from ai.response_parser import parse_response
    assert parse_response("  some response  \n").text == "some response"


def test_parser_detects_forbidden_phrase():
    from ai.response_parser import parse_response
    r = parse_response("This is a great setup — buy here and profit!")
    assert r.guardrail_passed is False
    assert "buy here" in r.guardrail_violations


def test_parser_redacts_forbidden_phrase():
    from ai.response_parser import parse_response
    r = parse_response("You should buy here immediately.")
    assert "buy here" not in r.text.lower() or "[redacted]" in r.text


def test_parser_confidence_high():
    from ai.response_parser import parse_response
    r = parse_response("Clearly a high confidence setup — all timeframes aligned and confirmed.")
    assert r.confidence_label == "high"


def test_parser_confidence_low():
    from ai.response_parser import parse_response
    r = parse_response("Unclear market conditions — mixed signals and low confidence. Stand aside.")
    assert r.confidence_label == "low"


def test_parser_confidence_medium_default():
    from ai.response_parser import parse_response
    assert parse_response("The setup shows decent alignment.").confidence_label == "medium"


def test_parser_risk_flags_propagated():
    from ai.response_parser import parse_response
    r = parse_response("Neutral observation.", risk_flags=["not_ready", "low_score"])
    assert "not_ready" in r.risk_flags and "low_score" in r.risk_flags


def test_parser_to_dict_structure():
    from ai.response_parser import parse_response
    d = parse_response("Some analysis here.").to_dict()
    for k in ("text","recommendation_type","confidence_label",
              "risk_flags","guardrail_passed","is_fallback","generated_at"):
        assert k in d


def test_parser_is_usable_normal():
    from ai.response_parser import parse_response
    assert parse_response("Normal response text.").is_usable is True


def test_parser_fallback_response():
    from ai.response_parser import make_fallback_response
    r = make_fallback_response("Connection timeout")
    assert r.is_fallback is True
    assert "unavailable" in r.text.lower() or "timeout" in r.text.lower()
    assert r.guardrail_passed is True
    assert r.is_usable is False


def test_parser_fallback_has_risk_flags():
    from ai.response_parser import make_fallback_response
    r = make_fallback_response("test", risk_flags=["not_ready"])
    assert "not_ready" in r.risk_flags


def test_parser_validate_clean():
    from ai.response_parser import parse_response, validate_response
    assert validate_response(parse_response("Clean analysis text.")) == []


def test_parser_validate_fallback():
    from ai.response_parser import make_fallback_response, validate_response
    issues = validate_response(make_fallback_response("test error"))
    assert any("Fallback" in i for i in issues)


def test_parser_has_risk_flags_property():
    from ai.response_parser import parse_response
    assert parse_response("text", risk_flags=["not_ready"]).has_risk_flags is True


def test_parser_no_risk_flags():
    from ai.response_parser import parse_response
    assert parse_response("text", risk_flags=[]).has_risk_flags is False


# ═══════════════════════════════════════════════════════════
# ai/claude_client.py — mock mode
# ═══════════════════════════════════════════════════════════

def test_client_mock_available():
    from ai.claude_client import ClaudeClient
    assert ClaudeClient(mock=True).is_available is True


def test_client_no_key_unavailable():
    from ai.claude_client import ClaudeClient
    assert ClaudeClient(api_key="", mock=False).is_available is False


def test_client_mock_startup_briefing():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(ClaudeClient(mock=True).ask_with_package(_package(), "startup_briefing"))
    assert r.recommendation_type == "startup_briefing"
    assert not r.is_fallback and len(r.text) > 20


def test_client_mock_market_commentary():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(ClaudeClient(mock=True).ask_with_package(_package(), "market_commentary"))
    assert not r.is_fallback
    assert "trend" in r.text.lower() or "narrative" in r.text.lower()


def test_client_mock_setup_explanation():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(
        ClaudeClient(mock=True).ask_with_package(
            _package(), "setup_explanation", direction="calls"
        )
    )
    assert not r.is_fallback
    assert "setup" in r.text.lower() or "grade" in r.text.lower()


def test_client_mock_trade_review():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(
        ClaudeClient(mock=True).ask_with_package(_package(trade=_trade_state()), "trade_review")
    )
    assert not r.is_fallback


def test_client_mock_end_of_day():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(ClaudeClient(mock=True).ask_with_package(_package(), "end_of_day"))
    assert not r.is_fallback


def test_client_mock_user_question():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(
        ClaudeClient(mock=True).ask_with_package(
            _package(), "user_question", user_question="Is this setup valid?"
        )
    )
    assert not r.is_fallback


def test_client_fallback_when_unavailable():
    import asyncio
    from ai.claude_client import ClaudeClient
    r = asyncio.run(ClaudeClient(api_key="", mock=False).ask_with_package(
        _package(), "market_commentary"
    ))
    assert r.is_fallback is True
    assert "unavailable" in r.text.lower()


def test_client_mock_history_not_grown():
    import asyncio
    from ai.claude_client import ClaudeClient
    c = ClaudeClient(mock=True)
    asyncio.run(c.ask_with_package(_package(), "market_commentary"))
    assert len(c.get_history()) == 0


def test_client_clear_history():
    from ai.claude_client import ClaudeClient
    c = ClaudeClient(mock=True)
    c._history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    c.clear_history()
    assert c.get_history() == []


# ═══════════════════════════════════════════════════════════
# ai/claude_client.py — sync public integration functions
# ═══════════════════════════════════════════════════════════

def test_sync_generate_startup_commentary():
    from ai.claude_client import generate_startup_commentary
    r = generate_startup_commentary(_package(), mock=True)
    assert isinstance(r.text, str) and len(r.text) > 10 and not r.is_fallback


def test_sync_generate_market_commentary():
    from ai.claude_client import generate_market_commentary
    assert not generate_market_commentary(_package(), mock=True).is_fallback


def test_sync_generate_setup_explanation():
    from ai.claude_client import generate_setup_explanation
    assert not generate_setup_explanation(_package(), direction="calls", mock=True).is_fallback


def test_sync_generate_trade_review():
    from ai.claude_client import generate_trade_review
    assert not generate_trade_review(_package(trade=_trade_state()), mock=True).is_fallback


def test_sync_generate_end_of_day():
    from ai.claude_client import generate_end_of_day_reflection
    assert not generate_end_of_day_reflection(_package(), mock=True).is_fallback


def test_sync_answer_user_question():
    from ai.claude_client import answer_user_question
    r = answer_user_question(
        _package(), "What does the current regime mean for entries?", mock=True
    )
    assert not r.is_fallback and len(r.text) > 10


def test_sync_returns_parsed_response_type():
    from ai.claude_client import generate_market_commentary
    from ai.response_parser import ParsedResponse
    assert isinstance(generate_market_commentary(_package(), mock=True), ParsedResponse)


def test_sync_fallback_when_no_api():
    import asyncio
    from ai.claude_client import ClaudeClient
    c = ClaudeClient(api_key="", mock=False)
    r = asyncio.run(c.ask_with_package(_package(), "market_commentary"))
    assert r.is_fallback is True


# ═══════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════

def test_integration_full_pipeline_mock():
    """State → package → prompt → mock response → ParsedResponse."""
    from ai.state_packager import build_full_package
    from ai.prompt_builder import build_startup_briefing_prompt, estimate_tokens
    from ai.response_parser import parse_response, validate_response
    from engine.models import TradeDirection

    pkg = build_full_package(
        state=_market_state(), score=_setup_score(),
        direction=TradeDirection.CALLS, context_state=_context_state(),
        session_memory=_session_memory(),
    )
    assert pkg.market is not None

    system, user = build_startup_briefing_prompt(pkg)
    assert 100 < estimate_tokens(system, user) < 3000

    mock_text = "SPY is in a trend-bull regime with solid alignment. Worth watching for continuation setups."
    parsed = parse_response(mock_text, "startup_briefing", pkg.risk_flags)
    assert parsed.guardrail_passed is True
    assert validate_response(parsed) == []


def test_integration_warning_driven_risk_flags():
    """Warnings from cli feed through state_packager into risk_flags."""
    from ai.state_packager import build_full_package, package_warnings
    from cli.warnings import R2Warning, WarningSeverity, WarningCategory

    ws  = [R2Warning(WarningCategory.READINESS, WarningSeverity.CRITICAL, "Not ready")]
    pkg = build_full_package(state=_market_state(readiness="not_ready"), warnings=ws)
    assert "not_ready" in pkg.risk_flags
    assert "CRITICAL" in package_warnings(ws)["groups"]


def test_integration_response_with_risk_flags():
    """Risk flags from package propagate to ParsedResponse."""
    from ai.claude_client import generate_startup_commentary
    from ai.state_packager import build_full_package
    pkg = build_full_package(state=_market_state(readiness="not_ready"))
    r   = generate_startup_commentary(pkg, mock=True)
    assert isinstance(r.risk_flags, list)


def test_integration_guardrail_blocks_hype():
    """Guardrail catches forbidden phrases regardless of request type."""
    from ai.response_parser import parse_response
    for text in [
        "Buy here — this is a massive move incoming!",
        "Take the trade now, you can't lose!",
        "This is incredible setup — go long immediately!",
    ]:
        r = parse_response(text, "setup_explanation")
        assert r.guardrail_passed is False, f"Should have failed: {text[:50]}"


def test_integration_advisory_language_passes():
    """Well-formed advisory text passes guardrail."""
    from ai.response_parser import parse_response
    for text in [
        "The structure suggests a continuation setup if price holds above VWAP.",
        "Worth monitoring for a cleaner entry candle before sizing in.",
        "The thesis remains intact as long as the stop level at 541 holds.",
        "Consider the R:R carefully — the setup is B grade, not A+.",
    ]:
        r = parse_response(text, "market_commentary")
        assert r.guardrail_passed is True, f"Should have passed: {text[:50]}"


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nPhase 8 tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)