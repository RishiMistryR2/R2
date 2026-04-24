"""
tests/test_phase4.py — Phase 4: context layer.

Tests:
  - context/models.py          : NewsItem, NewsEvent, Sentiment, ImpactLevel
  - context/headline_scorer.py : score_headline(), classify_sentiment(), compute_vix_regime(),
                                  HeadlineScorer class
  - context/news_fetcher.py    : fetch_news() stub, injection, mock data
  - context/macro_calendar.py  : get_macro_events() stub, MacroCalendar helpers
  - context/context_engine.py  : build_context() logic, bias derivation, risk level
  - context/principal_alerts.py: get_alerts(), check_news_spike(), check_macro_imminent()
  - context/event_confirmation.py: confirm_setup(), score_adjustment_from_context()
  - Integration                : context wired into MarketState, scorer, StrategyEngine

Run: PYTHONPATH=. python tests/test_phase4.py
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


# ═══════════════════════════════════════════════════════════════════════════
# context/models.py
# ═══════════════════════════════════════════════════════════════════════════

def test_sentiment_enum_values():
    from context.models import Sentiment
    assert Sentiment.BULLISH.value == "bullish"
    assert Sentiment.BEARISH.value == "bearish"
    assert Sentiment.NEUTRAL.value == "neutral"


def test_impact_level_mirrors_headline_impact():
    from context.models import ImpactLevel
    from context.context_state import HeadlineImpact
    assert ImpactLevel.HIGH   == HeadlineImpact.HIGH
    assert ImpactLevel.MEDIUM == HeadlineImpact.MEDIUM
    assert ImpactLevel.LOW    == HeadlineImpact.LOW


def test_news_item_defaults():
    from context.models import NewsItem, SourceTier
    item = NewsItem(
        id="1", headline="SPY rallies", summary="",
        source="test", source_tier=SourceTier.TIER_2_CURATED,
        published_at=_now(),
    )
    assert item.importance_score == 0.0
    assert item.sentiment == 0.0
    assert item.is_market_moving is False


def test_news_item_to_dict():
    from context.models import NewsItem, SourceTier
    item = NewsItem(
        id="x", headline="test", summary="s",
        source="src", source_tier=SourceTier.TIER_1_AUTHORITATIVE,
        published_at=_now(),
    )
    d = item.to_dict()
    assert "headline" in d and "sentiment" in d and "market_moving" in d


def test_news_event_defaults():
    from context.models import NewsEvent
    e = NewsEvent(headline="Fed raises rates", timestamp=_now(), source="reuters")
    assert e.symbols == []
    assert e.sentiment_score is None


def test_news_event_with_sentiment():
    from context.models import NewsEvent
    e = NewsEvent(headline="h", timestamp=_now(), source="s",
                  symbols=["SPY"], sentiment_score=-0.5)
    d = e.to_dict()
    assert d["sentiment_score"] == -0.5
    assert "SPY" in d["symbols"]


# ═══════════════════════════════════════════════════════════════════════════
# context/headline_scorer.py
# ═══════════════════════════════════════════════════════════════════════════

def test_score_headline_bullish():
    from context.headline_scorer import score_headline
    s = score_headline("Fed announces emergency rate cut — markets rally")
    assert s > 0.1, f"expected positive sentiment, got {s}"


def test_score_headline_bearish():
    from context.headline_scorer import score_headline
    s = score_headline("Market crash fears grow as bank failure triggers sell-off")
    assert s < -0.1, f"expected negative sentiment, got {s}"


def test_score_headline_neutral():
    from context.headline_scorer import score_headline
    s = score_headline("Company announces quarterly earnings report")
    assert -0.2 <= s <= 0.2, f"expected near-neutral, got {s}"


def test_score_headline_empty():
    from context.headline_scorer import score_headline
    assert score_headline("") == 0.0
    assert score_headline("   ") == 0.0


def test_score_headline_range():
    from context.headline_scorer import score_headline
    for text in ["CPI beats expectations", "recession fears mount",
                 "earnings beat", "stock market crash"]:
        s = score_headline(text)
        assert -1.0 <= s <= 1.0, f"score out of range for '{text}': {s}"


def test_classify_sentiment_bullish():
    from context.headline_scorer import classify_sentiment
    from context.models import Sentiment
    assert classify_sentiment(0.5) == Sentiment.BULLISH
    assert classify_sentiment(0.16) == Sentiment.BULLISH


def test_classify_sentiment_bearish():
    from context.headline_scorer import classify_sentiment
    from context.models import Sentiment
    assert classify_sentiment(-0.5) == Sentiment.BEARISH
    assert classify_sentiment(-0.16) == Sentiment.BEARISH


def test_classify_sentiment_neutral():
    from context.headline_scorer import classify_sentiment
    from context.models import Sentiment
    assert classify_sentiment(0.0) == Sentiment.NEUTRAL
    assert classify_sentiment(0.10) == Sentiment.NEUTRAL
    assert classify_sentiment(-0.10) == Sentiment.NEUTRAL


def test_classify_sentiment_boundary():
    from context.headline_scorer import classify_sentiment
    from context.models import Sentiment
    # exactly at threshold: 0.15 is not > 0.15
    assert classify_sentiment(0.15) == Sentiment.NEUTRAL
    assert classify_sentiment(0.151) == Sentiment.BULLISH


def test_compute_vix_regime_low():
    from context.headline_scorer import compute_vix_regime
    assert compute_vix_regime(10.0) == "low"
    assert compute_vix_regime(12.9) == "low"


def test_compute_vix_regime_normal():
    from context.headline_scorer import compute_vix_regime
    assert compute_vix_regime(13.0) == "normal"
    assert compute_vix_regime(19.9) == "normal"


def test_compute_vix_regime_elevated():
    from context.headline_scorer import compute_vix_regime
    assert compute_vix_regime(20.0) == "elevated"
    assert compute_vix_regime(29.9) == "elevated"


def test_compute_vix_regime_extreme():
    from context.headline_scorer import compute_vix_regime
    assert compute_vix_regime(30.0) == "extreme"
    assert compute_vix_regime(50.0) == "extreme"


def test_headline_scorer_class_scores_items():
    from context.headline_scorer import HeadlineScorer
    from context.models import NewsItem, SourceTier
    scorer = HeadlineScorer()
    items = [
        NewsItem("1", "Rate cut announced", "", "fed.gov",
                 SourceTier.TIER_1_AUTHORITATIVE, _now()),
        NewsItem("2", "Stocks rally to record high", "", "reuters",
                 SourceTier.TIER_1_AUTHORITATIVE, _now()),
    ]
    scored = scorer.score_items(items)
    assert len(scored) == 2
    for item in scored:
        assert item.importance_score >= 0.0
        assert item.sentiment_label in ("bullish", "bearish", "neutral")


def test_headline_scorer_dedup():
    from context.headline_scorer import HeadlineScorer
    from context.models import NewsItem, SourceTier
    scorer = HeadlineScorer()
    item1 = NewsItem("1", "Fed cuts rates", "", "src", SourceTier.TIER_2_CURATED, _now())
    item2 = NewsItem("2", "Fed cuts rates", "", "src", SourceTier.TIER_2_CURATED, _now())
    result = scorer.score_items([item1, item2])
    assert len(result) == 1   # deduped


def test_classify_source_tier_tier1():
    from context.headline_scorer import HeadlineScorer
    from context.context_state import SourceTier
    scorer = HeadlineScorer()
    assert scorer.classify_source_tier("Reuters") == SourceTier.TIER_1_AUTHORITATIVE
    assert scorer.classify_source_tier("Bloomberg") == SourceTier.TIER_1_AUTHORITATIVE


# ═══════════════════════════════════════════════════════════════════════════
# context/news_fetcher.py
# ═══════════════════════════════════════════════════════════════════════════

def test_fetch_news_returns_list():
    from context.news_fetcher import fetch_news
    result = fetch_news("SPY")
    assert isinstance(result, list)


def test_fetch_news_default_mock_not_empty():
    from context.news_fetcher import fetch_news, inject_news
    inject_news(None)   # reset to mock
    result = fetch_news("SPY")
    assert len(result) >= 1


def test_fetch_news_mock_items_have_sentiment():
    from context.news_fetcher import fetch_news, inject_news
    inject_news(None)
    for event in fetch_news("SPY"):
        assert event.sentiment_score is not None
        assert -1.0 <= event.sentiment_score <= 1.0


def test_fetch_news_per_call_injection():
    from context.news_fetcher import fetch_news
    from context.models import NewsEvent
    custom = [
        NewsEvent("Bear market confirmed", _now(), "test", ["SPY"], -0.8),
    ]
    result = fetch_news("SPY", injected=custom)
    assert len(result) == 1
    assert result[0].sentiment_score == -0.8


def test_fetch_news_module_injection():
    from context.news_fetcher import fetch_news, inject_news
    from context.models import NewsEvent
    inject_news([
        NewsEvent("SPY surges on rate cut", _now(), "test", ["SPY"], 0.9),
    ])
    result = fetch_news("SPY")
    assert len(result) == 1
    assert result[0].sentiment_score == 0.9
    inject_news(None)   # reset


def test_fetch_news_empty_injection():
    from context.news_fetcher import fetch_news, inject_news
    inject_news([])
    result = fetch_news("SPY")
    assert result == []
    inject_news(None)


# ═══════════════════════════════════════════════════════════════════════════
# context/macro_calendar.py
# ═══════════════════════════════════════════════════════════════════════════

def test_get_macro_events_default_empty():
    from context.macro_calendar import get_macro_events, inject_macro_events
    inject_macro_events([])
    assert get_macro_events() == []


def test_get_macro_events_injection():
    from context.macro_calendar import get_macro_events, inject_macro_events
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    event = MacroEvent(
        event_type=MacroEventType.CPI,
        name="CPI Release",
        scheduled_at=_now() + timedelta(hours=1),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    inject_macro_events([event])
    result = get_macro_events()
    assert len(result) == 1
    assert result[0].name == "CPI Release"
    inject_macro_events([])


def test_get_macro_events_per_call_override():
    from context.macro_calendar import get_macro_events
    from context.context_state import MacroEvent, MacroEventType
    event = MacroEvent(
        event_type=MacroEventType.NFP,
        name="NFP",
        scheduled_at=_now() + timedelta(hours=2),
    )
    result = get_macro_events(injected=[event])
    assert len(result) == 1


def test_macro_calendar_assess_risk_high_event():
    from context.macro_calendar import MacroCalendar
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    cal = MacroCalendar()
    event = MacroEvent(
        event_type=MacroEventType.FED_DECISION,
        name="FOMC",
        scheduled_at=_now() + timedelta(minutes=45),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    elevated, note = cal.assess_risk([event], None)
    assert elevated is True
    assert "FOMC" in note


def test_macro_calendar_assess_risk_clear():
    from context.macro_calendar import MacroCalendar
    cal = MacroCalendar()
    elevated, note = cal.assess_risk([], vix=15.0)
    assert elevated is False
    assert note == ""


def test_macro_calendar_classify_regime_risk_on():
    from context.macro_calendar import MacroCalendar
    from context.context_state import MarketRegime
    cal = MacroCalendar()
    regime = cal.classify_regime(vix=11.0, events=[])
    assert regime == MarketRegime.RISK_ON


def test_macro_calendar_classify_regime_risk_off():
    from context.macro_calendar import MacroCalendar
    from context.context_state import MarketRegime
    cal = MacroCalendar()
    regime = cal.classify_regime(vix=28.0, events=[])
    assert regime == MarketRegime.RISK_OFF


# ═══════════════════════════════════════════════════════════════════════════
# context/context_engine.py
# ═══════════════════════════════════════════════════════════════════════════

def test_build_context_returns_context_state():
    from context.context_engine import build_context
    from context.context_state import ContextState
    ctx = build_context("SPY")
    assert isinstance(ctx, ContextState)
    assert ctx.context_valid is True


def test_build_context_has_required_fields():
    from context.context_engine import build_context
    ctx = build_context("SPY")
    assert hasattr(ctx, "news_sentiment")
    assert hasattr(ctx, "context_bias")
    assert hasattr(ctx, "volatility_risk_level")
    assert hasattr(ctx, "upcoming_events")


def test_build_context_bias_bullish_on_positive_news():
    from context.context_engine import build_context
    from context.models import NewsEvent
    events = [
        NewsEvent("SPY surges on rate cut", _now(), "t", ["SPY"], 0.8),
        NewsEvent("Markets rally to record high", _now(), "t", ["SPY"], 0.7),
    ]
    ctx = build_context("SPY", injected_news=events)
    assert ctx.news_sentiment > 0.0
    assert ctx.context_bias in ("bullish", "neutral")


def test_build_context_bias_bearish_on_negative_news():
    from context.context_engine import build_context
    from context.models import NewsEvent
    events = [
        NewsEvent("Market crash fears grow", _now(), "t", ["SPY"], -0.8),
        NewsEvent("Bank failure triggers sell-off", _now(), "t", ["SPY"], -0.9),
    ]
    ctx = build_context("SPY", injected_news=events)
    assert ctx.news_sentiment < 0.0
    assert ctx.context_bias in ("bearish", "neutral")


def test_build_context_elevated_risk_on_high_impact_event():
    from context.context_engine import build_context
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    event = MacroEvent(
        event_type=MacroEventType.CPI,
        name="CPI Release",
        scheduled_at=_now() + timedelta(minutes=30),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    ctx = build_context("SPY", injected_events=[event])
    assert ctx.volatility_risk_level == "elevated"
    assert ctx.elevated_risk is True
    assert len(ctx.upcoming_events) == 1


def test_build_context_neutral_bias_when_elevated_risk():
    """High-risk environment forces neutral bias regardless of sentiment."""
    from context.context_engine import build_context
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    from context.models import NewsEvent
    event = MacroEvent(
        event_type=MacroEventType.FED_DECISION,
        name="FOMC",
        scheduled_at=_now() + timedelta(minutes=60),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    news = [NewsEvent("Rate cut coming", _now(), "t", ["SPY"], 0.9)]
    ctx = build_context("SPY", injected_news=news, injected_events=[event])
    assert ctx.context_bias == "neutral"   # elevated risk overrides bullish news


def test_build_context_neutral_news():
    from context.context_engine import build_context
    from context.models import NewsEvent
    events = [NewsEvent("Market overview", _now(), "t", ["SPY"], 0.0)]
    ctx = build_context("SPY", injected_news=events)
    assert ctx.context_bias == "neutral"


def test_build_context_no_news_neutral():
    from context.context_engine import build_context
    ctx = build_context("SPY", injected_news=[])
    assert ctx.news_sentiment == 0.0
    assert ctx.context_bias == "neutral"


def test_build_context_never_raises():
    from context.context_engine import build_context
    ctx = build_context("INVALID_SYM", now=_now(), injected_news=[], injected_events=[])
    assert ctx is not None


def test_context_state_empty():
    from context.context_state import ContextState
    ctx = ContextState.empty()
    assert ctx.context_valid is False
    assert ctx.context_bias == "neutral"


def test_context_state_to_dict_has_new_fields():
    from context.context_engine import build_context
    ctx = build_context("SPY")
    d = ctx.to_dict()
    assert "news_sentiment" in d
    assert "context_bias" in d
    assert "volatility_risk_level" in d


# ═══════════════════════════════════════════════════════════════════════════
# context/principal_alerts.py
# ═══════════════════════════════════════════════════════════════════════════

def _make_ctx(**kwargs):
    from context.context_state import ContextState
    defaults = dict(
        timestamp=_now(),
        news_sentiment=0.0,
        context_bias="neutral",
        volatility_risk_level="normal",
        elevated_risk=False,
        risk_note="",
        upcoming_events=[],
    )
    defaults.update(kwargs)
    return ContextState(**defaults)


def test_alerts_empty_on_neutral_ctx():
    from context.principal_alerts import get_alerts
    ctx = _make_ctx()
    alerts = get_alerts(ctx)
    assert alerts == []


def test_alerts_news_spike_warning():
    from context.principal_alerts import get_alerts, AlertType, AlertSeverity
    ctx = _make_ctx(news_sentiment=0.55)
    alerts = get_alerts(ctx)
    spike = [a for a in alerts if a.alert_type == AlertType.NEWS_SPIKE]
    assert len(spike) == 1
    assert spike[0].severity == AlertSeverity.WARNING


def test_alerts_news_spike_critical():
    from context.principal_alerts import get_alerts, AlertType, AlertSeverity
    ctx = _make_ctx(news_sentiment=-0.80)
    alerts = get_alerts(ctx)
    spike = [a for a in alerts if a.alert_type == AlertType.NEWS_SPIKE]
    assert len(spike) == 1
    assert spike[0].severity == AlertSeverity.CRITICAL


def test_alerts_no_spike_below_threshold():
    from context.principal_alerts import get_alerts, AlertType
    ctx = _make_ctx(news_sentiment=0.40)
    alerts = get_alerts(ctx)
    spike = [a for a in alerts if a.alert_type == AlertType.NEWS_SPIKE]
    assert len(spike) == 0


def test_alerts_macro_imminent():
    from context.principal_alerts import get_alerts, AlertType, AlertSeverity
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    event = MacroEvent(
        event_type=MacroEventType.CPI,
        name="CPI",
        scheduled_at=_now() + timedelta(minutes=45),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    ctx = _make_ctx(upcoming_events=[event])
    alerts = get_alerts(ctx)
    macro = [a for a in alerts if a.alert_type == AlertType.MACRO_IMMINENT]
    assert len(macro) == 1
    assert macro[0].severity == AlertSeverity.CRITICAL
    assert "CPI" in macro[0].message


def test_alerts_macro_outside_window_ignored():
    from context.principal_alerts import get_alerts, AlertType
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    event = MacroEvent(
        event_type=MacroEventType.CPI,
        name="CPI",
        scheduled_at=_now() + timedelta(hours=5),   # 5h away
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    ctx = _make_ctx(upcoming_events=[event])
    alerts = get_alerts(ctx)
    macro = [a for a in alerts if a.alert_type == AlertType.MACRO_IMMINENT]
    assert len(macro) == 0


def test_alerts_high_volatility_warning():
    from context.principal_alerts import get_alerts, AlertType, AlertSeverity
    ctx = _make_ctx(volatility_risk_level="elevated", elevated_risk=True)
    alerts = get_alerts(ctx)
    vol = [a for a in alerts if a.alert_type == AlertType.HIGH_VOLATILITY]
    assert len(vol) == 1
    assert vol[0].severity == AlertSeverity.WARNING


def test_alerts_high_volatility_critical():
    from context.principal_alerts import get_alerts, AlertType, AlertSeverity
    ctx = _make_ctx(volatility_risk_level="extreme", elevated_risk=True)
    alerts = get_alerts(ctx)
    vol = [a for a in alerts if a.alert_type == AlertType.HIGH_VOLATILITY]
    assert vol[0].severity == AlertSeverity.CRITICAL


def test_alerts_sorted_critical_first():
    from context.principal_alerts import get_alerts, AlertSeverity
    from context.context_state import MacroEvent, MacroEventType, HeadlineImpact
    event = MacroEvent(
        event_type=MacroEventType.FED_DECISION,
        name="FOMC",
        scheduled_at=_now() + timedelta(minutes=30),
        status="upcoming",
        market_impact=HeadlineImpact.HIGH,
    )
    ctx = _make_ctx(
        news_sentiment=0.55,          # WARNING
        upcoming_events=[event],      # CRITICAL
        volatility_risk_level="elevated",  # WARNING
        elevated_risk=True,
    )
    alerts = get_alerts(ctx)
    assert len(alerts) >= 2
    assert alerts[0].severity == AlertSeverity.CRITICAL


def test_alert_to_dict():
    from context.principal_alerts import get_alerts, AlertType
    ctx = _make_ctx(news_sentiment=0.60)
    alerts = get_alerts(ctx)
    d = alerts[0].to_dict()
    for k in ("type", "severity", "message", "details"):
        assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# context/event_confirmation.py
# ═══════════════════════════════════════════════════════════════════════════

def test_confirm_setup_no_context():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    result = confirm_setup(TradeDirection.CALLS, None)
    assert result.confidence_multiplier == 1.0


def test_confirm_setup_aligned_bullish():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish")
    result = confirm_setup(TradeDirection.CALLS, ctx)
    assert result.aligned is True
    assert result.confidence_multiplier > 1.0


def test_confirm_setup_aligned_bearish():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bearish")
    result = confirm_setup(TradeDirection.PUTS, ctx)
    assert result.aligned is True
    assert result.confidence_multiplier > 1.0


def test_confirm_setup_mismatch_long_bearish_ctx():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bearish")
    result = confirm_setup(TradeDirection.CALLS, ctx)
    assert result.aligned is False
    assert result.confidence_multiplier < 1.0


def test_confirm_setup_mismatch_short_bullish_ctx():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish")
    result = confirm_setup(TradeDirection.PUTS, ctx)
    assert result.aligned is False
    assert result.confidence_multiplier < 1.0


def test_confirm_setup_high_risk_reduces_confidence():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish", volatility_risk_level="elevated")
    result_no_risk = confirm_setup(TradeDirection.CALLS, _make_ctx(context_bias="bullish"))
    result_risk    = confirm_setup(TradeDirection.CALLS, ctx)
    assert result_risk.confidence_multiplier < result_no_risk.confidence_multiplier


def test_confirm_setup_notes_populated():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish")
    result = confirm_setup(TradeDirection.CALLS, ctx)
    assert len(result.notes) >= 1


def test_score_adjustment_positive_on_alignment():
    from context.event_confirmation import score_adjustment_from_context
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish")
    adj = score_adjustment_from_context(TradeDirection.CALLS, ctx)
    assert adj > 0


def test_score_adjustment_negative_on_mismatch():
    from context.event_confirmation import score_adjustment_from_context
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bearish")
    adj = score_adjustment_from_context(TradeDirection.CALLS, ctx)
    assert adj < 0


def test_score_adjustment_negative_on_high_risk():
    from context.event_confirmation import score_adjustment_from_context
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish", volatility_risk_level="elevated")
    adj = score_adjustment_from_context(TradeDirection.CALLS, ctx)
    assert adj < 0  # high risk overrides alignment


def test_score_adjustment_zero_on_neutral():
    from context.event_confirmation import score_adjustment_from_context
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="neutral")
    adj = score_adjustment_from_context(TradeDirection.CALLS, ctx)
    assert adj == 0.0


def test_confirm_setup_to_dict():
    from context.event_confirmation import confirm_setup
    from engine.models import TradeDirection
    ctx = _make_ctx(context_bias="bullish")
    result = confirm_setup(TradeDirection.CALLS, ctx)
    d = result.to_dict()
    assert "confidence_multiplier" in d
    assert "notes" in d
    assert "aligned" in d


# ═══════════════════════════════════════════════════════════════════════════
# Integration — MarketState context fields
# ═══════════════════════════════════════════════════════════════════════════

def test_market_state_has_context_fields():
    from engine.models import MarketState
    state = MarketState(symbol="SPY", timestamp=_now())
    assert hasattr(state, "context")
    assert hasattr(state, "context_bias")
    assert hasattr(state, "context_risk_level")
    assert state.context is None
    assert state.context_bias == "neutral"
    assert state.context_risk_level == "normal"


def test_setup_score_has_context_adjustment():
    from engine.models import SetupScore
    score = SetupScore()
    assert hasattr(score, "context_adjustment")
    assert score.context_adjustment == 0.0


def test_setup_score_to_dict_has_context_adjustment():
    from engine.models import SetupScore
    score = SetupScore(context_adjustment=5.0)
    d = score.to_dict()
    assert "context_adjustment" in d


# ═══════════════════════════════════════════════════════════════════════════
# Integration — scorer context adjustment
# ═══════════════════════════════════════════════════════════════════════════

def _bull_state_with_context(context_bias="bullish", risk_level="normal"):
    from engine.models import (
        MarketState, IndicatorSnapshot, TrendState, Direction, TrendStrength,
        RegimeState, MarketRegime, SessionState, SessionPhase, Readiness,
        SessionIntent
    )
    from context.context_state import ContextState
    s = MarketState(symbol="SPY", timestamp=_now())
    for tf in ["1d","1h","15m","5m","1m"]:
        t = TrendState(tf, Direction.BULLISH, TrendStrength.STRONG)
        t.higher_highs = True; t.last_swing_low = 542.0
        s.trends[tf] = t
        ind = IndicatorSnapshot(tf, _now(), 545.0)
        ind.rsi=55.0; ind.macd_hist=0.01; ind.ema_stack="bull_aligned"
        ind.vwap=542.0; ind.atr=1.2; ind.ema_9=543.0; ind.ema_20=541.0
        ind.price_vs_vwap="above"; ind.macd_bias="bullish"
        s.indicators[tf] = ind
    s.session = SessionState(
        phase=SessionPhase.MIDDAY, market_open=True,
        entries_allowed=True, score_modifier=1.0,
        entry_filter="normal", phase_note="test",
        readiness=Readiness.FULLY_READY,
    )
    s.regime  = RegimeState(MarketRegime.TREND_BULL, 0.85, 1.0, True, "bull")
    s.intent  = SessionIntent(min_score=60)
    s.context = _make_ctx(context_bias=context_bias, volatility_risk_level=risk_level)
    s.context_bias = context_bias
    s.context_risk_level = risk_level
    return s


def test_scorer_context_boost_on_alignment():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    state_aligned = _bull_state_with_context("bullish", "normal")
    state_neutral  = _bull_state_with_context("neutral", "normal")
    s_aligned = score_setup(state_aligned, TradeDirection.CALLS)
    s_neutral  = score_setup(state_neutral, TradeDirection.CALLS)
    assert s_aligned.context_adjustment > 0
    assert s_aligned.adjusted_total > s_neutral.adjusted_total


def test_scorer_context_penalty_on_mismatch():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    state_mismatch = _bull_state_with_context("bearish", "normal")
    state_neutral  = _bull_state_with_context("neutral", "normal")
    s_mismatch = score_setup(state_mismatch, TradeDirection.CALLS)
    s_neutral   = score_setup(state_neutral, TradeDirection.CALLS)
    assert s_mismatch.context_adjustment < 0
    assert s_mismatch.adjusted_total < s_neutral.adjusted_total


def test_scorer_context_penalty_on_high_risk():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    state_elevated = _bull_state_with_context("bullish", "elevated")
    state_normal   = _bull_state_with_context("neutral", "normal")
    s_elevated = score_setup(state_elevated, TradeDirection.CALLS)
    assert s_elevated.context_adjustment < 0


def test_scorer_no_context_no_adjustment():
    from strategy.scorer import score_setup
    from engine.models import TradeDirection
    state = _bull_state_with_context()
    state.context = None
    score = score_setup(state, TradeDirection.CALLS)
    assert score.context_adjustment == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration — StrategyEngine Step 4b
# ═══════════════════════════════════════════════════════════════════════════

def test_strategy_engine_attaches_context():
    from strategy.strategy_engine import StrategyEngine
    eng   = StrategyEngine()
    state = _bull_state_with_context()
    state.context = None   # let engine set it
    result = eng.process(state)
    assert result.context is not None


def test_strategy_engine_sets_context_bias():
    from strategy.strategy_engine import StrategyEngine
    eng   = StrategyEngine()
    state = _bull_state_with_context()
    state.context = None
    result = eng.process(state)
    assert result.context_bias in ("bullish", "bearish", "neutral")


def test_strategy_engine_sets_context_risk_level():
    from strategy.strategy_engine import StrategyEngine
    eng   = StrategyEngine()
    state = _bull_state_with_context()
    state.context = None
    result = eng.process(state)
    assert result.context_risk_level in ("low", "normal", "elevated", "extreme")


def test_strategy_engine_context_does_not_break_existing_flow():
    """Ensure setup_direction is still set correctly when context is attached."""
    from strategy.strategy_engine import StrategyEngine
    from engine.models import TradeDirection
    eng   = StrategyEngine()
    state = _bull_state_with_context("neutral")  # neutral context = no interference
    result = eng.process(state)
    assert result.setup_direction == TradeDirection.CALLS


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"\nPhase 4 tests ({len(tests)} total)\n{'─'*56}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'='*56}")
    print(f"{passed} passed, {failed} failed  ({passed+failed} total)")
    sys.exit(0 if failed == 0 else 1)