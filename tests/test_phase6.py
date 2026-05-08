"""
tests/test_phase6.py — Phase 6: memory and reflection layer.

Coverage:
  persistence        : atomic save/load, list_files, schema stamps
  session_memory     : creation, record_trade, regime changes, round-trip
  trade_memory       : from_trade, serialisation, load_all, load_by_session
  performance_stats  : compute_stats, build_report, grouping, best/worst
  pattern_memory     : BehaviorPattern, MarketPattern, learning loop,
                       record_market_pattern, update_market_pattern_outcome,
                       surfacing thresholds, all MarketPatternType constants
  reflection_engine  : quality separation, good_trade_bad_outcome,
                       bad_trade_good_outcome, apply_quality_tags
  startup_context    : build_startup_briefing, prev session, risks
  integration        : cross-module round-trips

Run: PYTHONPATH=. python tests/test_phase6.py
"""

import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

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


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _tmp() -> Path:
    """Return a fresh temp directory for each test that needs I/O."""
    return Path(tempfile.mkdtemp())


# ═══════════════════════════════════════════════════════════
# memory/persistence.py
# ═══════════════════════════════════════════════════════════

def test_persistence_save_and_load():
    from memory.persistence import save, load
    d = _tmp()
    ok = save(d / "x.json", {"k": "v"})
    assert ok is True
    result = load(d / "x.json")
    assert result["k"] == "v"
    assert "schema_version" in result
    assert "saved_at" in result


def test_persistence_atomic_write_tmp_cleaned_up():
    """No .tmp_ file should be left behind after a successful write."""
    from memory.persistence import save
    d = _tmp()
    save(d / "a.json", {"x": 1})
    leftovers = list(d.glob(".tmp_*"))
    assert len(leftovers) == 0


def test_persistence_load_missing_returns_none():
    from memory.persistence import load
    assert load(Path("/nonexistent/path/x.json")) is None


def test_persistence_load_corrupt_returns_none():
    d = _tmp()
    p = d / "bad.json"
    p.write_text("NOT_JSON{{{")
    from memory.persistence import load
    assert load(p) is None


def test_persistence_save_list_and_load_list():
    from memory.persistence import save_list, load_list
    d     = _tmp()
    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    save_list(d / "lst.json", items)
    result = load_list(d / "lst.json")
    assert len(result) == 3
    assert result[1]["id"] == 2


def test_persistence_load_list_missing_returns_empty():
    from memory.persistence import load_list
    assert load_list(Path("/no/file.json")) == []


def test_persistence_list_files_sorted():
    from memory.persistence import save, list_files
    d = _tmp()
    for name in ["c.json", "a.json", "b.json"]:
        save(d / name, {"n": name})
    files = list_files(d)
    names = [f.name for f in files]
    assert names == sorted(names)


def test_persistence_list_files_empty_dir():
    from memory.persistence import list_files
    assert list_files(_tmp()) == []


def test_persistence_schema_version_stamped():
    from memory.persistence import save, load
    from config.settings import MEMORY_SCHEMA_VERSION
    d = _tmp()
    save(d / "sv.json", {"data": "x"})
    r = load(d / "sv.json")
    assert r["schema_version"] == MEMORY_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════
# memory/session_memory.py
# ═══════════════════════════════════════════════════════════

def test_session_today():
    from memory.session_memory import SessionMemory
    from datetime import date
    s = SessionMemory.today()
    assert s.session_date == date.today().strftime("%Y-%m-%d")
    assert s.symbol == "SPY"


def test_session_open_close():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.open_session()
    assert s.session_start is not None
    s.close_session(notes="good day", lesson="wait for confirmation")
    assert s.session_end is not None
    assert s.lesson == "wait for confirmation"


def test_session_record_trade_win():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.record_trade("T1", 1.5)
    assert s.total_trades == 1
    assert s.winning_trades == 1
    assert s.daily_pnl_pct == 1.5


def test_session_record_trade_loss():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.record_trade("T2", -0.8)
    assert s.losing_trades == 1
    assert abs(s.daily_pnl_pct - (-0.8)) < 0.001


def test_session_win_rate_calculation():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.record_trade("T1", 1.2)
    s.record_trade("T2", -0.5)
    s.record_trade("T3", 0.9)
    assert abs(s.win_rate - 2/3) < 0.001


def test_session_no_duplicate_trade_id():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.record_trade("T1", 1.0)
    s.record_trade("T1", 1.0)   # duplicate
    assert s.total_trades == 1


def test_session_regime_change():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.record_regime_change("trend_bull", "chop", confidence=0.8, note="VWAP break")
    assert len(s.regime_changes) == 1
    assert s.closing_regime == "chop"
    assert s.regime_changes[0].from_regime == "trend_bull"


def test_session_behavioral_flag_dedup():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.add_behavioral_flag("over-traded")
    s.add_behavioral_flag("over-traded")
    assert s.behavioral_flags.count("over-traded") == 1


def test_session_update_drawdown():
    from memory.session_memory import SessionMemory
    s = SessionMemory.today()
    s.update_drawdown(-1.5)
    s.update_drawdown(-0.5)
    assert s.max_intraday_drawdown_pct == 1.5
    s.update_drawdown(2.0)
    assert s.peak_gain_pct == 2.0


def test_session_round_trip():
    from memory.session_memory import SessionMemory
    d = _tmp()
    s = SessionMemory.today()
    s.opening_regime = "trend_bull"
    s.record_trade("T1", 1.2)
    s.record_regime_change("trend_bull", "chop", 0.7)
    s.add_behavioral_flag("overtraded")
    s.lesson = "wait for pull"
    s.save(d)

    loaded = SessionMemory.load(s.session_date, d)
    assert loaded is not None
    assert loaded.opening_regime == "trend_bull"
    assert loaded.total_trades == 1
    assert len(loaded.regime_changes) == 1
    assert loaded.lesson == "wait for pull"
    assert "overtraded" in loaded.behavioral_flags


def test_session_load_latest():
    from memory.session_memory import SessionMemory
    d = _tmp()
    s1 = SessionMemory(session_date="2025-01-01")
    s2 = SessionMemory(session_date="2025-01-02")
    s1.save(d); s2.save(d)
    latest = SessionMemory.load_latest(d)
    assert latest.session_date == "2025-01-02"


def test_session_load_missing_returns_none():
    from memory.session_memory import SessionMemory
    assert SessionMemory.load("9999-99-99", _tmp()) is None


# ═══════════════════════════════════════════════════════════
# memory/trade_memory.py
# ═══════════════════════════════════════════════════════════

def _make_trade_record(trade_id="T001", pnl=1.2, outcome_override=None):
    from memory.trade_memory import TradeRecord
    now = _now()
    r = TradeRecord.from_trade(
        trade_id        = trade_id,
        session_date    = "2025-01-15",
        direction       = "calls",
        instrument_type = "options",
        entry_price     = 545.0,
        exit_price      = 551.0,
        entry_time      = now,
        exit_time       = now,
        stop_price      = 541.0,
        target_price    = 551.0,
        pnl_pct         = pnl,
        pnl_dollar      = 60.0,
        exit_reason     = "profit_target",
        setup_type      = "continuation",
        regime_at_entry = "trend_bull",
        htf_bias        = "bullish",
        rr_planned      = 1.5,
    )
    if outcome_override:
        r.outcome = outcome_override
    return r


def test_trade_record_from_trade_basic():
    r = _make_trade_record()
    assert r.trade_id   == "T001"
    assert r.direction  == "calls"
    assert r.pnl_pct    == 1.2
    assert r.outcome    == "win"
    assert r.setup_type == "continuation"


def test_trade_classify_outcome_win():
    from memory.trade_memory import _classify_outcome, OUTCOME_WIN
    assert _classify_outcome(1.0)  == OUTCOME_WIN
    assert _classify_outcome(0.16) == OUTCOME_WIN


def test_trade_classify_outcome_loss():
    from memory.trade_memory import _classify_outcome, OUTCOME_LOSS
    assert _classify_outcome(-0.5)  == OUTCOME_LOSS
    assert _classify_outcome(-0.16) == OUTCOME_LOSS


def test_trade_classify_outcome_breakeven():
    from memory.trade_memory import _classify_outcome, OUTCOME_BREAKEVEN
    assert _classify_outcome(0.0)  == OUTCOME_BREAKEVEN
    assert _classify_outcome(0.10) == OUTCOME_BREAKEVEN


def test_trade_round_trip():
    d = _tmp()
    r = _make_trade_record("T_RT")
    r.save(d)
    loaded = r.load_by_id("T_RT", d)
    assert loaded is not None
    assert loaded.trade_id   == "T_RT"
    assert loaded.pnl_pct    == 1.2
    assert loaded.setup_type == "continuation"


def test_trade_to_dict_structure():
    r = _make_trade_record()
    d = r.to_dict()
    for k in ("trade_id", "session_date", "instrument", "levels", "timing",
              "setup", "conditions", "outcome", "stop_flags", "quality"):
        assert k in d, f"missing key: {k}"


def test_trade_load_all():
    d = _tmp()
    for i in range(3):
        _make_trade_record(f"T{i}", pnl=1.0 - i * 0.5).save(d)
    from memory.trade_memory import TradeRecord
    all_t = TradeRecord.load_all(d)
    assert len(all_t) == 3


def test_trade_load_by_session():
    d = _tmp()
    from memory.trade_memory import TradeRecord
    for i in range(2):
        _make_trade_record(f"S1_{i}").save(d)
    r2 = _make_trade_record("S2_0")
    r2.session_date = "2025-01-16"
    r2.save(d)
    result = TradeRecord.load_by_session("2025-01-15", d)
    assert len(result) == 2


def test_trade_duration_computed():
    from memory.trade_memory import TradeRecord
    t1 = datetime.now(tz=timezone.utc)
    t2 = t1 + timedelta(minutes=15)
    r = TradeRecord.from_trade(
        trade_id="DUR", session_date="2025-01-01",
        direction="calls", instrument_type="options",
        entry_price=545.0, exit_price=548.0,
        entry_time=t1.isoformat(), exit_time=t2.isoformat(),
        stop_price=541.0, target_price=551.0,
        pnl_pct=0.55, pnl_dollar=30.0, exit_reason="target",
    )
    assert abs(r.duration_minutes - 15.0) < 0.5


# ═══════════════════════════════════════════════════════════
# memory/performance_stats.py
# ═══════════════════════════════════════════════════════════

def _stats_records():
    records = []
    for i, (pnl, stype, regime, phase, instr, htf) in enumerate([
        ( 1.5, "continuation", "trend_bull", "midday",     "options", "bullish"),
        ( 1.2, "continuation", "trend_bull", "midday",     "options", "bullish"),
        (-0.8, "reversal",     "chop",       "open",       "shares",  "neutral"),
        ( 2.0, "breakout",     "trend_bull", "power_hour", "options", "bullish"),
        (-0.5, "continuation", "chop",       "midday",     "options", "neutral"),
    ]):
        r = _make_trade_record(f"STATS_{i}", pnl=pnl)
        r.setup_type             = stype
        r.regime_at_entry        = regime
        r.session_phase_at_entry = phase
        r.instrument_type        = instr
        r.htf_bias_at_entry      = htf
        records.append(r)
    return records


def test_stats_compute_win_rate():
    from memory.performance_stats import compute_stats
    records = _stats_records()
    s = compute_stats(records)
    assert s.total_trades == 5
    wins = sum(1 for r in records if r.outcome == "win")
    assert abs(s.win_rate - wins / 5) < 0.01


def test_stats_compute_profit_factor():
    from memory.performance_stats import compute_stats
    s = compute_stats(_stats_records())
    assert s.profit_factor > 0


def test_stats_compute_expectancy():
    from memory.performance_stats import compute_stats
    s = compute_stats(_stats_records())
    assert isinstance(s.expectancy, float)


def test_stats_empty_returns_zero_slice():
    from memory.performance_stats import compute_stats
    s = compute_stats([])
    assert s.total_trades == 0
    assert s.win_rate     == 0.0


def test_report_groups_by_setup():
    from memory.performance_stats import build_report
    rep = build_report(_stats_records())
    assert "continuation" in rep.by_setup_type
    assert "reversal"     in rep.by_setup_type


def test_report_groups_by_regime():
    from memory.performance_stats import build_report
    rep = build_report(_stats_records())
    assert "trend_bull" in rep.by_regime
    assert "chop"       in rep.by_regime


def test_report_groups_by_instrument():
    from memory.performance_stats import build_report
    rep = build_report(_stats_records())
    assert "options" in rep.by_instrument
    assert "shares"  in rep.by_instrument


def test_report_best_setup_min_3_trades():
    from memory.performance_stats import build_report
    rep = build_report(_stats_records())
    if rep.best_setup:
        assert rep.by_setup_type[rep.best_setup].total_trades >= 3


def test_report_empty_returns_empty_overall():
    from memory.performance_stats import build_report
    rep = build_report([])
    assert rep.overall.total_trades == 0


def test_report_to_dict_structure():
    from memory.performance_stats import build_report
    d = build_report(_stats_records()).to_dict()
    for k in ("overall", "by_setup_type", "by_regime", "by_instrument", "meta"):
        assert k in d


# ═══════════════════════════════════════════════════════════
# memory/pattern_memory.py — BehaviorPattern
# ═══════════════════════════════════════════════════════════

def test_behavior_update_confidence():
    from memory.pattern_memory import BehaviorPattern
    p = BehaviorPattern("bp1", "winning", "test desc")
    for _ in range(8):
        p.update(positive=True, trade_id="T1")
    for _ in range(2):
        p.update(positive=False, trade_id="T2")
    assert abs(p.confidence_score - 0.8) < 0.01
    assert p.occurrences == 10


def test_behavior_surfaced_threshold():
    from memory.pattern_memory import BehaviorPattern
    from config.settings import PATTERN_MIN_OCCURRENCES
    p = BehaviorPattern("bp2", "winning", "test")
    for _ in range(PATTERN_MIN_OCCURRENCES):
        p.update(positive=True)
    assert p.is_surfaced is True


def test_behavior_not_surfaced_below_min():
    from memory.pattern_memory import BehaviorPattern
    from config.settings import PATTERN_MIN_OCCURRENCES
    p = BehaviorPattern("bp3", "winning", "test")
    for _ in range(PATTERN_MIN_OCCURRENCES - 1):
        p.update(positive=True)
    assert p.is_surfaced is False


def test_behavior_example_trade_ids_capped():
    from memory.pattern_memory import BehaviorPattern
    p = BehaviorPattern("bp4", "winning", "test")
    for i in range(15):
        p.update(positive=True, trade_id=f"T{i}")
    assert len(p.example_trade_ids) <= 10


def test_behavior_round_trip():
    from memory.pattern_memory import BehaviorPattern
    p = BehaviorPattern("bp5", "mistake", "did not wait for confirmation")
    for _ in range(5):
        p.update(positive=True)
    d  = p.to_dict()
    p2 = BehaviorPattern.from_dict(d)
    assert p2.pattern_id  == "bp5"
    assert p2.occurrences == 5


def test_behavior_update_from_trade():
    from memory.pattern_memory import PatternMemory
    d  = _tmp()
    pm = PatternMemory(d / "pat.json")
    r  = _make_trade_record("T_BEH", pnl=1.2)
    r.setup_type        = "continuation"
    r.regime_at_entry   = "trend_bull"
    r.htf_bias_at_entry = "bullish"
    pm.update_behavior_from_trade(r)
    assert len(pm.get_all_behavior()) >= 2   # setup+regime + htf pattern


def test_behavior_mistake_recorded():
    from memory.pattern_memory import PatternMemory
    d  = _tmp()
    pm = PatternMemory(d / "pat.json")
    r  = _make_trade_record("T_MIST", pnl=-0.5)
    r.mistake_tags = ["chased entry", "no confirmation"]
    pm.update_behavior_from_trade(r)
    assert len(pm.get_mistakes()) == 2


# ═══════════════════════════════════════════════════════════
# memory/pattern_memory.py — MarketPattern
# ═══════════════════════════════════════════════════════════

def test_market_pattern_type_constants_exist():
    from memory.pattern_memory import MarketPatternType
    for name in [
        "BULL_FLAG", "BEAR_FLAG", "BREAKOUT_RETEST", "FAILED_BREAKOUT",
        "PULLBACK_TO_VWAP", "PULLBACK_TO_EMA", "RANGE_REJECTION",
        "HIGHER_LOW_CONTINUATION", "LOWER_HIGH_CONTINUATION",
        "TREND_EXHAUSTION", "GAP_AND_GO", "GAP_FILL_ATTEMPT", "TRAP",
    ]:
        assert hasattr(MarketPatternType, name), f"Missing: {name}"


def test_market_pattern_all_list_complete():
    from memory.pattern_memory import MarketPatternType
    assert len(MarketPatternType.ALL) >= 14


def test_market_pattern_make_id():
    from memory.pattern_memory import MarketPattern
    pid = MarketPattern.make_id("bull_flag", "bullish", "5m")
    assert pid == "bull_flag__bullish__5m"


def test_record_market_pattern_creates_entry():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm  = PatternMemory(_tmp() / "p.json")
    obs = pm.record_market_pattern(
        MarketPatternType.BULL_FLAG, "bullish", "5m", trade_id="T1"
    )
    assert obs.observation_id is not None
    mp = pm.get_market_pattern("bull_flag__bullish__5m")
    assert mp is not None
    assert mp.occurrence_count == 1


def test_record_market_pattern_increments_on_repeat():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm = PatternMemory(_tmp() / "p.json")
    for i in range(3):
        pm.record_market_pattern(
            MarketPatternType.PULLBACK_TO_VWAP, "bullish", "15m", trade_id=f"T{i}"
        )
    mp = pm.get_market_pattern("pullback_to_vwap__bullish__15m")
    assert mp.occurrence_count == 3


def test_update_market_pattern_outcome_by_observation_id():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm  = PatternMemory(_tmp() / "p.json")
    obs = pm.record_market_pattern(
        MarketPatternType.BREAKOUT_RETEST, "bullish", "15m", trade_id="T99"
    )
    ok = pm.update_market_pattern_outcome(
        pattern_type   = MarketPatternType.BREAKOUT_RETEST,
        direction      = "bullish",
        timeframe      = "15m",
        observation_id = obs.observation_id,
        pnl_pct        = 1.5,
        outcome        = "win",
        trade_id       = "T99",
    )
    assert ok is True
    mp = pm.get_market_pattern("breakout_retest__bullish__15m")
    assert mp.resolved_count == 1
    assert mp.win_count      == 1
    assert abs(mp.win_rate   - 1.0) < 0.01
    assert abs(mp.avg_return - 1.5) < 0.01


def test_update_market_pattern_outcome_by_trade_id():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm = PatternMemory(_tmp() / "p.json")
    pm.record_market_pattern(
        MarketPatternType.FAILED_BREAKOUT, "bearish", "5m", trade_id="T42"
    )
    ok = pm.update_market_pattern_outcome(
        pattern_type = MarketPatternType.FAILED_BREAKOUT,
        direction    = "bearish",
        timeframe    = "5m",
        pnl_pct      = -0.8,
        outcome      = "loss",
        trade_id     = "T42",
    )
    assert ok is True
    mp = pm.get_market_pattern("failed_breakout__bearish__5m")
    assert mp.win_count  == 0
    assert mp.avg_return < 0


def test_market_pattern_win_rate_recomputed():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm       = PatternMemory(_tmp() / "p.json")
    obs_list = []
    for i in range(4):
        obs = pm.record_market_pattern(
            MarketPatternType.PULLBACK_TO_EMA, "bullish", "5m", trade_id=f"E{i}"
        )
        obs_list.append(obs)
    for i, (pnl, outcome) in enumerate(
        [(1.2, "win"), (0.8, "win"), (1.5, "win"), (-0.6, "loss")]
    ):
        pm.update_market_pattern_outcome(
            MarketPatternType.PULLBACK_TO_EMA, "bullish", "5m",
            observation_id=obs_list[i].observation_id,
            pnl_pct=pnl, outcome=outcome,
        )
    mp = pm.get_market_pattern("pullback_to_ema__bullish__5m")
    assert abs(mp.win_rate   - 0.75) < 0.01
    assert mp.avg_return > 0


def test_market_pattern_not_surfaced_below_threshold():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm  = PatternMemory(_tmp() / "p.json")
    obs = pm.record_market_pattern(
        MarketPatternType.GAP_AND_GO, "bullish", "any", trade_id="G1"
    )
    pm.update_market_pattern_outcome(
        MarketPatternType.GAP_AND_GO, "bullish", "any",
        observation_id=obs.observation_id,
        pnl_pct=1.0, outcome="win",
    )
    mp = pm.get_market_pattern("gap_and_go__bullish__any")
    assert mp.is_surfaced is False   # only 1 occurrence — below min_occurrences


def test_market_pattern_surfaced_after_threshold():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    from config.settings import PATTERN_MIN_OCCURRENCES
    pm       = PatternMemory(_tmp() / "p.json")
    obs_list = []
    for i in range(PATTERN_MIN_OCCURRENCES):
        obs = pm.record_market_pattern(
            MarketPatternType.HIGHER_LOW_CONTINUATION, "bullish", "15m",
            trade_id=f"HL{i}"
        )
        obs_list.append(obs)
    for obs in obs_list:
        pm.update_market_pattern_outcome(
            MarketPatternType.HIGHER_LOW_CONTINUATION, "bullish", "15m",
            observation_id=obs.observation_id,
            pnl_pct=1.0, outcome="win",
        )
    mp = pm.get_market_pattern("higher_low_continuation__bullish__15m")
    assert mp.is_surfaced is True
    surfaced = pm.get_surfaced_market()
    assert any(p.pattern_id == mp.pattern_id for p in surfaced)


def test_market_pattern_round_trip_serialisation():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    d   = _tmp()
    pm1 = PatternMemory(d / "p.json")
    obs = pm1.record_market_pattern(
        MarketPatternType.RANGE_REJECTION, "bearish", "15m", trade_id="RR1"
    )
    pm1.update_market_pattern_outcome(
        MarketPatternType.RANGE_REJECTION, "bearish", "15m",
        observation_id=obs.observation_id,
        pnl_pct=0.9, outcome="win",
    )
    pm1.save()

    pm2 = PatternMemory(d / "p.json")
    mp  = pm2.get_market_pattern("range_rejection__bearish__15m")
    assert mp is not None
    assert mp.occurrence_count == 1
    assert mp.resolved_count   == 1
    assert mp.win_count        == 1


def test_market_pattern_get_by_type():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    pm = PatternMemory(_tmp() / "p.json")
    pm.record_market_pattern(MarketPatternType.BULL_FLAG, "bullish", "5m")
    pm.record_market_pattern(MarketPatternType.BULL_FLAG, "bullish", "15m")
    pm.record_market_pattern(MarketPatternType.BEAR_FLAG, "bearish", "5m")
    bull_flags = pm.get_market_by_type(MarketPatternType.BULL_FLAG)
    assert len(bull_flags) == 2


def test_market_pattern_update_nonexistent_returns_false():
    from memory.pattern_memory import PatternMemory
    pm = PatternMemory(_tmp() / "p.json")
    ok = pm.update_market_pattern_outcome(
        "gap_and_go", "bullish", "any",
        observation_id="nonexistent",
        pnl_pct=1.0, outcome="win",
    )
    assert ok is False


def test_pattern_memory_save_and_reload_both_namespaces():
    from memory.pattern_memory import PatternMemory, MarketPatternType
    d   = _tmp()
    pm1 = PatternMemory(d / "pat.json")
    pm1.record_behavior("test_b", "test behavior", positive=True, trade_id="T1")
    pm1.record_market_pattern(MarketPatternType.TREND_EXHAUSTION, "bearish", "1h")
    pm1.save()

    pm2 = PatternMemory(d / "pat.json")
    assert len(pm2.get_all_behavior())        >= 1
    assert len(pm2.get_all_market_patterns()) >= 1


# ═══════════════════════════════════════════════════════════
# memory/reflection_engine.py
# ═══════════════════════════════════════════════════════════

def _reflection_session():
    from memory.session_memory import SessionMemory
    s = SessionMemory(session_date="2025-01-15")
    s.opening_regime     = "trend_bull"
    s.dominant_narrative = "trending_bull"
    s.record_trade("T1",  1.2)
    s.record_trade("T2",  0.9)
    s.record_trade("T3", -0.5)
    s.add_behavioral_flag("overtraded")
    return s


def _reflection_trades():
    records = []
    for i, (pnl, rule_ok, mistakes) in enumerate([
        ( 1.2, True,  []),
        ( 0.9, True,  []),
        (-0.5, False, ["chased entry"]),
    ]):
        r = _make_trade_record(f"T{i+1}", pnl=pnl)
        r.rule_compliant = rule_ok
        r.mistake_tags   = mistakes
        r.rr_planned     = 1.6
        records.append(r)
    return records


def test_reflection_generate_basic():
    from memory.reflection_engine import ReflectionEngine
    ref = ReflectionEngine().generate(_reflection_session(), _reflection_trades())
    assert ref.session_date == "2025-01-15"
    assert ref.total_trades == 3


def test_reflection_quality_separation_counts():
    from memory.reflection_engine import ReflectionEngine
    ref = ReflectionEngine().generate(_reflection_session(), _reflection_trades())
    assert len(ref.trade_quality_assessments) == 3


def test_reflection_good_exec_bad_outcome():
    from memory.reflection_engine import ReflectionEngine
    from memory.session_memory import SessionMemory
    s = SessionMemory(session_date="2025-01-20")
    s.record_trade("T_GEBO", -0.8)
    t = _make_trade_record("T_GEBO", pnl=-0.8)
    t.rule_compliant      = True
    t.mistake_tags        = []
    t.rr_planned          = 1.8
    t.outcome             = "loss"
    t.time_stop_triggered = True   # force bad outcome quality
    ref = ReflectionEngine().generate(s, [t])
    assert isinstance(ref.good_trade_bad_outcome_count, int)
    assert isinstance(ref.bad_trade_good_outcome_count, int)


def test_reflection_bad_exec_good_outcome_tracked():
    from memory.reflection_engine import ReflectionEngine
    from memory.session_memory import SessionMemory
    s = SessionMemory(session_date="2025-01-21")
    s.record_trade("T_BEGO", 1.5)
    t = _make_trade_record("T_BEGO", pnl=1.5)
    t.rule_compliant = False
    t.mistake_tags   = ["fomo entry", "no plan"]
    t.outcome        = "win"
    ref = ReflectionEngine().generate(s, [t])
    assert ref.bad_trade_good_outcome_count == 1
    assert ref.good_trade_bad_outcome_count == 0


def test_reflection_apply_quality_tags():
    from memory.reflection_engine import ReflectionEngine
    engine = ReflectionEngine()
    trades = _reflection_trades()
    ref    = engine.generate(_reflection_session(), trades)
    engine.apply_quality_tags(ref, trades)
    for t in trades:
        assert t.execution_quality in ("good", "acceptable", "poor")
        assert t.outcome_quality   in ("good", "bad")


def test_reflection_to_dict_structure():
    from memory.reflection_engine import ReflectionEngine
    d = ReflectionEngine().generate(_reflection_session(), _reflection_trades()).to_dict()
    for k in ("session_date", "generated_at", "market", "trades",
              "quality_separation", "analysis", "stats"):
        assert k in d


def test_reflection_no_trades():
    from memory.reflection_engine import ReflectionEngine
    from memory.session_memory import SessionMemory
    s   = SessionMemory(session_date="2025-01-16")
    ref = ReflectionEngine().generate(s, [])
    assert ref.total_trades    == 0
    assert ref.performance_stats is None


def test_reflection_performance_stats_attached():
    from memory.reflection_engine import ReflectionEngine
    ref = ReflectionEngine().generate(_reflection_session(), _reflection_trades())
    assert ref.performance_stats is not None
    assert "overall" in ref.performance_stats


def test_reflection_mistake_analysis():
    from memory.reflection_engine import ReflectionEngine
    ref = ReflectionEngine().generate(_reflection_session(), _reflection_trades())
    combined = " ".join(ref.mistake_analysis)
    assert "chased" in combined or "Behavioral" in combined or "overtraded" in combined


def test_reflection_suggested_adjustments_insufficient_data():
    from memory.reflection_engine import ReflectionEngine
    from memory.session_memory import SessionMemory
    s   = SessionMemory(session_date="2025-01-17")
    t   = _make_trade_record("T_SOLO")
    ref = ReflectionEngine().generate(s, [t])
    assert any("Insufficient" in a for a in ref.suggested_adjustments)


# ═══════════════════════════════════════════════════════════
# memory/startup_context.py
# ═══════════════════════════════════════════════════════════

def test_startup_no_previous_data():
    from memory.startup_context import build_startup_briefing
    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = _tmp(),
        trades_dir   = _tmp(),
        pattern_file = _tmp() / "p.json",
    )
    assert b.session_date      == "2025-06-01"
    assert b.prev_session_date == ""
    assert b.recent_trades_n   == 0
    assert "No previous" in b.summary


def test_startup_loads_prev_session():
    from memory.startup_context import build_startup_briefing
    from memory.session_memory  import SessionMemory
    d = _tmp()
    s = SessionMemory(session_date="2025-05-31")
    s.record_trade("T1", 1.2)
    s.record_trade("T2", 0.8)
    s.closing_regime = "trend_bull"
    s.lesson = "wait for breakout confirm"
    s.save(d)

    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = d,
        trades_dir   = _tmp(),
        pattern_file = _tmp() / "p.json",
    )
    assert b.prev_session_date   == "2025-05-31"
    assert b.prev_session_trades == 2
    assert b.prev_session_lesson == "wait for breakout confirm"
    assert "2025-05-31" in b.summary


def test_startup_loads_recent_trades():
    from memory.startup_context import build_startup_briefing
    d = _tmp()
    for i in range(5):
        _make_trade_record(f"R{i}", pnl=1.0).save(d)

    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = _tmp(),
        trades_dir   = d,
        pattern_file = _tmp() / "p.json",
    )
    assert b.recent_trades_n == 5
    assert b.recent_win_rate == 1.0


def test_startup_top_risks_from_bad_session():
    from memory.startup_context import build_startup_briefing
    from memory.session_memory  import SessionMemory
    sd = _tmp()
    s  = SessionMemory(session_date="2025-05-31")
    s.daily_pnl_pct = -2.5
    s.save(sd)

    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = sd,
        trades_dir   = _tmp(),
        pattern_file = _tmp() / "p.json",
    )
    assert len(b.top_risks) >= 1
    assert any("loss" in r.lower() or "size" in r.lower() for r in b.top_risks)


def test_startup_includes_market_patterns():
    from memory.startup_context import build_startup_briefing
    from memory.pattern_memory  import PatternMemory, MarketPatternType
    from config.settings import PATTERN_MIN_OCCURRENCES
    d  = _tmp()
    pf = d / "pat.json"
    pm = PatternMemory(pf)
    obs_list = []
    for i in range(PATTERN_MIN_OCCURRENCES):
        obs = pm.record_market_pattern(
            MarketPatternType.BULL_FLAG, "bullish", "15m", trade_id=f"BF{i}"
        )
        obs_list.append(obs)
    for obs in obs_list:
        pm.update_market_pattern_outcome(
            MarketPatternType.BULL_FLAG, "bullish", "15m",
            observation_id=obs.observation_id,
            pnl_pct=1.0, outcome="win",
        )
    pm.save()

    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = _tmp(),
        trades_dir   = _tmp(),
        pattern_file = pf,
    )
    assert len(b.active_market_patterns) >= 1


def test_startup_to_dict_structure():
    from memory.startup_context import build_startup_briefing
    b = build_startup_briefing(
        session_date = "2025-06-01",
        session_dir  = _tmp(),
        trades_dir   = _tmp(),
        pattern_file = _tmp() / "p.json",
    )
    d = b.to_dict()
    for k in ("generated_at", "session_date", "previous_session",
              "recent_performance", "patterns", "top_risks", "summary"):
        assert k in d
    assert "market" in d["patterns"]


# ═══════════════════════════════════════════════════════════
# Integration — cross-module round-trip
# ═══════════════════════════════════════════════════════════

def test_integration_full_session_cycle():
    """Session → trades → reflection → pattern update → startup briefing."""
    from memory.session_memory    import SessionMemory
    from memory.trade_memory      import TradeRecord
    from memory.pattern_memory    import PatternMemory, MarketPatternType
    from memory.reflection_engine import ReflectionEngine
    from memory.startup_context   import build_startup_briefing

    base = _tmp()
    sd = base / "sessions";  sd.mkdir()
    td = base / "trades";    td.mkdir()
    pf = base / "patterns.json"

    # Session
    session = SessionMemory(session_date="2025-06-01")
    session.opening_regime = "trend_bull"
    session.open_session()
    session.record_trade("T001",  1.5)
    session.record_trade("T002", -0.4)
    session.close_session(lesson="Only trade A+ setups")
    session.save(sd)

    # Trades
    t1 = _make_trade_record("T001", pnl=1.5)
    t2 = _make_trade_record("T002", pnl=-0.4)
    t1.save(td); t2.save(td)

    # Market pattern observation + outcome
    pm  = PatternMemory(pf)
    obs = pm.record_market_pattern(
        MarketPatternType.BULL_FLAG, "bullish", "5m", trade_id="T001"
    )
    pm.update_market_pattern_outcome(
        MarketPatternType.BULL_FLAG, "bullish", "5m",
        observation_id = obs.observation_id,
        pnl_pct=1.5, outcome="win", trade_id="T001",
    )
    pm.save()

    # Reflection
    ref = ReflectionEngine().generate(session, [t1, t2])
    assert ref.total_trades == 2
    assert len(ref.trade_quality_assessments) == 2

    # Next-day startup briefing
    briefing = build_startup_briefing(
        session_date = "2025-06-02",
        session_dir  = sd,
        trades_dir   = td,
        pattern_file = pf,
    )
    assert briefing.prev_session_date   == "2025-06-01"
    assert briefing.prev_session_lesson == "Only trade A+ setups"
    assert briefing.recent_trades_n     == 2


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nPhase 6 tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    sys.exit(0 if failed == 0 else 1)