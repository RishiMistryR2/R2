"""
tests/test_confidence_engine.py — Unit tests for engine/confidence_engine.py

Tests evidence generation, confidence scoring, full thesis format, and
compact thesis format.  No API calls, no live data.

Run: PYTHONPATH=. python tests/test_confidence_engine.py
"""

import os, sys, traceback

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


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _uptrend():
    return {
        "direction":            "uptrend",
        "sequence":             "HH HL HH HL",
        "trend_intact":         True,
        "last_hh":              549.48,
        "last_hl":              544.22,
        "last_lh":              None,
        "last_ll":              None,
        "market_state":         "trend",
        "missing_confirmation": "Need break above 549.48.",
        "key_levels": {
            "session_high":       549.48,
            "session_low":        541.19,
            "prior_day_high":     565.43,
            "prior_day_low":      552.67,
            "vwap":               543.50,
            "nearest_support":    544.22,
            "nearest_resistance": None,
        },
        "recent_events": [],
    }


def _downtrend():
    return {
        "direction":            "downtrend",
        "sequence":             "LH LL LH LL",
        "trend_intact":         True,
        "last_hh":              None,
        "last_hl":              None,
        "last_lh":              738.39,
        "last_ll":              730.50,
        "market_state":         "trend",
        "missing_confirmation": "Need new low below 730.50.",
        "key_levels": {
            "session_high":       738.39,
            "session_low":        730.50,
            "prior_day_high":     741.00,
            "prior_day_low":      728.00,
            "vwap":               734.90,
            "nearest_support":    None,
            "nearest_resistance": 738.39,
        },
        "recent_events": [],
    }


def _range():
    return {
        "direction":            "ranging",
        "sequence":             "",
        "trend_intact":         False,
        "last_hh":              None,
        "last_hl":              None,
        "last_lh":              None,
        "last_ll":              None,
        "market_state":         "range",
        "missing_confirmation": "",
        "key_levels": {
            "session_high":       737.00,
            "session_low":        735.80,
            "prior_day_high":     738.50,
            "prior_day_low":      734.20,
            "vwap":               736.40,
            "nearest_support":    None,
            "nearest_resistance": None,
        },
        "recent_events": [],
    }


# ──────────────────────────────────────────────────────────────
# compute_confidence — basics
# ──────────────────────────────────────────────────────────────

def test_returns_confidence_result():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend())
    assert hasattr(cr, "score")
    assert hasattr(cr, "label")
    assert hasattr(cr, "supporting")
    assert hasattr(cr, "contradicting")
    assert hasattr(cr, "conviction")


def test_score_in_range():
    from engine.confidence_engine import compute_confidence
    for fields in (_uptrend(), _downtrend(), _range()):
        cr = compute_confidence(fields)
        assert 0 <= cr.score <= 100, f"Score {cr.score} out of range"


def test_score_clamped_15_to_90():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence({})   # empty fields — minimum conditions
    assert cr.score >= 15
    cr2 = compute_confidence(
        _uptrend(), regime_name="trend_bull", regime_conf=0.9,
        session_key="open", entries_allowed=True,
    )
    assert cr2.score <= 90


def test_not_empty_supporting_for_clear_trend():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend(), regime_name="trend_bull", regime_conf=0.75)
    assert len(cr.supporting) >= 2, (
        f"Expected ≥2 supporting items for strong uptrend, got {cr.supporting}"
    )


def test_not_empty_contradicting_for_chop_regime():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend(), regime_name="chop", regime_conf=0.6)
    has_chop = any("chop" in s.lower() for s in cr.contradicting)
    assert has_chop, f"Expected chop regime in contradicting:\n{cr.contradicting}"


# ──────────────────────────────────────────────────────────────
# Regime alignment
# ──────────────────────────────────────────────────────────────

def test_aligned_regime_raises_score():
    from engine.confidence_engine import compute_confidence
    cr_aligned = compute_confidence(
        _uptrend(), regime_name="trend_bull", regime_conf=0.75,
        session_key="open",
    )
    cr_chop = compute_confidence(
        _uptrend(), regime_name="chop", regime_conf=0.75,
        session_key="open",
    )
    assert cr_aligned.score > cr_chop.score, (
        "Aligned regime should produce higher score than chop regime"
    )


def test_opposing_regime_lowers_score():
    from engine.confidence_engine import compute_confidence
    cr_bull = compute_confidence(
        _uptrend(), regime_name="trend_bull", session_key="open",
    )
    cr_bear = compute_confidence(
        _uptrend(), regime_name="trend_bear", session_key="open",
    )
    assert cr_bear.score < cr_bull.score


def test_contradicting_has_chop_note():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_downtrend(), regime_name="chop")
    assert any("chop" in s.lower() for s in cr.contradicting)


def test_supporting_has_regime_note_when_aligned():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_downtrend(), regime_name="trend_bear", regime_conf=0.8)
    assert any("regime" in s.lower() for s in cr.supporting)


# ──────────────────────────────────────────────────────────────
# Session effects
# ──────────────────────────────────────────────────────────────

def test_after_hours_lowers_score():
    from engine.confidence_engine import compute_confidence
    cr_open   = compute_confidence(_uptrend(), session_key="open")
    cr_after  = compute_confidence(_uptrend(), session_key="after_hours")
    assert cr_after.score < cr_open.score


def test_after_hours_in_contradicting():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend(), session_key="after_hours")
    has_note = any(
        "extended-hours" in s.lower() or "after" in s.lower()
        for s in cr.contradicting
    )
    assert has_note, f"Expected after-hours note in contradicting:\n{cr.contradicting}"


def test_closed_lower_than_after_hours():
    from engine.confidence_engine import compute_confidence
    cr_after  = compute_confidence(_uptrend(), session_key="after_hours")
    cr_closed = compute_confidence(_uptrend(), session_key="closed")
    assert cr_closed.score < cr_after.score


# ──────────────────────────────────────────────────────────────
# Entries allowed
# ──────────────────────────────────────────────────────────────

# v2.7: entry readiness is an action filter, not directional evidence.
# compute_confidence() keeps the entries_allowed parameter for backward
# compatibility but no longer lets it change the score or evidence lists.

def test_entries_not_allowed_does_not_change_score():
    # Was test_entries_not_allowed_lowers_score — v2.7 removed the −5 penalty.
    from engine.confidence_engine import compute_confidence
    cr_yes = compute_confidence(_uptrend(), entries_allowed=True,  session_key="open")
    cr_no  = compute_confidence(_uptrend(), entries_allowed=False, session_key="open")
    assert cr_no.score == cr_yes.score


def test_entries_not_allowed_not_in_contradicting():
    # Was test_entries_not_allowed_in_contradicting — v2.7 removed that item.
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend(), entries_allowed=False)
    assert not any("permit" in s.lower() for s in cr.contradicting)


def test_entries_flag_no_effect_across_directions_and_sessions():
    from engine.confidence_engine import compute_confidence
    for fields in (_uptrend(), _downtrend(), _range()):
        for sess in ("premarket", "open", "midday", "power_hour", "after_hours", "closed"):
            cr_yes = compute_confidence(fields, entries_allowed=True,  session_key=sess)
            cr_no  = compute_confidence(fields, entries_allowed=False, session_key=sess)
            assert cr_yes.score == cr_no.score, f"{fields['direction']}/{sess}"
            assert cr_yes.label == cr_no.label
            assert cr_yes.supporting == cr_no.supporting
            assert cr_yes.contradicting == cr_no.contradicting


def test_entries_permission_not_in_supporting():
    from engine.confidence_engine import compute_confidence
    for fields in (_uptrend(), _downtrend()):
        cr = compute_confidence(fields, entries_allowed=True, session_key="open")
        assert not any("permit" in s.lower() for s in cr.supporting)


# ──────────────────────────────────────────────────────────────
# ConfidenceState (v2.7)
# ──────────────────────────────────────────────────────────────

def test_confidence_state_holds_all_fields():
    from datetime import datetime, timezone
    from engine.confidence_engine import ConfidenceState
    now = datetime(2026, 10, 22, 14, 30, tzinfo=timezone.utc)
    cs = ConfidenceState(
        structural_confidence=69.0,
        calibrated_confidence=72.5,
        regime_confidence=0.75,
        calibration_result=None,
        computed_at=now,
        source_cycle=42,
    )
    assert cs.structural_confidence == 69.0
    assert cs.calibrated_confidence == 72.5
    assert cs.regime_confidence == 0.75
    assert cs.calibration_result is None
    assert cs.computed_at == now
    assert cs.source_cycle == 42


def test_confidence_state_accepts_calibration_result():
    from datetime import datetime
    from engine.confidence_engine import ConfidenceState
    from memory.statistical_intelligence import CalibrationResult
    cal = CalibrationResult(
        base_confidence=69.0, final_confidence=72.5, adjustment=3.5,
        sufficient_history=True, win_rate=0.62, sample_size=30,
        explanation="Historical win rate 62% over 30 observations.",
    )
    cs = ConfidenceState(
        structural_confidence=cal.base_confidence,
        calibrated_confidence=cal.final_confidence,
        regime_confidence=0.75,
        calibration_result=cal,
        computed_at=datetime.now(),
        source_cycle=1,
    )
    assert cs.calibration_result is cal
    assert cs.calibrated_confidence - cs.structural_confidence == cal.adjustment


def test_confidence_engine_does_not_import_memory_at_runtime():
    # The CalibrationResult annotation is type-only; engine/ must not pull
    # memory/ in at import time.
    # Runs in a fresh interpreter so it can't disturb other tests' modules.
    import subprocess, sys as _sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    code = (
        "import sys; import engine.confidence_engine; "
        "sys.exit(1 if 'memory.statistical_intelligence' in sys.modules else 0)"
    )
    result = subprocess.run([_sys.executable, "-c", code], cwd=root)
    assert result.returncode == 0, "engine.confidence_engine imported memory/ at runtime"


# ──────────────────────────────────────────────────────────────
# Label + conviction mapping
# ──────────────────────────────────────────────────────────────

def test_high_confidence_label():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(
        _uptrend(), regime_name="trend_bull", regime_conf=0.9,
        session_key="open", entries_allowed=True,
    )
    if cr.score >= 75:
        assert cr.label == "High"
        assert cr.conviction == "High"


def test_low_score_label():
    from engine.confidence_engine import compute_confidence
    # Force low score: closed session + opposing regime + no direction
    cr = compute_confidence(
        {}, regime_name="trend_bull", session_key="closed",
        entries_allowed=False,
    )
    assert cr.label in ("Low", "Moderate")


def test_conviction_values_are_valid():
    from engine.confidence_engine import compute_confidence
    valid = {"Low", "Moderate", "High"}
    for fields in (_uptrend(), _downtrend(), _range()):
        cr = compute_confidence(fields)
        assert cr.conviction in valid, f"Invalid conviction: {cr.conviction}"


# ──────────────────────────────────────────────────────────────
# build_thesis — format checks
# ──────────────────────────────────────────────────────────────

def test_thesis_returns_string():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert isinstance(text, str)
    assert len(text) > 80


def test_thesis_has_confidence_line():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert "Confidence:" in text


def test_thesis_has_supporting_section():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend(), regime_name="trend_bull", regime_conf=0.8,
                        session="open", entries_allowed=True)
    assert "Supporting:" in text


def test_thesis_has_contradicting_section():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend(), regime_name="chop", regime_conf=0.5)
    assert "Contradicting:" in text


def test_thesis_has_conviction_line():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert "Conviction:" in text


def test_thesis_no_pre_market_for_after_hours():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_downtrend(), session="after_hours")
    assert "pre-market" not in text.lower()
    assert "after hours" in text.lower() or "extended-hours" in text.lower()


def test_thesis_has_specific_price_levels():
    """Thesis must contain at least one actual price level."""
    import re
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    prices = re.findall(r"\d{3,}\.\d{2}", text)
    assert len(prices) >= 1, f"Expected price levels in thesis:\n{text}"


def test_thesis_downtrend_says_trending_lower():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_downtrend())
    assert "trending lower" in text.lower() or "downtrend" in text.lower()


def test_thesis_uptrend_says_trending_higher():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert "trending higher" in text.lower() or "uptrend" in text.lower()


def test_thesis_range_has_range_levels():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_range())
    assert "737.00" in text or "735.80" in text, (
        f"Range thesis missing specific levels:\n{text}"
    )


def test_thesis_has_confirmation():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert "confirmation" in text.lower() or "confirm" in text.lower()


def test_thesis_has_invalidation():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend())
    assert "invalidation" in text.lower() or "degrades" in text.lower()


# ──────────────────────────────────────────────────────────────
# build_compact_thesis
# ──────────────────────────────────────────────────────────────

def test_compact_shorter_than_full():
    from engine.confidence_engine import build_thesis, build_compact_thesis
    full    = build_thesis(_uptrend())
    compact = build_compact_thesis(_uptrend())
    assert len(compact) < len(full), "Compact should be shorter than full thesis"


def test_compact_has_confidence_line():
    from engine.confidence_engine import build_compact_thesis
    text = build_compact_thesis(_uptrend())
    assert "confidence" in text.lower()
    assert "conviction" in text.lower()


def test_compact_no_bullets():
    from engine.confidence_engine import build_compact_thesis
    text = build_compact_thesis(_uptrend())
    assert "Supporting:" not in text, "Compact should not have full evidence bullets"


# ──────────────────────────────────────────────────────────────
# Package wrappers
# ──────────────────────────────────────────────────────────────

def test_thesis_from_package_no_crash():
    from engine.confidence_engine import build_thesis_from_package
    from ai.state_packager import build_full_package
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state  = build_demo_state()

    class _FMS:
        value = "after_hours"
    state.market_session = _FMS()
    pkg  = build_full_package(state=state, structure_state=analyze(state))
    text = build_thesis_from_package(pkg.to_dict())
    assert "Confidence:" in text
    assert "Conviction:" in text


def test_compact_from_package_no_crash():
    from engine.confidence_engine import build_compact_thesis_from_package
    from ai.state_packager import build_full_package
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    state = build_demo_state()
    pkg   = build_full_package(state=state, structure_state=analyze(state))
    text  = build_compact_thesis_from_package(pkg.to_dict())
    assert isinstance(text, str)
    assert len(text) > 30


def test_package_wrapper_empty_dict():
    from engine.confidence_engine import build_thesis_from_package
    text = build_thesis_from_package({})
    assert isinstance(text, str)


# ──────────────────────────────────────────────────────────────
# StructureState object path
# ──────────────────────────────────────────────────────────────

def test_accepts_structure_state_object():
    from engine.confidence_engine import build_thesis
    from engine.structure_engine import analyze
    from runtime.update_cycle import build_demo_state, reset_engine
    reset_engine()
    struct = analyze(build_demo_state())
    text   = build_thesis(
        struct, regime_name="trend_bull", regime_conf=0.72,
        session="open", entries_allowed=True,
    )
    assert "Confidence:" in text
    assert "Conviction:" in text


# ──────────────────────────────────────────────────────────────
# v2.7 — thesis text shows the stored Final Conviction
# ──────────────────────────────────────────────────────────────

def test_thesis_without_stored_value_unchanged():
    from engine.confidence_engine import build_thesis, compute_confidence
    cr   = compute_confidence(_uptrend())
    text = build_thesis(_uptrend())
    assert f"Confidence: {cr.score:.0f}% ({cr.label})" in text
    assert "historical adjustment" not in text.lower()


def test_thesis_shows_stored_final_confidence():
    from engine.confidence_engine import build_thesis, compute_confidence
    cr   = compute_confidence(_uptrend())
    text = build_thesis(_uptrend(), final_confidence=cr.score + 8,
                        structural_confidence=cr.score)
    assert f"Confidence: {cr.score + 8:.0f}%" in text
    assert "Historical adjustment +8" in text or "historical adjustment +8" in text
    assert f"Structural {cr.score:.0f}%" in text


def test_thesis_label_and_conviction_follow_final_value():
    from engine.confidence_engine import build_thesis
    text = build_thesis(_uptrend(), final_confidence=80.0, structural_confidence=70.0)
    assert "Confidence: 80% (High)" in text
    assert "Conviction: High." in text


def test_thesis_no_adjustment_note_when_equal():
    from engine.confidence_engine import build_thesis, compute_confidence
    cr   = compute_confidence(_uptrend())
    text = build_thesis(_uptrend(), final_confidence=cr.score,
                        structural_confidence=cr.score)
    assert "historical adjustment" not in text.lower()


def test_compact_thesis_shows_stored_final_confidence():
    from engine.confidence_engine import build_compact_thesis
    text = build_compact_thesis(_uptrend(), final_confidence=50.0,
                                structural_confidence=56.0)
    assert "Confidence: 50% (Moderate; structural 56%, historical adjustment -6)" in text


def test_thesis_from_package_reads_confidence_block():
    from engine.confidence_engine import (
        build_thesis_from_package, build_compact_thesis_from_package,
    )
    pkg = {"structure": _uptrend(),
           "confidence": {"structural": 62.0, "calibrated": 71.0}}
    assert "Confidence: 71%" in build_thesis_from_package(pkg)
    assert "Confidence: 71%" in build_compact_thesis_from_package(pkg)


def test_thesis_from_package_without_block_falls_back():
    from engine.confidence_engine import build_thesis_from_package, compute_confidence
    cr = compute_confidence(_uptrend())
    assert f"Confidence: {cr.score:.0f}%" in build_thesis_from_package({"structure": _uptrend()})


# ──────────────────────────────────────────────────────────────
# Structured evidence (regression features)
# ──────────────────────────────────────────────────────────────

_KNOWN_EVIDENCE_KEYS = {
    "direction_present", "sequence_strength", "trend_intact", "trend_weakening",
    "regime_aligned", "regime_chop_vs_trend", "regime_opposes",
    "regime_consistent_with_range", "regime_directional_no_sequence",
    "regime_confidence_high", "regime_confidence_low",
    "structural_reference_defined", "opposing_level_unbroken",
    "event_vwap_reclaim", "event_vwap_rejection",
    "event_failed_breakout", "event_failed_breakdown",
    "confirmation_pending",
    "session_midday", "session_power_hour", "session_premarket",
    "session_after_hours", "session_closed",
}


def _evidence_cases():
    import itertools
    events = [[], ["vwap_reclaim", "failed_breakdown"], ["vwap_rejection", "failed_breakout"]]
    for fields, regime, sess, conf, evs, intact in itertools.product(
            (_uptrend(), _downtrend(), _range()),
            ("trend_bull", "trend_bear", "chop", ""),
            ("open", "midday", "power_hour", "premarket", "after_hours", "closed"),
            (0.3, 0.6, 0.8), events, (True, False)):
        f = dict(fields, recent_events=evs, trend_intact=intact)
        yield f, regime, conf, sess


def test_evidence_accounts_for_score():
    # clamp(35 + sum(points)) must reproduce the score exactly, so the
    # recorded items are a complete explanation of every score.
    from engine.confidence_engine import compute_confidence
    for f, regime, conf, sess in _evidence_cases():
        cr = compute_confidence(f, regime, conf, sess)
        rebuilt = round(max(15.0, min(90.0, 35 + sum(e.points for e in cr.evidence))), 1)
        assert rebuilt == cr.score, (f["direction"], regime, conf, sess, cr.evidence)


def test_evidence_polarity_matches_sign():
    from engine.confidence_engine import compute_confidence
    for f, regime, conf, sess in _evidence_cases():
        for e in compute_confidence(f, regime, conf, sess).evidence:
            assert e.polarity in ("supporting", "contradicting")
            if e.points > 0:
                assert e.polarity == "supporting", e
            if e.points < 0:
                assert e.polarity == "contradicting", e


def test_evidence_keys_known_and_unique():
    from engine.confidence_engine import compute_confidence
    seen = set()
    for f, regime, conf, sess in _evidence_cases():
        keys = [e.key for e in compute_confidence(f, regime, conf, sess).evidence]
        assert len(keys) == len(set(keys)), keys
        assert set(keys) <= _KNOWN_EVIDENCE_KEYS, set(keys) - _KNOWN_EVIDENCE_KEYS
        seen |= set(keys)
    assert seen == _KNOWN_EVIDENCE_KEYS, _KNOWN_EVIDENCE_KEYS - seen


def test_text_only_evidence_has_zero_points():
    from engine.confidence_engine import compute_confidence
    cr = compute_confidence(_uptrend(), "trend_bull", 0.6, "open")
    by_key = {e.key: e for e in cr.evidence}
    assert by_key["opposing_level_unbroken"].points == 0
    assert by_key["opposing_level_unbroken"].polarity == "contradicting"
    assert by_key["confirmation_pending"].points == 0


def test_evidence_independent_of_entries_allowed():
    from engine.confidence_engine import compute_confidence
    a = compute_confidence(_uptrend(), "trend_bull", 0.8, "after_hours", True)
    b = compute_confidence(_uptrend(), "trend_bull", 0.8, "after_hours", False)
    assert a.evidence == b.evidence


def test_evidence_item_to_dict():
    from engine.confidence_engine import EvidenceItem
    assert EvidenceItem("trend_intact", 10, "supporting").to_dict() == {
        "key": "trend_intact", "points": 10, "polarity": "supporting",
    }


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nConfidence engine tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)