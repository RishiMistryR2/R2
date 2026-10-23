"""
tests/test_confidence_state.py — Unit tests for the v2.7 confidence pipeline
in runtime/update_cycle.py (_run_confidence_pipeline / _compute_confidence_state).

Covers: stored values, cycle stamping, calibration wiring (incl. conflict
dampening), and non-fatal failure behaviour.  Demo data only, in-memory
outcome tracker, no API calls.

Run: PYTHONPATH=. python tests/test_confidence_state.py
"""

import os, sys, traceback
from datetime import datetime, timezone
from unittest import mock

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

def _fresh():
    """Fresh in-memory tracker + strategy engine + RuntimeState."""
    from memory.outcome_tracker import reset_tracker
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import reset_engine
    tracker = reset_tracker(":memory:")
    reset_engine()
    return RuntimeState(), tracker


def _cycled(n=1):
    """RuntimeState after n successful demo update cycles."""
    from runtime.update_cycle import run_update_cycle
    state, tracker = _fresh()
    for _ in range(n):
        assert run_update_cycle(state) is True
    return state, tracker


def _inputs(state):
    """Re-derive the pipeline's inputs the same way event_engine does."""
    from engine.narrative_engine import _to_fields, _session_key
    fields = _to_fields(state.structure_state) if state.structure_state else {}
    sess   = _session_key(state.market_state.market_session)
    reg    = state.market_state.regime
    return fields, reg.regime.value, float(reg.confidence), sess


class _Pat:
    def __init__(self, name, bias):
        self.name = name
        self.bias = bias


def _seed_history(tracker, name, bias, outcome, n=30):
    """Record and resolve n observations for one pattern."""
    for _ in range(n):
        oid = tracker.record_observation(
            {"confidence_score": 60.0, "current_price": 545.0},
            [{"name": name, "bias": bias}],
        )
        tracker.record_outcome(oid, outcome, resolution_minutes=40.0,
                               mfe_atr=1.0, mae_atr=0.3)


# ──────────────────────────────────────────────────────────────
# Stored values
# ──────────────────────────────────────────────────────────────

def test_cycle_populates_confidence_state():
    from engine.confidence_engine import ConfidenceState
    state, _ = _cycled()
    cs = state.confidence_state
    assert isinstance(cs, ConfidenceState)
    assert isinstance(cs.structural_confidence, float)
    assert isinstance(cs.calibrated_confidence, float)
    assert isinstance(cs.regime_confidence,     float)
    assert 15.0 <= cs.structural_confidence <= 90.0
    assert 15.0 <= cs.calibrated_confidence <= 90.0


def test_confidence_state_none_before_first_cycle():
    state, _ = _fresh()
    assert state.confidence_state is None


def test_source_cycle_matches_update_count():
    from runtime.update_cycle import run_update_cycle
    state, _ = _fresh()
    for expected in (1, 2, 3):
        run_update_cycle(state)
        assert state.confidence_state.source_cycle == expected
        assert state.confidence_state.source_cycle == state.update_count


def test_computed_at_is_recent_utc():
    before = datetime.now(tz=timezone.utc)
    state, _ = _cycled()
    after = datetime.now(tz=timezone.utc)
    cs = state.confidence_state
    assert cs.computed_at.tzinfo is not None
    assert before <= cs.computed_at <= after


def test_new_object_each_cycle():
    from runtime.update_cycle import run_update_cycle
    state, _ = _cycled()
    first = state.confidence_state
    run_update_cycle(state)
    assert state.confidence_state is not first


def test_regime_confidence_from_market_state():
    state, _ = _cycled()
    assert state.confidence_state.regime_confidence == float(
        state.market_state.regime.confidence
    )


def test_structural_matches_direct_compute():
    from engine.confidence_engine import compute_confidence
    state, _ = _cycled()
    fields, reg_name, reg_conf, sess = _inputs(state)
    expected = compute_confidence(fields, reg_name, reg_conf, sess).score
    assert state.confidence_state.structural_confidence == float(expected)


def test_structural_ignores_entry_permission():
    # Demo state is outside trading hours, so entries are disabled — the
    # stored score must still equal the entries-allowed computation.
    from engine.confidence_engine import compute_confidence
    state, _ = _cycled()
    fields, reg_name, reg_conf, sess = _inputs(state)
    allowed  = compute_confidence(fields, reg_name, reg_conf, sess, True).score
    disabled = compute_confidence(fields, reg_name, reg_conf, sess, False).score
    assert state.confidence_state.structural_confidence == allowed == disabled


# ──────────────────────────────────────────────────────────────
# Calibration wiring
# ──────────────────────────────────────────────────────────────

def test_no_history_calibrated_equals_structural():
    state, _ = _cycled()
    cs = state.confidence_state
    assert cs.calibrated_confidence == cs.structural_confidence
    assert cs.calibration_result is not None
    assert cs.calibration_result.sufficient_history is False


def test_history_adjusts_calibrated():
    from runtime.update_cycle import run_update_cycle
    state, tracker = _fresh()
    _seed_history(tracker, "Test Pattern", "bullish", "bullish")
    with mock.patch("engine.pattern_engine.classify_patterns",
                    return_value=[_Pat("Test Pattern", "bullish")]):
        run_update_cycle(state)
    cs = state.confidence_state
    assert cs.calibration_result.sufficient_history is True
    assert cs.calibrated_confidence > cs.structural_confidence
    assert cs.calibrated_confidence == cs.calibration_result.final_confidence
    assert cs.calibration_result.base_confidence == cs.structural_confidence


def test_structure_direction_reaches_conflict_dampening():
    # Demo structure is an uptrend; bearish-resolving history opposes it.
    from runtime.update_cycle import run_update_cycle
    state, tracker = _fresh()
    _seed_history(tracker, "Test Pattern", "bearish", "bearish")
    with mock.patch("engine.pattern_engine.classify_patterns",
                    return_value=[_Pat("Test Pattern", "bearish")]):
        run_update_cycle(state)
    calib = state.confidence_state.calibration_result
    assert calib.conflict_dampened is True
    assert "uptrend" in calib.dampen_reason


def test_prediction_taken_from_primary_pattern_bias():
    import memory.statistical_intelligence as si
    from runtime.update_cycle import run_update_cycle
    state, _ = _fresh()
    pats = [_Pat("A", "bearish"), _Pat("B", "bullish")]
    with mock.patch("engine.pattern_engine.classify_patterns", return_value=pats), \
         mock.patch.object(si, "get_historical_context",
                           wraps=si.get_historical_context) as spy:
        run_update_cycle(state)
    snap = spy.call_args[0][0]
    assert snap["prediction"]    == "bearish"
    assert snap["pattern_names"] == ["A", "B"]


# ──────────────────────────────────────────────────────────────
# Non-fatal behaviour
# ──────────────────────────────────────────────────────────────

def test_calibration_failure_falls_back_to_structural():
    import runtime.update_cycle as uc
    state, _ = _fresh()
    with mock.patch("memory.statistical_intelligence.calibrate_confidence",
                    side_effect=RuntimeError("boom")), \
         mock.patch.object(uc.logger, "warning") as warn:
        assert uc.run_update_cycle(state) is True
    cs = state.confidence_state
    assert cs is not None
    assert cs.calibrated_confidence == cs.structural_confidence
    assert cs.calibration_result is None
    assert any("calibration" in str(c) for c in warn.call_args_list)


def test_pipeline_failure_sets_none_and_cycle_succeeds():
    import runtime.update_cycle as uc
    state, _ = _fresh()
    with mock.patch("engine.confidence_engine.compute_confidence",
                    side_effect=RuntimeError("boom")), \
         mock.patch.object(uc.logger, "warning") as warn:
        assert uc.run_update_cycle(state) is True
    assert state.confidence_state is None
    assert state.health.last_cycle_ok is True
    assert any("Confidence pipeline error" in str(c) for c in warn.call_args_list)


def test_failure_clears_previous_cycle_value():
    # A value from an earlier cycle must not survive a failed cycle.
    import runtime.update_cycle as uc
    state, _ = _cycled()
    assert state.confidence_state is not None
    with mock.patch("engine.confidence_engine.compute_confidence",
                    side_effect=RuntimeError("boom")):
        uc.run_update_cycle(state)
    assert state.confidence_state is None


def test_missing_structure_still_produces_state():
    import runtime.update_cycle as uc
    with mock.patch.object(uc, "_run_structure_engine", return_value=None):
        state, _ = _fresh()
        assert uc.run_update_cycle(state) is True
    assert state.structure_state is None
    assert state.confidence_state is not None


def test_pipeline_on_empty_state_never_raises():
    from runtime.update_cycle import _run_confidence_pipeline
    state, _ = _fresh()
    _run_confidence_pipeline(state)   # no market_state, no structure
    cs = state.confidence_state
    assert cs is not None
    assert cs.source_cycle      == 0
    assert cs.regime_confidence == 0.5   # neutral default when no regime


# ──────────────────────────────────────────────────────────────
# State packager — package["confidence"] (step 9)
# ──────────────────────────────────────────────────────────────

def _package(state):
    from ai.state_packager import build_full_package
    return build_full_package(
        state=state.market_state, structure_state=state.structure_state,
        confidence_state=state.confidence_state,
    ).to_dict()


def test_package_confidence_block_matches_state():
    state, _ = _cycled()
    cs  = state.confidence_state
    blk = _package(state)["confidence"]
    assert blk["structural"]   == round(cs.structural_confidence, 1)
    assert blk["calibrated"]   == round(cs.calibrated_confidence, 1)
    assert blk["regime"]       == round(cs.regime_confidence, 2)
    assert blk["source_cycle"] == cs.source_cycle
    assert blk["historical_adjustment"] == round(blk["calibrated"] - blk["structural"], 1)


def test_package_confidence_includes_calibration_explanation():
    state, _ = _cycled()
    cal = _package(state)["confidence"]["calibration"]
    assert cal["explanation"] == state.confidence_state.calibration_result.explanation
    for key in ("raw_adjustment", "conflict_dampened", "dampen_reason", "age_filtered_count"):
        assert key in cal


def test_package_omits_confidence_when_none():
    state, _ = _cycled()
    state.confidence_state = None
    assert "confidence" not in _package(state)


def test_package_without_calibration_result():
    state, _ = _cycled()
    state.confidence_state.calibration_result = None
    blk = _package(state)["confidence"]
    assert "calibration" not in blk
    assert blk["calibrated"] == round(state.confidence_state.calibrated_confidence, 1)


def test_commentary_cycle_passes_confidence_state():
    import runtime.commentary_cycle as cc
    state, _ = _cycled()
    with mock.patch.object(cc, "build_full_package",
                           side_effect=RuntimeError("stop here")) as bfp:
        cc.generate_commentary(state, trigger="user_request")
    assert bfp.call_args.kwargs["confidence_state"] is state.confidence_state


# ──────────────────────────────────────────────────────────────
# Claude client — mock commentary reads the stored value (step 10)
# ──────────────────────────────────────────────────────────────

def _built_package(state, request_type):
    from ai.state_packager import build_full_package
    return build_full_package(
        state=state.market_state, structure_state=state.structure_state,
        confidence_state=state.confidence_state, request_type=request_type,
    )


def test_mock_commentary_shows_stored_calibrated():
    from ai.claude_client import _generate_mock_response
    state, _ = _cycled()
    state.confidence_state.calibrated_confidence = 81.0
    for rt in ("startup_briefing", "market_commentary"):
        text = _generate_mock_response(rt, _built_package(state, rt))
        assert "Confidence: 81%" in text, rt


def test_startup_briefing_does_not_recalibrate():
    import memory.statistical_intelligence as si
    from ai.claude_client import _generate_mock_response
    from runtime.update_cycle import run_update_cycle
    state, tracker = _fresh()
    _seed_history(tracker, "Test Pattern", "bullish", "bullish")
    with mock.patch("engine.pattern_engine.classify_patterns",
                    return_value=[_Pat("Test Pattern", "bullish")]):
        run_update_cycle(state)
        pkg = _built_package(state, "startup_briefing")
        with mock.patch.object(si, "calibrate_confidence",
                               side_effect=AssertionError("recalibrated")), \
             mock.patch("engine.pattern_engine.build_pattern_narrative_from_package",
                        return_value=""):
            text = _generate_mock_response("startup_briefing", pkg)
    assert state.confidence_state.calibration_result.explanation in text


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nConfidence state tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)
