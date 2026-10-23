"""
tests/test_confidence_integration.py — v2.7 end-to-end: one confidence value
shared by every consumer.

After one update cycle, the calibrated (Final Conviction) value stored on
RuntimeState.confidence_state must be what the event snapshot, the state
package, the startup briefing, market commentary and the dashboard all
show — and outcome history must record the structural value.

History is seeded so calibrated != structural; otherwise a consumer that
silently recomputed the structural score would still pass.

Run: PYTHONPATH=. python tests/test_confidence_integration.py
"""

import os, re, sys, traceback
from contextlib import contextmanager
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

class _Pat:
    def __init__(self, name, bias):
        self.name = name
        self.bias = bias


@contextmanager
def _pattern(name="Test Pattern", bias="bullish"):
    """Force one active pattern everywhere it is classified."""
    with mock.patch("engine.pattern_engine.classify_patterns",
                    return_value=[_Pat(name, bias)]), \
         mock.patch("engine.pattern_engine.build_pattern_narrative_from_package",
                    return_value=""):
        yield


def _seed(tracker, outcome="bullish", n=30, name="Test Pattern", bias="bullish"):
    for _ in range(n):
        oid = tracker.record_observation(
            {"confidence_score": 60.0, "current_price": 545.0},
            [{"name": name, "bias": bias}],
        )
        tracker.record_outcome(oid, outcome, resolution_minutes=40.0,
                               mfe_atr=1.0, mae_atr=0.3)


def _calibrated_state():
    """
    RuntimeState after one cycle with 30 winning bullish observations:
    calibrated is pushed above structural.  Must be used inside _pattern().
    """
    from memory.outcome_tracker import reset_tracker
    from runtime.runtime_state import RuntimeState
    from runtime.update_cycle import run_update_cycle, reset_engine
    tracker = reset_tracker(":memory:")
    reset_engine()
    _seed(tracker)
    state = RuntimeState(mock_ai=True)
    assert run_update_cycle(state) is True
    cs = state.confidence_state
    assert cs.calibrated_confidence != cs.structural_confidence, \
        "fixture must produce a historical adjustment"
    return state, tracker


def _package(state, request_type):
    from ai.state_packager import build_full_package
    return build_full_package(
        state=state.market_state, structure_state=state.structure_state,
        confidence_state=state.confidence_state, request_type=request_type,
    )


def _shown_pct(text):
    """Every 'Confidence: N%' value in a block of text."""
    return [int(m) for m in re.findall(r"Confidence: (\d+)%", text)]


def _approved():
    from engine.event_engine import EventIntelligenceResult
    return EventIntelligenceResult(
        priority="important", score=50.0, changes=[], suppress=False,
        change_summary="", trigger="regime_change",
    )


# ──────────────────────────────────────────────────────────────
# One shared value
# ──────────────────────────────────────────────────────────────

def test_all_consumers_show_the_same_value():
    from ai.claude_client import _generate_mock_response
    from cli.dashboard import render_dashboard
    from engine.event_engine import build_snapshot

    with _pattern():
        state, _ = _calibrated_state()
        final    = state.confidence_state.calibrated_confidence
        expected = round(final)

        snap      = build_snapshot(state, "regime_change")
        pkg_block = _package(state, "market_commentary").to_dict()["confidence"]
        briefing  = _generate_mock_response("startup_briefing",
                                            _package(state, "startup_briefing"))
        comment   = _generate_mock_response("market_commentary",
                                            _package(state, "market_commentary"))
        dash      = render_dashboard(state.market_state,
                                     confidence_state=state.confidence_state)

    assert snap.confidence_score    == final
    assert pkg_block["calibrated"]  == round(final, 1)
    assert _shown_pct(briefing)     == [expected]
    assert _shown_pct(comment)      == [expected]
    assert f"Final Conviction {expected}%" in dash


def test_adjustment_explained_consistently():
    from ai.claude_client import _generate_mock_response
    from cli.dashboard import render_dashboard

    with _pattern():
        state, _ = _calibrated_state()
        cs    = state.confidence_state
        delta = cs.calibrated_confidence - cs.structural_confidence

        pkg_block = _package(state, "startup_briefing").to_dict()["confidence"]
        briefing  = _generate_mock_response("startup_briefing",
                                            _package(state, "startup_briefing"))
        dash      = render_dashboard(state.market_state, confidence_state=cs)

    assert pkg_block["historical_adjustment"] == round(
        pkg_block["calibrated"] - pkg_block["structural"], 1)
    assert f"historical adjustment {delta:+.0f}" in briefing.lower()
    assert f"Historical Adjustment {delta:+.0f} pts" in dash
    assert cs.calibration_result.explanation in briefing


def test_live_commentary_path_uses_stored_value():
    import runtime.commentary_cycle as cc
    with _pattern(), \
         mock.patch("engine.event_engine.evaluate_event", return_value=_approved()):
        state, _ = _calibrated_state()
        text = cc.generate_commentary(state, trigger="regime_change")
    assert text is not None
    assert _shown_pct(text) == [round(state.confidence_state.calibrated_confidence)]


def test_outcome_history_records_structural():
    import runtime.commentary_cycle as cc
    with _pattern(), \
         mock.patch("engine.event_engine.evaluate_event", return_value=_approved()):
        state, tracker = _calibrated_state()
        before = {o.obs_id for o in tracker.recent_observations(n=500)}
        cc.generate_commentary(state, trigger="regime_change")
    new = [o for o in tracker.recent_observations(n=500) if o.obs_id not in before]
    cs  = state.confidence_state
    assert len(new) == 1
    assert new[0].confidence_score      == cs.structural_confidence
    assert new[0].calibrated_confidence == cs.calibrated_confidence


def test_consumers_follow_latest_cycle():
    from engine.event_engine import build_snapshot
    from runtime.update_cycle import run_update_cycle
    with _pattern():
        state, tracker = _calibrated_state()
        first = state.confidence_state.calibrated_confidence
        # Flip history to losing: next cycle's calibration must drop.
        _seed(tracker, outcome="bearish", n=60)
        run_update_cycle(state)
        second = state.confidence_state.calibrated_confidence
        snap   = build_snapshot(state, "regime_change")
        block  = _package(state, "market_commentary").to_dict()["confidence"]
    assert second < first
    assert snap.confidence_score == second
    assert block["calibrated"]   == round(second, 1)
    assert block["source_cycle"] == state.update_count == 2


def test_entry_permission_never_changes_shared_value():
    # Demo state is outside trading hours (entries disabled).  Forcing
    # entries on must not move any consumer's number.
    from engine.event_engine import build_snapshot
    with _pattern():
        state, _ = _calibrated_state()
        before = build_snapshot(state, "regime_change").confidence_score
        state.market_state.session.entries_allowed = True
        after  = build_snapshot(state, "regime_change").confidence_score
    assert before == after == state.confidence_state.calibrated_confidence


def test_real_ai_prompts_carry_the_same_value():
    from ai import prompt_builder as pb
    with _pattern():
        state, _ = _calibrated_state()
        expected = round(state.confidence_state.calibrated_confidence)
        prompts = [
            pb.build_startup_briefing_prompt(_package(state, "startup_briefing")),
            pb.build_market_commentary_prompt(_package(state, "market_commentary")),
        ]
    for _, user in prompts:
        assert f"Final conviction: {expected}%" in user


# ──────────────────────────────────────────────────────────────
# Fallback stays consistent
# ──────────────────────────────────────────────────────────────

def test_fallback_consumers_agree_when_no_stored_value():
    # If the pipeline failed, every consumer falls back to the same
    # structural computation rather than disagreeing with each other.
    from ai.claude_client import _generate_mock_response
    from cli.dashboard import render_dashboard
    from engine.event_engine import build_snapshot
    with _pattern():
        state, _ = _calibrated_state()
        structural = state.confidence_state.structural_confidence
        state.confidence_state = None
        snap    = build_snapshot(state, "regime_change")
        comment = _generate_mock_response("market_commentary",
                                          _package(state, "market_commentary"))
        dash    = render_dashboard(state.market_state, confidence_state=None)
    assert snap.confidence_score == structural
    assert _shown_pct(comment)   == [round(structural)]
    assert "Final Conviction" not in dash


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    print(f"\nConfidence integration tests ({len(tests)} total)\n{'─' * 60}")
    for name, fn in tests:
        run(name, fn)
    print(f"\n{'=' * 60}")
    print(f"{passed} passed, {failed} failed  ({passed + failed} total)")
    import sys; sys.exit(0 if failed == 0 else 1)
