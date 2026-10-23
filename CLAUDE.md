# R2 — Real-time SPY Market Intelligence System

## What R2 is

R2 is an **explainable AI market intelligence platform**, not a price predictor.
It continuously observes live SPY market structure, forms evidence-based trading
theses, tracks their outcomes, learns statistically from its own history, and
explains every decision in plain language.

The system reasons the way a discretionary professional trader does:
observe structure → form a thesis → assess confidence → compare to history →
track outcome → learn. Every number has an explanation. Nothing is a black box.

**What R2 is NOT:** a signal generator, a machine-learning model, or an
auto-trader. All learning is deterministic statistics over recorded
observations. No external ML libraries. No trades are placed automatically.

## Architecture — the reasoning cycle

Every `--interval` seconds (default 30), `runtime/update_cycle.run_update_cycle()`
runs this pipeline in order:

```
Live Alpaca data        engine/alpaca_feed.py          → MarketState (price, VWAP, EMAs, RSI, ATR)
       ↓
Strategy enrichment     strategy/strategy_engine.py    → regime, HTF bias, setup score, narrative
       ↓
Structure engine        engine/structure_engine.py     → StructureState (HH/HL/LH/LL, market state, levels)
       ↓
Narrative engine        engine/narrative_engine.py     → trader-quality prose from structure
       ↓
Confidence engine       engine/confidence_engine.py    → evidence-based thesis score 15–90
                                                          (computed ONCE per cycle into
                                                          RuntimeState.confidence_state — see v2.7)
       ↓
Pattern engine          engine/pattern_engine.py       → named patterns with base-rate statistics
       ↓
Event intelligence      engine/event_engine.py         → "did anything materially change?" scoring
       ↓
Commentary              runtime/commentary_cycle.py    → AI commentary via ai/claude_client.py
       ↓
Outcome tracker         memory/outcome_tracker.py      → records events, resolves outcomes, MFE/MAE
       ↓
Statistical intelligence memory/statistical_intelligence.py → historical context, confidence calibration
```

### Module ownership (who computes what)

| Value | Computed by | Stored on |
|---|---|---|
| Price, VWAP, ATR, indicators | `engine/alpaca_feed.py`, `engine/indicator_engine.py` | `MarketState` |
| Market session (premarket/open/midday/power_hour/after_hours/closed) | `engine/market_session.py` | `MarketState.market_session` |
| Regime (chop/trend_bull/trend_bear) + regime confidence | `strategy/regime_engine.py` | `MarketState.regime` |
| HTF bias, setup score, entry signals | `strategy/strategy_engine.py` | `MarketState` |
| Structure (direction, sequence, levels, events) | `engine/structure_engine.py` | `RuntimeState.structure_state` |
| Structural thesis confidence | `engine/confidence_engine.py` via `runtime/update_cycle._run_confidence_pipeline()` | `RuntimeState.confidence_state` |
| Pattern classification | `engine/pattern_engine.py` | computed on demand |
| Event significance + priority | `engine/event_engine.py` | `MarketSnapshot` |
| Historical observations | `memory/outcome_tracker.py` | `~/.r2/outcome_tracker.json` |
| Calibrated confidence (Final Conviction) | `memory/statistical_intelligence.py` via `runtime/update_cycle._run_confidence_pipeline()` | `RuntimeState.confidence_state` |

### Distinct concepts — never collapse these into one number

- **regime_confidence** — the regime classifier's self-assessed probability (e.g. Trend Bear at 75%)
- **structural_confidence** — evidence-based thesis strength from structure, session, regime alignment
- **calibrated_confidence** — structural score adjusted by historical win rates (bounded ±15)
- **setup_quality** — how good the specific trade opportunity is (0–100, from `strategy/scorer.py`)
- **entry_readiness** — boolean action filter (session + regime + risk flags). NOT evidence about direction.

A market can be Trend Bear at 75% regime confidence, structural thesis 69%, and
entries disabled after hours. That is not a contradiction.

Outcome history (`PatternObservation.confidence_score`) stores **structural**
confidence, never calibrated — calibration buckets compare against it, so
storing calibrated values would feed past calibration back into future
calibration. The calibrated value is kept alongside in
`PatternObservation.calibrated_confidence` for audit only.

## Version history

| Version | What it added | Commit |
|---|---|---|
| Phases 1–8 | Core engines, strategy, context, memory, CLI, AI integration, runtime loop | `5b84589` and earlier |
| v1.1 | Live Alpaca data, session classification, dashboard | `f96b4b5` |
| v2.0A | Market structure engine (swing detection, market state, key levels, behavioral events) | `c220130` |
| v2.0B | Narrative engine — structure → trader prose | `d7d2585` |
| v2.1 | Key levels & confirmation/invalidation in narrative | *(in narrative_engine)* |
| v2.2 | Confidence engine — evidence-based thesis scoring | *(in confidence_engine)* |
| v2.3 | Pattern engine — named patterns with base rates | *(in pattern_engine)* |
| v2.4 | Event intelligence — materiality scoring, duplicate suppression | `cba69ad` |
| v2.5 | Outcome tracker — records events, resolves outcomes, strategy abstraction | *(in outcome_tracker)* |
| v2.6 | Statistical intelligence — historical context, confidence calibration | `a5588b1` |
| v2.7 | Single authoritative ConfidenceState (steps 1–15) | `4dd33c8` … `8fa99ab` |

## Current state (as of Oct 2026)

- **1390 tests** on the developer's machine; all pass except
  `test_package_market_session_differs_from_phase`, which is wall-clock
  dependent and fails outside trading hours — not a real bug
- Live Alpaca data works. Session transitions work. All engines integrated.
- Outcome tracker is recording and resolving observations in live sessions.
- **v2.7 is complete.** No next version has been approved yet.

## v2.7 — Single authoritative confidence (COMPLETE)

### What it solved
`compute_confidence()` used to be called independently by the event engine,
startup briefing and mock commentary, so the briefing, dashboard, commentary
and outcome records could show different numbers for the same market.

### How it works now
Once per update cycle, `runtime/update_cycle._run_confidence_pipeline(state)`
builds a `ConfidenceState` (in `engine/confidence_engine.py`) and stores it on
`RuntimeState.confidence_state`. It runs after `mark_update()` (so
`source_cycle == update_count`) and after outcome resolution (so calibration
sees outcomes resolved that tick). Every consumer reads the stored value:

| Consumer | Reads | Fallback when `confidence_state` is None |
|---|---|---|
| `event_engine.build_snapshot()` | `calibrated_confidence` | `compute_confidence()` (permanent) |
| `ai/state_packager` → `package["confidence"]` | all fields + calibration explanation | block omitted |
| `confidence_engine.build_thesis*/build_compact_thesis*` (mock commentary) | `package["confidence"]` | own structural computation |
| `ai/claude_client` startup briefing | stored calibration explanation | recalibrates (packages built outside the cycle) |
| `ai/prompt_builder` THESIS CONFIDENCE section (real Claude) | `package["confidence"]` | section omitted |
| `cli/dashboard` Final Conviction / Historical Adjustment | `ConfidenceState` | lines omitted |
| `memory/outcome_tracker` via `commentary_cycle` | **structural** (+ calibrated for audit) | snapshot score |

If the pipeline fails it logs a warning and sets `confidence_state = None`
(never leaves a value from an earlier cycle). If only calibration fails,
calibrated = structural.

### Decisions (approved)
- `entries_allowed` no longer affects `compute_confidence()` (parameter kept)
- Outcome history stores structural confidence, not calibrated (see "Distinct concepts")
- Statistical intelligence excludes observations older than 30 days and halves
  the adjustment when history's dominant outcome opposes current structure
  direction (bullish history vs downtrend, bearish vs uptrend);
  `CalibrationResult` exposes `raw_adjustment`, `conflict_dampened`,
  `dampen_reason`, `age_filtered_count`
- Thesis text shows the stored Final Conviction and spells out any
  historical adjustment; evidence lists still come from the structural evaluation
- Startup briefing commentary is a session-open snapshot, prefixed
  `[Generated HH:MM:SS ET · confidence from cycle N — session-open snapshot]`
- Old Recent Events keep their original confidence values — never rewritten

### Known follow-ups (not approved, not done)
- Promote `outcome_tracker._compute_stats` to public (imported privately by
  `statistical_intelligence._pattern_stats_from`)
- `statistical_intelligence.get_historical_context_from_package()` still calls
  `compute_confidence()` for its bucket lookup (equals structural; only used
  by the claude_client fallback path)
- The claude_client fallback recalibration does not pass `structure_direction`

Tests: `tests/test_confidence_state.py` (pipeline, packager, prompt, dashboard,
briefing stamp) and `tests/test_confidence_integration.py` (one shared value
across all consumers).

## Roadmap

See **[docs/ML_DECISION_POINT.md](docs/ML_DECISION_POINT.md)**.

- R2 uses **hand-coded weights by design**: confidence point values in
  `compute_confidence`, pattern base rates in `_CATALOG`, and deterministic
  calibration. This keeps every number explainable.
- The question of moving to machine learning is **deferred** until every item
  in the checklist at the top of that doc is met. Until then the job is to run
  live sessions and let the Outcome Tracker collect data.
- **Do not suggest adding ML before then.** That includes fitted or trained
  weights, scikit-learn or similar libraries, clustering, or using the LLM as
  the reasoner instead of the explainer. If asked, point to the doc and its
  checklist.

## Working conventions — follow these strictly

1. **Audit before editing.** Read the actual file before proposing any change.
   Never patch from memory of what a file "probably" looks like.
2. **Targeted patches only.** Use exact find/replace blocks. Never rewrite a
   whole file to change a few lines. Never use ellipses in patches.
3. **One file at a time.** Patch → run its tests → run full regression → commit → next file.
4. **Protected files — do not modify without explicit instruction:**
   - `engine/alpaca_feed.py` (live data — working, fragile)
   - `runtime/runtime_loop.py`, `runtime/run_r2.py`, `runtime/scheduler.py`
     (v2.7 added one approved line to `runtime_loop._render_dashboard`)
   - `cli/dashboard.py`
   - `strategy/*` and `context/*` (stable, not part of current work)
5. **Never break the test count.** All existing tests must pass before
   adding new ones. If a change legitimately invalidates a test assertion,
   update that test explicitly and say why.
6. **Non-fatal integration.** Anything added to `update_cycle.py` or
   `commentary_cycle.py` must be wrapped in `try/except` and log a warning on
   failure. Market processing never halts because an enrichment layer crashed.
7. **No ML, no new data sources, no external databases.** All learning is
   deterministic. Persistence is local JSON.
8. **Deterministic and explainable.** Every score, adjustment, and decision
   must carry a plain-English reason. If you can't explain why a number is
   what it is, it isn't done.
9. **Preserve backward compatibility.** New dataclass fields get defaults.
   Function signatures keep existing parameters even when behavior changes.

## Running R2

```bash
pip install -r requirements.txt

# Demo mode (synthetic data, mock AI) — safe, no keys needed
python -m runtime.run_r2

# Live data (needs ALPACA_API_KEY and ALPACA_SECRET_KEY in env)
python -m runtime.run_r2 --live

# Live data + real Claude commentary (also needs ANTHROPIC_API_KEY)
python -m runtime.run_r2 --live --no-mock-ai

# Other flags
--interval 30        # seconds between update cycles
--interactive        # blocking terminal REPL
--no-dashboard       # suppress dashboard
--log-level DEBUG    # verbose; shows [structure-debug] and [event] lines
```

## Running tests

```bash
PYTHONPATH=. python -m pytest tests/ -q          # full suite
PYTHONPATH=. python -m pytest tests/test_confidence_engine.py -v   # one module
PYTHONPATH=. python tests/test_event_engine.py   # standalone runner (no pytest needed)
```

Most test files have a `__main__` block so they run without pytest too.

## Key constants

- Confidence range: 15–90 (clamped)
- Calibration: min 10 observations, full weight at 30, max ±15 points
- Calibration staleness: observations older than 30 days excluded (`MAX_OBSERVATION_AGE_DAYS`)
- Conflict dampening: adjustment × 0.5 when history opposes structure (`CONFLICT_DAMPEN_FACTOR`)
- Event priority: critical ≥40, important 20–39, informational 5–19, suppressed <5
- Informational events rate-limited to one per 5 minutes
- Outcome tracker: rolling cap 500 observations, default timeout 120 min
- Session windows (ET): premarket 04:00–09:30, open 09:30–12:00, midday 12:00–15:00,
  power_hour 15:00–16:00, after_hours 16:00–20:00, closed otherwise

## Debugging live behavior

Set `--log-level DEBUG` and grep for:
- `[structure-debug]` — what structure data reached commentary
- `[event]` — suppression / approval decisions with scores
- `[outcome]` — observation recording and resolution
- `[commentary]` — generation and fallback events
- `[confidence]` — per-cycle structural/calibrated scores and calibration explanation

If the dashboard's Final Conviction differs from the startup briefing: expected.
The briefing is a session-open snapshot — check its `[Generated … cycle N]` line.

If commentary seems stale: that's usually event intelligence correctly
suppressing a non-material change. Check `[event] Suppressed` lines before
assuming a bug.

If regime says Chop while structure says downtrend: that's expected hysteresis.
Regime uses smoothed indicators and lags; structure uses raw swing points and
reacts in 1–2 bars. Both are correct on their own timescale.
