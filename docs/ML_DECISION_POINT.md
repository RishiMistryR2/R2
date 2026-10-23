# R2 — Machine Learning Decision Point

**Status:** Deferred. Revisit when the criteria below are met.
**Decided:** 2026-10-22
**Decision owner:** Rishi

## The decision

R2 currently uses hand-coded confidence weights, a hand-written pattern catalog
with estimated base rates, and deterministic statistical calibration. All
learning is counting outcomes — no model training.

This is deliberate. It keeps every number explainable. The decision is to
**keep it this way for the next several months of live operation**, collect
outcome data via the Outcome Tracker, and then decide whether to replace
hand-coded components with learned ones.

The long-term intent is to move toward a learned system. That intent may
change. This document exists so the decision gets made against evidence,
not against whatever mood the project is in at the time.

## When to open this document

Revisit when ALL of the following are true:

- [ ] At least 300 resolved observations in `~/.r2/outcome_tracker.json`
      (check: `python -c "from memory.outcome_tracker import get_tracker; print(get_tracker().resolved_count)"`)
- [ ] At least 30 distinct trading sessions have been run live
- [ ] At least 3 pattern types each have 30+ resolved observations
- [ ] The v2.7 confidence pipeline has run without a confidence-consistency
      bug being found for at least 20 sessions
- [ ] You have personally watched enough live sessions to have an opinion
      on whether the confidence numbers *feel* right

If any of these are false, the data is not ready. Keep collecting.

## Questions to answer with the data before deciding

These can all be answered from the Outcome Tracker JSON without building
anything new. Run them as a notebook or script. Do not change R2 to
answer them.

### 1. Are the hand-coded confidence weights predictive?

For every resolved observation, you have `confidence_score` (structural) and
`outcome`. Bucket by confidence and compute the directional hit rate per bucket.

- If 70–80% confidence observations resolve in the predicted direction ~75%
  of the time, and 40–50% ones resolve ~45%, the weights are calibrated.
  **Keep them.**
- If high-confidence observations resolve no better than low-confidence ones,
  the weights are not predictive. **Something is wrong with the evidence
  model, not just the numbers.** Investigate before learning anything.
- If the relationship is monotonic but off (e.g. 70% confidence resolves 55%
  of the time), the weights are directionally right but overconfident.
  **A fitted recalibration would help.**

### 2. Which evidence items actually matter?

Fit a logistic regression: outcome ~ structure_direction + regime_aligned +
session + trend_intact + has_vwap_reclaim + ... (one feature per evidence
item in `compute_confidence`). Compare the fitted coefficients to the
hand-coded point values.

- Coefficients that match the hand-coded signs and rough magnitudes
  confirm the intuition.
- Coefficients near zero mean that evidence item is noise. Remove it.
- Coefficients with the opposite sign mean the intuition was wrong.
  Investigate — could be a bug, could be a real finding.

### 3. Do the pattern base rates hold up?

`_CATALOG` in `engine/pattern_engine.py` has estimated resolution rates
(e.g. Failed Breakout → 72% bearish). Compare to observed rates from the
tracker via `get_tracker().all_stats()`.

- Within ±10 points → keep the estimate
- Off by more → update the catalog to observed rates
- A pattern with < 50% directional accuracy → consider removing it or
  reclassifying as neutral

### 4. Are there patterns the catalog doesn't name?

Cluster observations by (direction, market_state, regime, session) and look
for combinations with strong directional outcomes that don't map to a named
pattern. These are candidates for new catalog entries — still hand-written,
but discovered from data.

## The spectrum of "becoming ML"

In order of least to most change to the system's character:

**Level 0 — Recalibrate the hand-coded numbers from data.**
Update point values in `compute_confidence` and base rates in `_CATALOG`
to match observed rates. Still fully deterministic and explainable. No new
dependencies. This is almost certainly the right first step regardless of
what comes after.

**Level 1 — Fitted confidence weights.**
Replace hand-coded point values with coefficients from a logistic regression
fit on outcome data. Still explainable (a coefficient is a weight with a
reason), still deterministic at inference, but the weights now come from
evidence. Retrain monthly. Requires scikit-learn or equivalent.
Keep `compute_confidence` structure; swap the constants for a loaded model.

**Level 2 — Learned pattern discovery.**
Use clustering or decision trees on observation features to discover
patterns rather than hand-writing them. The catalog becomes partly
generated. Explainability drops — a decision tree is readable, a cluster
less so.

**Level 3 — LLM as reasoner, not just writer.**
Hand Claude the raw structure data and let it form the thesis, instead of
the deterministic engines doing it. Much more "modern AI." Also much less
deterministic, harder to test, and harder to trust for anything you'd
trade on. **Not recommended** for a system that makes trading decisions.
Keep the current split: deterministic reasoning, LLM explanation.

**Level 4 — Reinforcement learning on actions.**
Learn which actions (enter, hold, exit) produce the best outcomes. This is
a different project. R2 as currently built is an analyst, not an executor.

## Recommendation (as of 2026-10-22)

Go to Level 0 first, no matter what. Recalibrating hand-coded numbers from
observed data is pure upside and changes nothing about the architecture.

Then decide between staying at Level 0 or moving to Level 1 based on
question 1 above. If the hand-coded weights are already well-calibrated,
Level 1 adds complexity for little gain. If they're not, Level 1 is the
fix.

Do not go past Level 1 without a strong reason. The explainability is the
product.

## What is already in place to support this

- `memory/outcome_tracker.py` — records every significant event with
  features and resolves outcomes. This is the training dataset.
- `memory/statistical_intelligence.py` — already computes per-pattern,
  per-bucket, and per-direction statistics. Extend this before building
  anything new.
- `ConfidenceState.calibration_result` — carries sample size, raw and
  applied adjustment, and conflict dampening. Every calibration is already
  logged with its reasoning.
- The 30-day staleness filter means the dataset self-prunes to recent
  regime conditions.

## What to build when the time comes (sketch, not a plan)

- `scripts/analyze_outcomes.py` — loads the tracker JSON, answers
  questions 1–4 above, prints a report. Build this first. Run it monthly.
- If going to Level 1: `engine/confidence_model.py` — loads fitted
  coefficients from a JSON file, exposes the same `compute_confidence`
  signature. `update_cycle` chooses which to call via a config flag.
  Hand-coded version stays as the fallback.
- Tests comparing fitted and hand-coded outputs on the same inputs.

Nothing above should be started until the checklist at the top is complete.
