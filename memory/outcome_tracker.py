"""
memory/outcome_tracker.py — R2 Outcome Tracker & Historical Pattern Memory v2.5

Records meaningful market events and tracks their eventual outcomes.
Deterministic record-keeping — no ML, no external databases.

This module is intentionally separate from memory/pattern_memory.py, which
tracks user behaviour patterns and trade performance.  This module tracks
structural market patterns and their price resolution.

Architecture
------------
ResolutionStrategy (ABC)
    Decouples resolution logic from the tracker.  Every strategy implements
    a single method:
        resolve(obs, current_price, atr, elapsed_minutes) -> str | None
    Returns an outcome string ("bullish" | "bearish" | "inconclusive") when
    the observation has resolved, or None when it is still pending.

    Bundled strategies:
      SessionBreakStrategy (default)
          price >= session_high  → "bullish"
          price <= session_low   → "bearish"
          elapsed >= timeout     → "inconclusive"
      ATRTargetStrategy
          abs(current_price - entry) / atr >= target_atr → directional
          elapsed >= timeout                              → "inconclusive"
      TimeExpirationStrategy
          elapsed >= expiry → direction from final price vs entry
      StructureContinuationStrategy
          Skeleton for v2.6+.  Currently defers to session-break timeout.

PatternObservation
    One recorded event.  Stores all context at time of recording, running
    price trackers (price_high_seen, price_low_seen), and outcome fields
    that are filled in when the observation resolves.

PatternStats
    Aggregated statistics for one pattern name across all resolved
    observations.  Exposes bullish_rate, bearish_rate, prediction_accuracy.

OutcomeTracker
    Container with load/save/query/resolve interface.
    Delegates resolution to a ResolutionStrategy instance.

Persistence
-----------
  Default: ~/.r2/outcome_tracker.json  (configurable via path arg)
  ":memory:" → in-memory only (no file I/O — for tests)
  Writes are atomic: write to .tmp, then os.replace().
  Rolling cap: MAX_OBSERVATIONS (500) — oldest resolved evicted first,
  then oldest pending.

Singleton
---------
  get_tracker(path=None)           → global OutcomeTracker singleton
  reset_tracker(path=":memory:")   → fresh instance, replaces singleton
                                     (use in tests)

Public API
----------
  tracker.record_observation(snapshot, patterns, session, symbol) → obs_id
  tracker.update_pending(current_price, atr=None)                 → int
  tracker.record_outcome(obs_id, outcome, ...)                    → bool
  tracker.get_observation(obs_id)         → PatternObservation | None
  tracker.pending_observations()          → list[PatternObservation]
  tracker.recent_observations(n=20)       → list[PatternObservation]
  tracker.get_stats(pattern_name)         → PatternStats | None
  tracker.all_stats()                     → dict[str, PatternStats]
  tracker.similar_patterns(snapshot)      → list[PatternStats]
  tracker.summary_text(pattern_name)      → str
  tracker.save() / tracker.load()
  tracker.observation_count               → int (property)
  tracker.resolved_count                  → int (property)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory.outcome")

_DEFAULT_PATH     = os.path.expanduser("~/.r2/outcome_tracker.json")
_MAX_OBSERVATIONS = 500
_DEFAULT_TIMEOUT       = 120.0   # minutes
_MIN_ELAPSED_MINUTES   = 0.001   # 60 ms floor — ensures tiny timeout sentinels
                                  # (e.g. 0.0001 min) resolve deterministically
                                  # in tests without wall-clock sleeps


# ══════════════════════════════════════════════════════════════
# Resolution strategies
# ══════════════════════════════════════════════════════════════

class ResolutionStrategy(ABC):
    """
    Abstract base for outcome resolution strategies.

    Implementations decide when an observation has resolved and what the
    outcome is.  The tracker calls resolve() on every update_pending() tick.
    """

    @abstractmethod
    def resolve(
        self,
        obs:             "PatternObservation",
        current_price:   float,
        atr:             Optional[float],
        elapsed_minutes: float,
    ) -> Optional[str]:
        """
        Evaluate whether the observation has resolved.

        Parameters
        ----------
        obs              : the observation being evaluated
        current_price    : current market price
        atr              : current 5-minute ATR (may be None)
        elapsed_minutes  : minutes since the observation was recorded

        Returns
        -------
        "bullish"       — observation resolved in favour of buyers
        "bearish"       — observation resolved in favour of sellers
        "inconclusive"  — timeout or no clear resolution
        None            — not yet resolved; check again next tick
        """


class SessionBreakStrategy(ResolutionStrategy):
    """
    Default resolution strategy.

    Resolves when price crosses a key session level or timeout elapses.

      price >= session_high  →  "bullish"
      price <= session_low   →  "bearish"
      elapsed >= timeout     →  "inconclusive"
    """

    def __init__(self, timeout_minutes: float = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout_minutes

    def resolve(self, obs, current_price, atr, elapsed_minutes):
        if elapsed_minutes >= self.timeout:
            return "inconclusive"
        if obs.session_high is not None and current_price >= obs.session_high:
            return "bullish"
        if obs.session_low is not None and current_price <= obs.session_low:
            return "bearish"
        return None


class ATRTargetStrategy(ResolutionStrategy):
    """
    Resolves when price moves at least target_atr × ATR from the entry price.

    Direction is determined by which way price moved.
    Falls back to inconclusive on timeout.
    """

    def __init__(
        self,
        target_atr:      float = 1.0,
        timeout_minutes: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.target_atr = target_atr
        self.timeout    = timeout_minutes

    def resolve(self, obs, current_price, atr, elapsed_minutes):
        if elapsed_minutes >= self.timeout:
            return "inconclusive"

        effective_atr = atr or obs.atr_at_record
        if not effective_atr or not obs.price_at_record or effective_atr <= 0:
            return None

        move      = current_price - obs.price_at_record
        atr_ratio = abs(move) / effective_atr

        if atr_ratio >= self.target_atr - 1e-9:
            return "bullish" if move > 0 else "bearish"
        return None


class TimeExpirationStrategy(ResolutionStrategy):
    """
    Resolves only after a fixed time window.

    Direction is determined by comparing final price to entry price.
    Useful for time-based studies of pattern resolution.
    """

    def __init__(self, expiry_minutes: float = _DEFAULT_TIMEOUT) -> None:
        self.expiry = expiry_minutes

    def resolve(self, obs, current_price, atr, elapsed_minutes):
        if elapsed_minutes < self.expiry:
            return None

        if obs.price_at_record is None:
            return "inconclusive"

        if current_price > obs.price_at_record:
            return "bullish"
        if current_price < obs.price_at_record:
            return "bearish"
        return "inconclusive"


class StructureContinuationStrategy(ResolutionStrategy):
    """
    Resolves when structural state diverges from the state at recording time.

    v2.5 skeleton — currently behaves like SessionBreakStrategy until the
    runtime passes current market_state to update_pending() in v2.6+.
    """

    def __init__(self, timeout_minutes: float = _DEFAULT_TIMEOUT) -> None:
        self.timeout   = timeout_minutes
        self._delegate = SessionBreakStrategy(timeout_minutes)

    def resolve(self, obs, current_price, atr, elapsed_minutes):
        # TODO v2.6: compare obs.market_state to current_state arg
        return self._delegate.resolve(obs, current_price, atr, elapsed_minutes)


# ══════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════

@dataclass
class PatternObservation:
    """
    A single recorded structural event.

    Fields above the blank line are set at recording time and never change.
    Fields below (resolved, outcome, mfe_atr, mae_atr, …) are populated
    when the observation resolves.
    """

    # ── Identity ──────────────────────────────────────────────
    obs_id:      str    # 12-char UUID fragment
    recorded_at: str    # ISO 8601 UTC
    symbol:      str
    session:     str    # "open" | "after_hours" | "midday" | …
    trigger:     str    # commentary trigger that caused this recording

    # ── Structure at recording time ───────────────────────────
    regime_name:      str
    direction:        str    # "uptrend" | "downtrend" | "ranging" | ""
    market_state:     str    # "trend" | "pullback" | "range" | …
    sequence:         str    # "HH HL HH HL" | "LH LL LH LL" | ""
    confidence_score: float  # v2.7: structural thesis score — calibration
                             # buckets compare against this, so it must
                             # never include a historical adjustment
    pattern_names:    list   # list[str] — all active patterns at recording
    prediction:       str    # "bullish" | "bearish" | "neutral"

    # ── Key levels (resolution anchors) ──────────────────────
    session_high:    Optional[float]
    session_low:     Optional[float]
    price_at_record: Optional[float]
    atr_at_record:   Optional[float]

    # ── Outcome (populated by update_pending / record_outcome) ─
    resolved:           bool           = False
    outcome:            Optional[str]  = None   # bullish | bearish | inconclusive
    resolution_minutes: Optional[float] = None
    resolution_price:   Optional[float] = None
    mfe_atr:            Optional[float] = None
    mae_atr:            Optional[float] = None

    # ── Running price trackers ────────────────────────────────
    price_high_seen: Optional[float] = None
    price_low_seen:  Optional[float] = None

    # ── v2.7: Final Conviction at record time (audit/display only) ──
    calibrated_confidence: Optional[float] = None

    @property
    def elapsed_minutes(self) -> float:
        try:
            rec = datetime.fromisoformat(self.recorded_at)
            raw = (datetime.now(tz=timezone.utc) - rec).total_seconds() / 60.0
            return max(raw, _MIN_ELAPSED_MINUTES)
        except Exception:
            return _MIN_ELAPSED_MINUTES

    @property
    def prediction_correct(self) -> Optional[bool]:
        """True/False for directional predictions that resolved conclusively."""
        if self.outcome is None or self.outcome == "inconclusive":
            return None
        if self.prediction == "neutral":
            return None
        return self.prediction == self.outcome


@dataclass
class PatternStats:
    """Aggregated outcome statistics for one pattern name."""

    pattern_name:           str
    occurrences:            int    # total obs (including pending)
    resolved_count:         int
    bullish_outcomes:       int
    bearish_outcomes:       int
    inconclusive_count:     int
    correct_predictions:    int
    total_directional:      int    # non-neutral preds resolved conclusively
    avg_resolution_minutes: float
    avg_mfe_atr:            float
    avg_mae_atr:            float
    avg_confidence:         float

    @property
    def bullish_rate(self) -> float:
        return self.bullish_outcomes / self.resolved_count if self.resolved_count else 0.5

    @property
    def bearish_rate(self) -> float:
        return self.bearish_outcomes / self.resolved_count if self.resolved_count else 0.5

    @property
    def prediction_accuracy(self) -> float:
        return (self.correct_predictions / self.total_directional
                if self.total_directional else 0.5)

    def __str__(self) -> str:
        lines = [
            self.pattern_name,
            f"  Occurrences: {self.occurrences}  "
            f"Resolved: {self.resolved_count}",
            f"  Bullish: {self.bullish_outcomes}  "
            f"Bearish: {self.bearish_outcomes}  "
            f"Inconclusive: {self.inconclusive_count}",
            f"  Average resolution time: {self.avg_resolution_minutes:.0f} min  "
            f"Average move: {self.avg_mfe_atr:.1f} ATR  "
            f"Average adverse: {self.avg_mae_atr:.1f} ATR",
            f"  Average confidence: {self.avg_confidence:.0f}",
        ]
        if self.total_directional > 0:
            lines.append(
                f"  Prediction accuracy: {self.prediction_accuracy:.0%} "
                f"({self.total_directional} directional)"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# OutcomeTracker
# ══════════════════════════════════════════════════════════════

class OutcomeTracker:
    """
    Central store for PatternObservation objects.

    All resolution logic is delegated to a ResolutionStrategy.
    Persistence is via an atomic JSON write to a configurable file path.
    Use ":memory:" for in-process testing with no file I/O.
    """

    def __init__(
        self,
        path:             str                = _DEFAULT_PATH,
        max_observations: int                = _MAX_OBSERVATIONS,
        strategy:         ResolutionStrategy = None,
    ) -> None:
        self._path     = None if path == ":memory:" else os.path.expanduser(path)
        self._max      = max_observations
        self._strategy = strategy or SessionBreakStrategy()
        self._obs:     dict[str, PatternObservation] = {}
        self._dirty    = False

        if self._path:
            self.load()

    # ── Properties ────────────────────────────────────────────

    @property
    def observation_count(self) -> int:
        return len(self._obs)

    @property
    def resolved_count(self) -> int:
        return sum(1 for o in self._obs.values() if o.resolved)

    # ── Recording ─────────────────────────────────────────────

    def record_observation(
        self,
        snapshot,           # MarketSnapshot (duck-typed) or dict
        patterns:  list,    # list[PatternMatch] or list[dict]
        session:   str = "",
        symbol:    str = "SPY",
        structural_confidence: Optional[float] = None,
        calibrated_confidence: Optional[float] = None,
    ) -> str:
        """
        Record a new structural observation.

        snapshot may be any object with the relevant attributes (e.g. a
        MarketSnapshot from event_engine) or a plain dict.
        patterns may be PatternMatch objects (from pattern_engine) or plain
        dicts with a 'name' and 'bias' key.

        v2.7: structural_confidence, when given, is stored as
        confidence_score in place of the snapshot's value (the snapshot
        carries the calibrated Final Conviction).  Recording the calibrated
        value would feed past calibration back into future calibration.
        calibrated_confidence is stored alongside for audit.

        Returns the new observation ID.
        """
        def _g(attr: str, default=None):
            if isinstance(snapshot, dict):
                return snapshot.get(attr, default)
            return getattr(snapshot, attr, default)

        # Pattern names and prediction from primary pattern's bias
        pattern_names: list[str] = []
        prediction = "neutral"
        for p in (patterns or []):
            name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
            if name:
                pattern_names.append(name)
        if patterns:
            first = patterns[0]
            bias  = (first.get("bias") if isinstance(first, dict)
                     else getattr(first, "bias", "neutral"))
            prediction = bias or "neutral"

        p0 = _g("current_price")

        conf = (structural_confidence if structural_confidence is not None
                else _g("confidence_score", 35.0))

        obs = PatternObservation(
            obs_id           = str(uuid.uuid4())[:12],
            recorded_at      = datetime.now(tz=timezone.utc).isoformat(),
            symbol           = symbol,
            session          = session,
            trigger          = str(_g("trigger", "")),
            regime_name      = str(_g("regime_name", "")),
            direction        = str(_g("direction", "")),
            market_state     = str(_g("market_state", "")),
            sequence         = str(_g("sequence", "")),
            confidence_score = float(conf or 35.0),
            pattern_names    = pattern_names,
            prediction       = prediction,
            session_high     = _g("session_high"),
            session_low      = _g("session_low"),
            price_at_record  = p0,
            atr_at_record    = _g("atr_5m"),
            price_high_seen  = p0,
            price_low_seen   = p0,
            calibrated_confidence = (float(calibrated_confidence)
                                     if calibrated_confidence is not None
                                     else None),
        )

        self._obs[obs.obs_id] = obs
        self._trim()
        self._dirty = True
        self._autosave()

        logger.debug(
            f"[outcome] Recorded {obs.obs_id}  patterns={pattern_names}  "
            f"prediction={prediction}  session={session}  price={p0}"
        )
        return obs.obs_id

    def record_outcome(
        self,
        obs_id:             str,
        outcome:            str,
        resolution_price:   Optional[float] = None,
        resolution_minutes: Optional[float] = None,
        mfe_atr:            Optional[float] = None,
        mae_atr:            Optional[float] = None,
    ) -> bool:
        """
        Manually attach an outcome to a pending observation.
        Returns True if found; False if obs_id does not exist.
        """
        obs = self._obs.get(obs_id)
        if obs is None:
            return False

        obs.resolved           = True
        obs.outcome            = outcome
        obs.resolution_price   = resolution_price
        obs.resolution_minutes = (resolution_minutes
                                  if resolution_minutes is not None
                                  else obs.elapsed_minutes)
        if mfe_atr is not None:
            obs.mfe_atr = mfe_atr
        if mae_atr is not None:
            obs.mae_atr = mae_atr

        self._dirty = True
        self._autosave()
        logger.debug(f"[outcome] Manual outcome {obs_id}: {outcome}")
        return True

    # ── Auto-resolution ───────────────────────────────────────

    def update_pending(
        self,
        current_price: float,
        atr:           Optional[float] = None,
    ) -> int:
        """
        Evaluate all unresolved observations.

        For each pending observation:
          1. Update price_high_seen / price_low_seen trackers.
          2. Ask the strategy whether the observation has resolved.
          3. If resolved: compute MFE/MAE and mark resolved.

        Returns the number of newly resolved observations.
        """
        newly = 0

        for obs in self._obs.values():
            if obs.resolved:
                continue

            # Update running trackers
            if obs.price_high_seen is None or current_price > obs.price_high_seen:
                obs.price_high_seen = current_price
                self._dirty = True
            if obs.price_low_seen is None or current_price < obs.price_low_seen:
                obs.price_low_seen  = current_price
                self._dirty = True

            elapsed = obs.elapsed_minutes
            outcome = self._strategy.resolve(obs, current_price, atr, elapsed)

            if outcome is not None:
                self._resolve(obs, outcome, current_price, elapsed, atr)
                newly += 1

        if newly:
            self._autosave()
        return newly

    def _resolve(
        self,
        obs:              PatternObservation,
        outcome:          str,
        resolution_price: float,
        elapsed_minutes:  float,
        atr:              Optional[float],
    ) -> None:
        obs.resolved           = True
        obs.outcome            = outcome
        obs.resolution_price   = resolution_price
        obs.resolution_minutes = elapsed_minutes
        self._dirty            = True

        # Compute MFE / MAE in ATR units
        effective_atr = atr or obs.atr_at_record
        if (effective_atr and effective_atr > 0
                and obs.price_at_record is not None):
            mfe, mae = _compute_mfe_mae(obs, effective_atr)
            obs.mfe_atr = mfe
            obs.mae_atr = mae

        logger.debug(
            f"[outcome] Resolved {obs.obs_id}  outcome={outcome}  "
            f"elapsed={elapsed_minutes:.1f}min  "
            f"mfe={obs.mfe_atr}  mae={obs.mae_atr}"
        )

    # ── Queries ───────────────────────────────────────────────

    def get_observation(self, obs_id: str) -> Optional[PatternObservation]:
        return self._obs.get(obs_id)

    def pending_observations(self) -> list:
        """Unresolved observations, newest first."""
        return sorted(
            (o for o in self._obs.values() if not o.resolved),
            key=lambda o: o.recorded_at, reverse=True,
        )

    def recent_observations(self, n: int = 20) -> list:
        """Most recent N observations (any status), newest first."""
        return sorted(
            self._obs.values(), key=lambda o: o.recorded_at, reverse=True
        )[:n]

    def get_stats(self, pattern_name: str) -> Optional[PatternStats]:
        """Aggregated stats for one pattern name; None if no resolved obs."""
        resolved = [
            o for o in self._obs.values()
            if pattern_name in o.pattern_names and o.resolved
        ]
        if not resolved:
            return None
        total = sum(
            1 for o in self._obs.values() if pattern_name in o.pattern_names
        )
        return _compute_stats(pattern_name, resolved, total)

    def all_stats(self) -> dict:
        """Stats for every pattern name that has at least one resolved obs."""
        names: set = set()
        for obs in self._obs.values():
            names.update(obs.pattern_names)
        return {
            n: s
            for n in sorted(names)
            for s in [self.get_stats(n)]
            if s is not None
        }

    def similar_patterns(self, snapshot) -> list:
        """
        Historical stats for patterns matching the snapshot's pattern_names.
        Ordered by occurrence count descending.

        snapshot may be a MarketSnapshot, a dict with a 'pattern_names' key,
        or any object with a pattern_names attribute.
        """
        if isinstance(snapshot, dict):
            names = list(snapshot.get("pattern_names", []))
        else:
            names = list(getattr(snapshot, "pattern_names", ()))

        stats = [s for n in names for s in [self.get_stats(n)] if s is not None]
        return sorted(stats, key=lambda s: s.occurrences, reverse=True)

    def summary_text(self, pattern_name: str) -> str:
        """Human-readable stats block for one pattern."""
        s = self.get_stats(pattern_name)
        return str(s) if s else f"{pattern_name}: no resolved observations yet."

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Atomic write to disk.  No-op for :memory: instances."""
        if self._path is None:
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            data = {
                "version":      1,
                "observations": {oid: asdict(o) for oid, o in self._obs.items()},
            }
            tmp = self._path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp, self._path)
            self._dirty = False
            logger.debug(f"[outcome] Saved {len(self._obs)} obs → {self._path}")
        except Exception as exc:
            logger.warning(f"[outcome] Save failed (non-fatal): {exc}")

    def load(self) -> None:
        """Load from disk.  No-op when file does not exist."""
        if self._path is None or not os.path.exists(self._path):
            return
        try:
            with open(self._path) as fh:
                data = json.load(fh)
            raw = data.get("observations") or {}
            self._obs = {oid: _dict_to_obs(d) for oid, d in raw.items()}
            logger.debug(f"[outcome] Loaded {len(self._obs)} obs ← {self._path}")
        except Exception as exc:
            logger.warning(f"[outcome] Load failed (non-fatal): {exc}")
            self._obs = {}

    # ── Internals ─────────────────────────────────────────────

    def _autosave(self) -> None:
        if self._dirty and self._path:
            self.save()

    def _trim(self) -> None:
        """Evict oldest observations when cap is exceeded."""
        if len(self._obs) <= self._max:
            return
        # resolved before pending (oldest first within each group)
        ordered = sorted(
            self._obs.items(),
            key=lambda t: (0 if t[1].resolved else 1, t[1].recorded_at),
        )
        for oid, _ in ordered[: len(ordered) - self._max]:
            del self._obs[oid]


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _compute_mfe_mae(
    obs: PatternObservation,
    effective_atr: float,
) -> tuple:
    """Return (mfe_atr, mae_atr) tuple.  Both may be 0.0."""
    p0 = obs.price_at_record   # guaranteed not None by caller
    ph = obs.price_high_seen if obs.price_high_seen is not None else p0
    pl = obs.price_low_seen  if obs.price_low_seen  is not None else p0

    if obs.prediction == "bullish":
        mfe_pts = max(ph - p0, 0.0)
        mae_pts = max(p0 - pl, 0.0)
    elif obs.prediction == "bearish":
        mfe_pts = max(p0 - pl, 0.0)
        mae_pts = max(ph - p0, 0.0)
    else:   # neutral
        mfe_pts = max(ph - p0, p0 - pl, 0.0)
        mae_pts = min(abs(ph - p0), abs(p0 - pl), mfe_pts)

    return (
        round(mfe_pts / effective_atr, 2),
        round(mae_pts / effective_atr, 2),
    )


def _compute_stats(
    pattern_name: str,
    resolved:     list,
    total:        int,
) -> PatternStats:
    bullish = sum(1 for o in resolved if o.outcome == "bullish")
    bearish = sum(1 for o in resolved if o.outcome == "bearish")
    inconc  = sum(1 for o in resolved if o.outcome == "inconclusive")

    directional = [
        o for o in resolved
        if o.prediction != "neutral" and o.outcome not in (None, "inconclusive")
    ]
    correct = sum(1 for o in directional if o.prediction == o.outcome)

    res_times   = [o.resolution_minutes  for o in resolved if o.resolution_minutes  is not None]
    mfes        = [o.mfe_atr             for o in resolved if o.mfe_atr             is not None]
    maes        = [o.mae_atr             for o in resolved if o.mae_atr             is not None]
    confidences = [o.confidence_score    for o in resolved]

    def _avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0

    return PatternStats(
        pattern_name           = pattern_name,
        occurrences            = total,
        resolved_count         = len(resolved),
        bullish_outcomes       = bullish,
        bearish_outcomes       = bearish,
        inconclusive_count     = inconc,
        correct_predictions    = correct,
        total_directional      = len(directional),
        avg_resolution_minutes = _avg(res_times),
        avg_mfe_atr            = _avg(mfes),
        avg_mae_atr            = _avg(maes),
        avg_confidence         = _avg(confidences),
    )


def _dict_to_obs(d: dict) -> PatternObservation:
    defaults = {
        "resolved": False, "outcome": None,
        "resolution_minutes": None, "resolution_price": None,
        "mfe_atr": None, "mae_atr": None,
        "price_high_seen": None, "price_low_seen": None,
    }
    merged = {**defaults, **d}
    known  = set(PatternObservation.__dataclass_fields__)
    return PatternObservation(**{k: v for k, v in merged.items() if k in known})


# ══════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════

_tracker: Optional[OutcomeTracker] = None


def get_tracker(path: Optional[str] = None) -> OutcomeTracker:
    """Return the global OutcomeTracker singleton, creating it on first call."""
    global _tracker
    if _tracker is None:
        _tracker = OutcomeTracker(path or _DEFAULT_PATH)
    return _tracker


def reset_tracker(path: str = ":memory:") -> OutcomeTracker:
    """Reset the singleton and return a fresh instance.  Use in tests."""
    global _tracker
    _tracker = OutcomeTracker(path)
    return _tracker