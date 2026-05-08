"""
memory/trade_memory.py — Phase 6 structured trade record.

TradeRecord is the canonical Phase 6 persistence format for completed trades.
Richer than the existing memory_state.TradeMemory (compact Phase-1 format).

Storage: data/memory/trades/<trade_id>.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.settings import MEMORY_DIR, MEMORY_SCHEMA_VERSION
from memory.persistence import list_files, load, save

TRADES_DIR    = Path(MEMORY_DIR) / "trades"
BREAKEVEN_PCT = 0.15   # ±0.15 % counts as breakeven

OUTCOME_WIN       = "win"
OUTCOME_LOSS      = "loss"
OUTCOME_BREAKEVEN = "breakeven"


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Persistent record of one completed trade."""

    trade_id:       str
    session_date:   str
    schema_version: str = MEMORY_SCHEMA_VERSION

    # Instrument
    symbol:          str = "SPY"
    instrument_type: str = "options"   # "options" | "shares"
    direction:       str = "calls"     # "calls" | "puts"

    # Levels
    entry_price:             float = 0.0
    exit_price:              float = 0.0
    underlying_entry_price:  float = 0.0
    underlying_stop_price:   float = 0.0
    underlying_target_price: float = 0.0
    option_entry_price:      Optional[float] = None

    # Timing
    entry_time:       str   = ""
    exit_time:        str   = ""
    duration_minutes: float = 0.0

    # Setup
    setup_type:  str   = ""
    setup_score: float = 0.0
    setup_grade: str   = ""

    # Market conditions at entry
    regime_at_entry:         str            = ""
    narrative_at_entry:      str            = ""
    htf_bias_at_entry:       str            = ""
    session_phase_at_entry:  str            = ""
    context_bias_at_entry:   str            = "neutral"
    vix_at_entry:            Optional[float] = None

    # Sizing
    position_size:    float = 0.0
    risk_amount:      float = 0.0
    sizing_compliant: bool  = True

    # Outcome
    pnl_pct:    float         = 0.0
    pnl_dollar: float         = 0.0
    outcome:    str           = OUTCOME_LOSS
    exit_reason: str          = ""
    stop_cause:  Optional[str] = None

    # Performance metrics
    mae_pct:      float = 0.0
    mfe_pct:      float = 0.0
    rr_planned:   float = 0.0
    rr_achieved:  float = 0.0

    # Stop triggers
    thesis_stop_triggered:  bool = False
    premium_stop_triggered: bool = False
    time_stop_triggered:    bool = False

    # Quality (set by ReflectionEngine)
    rule_compliant:    bool      = True
    mistake_tags:      List[str] = field(default_factory=list)
    execution_quality: str       = ""   # "good" | "acceptable" | "poor"
    outcome_quality:   str       = ""   # "good" | "bad"
    execution_notes:   str       = ""
    lesson:            str       = ""

    # ──────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────

    @classmethod
    def from_trade(
        cls,
        trade_id:      str,
        session_date:  str,
        direction:     str,
        instrument_type: str,
        entry_price:   float,
        exit_price:    float,
        entry_time:    str,
        exit_time:     str,
        stop_price:    float,
        target_price:  float,
        pnl_pct:       float,
        pnl_dollar:    float,
        exit_reason:   str,
        **kwargs,
    ) -> "TradeRecord":
        """Build a TradeRecord from trade-close primitives."""
        outcome = _classify_outcome(pnl_pct)

        duration = 0.0
        try:
            t1 = datetime.fromisoformat(entry_time)
            t2 = datetime.fromisoformat(exit_time)
            duration = round((t2 - t1).total_seconds() / 60, 1)
        except (ValueError, TypeError):
            pass

        stop_dist = abs(entry_price - stop_price)
        rr_achieved = 0.0
        if stop_dist > 0:
            rr_achieved = round(
                abs(pnl_pct) / (stop_dist / max(entry_price, 0.01) * 100), 2
            )

        r = cls(
            trade_id        = trade_id,
            session_date    = session_date,
            direction       = direction,
            instrument_type = instrument_type,
            entry_price     = entry_price,
            exit_price      = exit_price,
            underlying_entry_price  = entry_price,
            underlying_stop_price   = stop_price,
            underlying_target_price = target_price,
            entry_time      = entry_time,
            exit_time       = exit_time,
            duration_minutes = duration,
            pnl_pct         = round(pnl_pct, 3),
            pnl_dollar      = round(pnl_dollar, 2),
            outcome         = outcome,
            exit_reason     = exit_reason,
            rr_achieved     = rr_achieved,
        )
        # Apply keyword overrides
        for k, v in kwargs.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r

    # ──────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trade_id":       self.trade_id,
            "session_date":   self.session_date,
            "instrument": {
                "symbol":    self.symbol,
                "type":      self.instrument_type,
                "direction": self.direction,
            },
            "levels": {
                "entry":             self.entry_price,
                "exit":              self.exit_price,
                "underlying_entry":  self.underlying_entry_price,
                "underlying_stop":   self.underlying_stop_price,
                "underlying_target": self.underlying_target_price,
                "option_entry":      self.option_entry_price,
            },
            "timing": {
                "entry_time":       self.entry_time,
                "exit_time":        self.exit_time,
                "duration_minutes": self.duration_minutes,
            },
            "setup": {
                "type":  self.setup_type,
                "score": self.setup_score,
                "grade": self.setup_grade,
            },
            "conditions": {
                "regime":        self.regime_at_entry,
                "narrative":     self.narrative_at_entry,
                "htf_bias":      self.htf_bias_at_entry,
                "session_phase": self.session_phase_at_entry,
                "context_bias":  self.context_bias_at_entry,
                "vix":           self.vix_at_entry,
            },
            "sizing": {
                "position_size": self.position_size,
                "risk_amount":   self.risk_amount,
                "compliant":     self.sizing_compliant,
            },
            "outcome": {
                "pnl_pct":    self.pnl_pct,
                "pnl_dollar": self.pnl_dollar,
                "result":     self.outcome,
                "exit_reason": self.exit_reason,
                "stop_cause": self.stop_cause,
            },
            "performance": {
                "mae_pct":     self.mae_pct,
                "mfe_pct":     self.mfe_pct,
                "rr_planned":  self.rr_planned,
                "rr_achieved": self.rr_achieved,
            },
            "stop_flags": {
                "thesis":  self.thesis_stop_triggered,
                "premium": self.premium_stop_triggered,
                "time":    self.time_stop_triggered,
            },
            "quality": {
                "rule_compliant":    self.rule_compliant,
                "execution_quality": self.execution_quality,
                "outcome_quality":   self.outcome_quality,
                "mistake_tags":      self.mistake_tags,
                "execution_notes":   self.execution_notes,
                "lesson":            self.lesson,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeRecord":
        r = cls(
            trade_id     = data.get("trade_id", ""),
            session_date = data.get("session_date", ""),
        )
        r.schema_version = data.get("schema_version", MEMORY_SCHEMA_VERSION)

        i = data.get("instrument", {})
        r.symbol          = i.get("symbol", "SPY")
        r.instrument_type = i.get("type", "options")
        r.direction       = i.get("direction", "calls")

        l = data.get("levels", {})
        r.entry_price             = l.get("entry", 0.0)
        r.exit_price              = l.get("exit", 0.0)
        r.underlying_entry_price  = l.get("underlying_entry", 0.0)
        r.underlying_stop_price   = l.get("underlying_stop", 0.0)
        r.underlying_target_price = l.get("underlying_target", 0.0)
        r.option_entry_price      = l.get("option_entry")

        t = data.get("timing", {})
        r.entry_time       = t.get("entry_time", "")
        r.exit_time        = t.get("exit_time", "")
        r.duration_minutes = t.get("duration_minutes", 0.0)

        s = data.get("setup", {})
        r.setup_type  = s.get("type", "")
        r.setup_score = s.get("score", 0.0)
        r.setup_grade = s.get("grade", "")

        c = data.get("conditions", {})
        r.regime_at_entry        = c.get("regime", "")
        r.narrative_at_entry     = c.get("narrative", "")
        r.htf_bias_at_entry      = c.get("htf_bias", "")
        r.session_phase_at_entry = c.get("session_phase", "")
        r.context_bias_at_entry  = c.get("context_bias", "neutral")
        r.vix_at_entry           = c.get("vix")

        sz = data.get("sizing", {})
        r.position_size    = sz.get("position_size", 0.0)
        r.risk_amount      = sz.get("risk_amount", 0.0)
        r.sizing_compliant = sz.get("compliant", True)

        o = data.get("outcome", {})
        r.pnl_pct     = o.get("pnl_pct", 0.0)
        r.pnl_dollar  = o.get("pnl_dollar", 0.0)
        r.outcome     = o.get("result", OUTCOME_LOSS)
        r.exit_reason = o.get("exit_reason", "")
        r.stop_cause  = o.get("stop_cause")

        p = data.get("performance", {})
        r.mae_pct     = p.get("mae_pct", 0.0)
        r.mfe_pct     = p.get("mfe_pct", 0.0)
        r.rr_planned  = p.get("rr_planned", 0.0)
        r.rr_achieved = p.get("rr_achieved", 0.0)

        sf = data.get("stop_flags", {})
        r.thesis_stop_triggered  = sf.get("thesis", False)
        r.premium_stop_triggered = sf.get("premium", False)
        r.time_stop_triggered    = sf.get("time", False)

        q = data.get("quality", {})
        r.rule_compliant    = q.get("rule_compliant", True)
        r.execution_quality = q.get("execution_quality", "")
        r.outcome_quality   = q.get("outcome_quality", "")
        r.mistake_tags      = q.get("mistake_tags", [])
        r.execution_notes   = q.get("execution_notes", "")
        r.lesson            = q.get("lesson", "")
        return r

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self, directory: Path = TRADES_DIR) -> bool:
        return save(Path(directory) / f"{self.trade_id}.json", self.to_dict())

    @classmethod
    def load_by_id(
        cls,
        trade_id:  str,
        directory: Path = TRADES_DIR,
    ) -> Optional["TradeRecord"]:
        data = load(Path(directory) / f"{trade_id}.json")
        return cls.from_dict(data) if data else None

    @classmethod
    def load_all(cls, directory: Path = TRADES_DIR) -> List["TradeRecord"]:
        out = []
        for f in list_files(Path(directory)):
            data = load(f)
            if data and "trade_id" in data:
                try:
                    out.append(cls.from_dict(data))
                except Exception:
                    pass
        return out

    @classmethod
    def load_by_session(
        cls,
        session_date: str,
        directory:    Path = TRADES_DIR,
    ) -> List["TradeRecord"]:
        return [r for r in cls.load_all(directory) if r.session_date == session_date]


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _classify_outcome(pnl_pct: float) -> str:
    if pnl_pct > BREAKEVEN_PCT:
        return OUTCOME_WIN
    if pnl_pct < -BREAKEVEN_PCT:
        return OUTCOME_LOSS
    return OUTCOME_BREAKEVEN