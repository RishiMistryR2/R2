"""
memory/session_memory.py — one trading session record.

Captures everything meaningful about a single trading day:
  - Market summary (regime, narrative, key levels)
  - Trades taken (IDs + running P&L)
  - Regime changes
  - Context / news highlights
  - Behavioral flags
  - P&L and drawdown
  - User notes / lesson

Persisted to  data/memory/sessions/<date>.json  at session end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from config.settings import MEMORY_DIR, MEMORY_SCHEMA_VERSION
from memory.persistence import list_files, load, save

SESSION_DIR = Path(MEMORY_DIR) / "sessions"


# ──────────────────────────────────────────────
# Supporting types
# ──────────────────────────────────────────────

@dataclass
class RegimeChange:
    timestamp:   str
    from_regime: str
    to_regime:   str
    confidence:  float = 0.0
    note:        str   = ""

    def to_dict(self) -> dict:
        return {
            "timestamp":  self.timestamp,
            "from":       self.from_regime,
            "to":         self.to_regime,
            "confidence": round(self.confidence, 3),
            "note":       self.note,
        }


# ──────────────────────────────────────────────
# Main dataclass
# ──────────────────────────────────────────────

@dataclass
class SessionMemory:
    """Complete record of one trading session."""

    session_date:   str              # "YYYY-MM-DD"
    symbol:         str = "SPY"
    schema_version: str = MEMORY_SCHEMA_VERSION

    # Market
    opening_regime:     str   = ""
    closing_regime:     str   = ""
    dominant_narrative: str   = ""
    htf_bias:           str   = "neutral"
    vix_at_open:        Optional[float] = None
    vix_at_close:       Optional[float] = None
    key_levels:         List[str] = field(default_factory=list)

    # Trades
    trade_ids:     List[str] = field(default_factory=list)
    total_trades:  int       = 0
    winning_trades: int      = 0
    losing_trades:  int      = 0

    # P&L
    daily_pnl_pct:             float = 0.0
    daily_pnl_dollar:          float = 0.0
    max_intraday_drawdown_pct: float = 0.0
    peak_gain_pct:             float = 0.0

    # Regime / narrative tracking
    regime_changes: List[RegimeChange] = field(default_factory=list)

    # Context
    macro_events_active: List[str] = field(default_factory=list)
    news_headlines:      List[str] = field(default_factory=list)
    context_bias:        str       = "neutral"
    elevated_risk:       bool      = False

    # Behavioral
    behavioral_flags: List[str] = field(default_factory=list)
    mistake_tags:     List[str] = field(default_factory=list)

    # Notes
    pre_market_notes:  str = ""
    post_market_notes: str = ""
    lesson:            str = ""

    # Timestamps
    session_start: Optional[str] = None
    session_end:   Optional[str] = None

    # ──────────────────────────────────────────
    # Lifecycle helpers
    # ──────────────────────────────────────────

    def open_session(self) -> None:
        self.session_start = datetime.now(tz=timezone.utc).isoformat()

    def close_session(self, notes: str = "", lesson: str = "") -> None:
        self.session_end = datetime.now(tz=timezone.utc).isoformat()
        if notes:
            self.post_market_notes = notes
        if lesson:
            self.lesson = lesson

    def record_trade(self, trade_id: str, pnl_pct: float) -> None:
        if trade_id not in self.trade_ids:
            self.trade_ids.append(trade_id)
            self.total_trades += 1
        self.daily_pnl_pct += pnl_pct
        if pnl_pct > 0.15:
            self.winning_trades += 1
        elif pnl_pct < -0.15:
            self.losing_trades += 1

    def record_regime_change(
        self,
        from_regime: str,
        to_regime:   str,
        confidence:  float = 0.0,
        note:        str   = "",
    ) -> None:
        self.regime_changes.append(RegimeChange(
            timestamp   = datetime.now(tz=timezone.utc).isoformat(),
            from_regime = from_regime,
            to_regime   = to_regime,
            confidence  = confidence,
            note        = note,
        ))
        self.closing_regime = to_regime

    def add_behavioral_flag(self, flag: str) -> None:
        if flag not in self.behavioral_flags:
            self.behavioral_flags.append(flag)

    def update_drawdown(self, current_pnl_pct: float) -> None:
        if current_pnl_pct < -abs(self.max_intraday_drawdown_pct):
            self.max_intraday_drawdown_pct = abs(current_pnl_pct)
        if current_pnl_pct > self.peak_gain_pct:
            self.peak_gain_pct = current_pnl_pct

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round(self.winning_trades / self.total_trades, 3)

    # ──────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_date":   self.session_date,
            "symbol":         self.symbol,
            "market": {
                "opening_regime":     self.opening_regime,
                "closing_regime":     self.closing_regime,
                "dominant_narrative": self.dominant_narrative,
                "htf_bias":           self.htf_bias,
                "vix_at_open":        self.vix_at_open,
                "vix_at_close":       self.vix_at_close,
                "key_levels":         self.key_levels,
            },
            "trades": {
                "trade_ids":   self.trade_ids,
                "total":       self.total_trades,
                "winning":     self.winning_trades,
                "losing":      self.losing_trades,
                "win_rate":    self.win_rate,
            },
            "pnl": {
                "daily_pct":        round(self.daily_pnl_pct, 3),
                "daily_dollar":     round(self.daily_pnl_dollar, 2),
                "max_drawdown_pct": round(self.max_intraday_drawdown_pct, 3),
                "peak_gain_pct":    round(self.peak_gain_pct, 3),
            },
            "regime_changes": [r.to_dict() for r in self.regime_changes],
            "context": {
                "macro_events":  self.macro_events_active,
                "news":          self.news_headlines[:5],
                "bias":          self.context_bias,
                "elevated_risk": self.elevated_risk,
            },
            "behavioral": {
                "flags":        self.behavioral_flags,
                "mistake_tags": self.mistake_tags,
            },
            "notes": {
                "pre_market":  self.pre_market_notes,
                "post_market": self.post_market_notes,
                "lesson":      self.lesson,
            },
            "timestamps": {
                "start": self.session_start,
                "end":   self.session_end,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        sm = cls(
            session_date   = data.get("session_date", ""),
            symbol         = data.get("symbol", "SPY"),
            schema_version = data.get("schema_version", MEMORY_SCHEMA_VERSION),
        )
        m = data.get("market", {})
        sm.opening_regime     = m.get("opening_regime", "")
        sm.closing_regime     = m.get("closing_regime", "")
        sm.dominant_narrative = m.get("dominant_narrative", "")
        sm.htf_bias           = m.get("htf_bias", "neutral")
        sm.vix_at_open        = m.get("vix_at_open")
        sm.vix_at_close       = m.get("vix_at_close")
        sm.key_levels         = m.get("key_levels", [])

        t = data.get("trades", {})
        sm.trade_ids       = t.get("trade_ids", [])
        sm.total_trades    = t.get("total", 0)
        sm.winning_trades  = t.get("winning", 0)
        sm.losing_trades   = t.get("losing", 0)

        p = data.get("pnl", {})
        sm.daily_pnl_pct              = p.get("daily_pct", 0.0)
        sm.daily_pnl_dollar           = p.get("daily_dollar", 0.0)
        sm.max_intraday_drawdown_pct  = p.get("max_drawdown_pct", 0.0)
        sm.peak_gain_pct              = p.get("peak_gain_pct", 0.0)

        for rc in data.get("regime_changes", []):
            sm.regime_changes.append(RegimeChange(
                timestamp   = rc.get("timestamp", ""),
                from_regime = rc.get("from", ""),
                to_regime   = rc.get("to", ""),
                confidence  = rc.get("confidence", 0.0),
                note        = rc.get("note", ""),
            ))

        ctx = data.get("context", {})
        sm.macro_events_active = ctx.get("macro_events", [])
        sm.news_headlines      = ctx.get("news", [])
        sm.context_bias        = ctx.get("bias", "neutral")
        sm.elevated_risk       = ctx.get("elevated_risk", False)

        beh = data.get("behavioral", {})
        sm.behavioral_flags = beh.get("flags", [])
        sm.mistake_tags     = beh.get("mistake_tags", [])

        n = data.get("notes", {})
        sm.pre_market_notes  = n.get("pre_market", "")
        sm.post_market_notes = n.get("post_market", "")
        sm.lesson            = n.get("lesson", "")

        ts = data.get("timestamps", {})
        sm.session_start = ts.get("start")
        sm.session_end   = ts.get("end")
        return sm

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self, directory: Path = SESSION_DIR) -> bool:
        return save(Path(directory) / f"{self.session_date}.json", self.to_dict())

    @classmethod
    def load(
        cls,
        session_date: str,
        directory:    Path = SESSION_DIR,
    ) -> Optional["SessionMemory"]:
        data = load(Path(directory) / f"{session_date}.json")
        return cls.from_dict(data) if data else None

    @classmethod
    def load_latest(cls, directory: Path = SESSION_DIR) -> Optional["SessionMemory"]:
        files = list_files(Path(directory))
        if not files:
            return None
        data = load(files[-1])
        return cls.from_dict(data) if data else None

    @classmethod
    def today(cls, symbol: str = "SPY") -> "SessionMemory":
        return cls(session_date=date.today().strftime("%Y-%m-%d"), symbol=symbol)