"""
memory/startup_context.py — pre-market briefing builder.

Assembles a compact StartupBriefing from:
  - previous session summary
  - recent trade performance
  - surfaced patterns (market + behavioral)
  - top risks for today

Passed to the AI context and CLI at session open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config.settings import MEMORY_DIR
from memory.session_memory import SessionMemory
from memory.trade_memory import TradeRecord
from memory.performance_stats import build_report
from memory.pattern_memory import PatternMemory, BTYPE_MISTAKE


@dataclass
class StartupBriefing:
    generated_at: str
    session_date: str

    # Previous session
    prev_session_date:     str   = ""
    prev_session_pnl_pct:  float = 0.0
    prev_session_win_rate: float = 0.0
    prev_session_trades:   int   = 0
    prev_session_regime:   str   = ""
    prev_session_lesson:   str   = ""

    # Recent performance
    recent_trades_n:    int   = 0
    recent_win_rate:    float = 0.0
    recent_expectancy:  float = 0.0
    recent_best_setup:  str   = ""
    recent_worst_setup: str   = ""

    # Patterns — three separate buckets
    active_patterns:        List[dict] = field(default_factory=list)   # behavioral
    active_mistakes:        List[dict] = field(default_factory=list)   # mistake tags
    active_market_patterns: List[dict] = field(default_factory=list)   # market patterns

    # Risks
    top_risks: List[str] = field(default_factory=list)

    # User preferences
    user_focus:        str       = ""
    user_restrictions: List[str] = field(default_factory=list)

    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "session_date": self.session_date,
            "previous_session": {
                "date":     self.prev_session_date,
                "pnl_pct":  self.prev_session_pnl_pct,
                "win_rate": round(self.prev_session_win_rate * 100, 1),
                "trades":   self.prev_session_trades,
                "regime":   self.prev_session_regime,
                "lesson":   self.prev_session_lesson,
            },
            "recent_performance": {
                "trades_n":    self.recent_trades_n,
                "win_rate":    round(self.recent_win_rate * 100, 1),
                "expectancy":  self.recent_expectancy,
                "best_setup":  self.recent_best_setup,
                "worst_setup": self.recent_worst_setup,
            },
            "patterns": {
                "behavioral": self.active_patterns,
                "mistakes":   self.active_mistakes,
                "market":     self.active_market_patterns,
            },
            "top_risks": self.top_risks,
            "user": {
                "focus":        self.user_focus,
                "restrictions": self.user_restrictions,
            },
            "summary": self.summary,
        }


def build_startup_briefing(
    session_date:  str            = "",
    session_dir:   Optional[Path] = None,
    trades_dir:    Optional[Path] = None,
    pattern_file:  Optional[Path] = None,
    recent_n:      int            = 20,
) -> StartupBriefing:
    from memory.session_memory import SESSION_DIR
    from memory.trade_memory   import TRADES_DIR
    from memory.pattern_memory import PATTERN_FILE
    from datetime import date

    if not session_date:
        session_date = date.today().strftime("%Y-%m-%d")

    now      = datetime.now(tz=timezone.utc).isoformat()
    briefing = StartupBriefing(generated_at=now, session_date=session_date)

    # Previous session
    prev = SessionMemory.load_latest(session_dir or SESSION_DIR)
    if prev:
        briefing.prev_session_date     = prev.session_date
        briefing.prev_session_pnl_pct  = prev.daily_pnl_pct
        briefing.prev_session_win_rate = prev.win_rate
        briefing.prev_session_trades   = prev.total_trades
        briefing.prev_session_regime   = prev.closing_regime
        briefing.prev_session_lesson   = prev.lesson

    # Recent performance
    all_trades = TradeRecord.load_all(trades_dir or TRADES_DIR)
    recent     = all_trades[-recent_n:] if all_trades else []
    if recent:
        rep = build_report(recent)
        briefing.recent_trades_n    = len(recent)
        briefing.recent_win_rate    = rep.overall.win_rate
        briefing.recent_expectancy  = rep.overall.expectancy
        briefing.recent_best_setup  = rep.best_setup
        briefing.recent_worst_setup = rep.worst_setup

    # Patterns — behavioral, mistakes, market (separate buckets)
    pm         = PatternMemory(pattern_file or PATTERN_FILE)
    surfaced_b = pm.get_surfaced_behavior()
    surfaced_m = pm.get_surfaced_market()
    briefing.active_patterns = [
        p.to_dict() for p in surfaced_b if p.pattern_type != BTYPE_MISTAKE
    ][:5]
    briefing.active_mistakes = [
        p.to_dict() for p in pm.get_mistakes() if p.is_surfaced
    ][:3]
    briefing.active_market_patterns = [p.to_dict() for p in surfaced_m][:5]

    # Top risks
    briefing.top_risks = _top_risks(briefing, prev, recent)

    # Summary text
    briefing.summary = _summary(briefing)
    return briefing


def _top_risks(
    b:      StartupBriefing,
    prev:   Optional[SessionMemory],
    recent: List[TradeRecord],
) -> List[str]:
    risks: List[str] = []
    if prev and prev.daily_pnl_pct < -1.0:
        risks.append(
            f"Previous session loss ({prev.daily_pnl_pct:+.2f}%) — consider reducing size."
        )
    if b.active_mistakes:
        m = b.active_mistakes[0]
        risks.append(f"Recurring mistake: {m.get('description', 'unknown')}.")
    if b.recent_trades_n >= 5 and b.recent_win_rate < 0.40:
        risks.append(f"Recent win rate {b.recent_win_rate * 100:.0f}% — A+ setups only.")
    if recent:
        ts_rate = sum(1 for t in recent if t.time_stop_triggered) / len(recent)
        if ts_rate > 0.30:
            risks.append(
                f"{ts_rate * 100:.0f}% of recent trades hit time stop — "
                "setups may not be developing."
            )
    return risks[:4]


def _summary(b: StartupBriefing) -> str:
    parts: List[str] = []
    if b.prev_session_date:
        parts.append(
            f"Previous session ({b.prev_session_date}): "
            f"{b.prev_session_trades} trade(s), "
            f"{b.prev_session_win_rate * 100:.0f}% win rate, "
            f"P&L {b.prev_session_pnl_pct:+.2f}%."
        )
        if b.prev_session_lesson:
            parts.append(f"Lesson: {b.prev_session_lesson}")
    else:
        parts.append("No previous session data.")
    if b.recent_trades_n > 0:
        parts.append(
            f"Recent {b.recent_trades_n} trades: "
            f"{b.recent_win_rate * 100:.0f}% win rate, "
            f"expectancy {b.recent_expectancy:+.3f}%."
        )
    if b.top_risks:
        parts.append(f"Top risk: {b.top_risks[0]}")
    return " ".join(parts)