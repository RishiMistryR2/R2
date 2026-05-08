"""
memory/reflection_engine.py — end-of-day reflection generator.

CRITICAL: execution_quality ≠ outcome_quality

  Good exec / bad outcome  → process was right; market uncertain
  Bad exec  / good outcome → lucky win; do NOT reinforce bad execution

Tracks good_trade_bad_outcome_count and bad_trade_good_outcome_count
independently from win / loss counts.

Output: DayReflection (pure dataclass — caller persists if desired)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from memory.session_memory import SessionMemory
from memory.trade_memory import TradeRecord, OUTCOME_WIN, OUTCOME_LOSS
from memory.performance_stats import build_report


@dataclass
class TradeQualityAssessment:
    trade_id:          str
    execution_quality: str   # "good" | "acceptable" | "poor"
    outcome_quality:   str   # "good" | "bad"
    is_good_execution_bad_outcome: bool = False
    is_bad_execution_good_outcome: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_id":          self.trade_id,
            "execution_quality": self.execution_quality,
            "outcome_quality":   self.outcome_quality,
            "good_exec_bad_out": self.is_good_execution_bad_outcome,
            "bad_exec_good_out": self.is_bad_execution_good_outcome,
            "notes":             self.notes,
        }


@dataclass
class DayReflection:
    session_date: str
    generated_at: str

    market_summary: str   = ""
    regime_notes:   str   = ""
    trade_summary:  str   = ""
    total_trades:   int   = 0
    win_rate:       float = 0.0
    daily_pnl_pct:  float = 0.0

    # Quality separation — the critical counters
    good_trade_bad_outcome_count: int = 0
    bad_trade_good_outcome_count: int = 0
    trade_quality_assessments:    List[TradeQualityAssessment] = field(default_factory=list)

    # Analysis
    mistake_analysis:      List[str] = field(default_factory=list)
    performance_insights:  List[str] = field(default_factory=list)
    lessons_learned:       List[str] = field(default_factory=list)
    suggested_adjustments: List[str] = field(default_factory=list)

    performance_stats: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "session_date": self.session_date,
            "generated_at": self.generated_at,
            "market": {
                "summary":      self.market_summary,
                "regime_notes": self.regime_notes,
            },
            "trades": {
                "summary":      self.trade_summary,
                "total":        self.total_trades,
                "win_rate_pct": round(self.win_rate * 100, 1),
                "daily_pnl":    self.daily_pnl_pct,
            },
            "quality_separation": {
                "good_execution_bad_outcome": self.good_trade_bad_outcome_count,
                "bad_execution_good_outcome": self.bad_trade_good_outcome_count,
                "assessments": [a.to_dict() for a in self.trade_quality_assessments],
            },
            "analysis": {
                "mistakes":    self.mistake_analysis,
                "insights":    self.performance_insights,
                "lessons":     self.lessons_learned,
                "adjustments": self.suggested_adjustments,
            },
            "stats": self.performance_stats,
        }


class ReflectionEngine:
    """Generates end-of-day structured reflection."""

    def generate(
        self,
        session: SessionMemory,
        trades:  List[TradeRecord],
    ) -> DayReflection:
        now = datetime.now(tz=timezone.utc).isoformat()
        ref = DayReflection(session_date=session.session_date, generated_at=now)

        ref.market_summary = self._market_summary(session)
        ref.regime_notes   = self._regime_notes(session)
        ref.total_trades   = session.total_trades
        ref.win_rate       = session.win_rate
        ref.daily_pnl_pct  = session.daily_pnl_pct
        ref.trade_summary  = self._trade_summary(session, trades)

        assessments = [self._assess(t) for t in trades]
        ref.trade_quality_assessments    = assessments
        ref.good_trade_bad_outcome_count = sum(
            1 for a in assessments if a.is_good_execution_bad_outcome
        )
        ref.bad_trade_good_outcome_count = sum(
            1 for a in assessments if a.is_bad_execution_good_outcome
        )

        ref.mistake_analysis      = self._mistakes(session, trades)
        ref.performance_insights  = self._insights(session, trades)
        ref.lessons_learned       = self._lessons(session, trades, assessments)
        ref.suggested_adjustments = self._adjustments(session, trades, assessments)

        if trades:
            ref.performance_stats = build_report(trades).to_dict()

        return ref

    def apply_quality_tags(
        self,
        reflection: DayReflection,
        trades:     List[TradeRecord],
    ) -> None:
        """Write execution/outcome quality back onto TradeRecord objects."""
        for a in reflection.trade_quality_assessments:
            for t in trades:
                if t.trade_id == a.trade_id:
                    t.execution_quality = a.execution_quality
                    t.outcome_quality   = a.outcome_quality
                    break

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _market_summary(self, s: SessionMemory) -> str:
        parts = [f"Date: {s.session_date}"]
        if s.opening_regime:
            parts.append(f"Regime: {s.opening_regime}")
            if s.closing_regime and s.closing_regime != s.opening_regime:
                parts[-1] += f" → {s.closing_regime}"
        if s.dominant_narrative:
            parts.append(f"Narrative: {s.dominant_narrative}")
        if s.elevated_risk:
            parts.append("⚠ Elevated risk")
        return " | ".join(parts)

    def _regime_notes(self, s: SessionMemory) -> str:
        if not s.regime_changes:
            return "No regime changes."
        lines = [f"{r.from_regime}→{r.to_regime}" for r in s.regime_changes]
        return f"{len(s.regime_changes)} change(s): {', '.join(lines)}"

    def _trade_summary(self, s: SessionMemory, trades: List[TradeRecord]) -> str:
        if not trades:
            return "No trades taken."
        wins   = sum(1 for t in trades if t.outcome == OUTCOME_WIN)
        losses = sum(1 for t in trades if t.outcome == OUTCOME_LOSS)
        return (
            f"{len(trades)} trade(s): {wins}W/{losses}L — "
            f"Win rate {s.win_rate * 100:.0f}% — P&L {s.daily_pnl_pct:+.2f}%"
        )

    def _assess(self, trade: TradeRecord) -> TradeQualityAssessment:
        """
        Execution quality:
          good        → rule_compliant, 0 mistakes, rr_planned ≥ 1.5
          acceptable  → compliant, ≤ 1 mistake
          poor        → not compliant OR ≥ 2 mistakes

        Outcome quality (independent of execution):
          good  → win without time-stop, OR thesis stop with |pnl| < 1.5%
          bad   → time/premium stop, OR loss > 2%
        """
        n_mistakes = len(trade.mistake_tags)
        if trade.rule_compliant and n_mistakes == 0 and trade.rr_planned >= 1.5:
            exec_q = "good"
        elif trade.rule_compliant and n_mistakes <= 1:
            exec_q = "acceptable"
        else:
            exec_q = "poor"

        if trade.outcome == OUTCOME_WIN and not trade.time_stop_triggered:
            out_q = "good"
        elif trade.thesis_stop_triggered and abs(trade.pnl_pct) < 1.5:
            out_q = "good"
        elif trade.time_stop_triggered or trade.premium_stop_triggered:
            out_q = "bad"
        elif trade.outcome == OUTCOME_LOSS and abs(trade.pnl_pct) > 2.0:
            out_q = "bad"
        else:
            out_q = "good" if trade.outcome == OUTCOME_WIN else "bad"

        good_exec_bad_out = (exec_q == "good" and out_q == "bad")
        bad_exec_good_out = (exec_q == "poor" and out_q == "good")

        notes = ""
        if bad_exec_good_out:
            notes = "Lucky outcome — do not reinforce bad execution."
        elif good_exec_bad_out:
            notes = "Good process — market did not cooperate. Maintain rules."

        return TradeQualityAssessment(
            trade_id                      = trade.trade_id,
            execution_quality             = exec_q,
            outcome_quality               = out_q,
            is_good_execution_bad_outcome = good_exec_bad_out,
            is_bad_execution_good_outcome = bad_exec_good_out,
            notes                         = notes,
        )

    def _mistakes(self, s: SessionMemory, trades: List[TradeRecord]) -> List[str]:
        tags = list(s.mistake_tags)
        for t in trades:
            tags.extend(t.mistake_tags)
        if not tags:
            return ["No mistakes tagged."]
        counts = Counter(tags)
        out = [f"'{tag}' × {n}" for tag, n in counts.most_common(5)]
        for f in s.behavioral_flags:
            out.append(f"Behavioral flag: {f}")
        return out

    def _insights(self, s: SessionMemory, trades: List[TradeRecord]) -> List[str]:
        if not trades:
            return ["No trades to analyse."]
        ins: List[str] = []
        regime_wins = [t for t in trades
                       if t.regime_at_entry in ("trend_bull", "trend_bear")
                       and t.outcome == OUTCOME_WIN]
        if len(regime_wins) / len(trades) > 0.7:
            ins.append("Strong performance in trending regime — alignment working.")
        ts = sum(1 for t in trades if t.time_stop_triggered)
        if ts > 1:
            ins.append(f"{ts} time stop(s) — consider tighter setup criteria.")
        wide = [t for t in trades if t.mae_pct > 0.5]
        if wide:
            ins.append(f"{len(wide)} trade(s) had MAE > 0.5% — review stop placement.")
        return ins or ["No notable insights."]

    def _lessons(
        self,
        s:      SessionMemory,
        trades: List[TradeRecord],
        asmt:   List[TradeQualityAssessment],
    ) -> List[str]:
        out: List[str] = []
        if s.lesson:
            out.append(s.lesson)
        for t in trades:
            if t.lesson:
                out.append(f"[{t.trade_id[:6]}] {t.lesson}")
        n_bad_exec_good = sum(1 for a in asmt if a.is_bad_execution_good_outcome)
        if n_bad_exec_good:
            out.append(
                f"{n_bad_exec_good} poor-execution trade(s) won — "
                "process, not result, determines long-run edge."
            )
        n_good_exec_bad = sum(1 for a in asmt if a.is_good_execution_bad_outcome)
        if n_good_exec_bad:
            out.append(
                f"{n_good_exec_bad} well-executed trade(s) lost — "
                "stay the course; market randomness is normal."
            )
        return out or ["No specific lessons logged."]

    def _adjustments(
        self,
        s:      SessionMemory,
        trades: List[TradeRecord],
        asmt:   List[TradeQualityAssessment],
    ) -> List[str]:
        if len(trades) < 3:
            return ["Insufficient data (need ≥ 3 trades)."]
        out: List[str] = []
        if s.win_rate < 0.40:
            out.append("Win rate < 40% — review entry criteria (one session is not a sample).")
        ts = sum(1 for t in trades if t.time_stop_triggered)
        if ts >= 2:
            out.append(f"{ts} time stops — consider A+ setups only until rate improves.")
        poor = sum(1 for a in asmt if a.execution_quality == "poor")
        if poor >= 2:
            out.append(f"{poor} poor-execution trades — review rules checklist before entry.")
        return out or ["No adjustments suggested — maintain current approach."]