"""
memory/performance_stats.py — pure aggregate statistics over TradeRecord lists.

No I/O, no state.  All functions are pure.

Aggregates by: setup_type | regime | session_phase | instrument | htf_bias
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from memory.trade_memory import TradeRecord, OUTCOME_WIN, OUTCOME_LOSS


@dataclass
class StatsSlice:
    label:     str
    dimension: str
    total_trades:        int   = 0
    win_rate:            float = 0.0
    avg_return_pct:      float = 0.0
    avg_win_pct:         float = 0.0
    avg_loss_pct:        float = 0.0
    profit_factor:       float = 0.0
    expectancy:          float = 0.0
    avg_mae:             float = 0.0
    avg_mfe:             float = 0.0
    avg_rr_planned:      float = 0.0
    avg_rr_achieved:     float = 0.0
    avg_duration_minutes: float = 0.0
    max_drawdown_contribution: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label":            self.label,
            "dimension":        self.dimension,
            "total_trades":     self.total_trades,
            "win_rate_pct":     round(self.win_rate * 100, 1),
            "avg_return_pct":   round(self.avg_return_pct, 3),
            "avg_win_pct":      round(self.avg_win_pct, 3),
            "avg_loss_pct":     round(self.avg_loss_pct, 3),
            "profit_factor":    round(self.profit_factor, 2),
            "expectancy":       round(self.expectancy, 3),
            "avg_mae":          round(self.avg_mae, 3),
            "avg_mfe":          round(self.avg_mfe, 3),
            "avg_rr_planned":   round(self.avg_rr_planned, 2),
            "avg_rr_achieved":  round(self.avg_rr_achieved, 2),
            "avg_duration_min": round(self.avg_duration_minutes, 1),
        }


@dataclass
class PerformanceReport:
    overall:          StatsSlice
    by_setup_type:    Dict[str, StatsSlice] = field(default_factory=dict)
    by_regime:        Dict[str, StatsSlice] = field(default_factory=dict)
    by_session_phase: Dict[str, StatsSlice] = field(default_factory=dict)
    by_instrument:    Dict[str, StatsSlice] = field(default_factory=dict)
    by_htf_bias:      Dict[str, StatsSlice] = field(default_factory=dict)
    total_trades:     int = 0
    best_setup:       str = ""
    worst_setup:      str = ""

    def to_dict(self) -> dict:
        def sm(d: dict) -> dict:
            return {k: v.to_dict() for k, v in d.items()}
        return {
            "overall":          self.overall.to_dict(),
            "by_setup_type":    sm(self.by_setup_type),
            "by_regime":        sm(self.by_regime),
            "by_session_phase": sm(self.by_session_phase),
            "by_instrument":    sm(self.by_instrument),
            "by_htf_bias":      sm(self.by_htf_bias),
            "meta": {
                "total_trades": self.total_trades,
                "best_setup":   self.best_setup,
                "worst_setup":  self.worst_setup,
            },
        }


def compute_stats(
    records:   List[TradeRecord],
    label:     str = "all",
    dimension: str = "overall",
) -> StatsSlice:
    if not records:
        return StatsSlice(label=label, dimension=dimension)

    n      = len(records)
    wins   = [r for r in records if r.outcome == OUTCOME_WIN]
    losses = [r for r in records if r.outcome == OUTCOME_LOSS]

    win_rate   = len(wins) / n
    avg_return = sum(r.pnl_pct for r in records) / n
    avg_win    = sum(r.pnl_pct for r in wins)   / len(wins)   if wins   else 0.0
    avg_loss   = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0.0

    gross_profit  = sum(r.pnl_pct for r in wins)
    gross_loss    = abs(sum(r.pnl_pct for r in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

    return StatsSlice(
        label          = label,
        dimension      = dimension,
        total_trades   = n,
        win_rate       = round(win_rate, 4),
        avg_return_pct = round(avg_return, 4),
        avg_win_pct    = round(avg_win, 4),
        avg_loss_pct   = round(avg_loss, 4),
        profit_factor  = round(min(profit_factor, 999.0), 3),
        expectancy     = round(expectancy, 4),
        avg_mae        = round(sum(r.mae_pct for r in records) / n, 4),
        avg_mfe        = round(sum(r.mfe_pct for r in records) / n, 4),
        avg_rr_planned  = round(sum(r.rr_planned  for r in records) / n, 3),
        avg_rr_achieved = round(sum(r.rr_achieved for r in records) / n, 3),
        avg_duration_minutes       = round(sum(r.duration_minutes for r in records) / n, 1),
        max_drawdown_contribution  = round(abs(min((r.pnl_pct for r in records), default=0.0)), 3),
    )


def build_report(records: List[TradeRecord]) -> PerformanceReport:
    if not records:
        return PerformanceReport(overall=StatsSlice(label="all", dimension="overall"))

    report = PerformanceReport(
        overall      = compute_stats(records),
        total_trades = len(records),
    )

    for dim, key_fn in [
        ("setup_type",    lambda r: r.setup_type or "unknown"),
        ("regime",        lambda r: r.regime_at_entry or "unknown"),
        ("session_phase", lambda r: r.session_phase_at_entry or "unknown"),
        ("instrument",    lambda r: r.instrument_type or "unknown"),
        ("htf_bias",      lambda r: r.htf_bias_at_entry or "neutral"),
    ]:
        groups: Dict[str, List[TradeRecord]] = {}
        for r in records:
            k = key_fn(r)
            groups.setdefault(k, []).append(r)
        bucket = getattr(report, f"by_{dim}")
        for k, g in groups.items():
            bucket[k] = compute_stats(g, label=k, dimension=dim)

    qualified = {k: v for k, v in report.by_setup_type.items() if v.total_trades >= 3}
    if qualified:
        report.best_setup  = max(qualified, key=lambda k: qualified[k].expectancy)
        report.worst_setup = min(qualified, key=lambda k: qualified[k].expectancy)

    return report