"""
Memory layer data models.

The memory system is Tier 2 — high-quality, curated, personally authored.
It gives the AI access to:
  - Your past trades in similar setups (performance context)
  - Journal notes that match current conditions
  - Patterns you've observed and written down

This is your personal edge. None of this leaks to a third party.
All data lives locally in data/journal/ and data/trades/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TradeMemory:
    """
    A compressed record of a past trade, used for pattern matching.
    Stored separately from the full Trade log to keep retrieval fast.
    """
    trade_id:    str
    direction:   str         # "calls" | "puts"
    entry_date:  datetime
    setup_type:  str
    narrative:   str         # market narrative at time of trade
    score:       float       # setup quality score at entry

    # Outcome
    pnl_pct:     float = 0.0
    outcome:     str   = "unknown"   # "win" | "loss" | "breakeven"
    exit_reason: str   = ""

    # Context at entry (compact)
    htf_bias:    str = ""
    rsi_at_entry: Optional[float] = None
    vix_at_entry: Optional[float] = None

    # What you learned
    notes:       str = ""

    def to_dict(self) -> dict:
        return {
            "trade_id":   self.trade_id,
            "direction":  self.direction,
            "date":       self.entry_date.strftime("%Y-%m-%d"),
            "setup_type": self.setup_type,
            "narrative":  self.narrative,
            "score":      self.score,
            "pnl_pct":    round(self.pnl_pct, 2),
            "outcome":    self.outcome,
            "exit_reason": self.exit_reason,
            "notes":      self.notes[:200] if self.notes else "",
        }


@dataclass
class JournalEntry:
    """
    A single journal entry — observations, lessons, market notes.
    Written by you during or after a session.
    """
    entry_id:    str
    created_at:  datetime
    content:     str

    # Optional metadata for search
    tags:        List[str] = field(default_factory=list)
    setup_types: List[str] = field(default_factory=list)
    narratives:  List[str] = field(default_factory=list)
    symbols:     List[str] = field(default_factory=list)
    relevance:   float     = 0.0   # set by retrieval engine

    def to_dict(self) -> dict:
        return {
            "entry_id":   self.entry_id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "content":    self.content[:500],
            "tags":       self.tags,
        }


@dataclass
class MemoryContext:
    """
    Assembled memory context passed to the AI.

    Contains:
      similar_setups:    past trades in similar conditions (for pattern recognition)
      journal_notes:     your own observations that match current setup/narrative
      performance_note:  compact P&L summary for similar historical setups
      patterns:          specific patterns you've named and documented
    """
    similar_setups:    List[TradeMemory]  = field(default_factory=list)
    journal_notes:     List[JournalEntry] = field(default_factory=list)
    performance_note:  str               = ""
    patterns:          List[str]         = field(default_factory=list)
    search_query:      str               = ""   # what was searched for (debug)

    def is_empty(self) -> bool:
        return (not self.similar_setups
                and not self.journal_notes
                and not self.patterns)

    def to_dict(self) -> dict:
        return {
            "similar_setups": [t.to_dict() for t in self.similar_setups[:3]],
            "journal_notes":  [j.to_dict() for j in self.journal_notes[:3]],
            "performance_note": self.performance_note,
            "patterns":       self.patterns[:5],
        }
