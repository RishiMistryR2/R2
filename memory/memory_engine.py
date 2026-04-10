"""
MemoryEngine: local journal and trade history retrieval.

Storage:
  data/journal/  — JSON files, one per entry
  data/trades/   — JSON files, one per completed trade

Retrieval is keyword + metadata matching (no vector embeddings in v1).
Fast enough for the trade volumes we're dealing with.
Vector search can be added later if pattern corpus grows large.

The memory engine is read-mostly during live trading.
Writes happen after session end and during journal entry.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from memory.memory_state import JournalEntry, MemoryContext, TradeMemory
from engine.models import MarketState, TradeState

logger = logging.getLogger(__name__)

# Default data directories
DEFAULT_JOURNAL_DIR = Path("data/journal")
DEFAULT_TRADES_DIR  = Path("data/trades")


class MemoryEngine:
    """
    Manages the local journal and trade history.

    Usage:
        mem = MemoryEngine()
        mem.add_journal_entry("Today I noticed that false breakouts...")
        context = mem.retrieve(state, active_trade)
    """

    def __init__(
        self,
        journal_dir: Path = DEFAULT_JOURNAL_DIR,
        trades_dir:  Path = DEFAULT_TRADES_DIR,
    ) -> None:
        self._journal_dir = Path(journal_dir)
        self._trades_dir  = Path(trades_dir)
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._trades_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index (loaded at startup)
        self._journal:   List[JournalEntry] = []
        self._trades:    List[TradeMemory]  = []
        self._load_all()

    # ──────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────

    def retrieve(
        self,
        state:        MarketState,
        active_trade: Optional[TradeState] = None,
        max_results:  int = 3,
    ) -> MemoryContext:
        """
        Retrieve memory context relevant to the current market state.

        Matching criteria:
          - Setup type matches current setup_type
          - Narrative matches current narrative
          - Trade direction matches current bias
          - Journal tags match current conditions
        """
        narrative_val = (state.narrative.narrative.value
                         if state.narrative else "")
        setup_val     = state.setup_type.value if state.setup_type else ""
        direction_val = (active_trade.direction.value
                         if active_trade else
                         (state.setup_direction.value if state.setup_direction else ""))

        # Match trades
        similar = self._match_trades(
            narrative=narrative_val,
            setup_type=setup_val,
            direction=direction_val,
            max_n=max_results,
        )

        # Match journal entries
        notes = self._match_journal(
            narrative=narrative_val,
            setup_type=setup_val,
            max_n=max_results,
        )

        # Performance summary
        perf_note = self._performance_summary(similar)

        # Patterns
        patterns = self._extract_patterns(notes)

        search_query = f"narrative={narrative_val} setup={setup_val} dir={direction_val}"

        ctx = MemoryContext(
            similar_setups=similar,
            journal_notes=notes,
            performance_note=perf_note,
            patterns=patterns,
            search_query=search_query,
        )
        return ctx

    # ──────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────

    def add_journal_entry(
        self,
        content:    str,
        tags:       List[str]  = None,
        narratives: List[str]  = None,
        setup_types: List[str] = None,
    ) -> JournalEntry:
        """Write a new journal entry to disk and index."""
        entry = JournalEntry(
            entry_id=str(uuid.uuid4())[:8],
            created_at=datetime.now(tz=timezone.utc),
            content=content,
            tags=tags or [],
            narratives=narratives or [],
            setup_types=setup_types or [],
        )
        self._journal.append(entry)
        self._save_journal_entry(entry)
        logger.info(f"Journal entry saved: {entry.entry_id}")
        return entry

    def save_trade_memory(self, trade: TradeState, notes: str = "") -> TradeMemory:
        """Save a completed trade to the memory index."""
        from engine.models import ExitReason
        mem = TradeMemory(
            trade_id=trade.trade_id,
            direction=trade.direction.value,
            entry_date=trade.entry_time,
            setup_type=trade.setup_type.value,
            narrative="",   # filled by caller if available
            score=0.0,
            pnl_pct=trade.pnl_pct,
            outcome="win" if trade.pnl_pct > 0 else ("loss" if trade.pnl_pct < 0 else "breakeven"),
            exit_reason=trade.exit_reason.value if trade.exit_reason else "",
            notes=notes,
        )
        self._trades.append(mem)
        self._save_trade_memory(mem)
        return mem

    # ──────────────────────────────────────────
    # Matching helpers
    # ──────────────────────────────────────────

    def _match_trades(
        self,
        narrative:  str,
        setup_type: str,
        direction:  str,
        max_n:      int,
    ) -> List[TradeMemory]:
        scored = []
        for t in self._trades:
            score = 0
            if t.narrative == narrative:      score += 3
            if t.setup_type == setup_type:    score += 2
            if t.direction == direction:      score += 1
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:max_n]]

    def _match_journal(
        self,
        narrative:  str,
        setup_type: str,
        max_n:      int,
    ) -> List[JournalEntry]:
        scored = []
        for entry in self._journal:
            score = 0
            if narrative  in entry.narratives:  score += 3
            if setup_type in entry.setup_types:  score += 2
            # Simple keyword match in content
            if narrative.replace("_", " ")  in entry.content.lower(): score += 1
            if setup_type.replace("_", " ") in entry.content.lower(): score += 1
            if score > 0:
                entry.relevance = float(score)
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:max_n]]

    def _performance_summary(self, trades: List[TradeMemory]) -> str:
        if not trades:
            return ""
        wins = sum(1 for t in trades if t.outcome == "win")
        losses = sum(1 for t in trades if t.outcome == "loss")
        avg_pnl = sum(t.pnl_pct for t in trades) / len(trades)
        return (f"Last {len(trades)} similar setups: "
                f"{wins}W {losses}L, avg P&L {avg_pnl:+.2f}%")

    def _extract_patterns(self, entries: List[JournalEntry]) -> List[str]:
        """Return any explicitly tagged patterns from matching journal entries."""
        patterns = []
        for entry in entries:
            for tag in entry.tags:
                if tag.startswith("pattern:"):
                    patterns.append(tag.replace("pattern:", "").strip())
        return patterns[:5]

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def _load_all(self) -> None:
        self._load_journal()
        self._load_trades()

    def _load_journal(self) -> None:
        for path in sorted(self._journal_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                entry = JournalEntry(
                    entry_id=data["entry_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    content=data["content"],
                    tags=data.get("tags", []),
                    narratives=data.get("narratives", []),
                    setup_types=data.get("setup_types", []),
                )
                self._journal.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load journal entry {path}: {e}")
        logger.info(f"Loaded {len(self._journal)} journal entries")

    def _load_trades(self) -> None:
        for path in sorted(self._trades_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                mem = TradeMemory(
                    trade_id=data["trade_id"],
                    direction=data["direction"],
                    entry_date=datetime.fromisoformat(data["entry_date"]),
                    setup_type=data["setup_type"],
                    narrative=data.get("narrative", ""),
                    score=data.get("score", 0.0),
                    pnl_pct=data.get("pnl_pct", 0.0),
                    outcome=data.get("outcome", "unknown"),
                    exit_reason=data.get("exit_reason", ""),
                    notes=data.get("notes", ""),
                )
                self._trades.append(mem)
            except Exception as e:
                logger.warning(f"Failed to load trade memory {path}: {e}")
        logger.info(f"Loaded {len(self._trades)} trade memories")

    def _save_journal_entry(self, entry: JournalEntry) -> None:
        path = self._journal_dir / f"{entry.entry_id}.json"
        data = {
            "entry_id":   entry.entry_id,
            "created_at": entry.created_at.isoformat(),
            "content":    entry.content,
            "tags":       entry.tags,
            "narratives": entry.narratives,
            "setup_types": entry.setup_types,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_trade_memory(self, mem: TradeMemory) -> None:
        path = self._trades_dir / f"{mem.trade_id}.json"
        data = {
            "trade_id":   mem.trade_id,
            "direction":  mem.direction,
            "entry_date": mem.entry_date.isoformat(),
            "setup_type": mem.setup_type,
            "narrative":  mem.narrative,
            "score":      mem.score,
            "pnl_pct":    mem.pnl_pct,
            "outcome":    mem.outcome,
            "exit_reason": mem.exit_reason,
            "notes":      mem.notes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
