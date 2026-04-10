"""
Journal: SQLite-backed persistent trade journal.

Stores JournalEntries locally on disk. No cloud dependency.
Schema is append-only — entries are never modified, only added.

Search is keyword-based for v1. Vector search can be added later
if the journal grows large enough to need semantic retrieval.

DB location: logs/journal.db (configurable via settings)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from memory.models import JournalEntry, JournalEntryType, PerformanceSummary

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "logs" / "journal.db"


class Journal:
    """
    Local trade journal backed by SQLite.

    Thread-safety: SQLite connections are not thread-safe.
    Use one Journal instance per thread, or serialize access.
    For the trading system (single asyncio thread), this is fine.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Journal opened: {db_path}")

    # ──────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────

    def add_entry(self, entry: JournalEntry) -> None:
        """Persist a JournalEntry."""
        sql = """
            INSERT INTO journal_entries
              (id, entry_type, timestamp, content, tags,
               trade_id, symbol, direction, entry_price, exit_price,
               pnl_pct, setup_type, setup_score,
               was_correct, what_worked, what_failed)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        self._conn.execute(sql, (
            entry.id,
            entry.entry_type.value,
            entry.timestamp.isoformat(),
            entry.content,
            json.dumps(entry.tags),
            entry.trade_id,
            entry.symbol,
            entry.direction,
            entry.entry_price,
            entry.exit_price,
            entry.pnl_pct,
            entry.setup_type,
            entry.setup_score,
            int(entry.was_correct) if entry.was_correct is not None else None,
            entry.what_worked,
            entry.what_failed,
        ))
        self._conn.commit()

    def add_observation(self, content: str, tags: List[str] = None) -> JournalEntry:
        """Quick add: write a plain observation."""
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            entry_type=JournalEntryType.OBSERVATION,
            timestamp=datetime.now(tz=timezone.utc),
            content=content,
            tags=tags or [],
        )
        self.add_entry(entry)
        return entry

    def add_ai_insight(self, content: str, tags: List[str] = None) -> JournalEntry:
        """Save a Claude-generated insight that the user wants to keep."""
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            entry_type=JournalEntryType.AI_INSIGHT,
            timestamp=datetime.now(tz=timezone.utc),
            content=content,
            tags=tags or ["ai"],
        )
        self.add_entry(entry)
        return entry

    # ──────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────

    def get_recent(self, n: int = 20, entry_type: Optional[JournalEntryType] = None) -> List[JournalEntry]:
        """Return the most recent n entries, optionally filtered by type."""
        if entry_type:
            sql = "SELECT * FROM journal_entries WHERE entry_type=? ORDER BY timestamp DESC LIMIT ?"
            rows = self._conn.execute(sql, (entry_type.value, n)).fetchall()
        else:
            sql = "SELECT * FROM journal_entries ORDER BY timestamp DESC LIMIT ?"
            rows = self._conn.execute(sql, (n,)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search(self, query: str, limit: int = 10) -> List[JournalEntry]:
        """
        Keyword search across content, tags, and setup_type.
        Case-insensitive. Returns most recent matches first.
        """
        terms = query.lower().split()
        results = []

        all_rows = self._conn.execute(
            "SELECT * FROM journal_entries ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()

        for row in all_rows:
            text = " ".join([
                (row["content"] or "").lower(),
                (row["tags"] or "").lower(),
                (row["setup_type"] or "").lower(),
                (row["direction"] or "").lower(),
                (row["symbol"] or "").lower(),
            ])
            if all(t in text for t in terms):
                results.append(self._row_to_entry(row))
            if len(results) >= limit:
                break
        return results

    def get_trades_by_setup(self, setup_type: str, limit: int = 10) -> List[JournalEntry]:
        """Return completed trades for a specific setup type."""
        sql = """
            SELECT * FROM journal_entries
            WHERE entry_type = 'trade_exit' AND setup_type = ?
            ORDER BY timestamp DESC LIMIT ?
        """
        rows = self._conn.execute(sql, (setup_type, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def compute_performance(self, last_n_trades: int = 30) -> PerformanceSummary:
        """Compute rolling performance stats from completed trades."""
        sql = """
            SELECT pnl_pct, setup_type, was_correct
            FROM journal_entries
            WHERE entry_type = 'trade_exit' AND pnl_pct IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """
        rows = self._conn.execute(sql, (last_n_trades,)).fetchall()
        if not rows:
            return PerformanceSummary()

        pnls      = [r["pnl_pct"] for r in rows]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]
        win_rate  = len(wins) / len(pnls) if pnls else 0.0

        # Best/worst setup type
        from collections import defaultdict
        setup_wins: dict = defaultdict(int)
        setup_total: dict = defaultdict(int)
        for r in rows:
            st = r["setup_type"] or "unknown"
            setup_total[st] += 1
            if r["pnl_pct"] and r["pnl_pct"] > 0:
                setup_wins[st] += 1

        best_setup  = max(setup_wins, key=lambda s: setup_wins[s] / setup_total[s], default="")
        worst_setup = min(setup_wins, key=lambda s: setup_wins[s] / setup_total[s], default="")

        return PerformanceSummary(
            total_trades=len(rows),
            win_rate=win_rate,
            avg_win_pct=sum(wins) / len(wins) if wins else 0.0,
            avg_loss_pct=sum(losses) / len(losses) if losses else 0.0,
            best_setup_type=best_setup,
            worst_setup_type=worst_setup,
        )

    # ──────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id           TEXT PRIMARY KEY,
                entry_type   TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                content      TEXT,
                tags         TEXT,
                trade_id     TEXT,
                symbol       TEXT,
                direction    TEXT,
                entry_price  REAL,
                exit_price   REAL,
                pnl_pct      REAL,
                setup_type   TEXT,
                setup_score  REAL,
                was_correct  INTEGER,
                what_worked  TEXT,
                what_failed  TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON journal_entries(timestamp DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entry_type ON journal_entries(entry_type)"
        )
        self._conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        tags = []
        try:
            tags = json.loads(row["tags"] or "[]")
        except (json.JSONDecodeError, TypeError):
            pass

        was_correct = None
        if row["was_correct"] is not None:
            was_correct = bool(row["was_correct"])

        return JournalEntry(
            id=row["id"],
            entry_type=JournalEntryType(row["entry_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            content=row["content"] or "",
            tags=tags,
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            direction=row["direction"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            pnl_pct=row["pnl_pct"],
            setup_type=row["setup_type"],
            setup_score=row["setup_score"],
            was_correct=was_correct,
            what_worked=row["what_worked"] or "",
            what_failed=row["what_failed"] or "",
        )

    def close(self) -> None:
        self._conn.close()
