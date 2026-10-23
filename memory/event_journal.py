"""
memory/event_journal.py — append-only JSONL journal of event-intelligence decisions.

Every time commentary_cycle runs evaluate_event(), the decision is written
here, whether R2 spoke or stayed silent. Suppressed events are the valuable
ones: they show what the system chose NOT to say, and why.

One file per trading day (Eastern date):
    data/sessions/YYYY-MM-DD-events.jsonl

Each line is one JSON object:
    timestamp            ISO-8601 UTC
    cycle                RuntimeState.update_count
    trigger              e.g. "regime_change", "periodic_check"
    priority             "critical" | "important" | "informational" | "suppressed"
    score                event significance score (0–100)
    suppressed           True when event intelligence stayed silent
                         (below threshold, or informational rate limit)
    rate_limited         True when the event passed scoring but was held
                         back by the informational rate limit
    change_summary       human-readable summary from evaluate_event()
    change_keys          list of ChangeItem keys
    confidence           {structural, calibrated, regime_confidence,
                          source_cycle} or None if not computed
    commentary_generated True when commentary was produced and shown
    commentary           first 200 chars of the commentary, or ""

Writes never raise: a failure logs a warning and the loop carries on.

CLI:
    python -m memory.event_journal [YYYY-MM-DD]     (default: today, ET)
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("memory.event_journal")

try:
    from zoneinfo import ZoneInfo
    _ET: Optional[ZoneInfo] = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    _ET = None

# Repo-root data/sessions, independent of the working directory.
JOURNAL_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "sessions"

COMMENTARY_PREVIEW_CHARS = 200

_write_lock = threading.Lock()

DateLike = Union[date, str, None]


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

def et_date(dt: Optional[datetime] = None) -> date:
    """Eastern-time calendar date for dt (default: now)."""
    dt = dt or datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if _ET is not None:
        return dt.astimezone(_ET).date()
    return (dt.astimezone(timezone.utc) - timedelta(hours=5)).date()


def _as_date(d: DateLike) -> date:
    if d is None:
        return et_date()
    if isinstance(d, datetime):
        return et_date(d)
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))


def journal_path(d: DateLike = None, directory: Optional[Path] = None) -> Path:
    """Path of the journal file for an ET date."""
    base = Path(directory) if directory is not None else JOURNAL_DIR
    return base / f"{_as_date(d).isoformat()}-events.jsonl"


# ──────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────

def _confidence_snapshot(confidence_state: Any) -> Optional[dict]:
    if confidence_state is None:
        return None
    return {
        "structural":        getattr(confidence_state, "structural_confidence", None),
        "calibrated":        getattr(confidence_state, "calibrated_confidence", None),
        "regime_confidence": getattr(confidence_state, "regime_confidence", None),
        "source_cycle":      getattr(confidence_state, "source_cycle", None),
    }


def build_entry(
    event_result:     Any,
    *,
    cycle:            int = 0,
    trigger:          str = "",
    confidence_state: Any = None,
    rate_limited:     bool = False,
    commentary:       Optional[str] = None,
    now:              Optional[datetime] = None,
) -> dict:
    """Build one journal entry from an EventResult. Pure — no I/O."""
    now = now or datetime.now(tz=timezone.utc)
    suppress = bool(getattr(event_result, "suppress", False))
    changes  = getattr(event_result, "changes", None) or []
    return {
        "timestamp":            now.astimezone(timezone.utc).isoformat(),
        "cycle":                int(cycle or 0),
        "trigger":              trigger or getattr(event_result, "trigger", ""),
        "priority":             getattr(event_result, "priority", ""),
        "score":                float(getattr(event_result, "score", 0.0) or 0.0),
        "suppressed":           suppress or rate_limited,
        "rate_limited":         bool(rate_limited),
        "change_summary":       getattr(event_result, "change_summary", "") or "",
        "change_keys":          [getattr(c, "key", str(c)) for c in changes],
        "confidence":           _confidence_snapshot(confidence_state),
        "commentary_generated": bool(commentary),
        "commentary":           (commentary or "")[:COMMENTARY_PREVIEW_CHARS],
    }


def append_entry(entry: dict, directory: Optional[Path] = None) -> bool:
    """
    Append one entry to its day's file (date taken from entry["timestamp"]).
    Returns True on success. Never raises.
    """
    try:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except Exception:
            ts = None
        path = journal_path(et_date(ts), directory)
        line = json.dumps(entry, default=str, separators=(",", ":"))
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        return True
    except Exception as e:
        logger.warning(f"[journal] write failed (non-fatal): {e}")
        return False


def record_decision(
    event_result:     Any,
    *,
    cycle:            int = 0,
    trigger:          str = "",
    confidence_state: Any = None,
    rate_limited:     bool = False,
    commentary:       Optional[str] = None,
    directory:        Optional[Path] = None,
) -> bool:
    """Build and append a journal entry. Never raises."""
    try:
        entry = build_entry(
            event_result,
            cycle=cycle, trigger=trigger, confidence_state=confidence_state,
            rate_limited=rate_limited, commentary=commentary,
        )
    except Exception as e:
        logger.warning(f"[journal] could not build entry (non-fatal): {e}")
        return False
    return append_entry(entry, directory)


# ──────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────

def load_session(d: DateLike = None, directory: Optional[Path] = None) -> list[dict]:
    """All entries for an ET date, in write order. Missing file → []."""
    path = journal_path(d, directory)
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial last line (e.g. crash mid-write) shouldn't hide the rest.
                logger.warning(f"[journal] skipping malformed line {n} in {path.name}")
    return entries


def _calibrated(entry: dict) -> Optional[float]:
    conf = entry.get("confidence") or {}
    val = conf.get("calibrated")
    return float(val) if isinstance(val, (int, float)) else None


def session_stats(entries: list[dict]) -> dict:
    """Summary numbers for a list of entries. Pure."""
    approved = [e for e in entries if not e.get("suppressed")]
    by_priority: dict[str, int] = {}
    for e in approved:
        p = e.get("priority") or "unknown"
        by_priority[p] = by_priority.get(p, 0) + 1

    confs = [c for c in (_calibrated(e) for e in entries) if c is not None]
    keys: set[str] = set()
    for e in entries:
        keys.update(e.get("change_keys") or [])

    return {
        "total":                len(entries),
        "approved":             len(approved),
        "approved_by_priority": by_priority,
        "suppressed":           sum(1 for e in entries if e.get("suppressed")),
        "rate_limited":         sum(1 for e in entries if e.get("rate_limited")),
        "confidence_min":       min(confs) if confs else None,
        "confidence_max":       max(confs) if confs else None,
        "confidence_final":     confs[-1] if confs else None,
        "change_keys":          sorted(keys),
        "commentary_count":     sum(1 for e in entries if e.get("commentary_generated")),
    }


def format_summary(d: date, stats: dict) -> str:
    def pct(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.0f}%"

    lines = [f"R2 event journal — {d.isoformat()} (ET)"]
    if stats["total"] == 0:
        lines.append("  No decisions recorded.")
        return "\n".join(lines)

    order = ["critical", "important", "informational"]
    prios = stats["approved_by_priority"]
    ordered = [p for p in order if p in prios] + sorted(p for p in prios if p not in order)
    prio_txt = ", ".join(f"{p} {prios[p]}" for p in ordered) or "none"

    lines += [
        f"  Total decisions:   {stats['total']}",
        f"  Approved:          {stats['approved']}  ({prio_txt})",
        f"  Suppressed:        {stats['suppressed']}"
        + (f"  (incl. {stats['rate_limited']} rate-limited)" if stats["rate_limited"] else ""),
        f"  Final Conviction:  min {pct(stats['confidence_min'])}  "
        f"max {pct(stats['confidence_max'])}  final {pct(stats['confidence_final'])}",
        f"  Change keys seen:  {', '.join(stats['change_keys']) or 'none'}",
        f"  Commentary:        {stats['commentary_count']}",
    ]
    return "\n".join(lines)


def summarize_session(d: DateLike = None, directory: Optional[Path] = None) -> dict:
    """Print a summary of one ET day's decisions and return the numbers."""
    day   = _as_date(d)
    stats = session_stats(load_session(day, directory))
    print(format_summary(day, stats))
    return stats


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        print("usage: python -m memory.event_journal [YYYY-MM-DD]", file=sys.stderr)
        return 2
    try:
        day = _as_date(argv[0] if argv else None)
    except ValueError:
        print(f"invalid date: {argv[0]!r} (expected YYYY-MM-DD)", file=sys.stderr)
        return 2
    summarize_session(day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
