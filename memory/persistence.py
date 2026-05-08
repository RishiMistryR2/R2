"""
memory/persistence.py — atomic JSON persistence for the memory layer.

Design rules:
  - Atomic writes: write to .tmp, then os.replace — never corrupt an existing file
  - schema_version stamped on every persisted object
  - No database — plain JSON files under data/memory/
  - load() returns None on missing / corrupt (never raises)

Public API:
    save(path, data)              — atomic dict → JSON
    load(path)                    — JSON → dict | None
    save_list(path, items)        — save list of dicts
    load_list(path)               — load list of dicts ([] on missing)
    list_files(directory, pat)    — sorted Path list
    ensure_dir(path)              — mkdir -p
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import MEMORY_SCHEMA_VERSION

logger   = logging.getLogger(__name__)
ENCODING = "utf-8"


# ──────────────────────────────────────────────
# Core save / load
# ──────────────────────────────────────────────

def save(path: Path, data: Dict[str, Any]) -> bool:
    """
    Atomically write *data* as JSON to *path*.

    Writes to a temp file in the same directory then renames
    so a crash mid-write never leaves a half-written file.
    Returns True on success, False on any failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(data)
    data.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
    data.setdefault("saved_at", datetime.now(tz=timezone.utc).isoformat())

    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding=ENCODING) as f:
                json.dump(data, f, indent=2, default=_json_default)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error(f"persistence.save({path}): {e}")
        return False


def load(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from *path*.  Returns None if missing or corrupt."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding=ENCODING) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"persistence.load({path}): {e}")
        return None


def save_list(path: Path, items: List[Dict[str, Any]]) -> bool:
    """Save a list of dicts wrapped in a single JSON object.  Atomic."""
    return save(path, {"items": items, "count": len(items)})


def load_list(path: Path) -> List[Dict[str, Any]]:
    """Load a list saved by save_list.  Returns [] on missing / corrupt."""
    data = load(path)
    if data is None:
        return []
    return data.get("items", [])


def list_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    """Sorted list of files in *directory* matching *pattern*."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob(pattern))


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing.  Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────
# JSON serialisation helper
# ──────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):               # Enum
        return obj.value
    if hasattr(obj, "to_dict"):             # dataclass with to_dict
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    raise TypeError(f"Not JSON serialisable: {type(obj)}")