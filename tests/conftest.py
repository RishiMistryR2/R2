"""
Shared pytest fixtures.

The event journal appends to data/sessions/ by default. Redirect it to a
temp directory for every test so the suite never writes into a real
trading day's journal.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_event_journal(tmp_path, monkeypatch):
    import memory.event_journal as ej
    monkeypatch.setattr(ej, "JOURNAL_DIR", tmp_path / "sessions")
    yield
