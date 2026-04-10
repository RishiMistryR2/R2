"""
config/logging_setup.py — configure logging for the entire R2 process.

Call setup_logging() once at application startup (main.py or CLI entry point).
All other modules use: logger = logging.getLogger(__name__)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import (
    LOG_LEVEL,
    LOG_TO_FILE,
    LOG_FILE,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    ASSISTANT_NAME,
)


def setup_logging() -> None:
    """
    Configure root logger for the R2 process.

    - Always logs to stdout at the configured level.
    - Optionally logs to a rotating file (LOG_TO_FILE=1).
    - Suppresses noisy third-party loggers.
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Stdout handler ──────────────────────────────────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(LOG_LEVEL)
    root.addHandler(stdout_handler)

    # ── File handler (optional) ─────────────────────────────────────────────
    if LOG_TO_FILE:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVEL)
        root.addHandler(file_handler)

    # ── Suppress noisy third-party loggers ──────────────────────────────────
    for noisy in ("websockets", "aiohttp", "urllib3", "asyncio",
                  "httpcore", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"{ASSISTANT_NAME} logging initialized (level={LOG_LEVEL}, "
        f"file={'on' if LOG_TO_FILE else 'off'})"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper. Usage:
        from config.logging_setup import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
