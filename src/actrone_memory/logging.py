"""Logging configuration helpers for actrone-memory.

As a library, actrone-memory does NOT configure structlog globally — that is
the responsibility of the application that imports this library.

Call ``configure_json_logging()`` once at application startup to emit
production-ready structured JSON logs. For local development, call
``configure_dev_logging()`` instead for human-readable coloured output.

If you have your own structlog configuration, there is nothing to call here —
actrone-memory's loggers will inherit it automatically.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import cast

import structlog

# ── Standard field schema (CLAUDE.md §6.3) ────────────────────────────────────
# Reserved keys every log line should carry. Bind via `bind_logger` instead of
# adding ad-hoc kwargs so log aggregators can filter on a stable shape.
#
#   timestamp / level   — added by structlog
#   service             — always "actrone-memory"
#   request_id          — embedder-supplied correlation ID
#   trace_id            — OTel trace ID, when available
#   event               — machine-readable event name (e.g. "memory.context.retrieved")
#   agent_id / session_id — per-call identifiers

_SERVICE_NAME = "actrone-memory"

_request_id_var: ContextVar[str | None] = ContextVar("actrone_memory_request_id", default=None)
_trace_id_var: ContextVar[str | None] = ContextVar("actrone_memory_trace_id", default=None)


def set_request_id(value: str | None) -> None:
    """Set the correlation ID for log lines emitted in the current async task."""
    _request_id_var.set(value)


def set_trace_id(value: str | None) -> None:
    """Set the OTel trace ID for log lines emitted in the current async task."""
    _trace_id_var.set(value)


def bind_logger(name: str, **fields: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger pre-bound with the standard field schema.

    Use this in new call sites instead of ``structlog.get_logger(name)``; the
    standard fields will appear on every line emitted by the returned logger
    without callers having to remember them.
    """
    logger = structlog.get_logger(name)
    # structlog.get_logger / .bind() are typed as returning Any; cast back to the
    # documented return type for callers under strict mypy.
    return cast(
        "structlog.stdlib.BoundLogger",
        logger.bind(
            service=_SERVICE_NAME,
            request_id=_request_id_var.get(),
            trace_id=_trace_id_var.get(),
            **fields,
        ),
    )


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure structlog for production: JSON output to stdout.

    Every log line will include: timestamp, level, logger name, and all
    key-value pairs passed to the logger. This format is machine-parseable
    by log aggregators (Loki, Datadog, CloudWatch, etc.).

    Call this once at application startup, before any other code runs.

    Example::

        from actrone_memory.logging import configure_json_logging
        configure_json_logging()
    """
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level)


def configure_dev_logging(level: int = logging.DEBUG) -> None:
    """Configure structlog for local development: coloured, human-readable output.

    Call this once at application startup when running locally.

    Example::

        from actrone_memory.logging import configure_dev_logging
        configure_dev_logging()
    """
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level)
