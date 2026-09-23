"""Logging configuration helpers for actrone-memory.

As a library, actrone-memory does NOT configure structlog globally, that is
the responsibility of the application that imports this library.

Call ``configure_json_logging()`` once at application startup to emit
production-ready structured JSON logs. For local development, call
``configure_dev_logging()`` instead for human-readable coloured output.

If you have your own structlog configuration, there is nothing to call here: actrone-memory's
loggers will inherit it automatically. With no configuration at all, the library is quiet: its
events go to the standard ``logging`` module, so only warnings reach stderr.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog

# ── Standard field schema ──────────────────────────────────────────────────
# Reserved keys every log line should carry. Bind via `bind_logger` instead of
# adding ad-hoc kwargs so log aggregators can filter on a stable shape.
#
#   timestamp / level, added by structlog
#   service, always "actrone-memory"
#   request_id, embedder-supplied correlation ID
#   trace_id, OTel trace ID, when available
#   event, machine-readable event name (e.g. "memory.context.retrieved")
#   agent_id / session_id, per-call identifiers

_SERVICE_NAME = "actrone-memory"


def set_request_id(value: str | None) -> None:
    """Set the correlation ID carried by log lines in the current context.

    The ID is stored in structlog's own context-local store, so it is picked up
    by the ``merge_contextvars`` processor when each event is rendered. That is
    what makes it apply to loggers created before the ID was known (module-level
    loggers, which is how this library creates them) instead of only to loggers
    bound afterwards. Pass ``None`` to clear it, e.g. when a request ends.

    Requires a structlog configuration that includes ``merge_contextvars``;
    :func:`configure_json_logging` and :func:`configure_dev_logging` both do.
    """
    if value is None:
        structlog.contextvars.unbind_contextvars("request_id")
    else:
        structlog.contextvars.bind_contextvars(request_id=value)


def set_trace_id(value: str | None) -> None:
    """Set the OTel trace ID carried by log lines in the current context.

    Same context-local mechanism as :func:`set_request_id`. Pass ``None`` to clear.
    """
    if value is None:
        structlog.contextvars.unbind_contextvars("trace_id")
    else:
        structlog.contextvars.bind_contextvars(trace_id=value)


def clear_log_context() -> None:
    """Drop the request/trace IDs bound for the current context.

    Call this when a unit of work finishes, so IDs cannot leak into the next one
    on a reused worker task.
    """
    structlog.contextvars.unbind_contextvars("request_id", "trace_id")


def bind_logger(name: str, **fields: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger carrying the standard field schema.

    Use this instead of ``structlog.get_logger(name)`` so every line carries
    ``service``. ``request_id`` and ``trace_id`` are deliberately not attached
    here: they are resolved per event from the context (see
    :func:`set_request_id`), so a logger created at import time still picks up an
    ID that is set later, per request.

    The fields are passed as ``get_logger`` initial values rather than through
    ``.bind()``, which matters for module-level loggers: ``.bind()`` materialises
    the logger against whatever configuration exists at import time, freezing the
    renderer, so an application calling :func:`configure_json_logging` afterwards
    would keep getting console output. Initial values stay lazy and pick up the
    configuration in force when the first line is actually emitted.
    """
    # The proxy is duck-typed to BoundLogger (it forwards every attribute); cast back to the
    # documented return type for callers under strict mypy.
    return cast(
        "structlog.stdlib.BoundLogger",
        _LibraryLogger(name, {"service": _SERVICE_NAME, **fields}),
    )


# Used only while the application has not configured structlog. Rendered lines go to the standard
# logging module, which drops anything below WARNING until the application configures it and prints
# warnings to stderr through logging's last-resort handler.
_UNCONFIGURED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_log_level,
    structlog.dev.ConsoleRenderer(colors=False),
]


class _LibraryLogger:
    """A logger that follows the application's logging setup, and stays quiet without one.

    structlog's unconfigured default prints every level, debug included, to stdout, which would put
    this library's internal events into the output of any program that simply imports it. So each
    call checks whether the application has configured structlog: if it has, the call goes to
    structlog and inherits the application's processors, renderer and level; if it has not, the call
    goes to the standard ``logging`` logger of the same name, where the application's (or Python's
    default) level applies.
    """

    def __init__(self, name: str, fields: dict[str, object]) -> None:
        self._name = name
        self._fields = fields

    def _target(self) -> Any:  # noqa: ANN401  (a structlog BoundLogger, typed Any by structlog)
        if structlog.is_configured():
            return structlog.get_logger(self._name, **self._fields)
        return structlog.wrap_logger(
            logging.getLogger(self._name),
            processors=_UNCONFIGURED_PROCESSORS,
            wrapper_class=structlog.stdlib.BoundLogger,
        ).bind(**self._fields)

    def __getattr__(self, attribute: str) -> Any:  # noqa: ANN401  (forwards any logger method)
        return getattr(self._target(), attribute)


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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        # Must be the stdlib factory, not PrintLoggerFactory: the stdlib
        # processors above read `logger.name` and the stdlib level, which a
        # PrintLogger does not have. Pairing them raises AttributeError on the
        # first line emitted, and makes the `level` argument do nothing.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # format="%(message)s" keeps the rendered JSON as the entire line, with no
    # stdlib prefix wrapped around it.
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout, force=True)


def configure_dev_logging(level: int = logging.DEBUG) -> None:
    """Configure structlog for local development: coloured, human-readable output.

    Call this once at application startup when running locally.

    Example::

        from actrone_memory.logging import configure_dev_logging
        configure_dev_logging()
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        # See configure_json_logging: the stdlib processors above require the
        # stdlib logger factory.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout, force=True)
