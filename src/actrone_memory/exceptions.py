from __future__ import annotations


class ActroneMemoryError(Exception):
    """Base error for all actrone-memory exceptions.

    Every error carries:
        code, machine-readable string for programmatic handling (e.g. in logs or alerts)
        message, human-readable description
        details, structured dict of context that helps locate the failure
    """

    def __init__(
        self, message: str, *, code: str, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, object] = details or {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r}, details={self.details!r})"
        )


class ConfigurationError(ActroneMemoryError):
    """Raised when required configuration is missing or invalid.

    This always means something is wrong with how the library is set up, not a runtime failure.
    It should cause the application to exit.
    """

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message, code="ERR_CONFIGURATION", details=details)


class StoreConnectionError(ActroneMemoryError):
    """Raised when a connection to Redis or Qdrant cannot be established or maintained.

    Includes the name of the store that failed and the underlying exception.
    """

    def __init__(
        self, store: str, cause: Exception, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(
            f"Failed to connect to {store}: {cause}",
            code="ERR_STORE_CONNECTION",
            details={"store": store, "cause": str(cause), **(details or {})},
        )
        self.store = store
        self.cause = cause


class EmbeddingError(ActroneMemoryError):
    """Raised when an embedding provider returns an error or an unexpected response."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message, code="ERR_EMBEDDING", details=details)


class MemoryNotFoundError(ActroneMemoryError):
    """Raised when ``delete_memory()`` is called with an ID that does not exist."""

    def __init__(self, memory_id: str) -> None:
        super().__init__(
            f"Memory not found: {memory_id}",
            code="ERR_MEMORY_NOT_FOUND",
            details={"memory_id": memory_id},
        )
        self.memory_id = memory_id


class TokenBudgetError(ActroneMemoryError):
    """Raised when ``token_budget`` is zero or negative in ``retrieve_context()``."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message, code="ERR_TOKEN_BUDGET", details=details)


class ValidationError(ActroneMemoryError):
    """Raised when a public method receives an invalid argument at the API boundary."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(
            f"Invalid value for '{field}': {message}",
            code="ERR_VALIDATION",
            details={"field": field},
        )
        self.field = field
