from __future__ import annotations

from actrone_memory.exceptions import (
    ActroneMemoryError,
    ConfigurationError,
    EmbeddingError,
    MemoryNotFoundError,
    StoreConnectionError,
    TokenBudgetError,
    ValidationError,
)


def test_repr_includes_code_and_message():
    err = ConfigurationError("bad config")
    assert "ERR_CONFIGURATION" in repr(err)
    assert "bad config" in repr(err)


def test_embedding_error_sets_code():
    err = EmbeddingError("provider timeout")
    assert err.code == "ERR_EMBEDDING"
    assert err.message == "provider timeout"


def test_memory_not_found_error_sets_memory_id():
    err = MemoryNotFoundError("mem-abc")
    assert err.code == "ERR_MEMORY_NOT_FOUND"
    assert err.memory_id == "mem-abc"
    assert "mem-abc" in str(err)


def test_store_connection_error_sets_store_and_cause():
    cause = RuntimeError("refused")
    err = StoreConnectionError("Redis", cause)
    assert err.store == "Redis"
    assert err.cause is cause
    assert "Redis" in str(err)


def test_token_budget_error():
    err = TokenBudgetError("budget must be > 0")
    assert err.code == "ERR_TOKEN_BUDGET"


def test_all_errors_inherit_from_base():
    for cls in (ConfigurationError, EmbeddingError, MemoryNotFoundError, TokenBudgetError, ValidationError):
        assert issubclass(cls, ActroneMemoryError)


def test_error_details_dict_populated():
    err = StoreConnectionError("Redis", RuntimeError("refused"))
    assert err.details["store"] == "Redis"
    assert "cause" in err.details


def test_validation_error_sets_field():
    err = ValidationError("agent_id", "must not be empty")
    assert err.field == "agent_id"
    assert err.code == "ERR_VALIDATION"
    assert err.details["field"] == "agent_id"


def test_configuration_error_with_details():
    err = ConfigurationError("bad config", details={"key": "ACTRONE_OPENAI_API_KEY"})
    assert err.details["key"] == "ACTRONE_OPENAI_API_KEY"
