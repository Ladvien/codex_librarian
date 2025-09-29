"""
Basic unit tests for CORE-006 error handling implementation.

This tests the core error handling functionality without requiring
external dependencies like pytest-asyncio.
"""


import pytest


def test_basic_error_creation():
    """Test basic error creation and sanitization."""
    from src.pdf_to_markdown_mcp.core.errors import (
        PDFToMarkdownError,
        create_correlation_id,
    )

    # Test basic error creation
    correlation_id = create_correlation_id()
    error = PDFToMarkdownError(
        message="Test error", correlation_id=correlation_id, error_code="TEST001"
    )

    assert str(error) == "Test error"
    assert error.correlation_id == correlation_id
    assert error.error_code == "TEST001"
    assert error.timestamp is not None

    # Test user message sanitization
    user_message = error.get_user_message()
    assert isinstance(user_message, str)
    assert len(user_message) > 0


def test_security_error_sanitization():
    """Test that security errors never expose sensitive information."""
    from src.pdf_to_markdown_mcp.core.errors import (
        SecurityError,
        sanitize_error_for_user,
    )

    # Create security error with sensitive internal details
    error = SecurityError(
        message="Authentication failed", internal_details="Database password: secret123"
    )

    # Test sanitization
    user_message = sanitize_error_for_user(error)
    assert "secret123" not in user_message
    assert "password" not in user_message.lower()
    assert (
        "security error" in user_message.lower()
        or "authentication failed" in user_message.lower()
    )


def test_error_categorization():
    """Test error categorization for retry logic."""
    from src.pdf_to_markdown_mcp.core.errors import (
        ErrorCategory,
        ResourceError,
        SecurityError,
        TransientError,
        ValidationError,
        categorize_error,
        is_retryable_error,
    )

    # Test categorization
    assert (
        categorize_error(ValidationError("Invalid input")) == ErrorCategory.VALIDATION
    )
    assert categorize_error(SecurityError("Access denied")) == ErrorCategory.SECURITY
    assert (
        categorize_error(TransientError("Temporary failure")) == ErrorCategory.TRANSIENT
    )
    assert categorize_error(ResourceError("Out of memory")) == ErrorCategory.RESOURCE

    # Test retryable detection
    assert not is_retryable_error(ValidationError("Invalid input"))
    assert not is_retryable_error(SecurityError("Access denied"))
    assert is_retryable_error(TransientError("Temporary failure"))
    assert is_retryable_error(ResourceError("Out of memory"))


def test_correlation_id_uniqueness():
    """Test that correlation IDs are unique."""
    from src.pdf_to_markdown_mcp.core.errors import create_correlation_id

    id1 = create_correlation_id()
    id2 = create_correlation_id()

    assert id1 != id2
    assert len(id1) >= 8
    assert "-" in id1


def test_retry_config_and_strategy():
    """Test retry configuration and strategy selection."""
    from src.pdf_to_markdown_mcp.core.errors import (
        ExponentialBackoffRetry,
        ResourceError,
        RetryConfig,
        TransientError,
        ValidationError,
        get_retry_strategy,
    )

    # Test retry config
    config = RetryConfig(max_retries=3, base_delay=1.0, backoff_multiplier=2.0)
    retry_mechanism = ExponentialBackoffRetry(config)

    # Test delay calculation
    assert retry_mechanism.get_delay(0) == 1.0
    assert retry_mechanism.get_delay(1) == 2.0
    assert retry_mechanism.get_delay(2) == 4.0

    # Test strategy selection
    validation_strategy = get_retry_strategy(ValidationError("Invalid"))
    assert validation_strategy.max_retries == 0  # No retries for validation errors

    transient_strategy = get_retry_strategy(TransientError("Temporary"))
    assert transient_strategy.max_retries > 0  # Should allow retries

    resource_strategy = get_retry_strategy(ResourceError("Out of memory"))
    assert resource_strategy.max_retries > 0  # Should allow retries


def test_error_tracking():
    """Test basic error tracking functionality."""
    from src.pdf_to_markdown_mcp.core.errors import (
        ErrorTracker,
        TransientError,
        ValidationError,
    )

    tracker = ErrorTracker(window_size=60)  # 1 minute window

    # Record some errors
    tracker.record_error(
        error=ValidationError("Invalid input"),
        operation="test_operation",
        component="test_component",
    )

    tracker.record_error(
        error=TransientError("Temporary failure"),
        operation="test_operation",
        component="test_component",
    )

    # Get metrics
    metrics = tracker.get_metrics()
    assert metrics.total_errors == 2
    assert "ValidationError" in metrics.errors_by_type
    assert "TransientError" in metrics.errors_by_type
    assert "test_operation" in metrics.errors_by_operation


def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality."""
    from src.pdf_to_markdown_mcp.core.errors import CircuitBreaker, CircuitBreakerError

    circuit_breaker = CircuitBreaker(failure_threshold=2, name="test_service")

    # Initially should be closed
    assert circuit_breaker.state == "CLOSED"

    # Record failures to trigger opening
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()

    # Should now be open
    assert circuit_breaker.state == "OPEN"

    # Should raise error when trying to call
    with pytest.raises(CircuitBreakerError):
        circuit_breaker.call(lambda: "test")


def test_structured_logging():
    """Test structured logging format."""
    from src.pdf_to_markdown_mcp.core.errors import (
        PDFToMarkdownError,
        sanitize_log_message,
    )

    error = PDFToMarkdownError(
        message="Test error with sensitive data: password=secret123",
        error_code="TEST001",
        metadata={"connection_string": "postgres://user:pass@host/db"},
    )

    # Test structured log format
    log_data = error.to_structured_log()
    assert log_data["error_type"] == "PDFToMarkdownError"
    assert log_data["error_code"] == "TEST001"
    assert log_data["correlation_id"] is not None
    assert "pass" not in str(log_data)  # Sensitive data should be removed

    # Test message sanitization
    sanitized = sanitize_log_message("password=secret123 and token=abc123")
    assert "secret123" not in sanitized
    assert "abc123" not in sanitized
    assert "[REDACTED]" in sanitized


def test_security_validation_helpers():
    """Test security validation helper functions."""
    from src.pdf_to_markdown_mcp.core.errors import (
        InputValidationError,
        validate_file_path,
        validate_input_size,
    )

    # Test valid file path
    valid_path = validate_file_path("document.pdf")
    assert valid_path == "document.pdf"

    # Test directory traversal detection
    with pytest.raises(InputValidationError):
        validate_file_path("../../../etc/passwd")

    with pytest.raises(InputValidationError):
        validate_file_path("..\\windows\\system32")

    # Test input size validation
    valid_input = validate_input_size("short input", max_size=100)
    assert valid_input == "short input"

    with pytest.raises(InputValidationError):
        validate_input_size("x" * 1000, max_size=100)


if __name__ == "__main__":
    # Run tests manually if called directly
    test_basic_error_creation()
    test_security_error_sanitization()
    test_error_categorization()
    test_correlation_id_uniqueness()
    test_retry_config_and_strategy()
    test_error_tracking()
    test_circuit_breaker_basic()
    test_structured_logging()
    test_security_validation_helpers()
    print("All basic error handling tests passed!")
