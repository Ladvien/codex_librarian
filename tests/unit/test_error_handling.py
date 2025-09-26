"""
Tests for comprehensive error handling and retry mechanisms.

Following TDD approach for CORE-006 implementation.
"""

import time
from unittest.mock import Mock

import pytest

from src.pdf_to_markdown_mcp.core.errors import (
    AuthenticationError,
    CircuitBreaker,
    CircuitBreakerError,
    DatabaseError,
    ErrorCategory,
    ErrorTracker,
    ExponentialBackoffRetry,
    InputValidationError,
    # Basic error hierarchy
    PDFToMarkdownError,
    PermanentError,
    ProcessingError,
    RateLimitError,
    ResourceError,
    # Retry mechanisms
    RetryConfig,
    RetryManager,
    # Security-focused errors
    SecurityError,
    TimeoutError,
    # Retry and resilience errors
    TransientError,
    ValidationError,
    categorize_error,
    create_correlation_id,
    get_retry_strategy,
    is_retryable_error,
    sanitize_error_for_user,
)


class TestErrorHierarchy:
    """Test error hierarchy and categorization following TDD."""

    def test_base_error_creation(self):
        """Test that base error can be created with security context."""
        # Given
        message = "Test error message"
        correlation_id = "test-123"

        # When
        error = PDFToMarkdownError(
            message=message, correlation_id=correlation_id, error_code="PDF001"
        )

        # Then
        assert str(error) == message
        assert error.correlation_id == correlation_id
        assert error.error_code == "PDF001"
        assert error.timestamp is not None
        assert not error.sensitive_data  # Should be empty by default

    def test_security_error_never_exposes_sensitive_data(self):
        """Test that security errors never expose sensitive information."""
        # Given
        internal_details = "Database password: secret123"

        # When
        error = SecurityError(
            message="Authentication failed", internal_details=internal_details
        )

        # Then
        user_message = sanitize_error_for_user(error)
        assert "secret123" not in user_message
        assert "password" not in user_message
        assert (
            "Authentication failed" in user_message or "Security error" in user_message
        )

    def test_error_categorization(self):
        """Test that errors are correctly categorized for retry logic."""
        # Given & When & Then
        assert (
            categorize_error(ValidationError("Invalid input"))
            == ErrorCategory.VALIDATION
        )
        assert (
            categorize_error(TransientError("Temporary failure"))
            == ErrorCategory.TRANSIENT
        )
        assert (
            categorize_error(DatabaseError("Connection timeout"))
            == ErrorCategory.TRANSIENT
        )
        assert (
            categorize_error(PermanentError("Invalid file format"))
            == ErrorCategory.PERMANENT
        )
        assert (
            categorize_error(ResourceError("Out of memory")) == ErrorCategory.RESOURCE
        )
        assert (
            categorize_error(SecurityError("Access denied")) == ErrorCategory.SECURITY
        )

    def test_retryable_error_detection(self):
        """Test that retryable errors are correctly identified."""
        # Given & When & Then
        assert is_retryable_error(TransientError("Temporary failure"))
        assert is_retryable_error(DatabaseError("Connection timeout"))
        assert is_retryable_error(ResourceError("Temporary resource limit"))

        assert not is_retryable_error(ValidationError("Invalid input"))
        assert not is_retryable_error(SecurityError("Access denied"))
        assert not is_retryable_error(PermanentError("Invalid file format"))

    def test_correlation_id_generation(self):
        """Test correlation ID generation for error tracking."""
        # Given & When
        correlation_id1 = create_correlation_id()
        correlation_id2 = create_correlation_id()

        # Then
        assert correlation_id1 != correlation_id2
        assert len(correlation_id1) >= 8  # Should be reasonably long
        assert "-" in correlation_id1  # Should have readable format


class TestRetryMechanisms:
    """Test retry mechanisms and circuit breaker patterns."""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff retry delay calculation."""
        # Given
        retry_config = RetryConfig(
            max_retries=5, base_delay=1.0, max_delay=60.0, backoff_multiplier=2.0
        )
        retry_mechanism = ExponentialBackoffRetry(retry_config)

        # When & Then
        assert retry_mechanism.get_delay(0) == 1.0
        assert retry_mechanism.get_delay(1) == 2.0
        assert retry_mechanism.get_delay(2) == 4.0
        assert retry_mechanism.get_delay(3) == 8.0
        assert retry_mechanism.get_delay(10) <= 60.0  # Should cap at max_delay

    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        # Given
        circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30, success_threshold=2
        )

        # Then - Initial state should be CLOSED
        assert circuit_breaker.state == "CLOSED"

        # When - Record failures up to threshold
        for _ in range(3):
            circuit_breaker.record_failure()

        # Then - Should transition to OPEN
        assert circuit_breaker.state == "OPEN"

        # When - Try to call while OPEN
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(lambda: "test")

    @pytest.mark.asyncio
    async def test_async_retry_with_circuit_breaker(self):
        """Test async retry mechanism with circuit breaker integration."""
        # Given
        retry_manager = RetryManager()
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        # When
        result = await retry_manager.execute_with_retry(
            failing_operation, operation_name="test_operation"
        )

        # Then
        assert result == "success"
        assert call_count == 3

    def test_retry_strategy_selection(self):
        """Test that appropriate retry strategy is selected based on error type."""
        # Given & When & Then
        strategy = get_retry_strategy(DatabaseError("Connection failed"))
        assert strategy.max_retries >= 3
        assert strategy.base_delay > 0

        strategy = get_retry_strategy(ValidationError("Invalid input"))
        assert strategy.max_retries == 0  # No retries for validation errors

        strategy = get_retry_strategy(SecurityError("Access denied"))
        assert strategy.max_retries == 0  # No retries for security errors


class TestErrorTracking:
    """Test error tracking and metrics collection."""

    def test_error_tracker_records_metrics(self):
        """Test that error tracker properly records error metrics."""
        # Given
        error_tracker = ErrorTracker()

        # When
        error_tracker.record_error(
            error=DatabaseError("Connection failed"),
            operation="database_query",
            component="db_session",
        )

        # Then
        metrics = error_tracker.get_metrics()
        assert metrics.total_errors == 1
        assert "database_query" in metrics.errors_by_operation
        assert "DatabaseError" in metrics.errors_by_type
        assert "TRANSIENT" in metrics.errors_by_category

    def test_error_rate_limiting_tracking(self):
        """Test error rate tracking for potential abuse detection."""
        # Given
        error_tracker = ErrorTracker()

        # When - Record multiple validation errors quickly
        for _ in range(10):
            error_tracker.record_error(
                error=ValidationError("Invalid input"),
                operation="api_endpoint",
                user_context={"user_id": "test_user"},
            )

        # Then
        metrics = error_tracker.get_metrics()
        assert metrics.should_rate_limit("test_user")
        assert metrics.get_error_rate("test_user") > 0.5  # High error rate

    def test_structured_error_logging(self):
        """Test that errors are logged in structured format without sensitive data."""
        # Given
        error = DatabaseError(
            message="Query failed",
            operation="select_documents",
            internal_details="Connection string: postgres://user:pass@host/db",
        )

        # When
        log_data = error.to_structured_log()

        # Then
        assert log_data["error_type"] == "DatabaseError"
        assert log_data["message"] == "Query failed"
        assert log_data["correlation_id"] is not None
        assert "pass" not in str(log_data)  # Sensitive data should be removed
        assert "postgres" not in str(log_data)  # Connection details should be removed


class TestGracefulDegradation:
    """Test graceful degradation mechanisms."""

    @pytest.mark.asyncio
    async def test_service_fallback_mechanism(self):
        """Test that services gracefully fall back when primary fails."""
        # Given
        primary_service = Mock()
        primary_service.process.side_effect = TransientError("Service unavailable")

        fallback_service = Mock()
        fallback_service.process.return_value = "fallback_result"

        # When
        from src.pdf_to_markdown_mcp.core.errors import GracefulDegradationManager

        manager = GracefulDegradationManager()

        result = await manager.execute_with_fallback(
            primary_operation=primary_service.process,
            fallback_operation=fallback_service.process,
            operation_name="test_service",
        )

        # Then
        assert result == "fallback_result"
        assert primary_service.process.called
        assert fallback_service.process.called

    def test_timeout_handling_with_resource_cleanup(self):
        """Test that timeouts are handled with proper resource cleanup."""
        # Given
        resource_manager = Mock()

        def long_running_operation():
            time.sleep(10)  # Simulates long operation
            return "should_not_reach"

        # When
        with pytest.raises(TimeoutError):
            from src.pdf_to_markdown_mcp.core.errors import TimeoutManager

            timeout_manager = TimeoutManager(timeout_seconds=1)
            timeout_manager.execute_with_timeout(
                operation=long_running_operation, resource_manager=resource_manager
            )

        # Then
        resource_manager.cleanup.assert_called_once()

    def test_partial_success_handling(self):
        """Test handling of operations that partially succeed."""
        # Given
        batch_items = ["item1", "item2", "item3", "item4"]

        def batch_processor(items):
            results = []
            errors = []
            for i, item in enumerate(items):
                if i == 2:  # Simulate failure on third item
                    errors.append(ProcessingError(f"Failed to process {item}"))
                else:
                    results.append(f"processed_{item}")
            return results, errors

        # When
        from src.pdf_to_markdown_mcp.core.errors import PartialSuccessHandler

        handler = PartialSuccessHandler()

        result = handler.handle_batch_operation(
            items=batch_items, processor=batch_processor, min_success_rate=0.7
        )

        # Then
        assert result.success_count == 3
        assert result.error_count == 1
        assert result.success_rate >= 0.7
        assert len(result.successful_items) == 3
        assert len(result.failed_items) == 1


class TestSecurityErrorHandling:
    """Test security-focused error handling."""

    def test_rate_limiting_error_generation(self):
        """Test rate limiting error with security headers."""
        # Given
        user_id = "test_user"

        # When
        error = RateLimitError(
            message="Too many requests", user_id=user_id, retry_after=60
        )

        # Then
        assert error.user_id == user_id
        assert error.retry_after == 60
        assert error.get_security_headers()["Retry-After"] == "60"
        assert "X-RateLimit-Remaining" in error.get_security_headers()

    def test_input_validation_prevents_injection(self):
        """Test that input validation errors prevent potential injection attacks."""
        # Given
        malicious_input = "'; DROP TABLE users; --"

        # When
        error = InputValidationError(
            message="Invalid input detected",
            input_value=malicious_input,
            validation_rule="sql_injection_check",
        )

        # Then
        sanitized_message = sanitize_error_for_user(error)
        assert "DROP TABLE" not in sanitized_message
        assert "Invalid input" in sanitized_message

    def test_authentication_error_doesnt_leak_user_existence(self):
        """Test that authentication errors don't reveal whether users exist."""
        # Given
        nonexistent_user = "nonexistent@example.com"

        # When
        error = AuthenticationError(
            message="Authentication failed", user_identifier=nonexistent_user
        )

        # Then
        user_message = sanitize_error_for_user(error)
        assert nonexistent_user not in user_message
        assert (
            "Authentication failed" in user_message
            or "Invalid credentials" in user_message
        )


class TestErrorRecovery:
    """Test error recovery and dead letter queue mechanisms."""

    @pytest.mark.asyncio
    async def test_dead_letter_queue_handling(self):
        """Test that failed tasks are properly queued in dead letter queue."""
        # Given
        from src.pdf_to_markdown_mcp.core.errors import DeadLetterQueueManager

        dlq_manager = DeadLetterQueueManager()

        failed_task = {
            "task_id": "test-task-123",
            "operation": "process_pdf",
            "error": ProcessingError("Processing failed"),
            "retry_count": 3,
            "original_args": {"file_path": "/test/file.pdf"},
        }

        # When
        dlq_manager.add_failed_task(failed_task)

        # Then
        queued_tasks = dlq_manager.get_failed_tasks()
        assert len(queued_tasks) == 1
        assert queued_tasks[0]["task_id"] == "test-task-123"
        assert queued_tasks[0]["retry_count"] == 3

    def test_error_recovery_strategy_selection(self):
        """Test that appropriate recovery strategies are selected."""
        # Given
        from src.pdf_to_markdown_mcp.core.errors import RecoveryStrategyManager

        recovery_manager = RecoveryStrategyManager()

        # When & Then
        strategy = recovery_manager.get_recovery_strategy(
            DatabaseError("Connection lost")
        )
        assert strategy.strategy_type == "reconnect_and_retry"

        strategy = recovery_manager.get_recovery_strategy(
            ValidationError("Invalid input")
        )
        assert strategy.strategy_type == "no_recovery"

        strategy = recovery_manager.get_recovery_strategy(
            ResourceError("Out of memory")
        )
        assert strategy.strategy_type == "wait_and_retry"


if __name__ == "__main__":
    pytest.main([__file__])
