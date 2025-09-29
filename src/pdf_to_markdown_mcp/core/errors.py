"""
Comprehensive error handling and retry mechanisms for PDF to Markdown MCP Server.

This module provides a security-focused, comprehensive error handling system with:
- Hierarchical error classification
- Security-focused error sanitization
- Retry mechanisms with exponential backoff
- Circuit breaker patterns
- Graceful degradation support
- Error tracking and metrics
- Rate limiting and abuse prevention

Security Features:
- Never exposes sensitive information to users
- Sanitizes all error messages before external exposure
- Implements rate limiting to prevent abuse
- Provides correlation IDs for internal tracking
- Logs security events for monitoring

CORE-006 Implementation - Security Auditor
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, TypeVar

# Type variables for generic retry mechanisms
T = TypeVar("T")

logger = logging.getLogger(__name__)

# =============================================================================
# Error Categories and Classification
# =============================================================================


class ErrorCategory(Enum):
    """Security-focused error categorization for retry and handling strategies."""

    VALIDATION = auto()  # Input validation failures - never retry
    SECURITY = auto()  # Security violations - never retry, log for monitoring
    AUTHENTICATION = auto()  # Auth failures - never retry, monitor for attacks
    AUTHORIZATION = auto()  # Permission failures - never retry
    RATE_LIMIT = auto()  # Rate limiting - temporary, suggest retry after
    TRANSIENT = auto()  # Temporary failures - safe to retry
    RESOURCE = auto()  # Resource exhaustion - retry with backoff
    PERMANENT = auto()  # Permanent failures - never retry
    CONFIGURATION = auto()  # Config errors - never retry until fixed
    EXTERNAL = auto()  # External service failures - retry with circuit breaker


def create_correlation_id() -> str:
    """Create a unique correlation ID for error tracking."""
    return f"{uuid.uuid4().hex[:8]}-{int(time.time())}"


def sanitize_error_for_user(error: "PDFToMarkdownError") -> str:
    """
    Sanitize error message for external user consumption.

    Security: Never expose internal details, paths, credentials, or system info.
    """
    if isinstance(error, SecurityError):
        return "A security error occurred. Please contact support if this persists."

    if isinstance(error, AuthenticationError):
        return "Authentication failed. Please check your credentials."

    if isinstance(error, AuthorizationError):
        return "You don't have permission to perform this operation."

    if isinstance(error, RateLimitError):
        return f"Rate limit exceeded. Please try again in {error.retry_after} seconds."

    if isinstance(error, ValidationError):
        # Only show safe validation messages
        safe_message = error.message
        # Remove any potential sensitive information
        for sensitive_term in ["password", "token", "key", "secret", "credential"]:
            if sensitive_term.lower() in safe_message.lower():
                return "Invalid input provided. Please check your request."
        return safe_message

    if isinstance(error, ResourceError):
        return "System is temporarily overloaded. Please try again later."

    if isinstance(error, TransientError):
        return "A temporary error occurred. Please try again."

    # Default safe message for any other errors
    return "An error occurred while processing your request. Please contact support if this persists."


# =============================================================================
# Base Error Hierarchy
# =============================================================================


class PDFToMarkdownError(Exception):
    """
    Base exception for all PDF to Markdown MCP Server errors.

    Security Features:
    - Correlation ID for internal tracking
    - Sanitized user-facing messages
    - Structured logging support
    - No sensitive data exposure
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        correlation_id: str | None = None,
        internal_details: str | None = None,
        user_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.correlation_id = correlation_id or create_correlation_id()
        self.internal_details = internal_details  # Never exposed to users
        self.user_message = user_message  # Pre-approved safe message
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.sensitive_data: list[str] = []  # Track sensitive data to redact

    def get_user_message(self) -> str:
        """Get sanitized message safe for user consumption."""
        return self.user_message or sanitize_error_for_user(self)

    def to_structured_log(self) -> dict[str, Any]:
        """Convert to structured log format with sensitive data removed."""
        log_data = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self._sanitize_metadata(self.metadata),
        }

        # Never include internal_details in logs
        return log_data

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive information from metadata."""
        sanitized = {}
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "key",
            "credential",
            "auth",
            "session",
            "cookie",
            "connection_string",
        }

        for key, value in metadata.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings that might contain sensitive data
                sanitized[key] = value[:97] + "..."
            else:
                sanitized[key] = value

        return sanitized


# =============================================================================
# Security-Focused Error Types
# =============================================================================


class SecurityError(PDFToMarkdownError):
    """Base class for all security-related errors. Never retried."""

    def __init__(self, message: str, security_event_type: str = "generic", **kwargs):
        super().__init__(message, **kwargs)
        self.security_event_type = security_event_type
        # Always log security events
        logger.warning(
            f"Security event: {security_event_type}",
            extra={
                "security_event": True,
                "event_type": security_event_type,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp.isoformat(),
            },
        )


class AuthenticationError(SecurityError):
    """Authentication failures. Monitor for potential attacks."""

    def __init__(self, message: str, user_identifier: str | None = None, **kwargs):
        super().__init__(
            message, security_event_type="authentication_failure", **kwargs
        )
        self.user_identifier = user_identifier
        # Don't include user_identifier in metadata to prevent leaks


class AuthorizationError(SecurityError):
    """Authorization failures. Monitor for privilege escalation attempts."""

    def __init__(self, message: str, resource: str | None = None, **kwargs):
        super().__init__(message, security_event_type="authorization_failure", **kwargs)
        self.resource = resource


class InputValidationError(SecurityError):
    """Input validation failures. Potential injection attempts."""

    def __init__(
        self,
        message: str,
        input_value: str | None = None,
        validation_rule: str | None = None,
        **kwargs,
    ):
        super().__init__(message, security_event_type="validation_failure", **kwargs)
        self.input_value = input_value  # Never exposed in logs
        self.validation_rule = validation_rule

        # Check for potential injection attempts
        if input_value and self._detect_injection_attempt(input_value):
            logger.error(
                "Potential injection attempt detected",
                extra={
                    "security_alert": True,
                    "validation_rule": validation_rule,
                    "correlation_id": self.correlation_id,
                },
            )

    def _detect_injection_attempt(self, input_value: str) -> bool:
        """Detect potential injection patterns."""
        injection_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "UPDATE SET",
            "<script>",
            "javascript:",
            "eval(",
            "exec(",
            "../",
            "..\\",
            "/etc/passwd",
            "cmd.exe",
        ]
        return any(
            pattern.lower() in input_value.lower() for pattern in injection_patterns
        )


class RateLimitError(SecurityError):
    """Rate limiting violations. Implement progressive penalties."""

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        retry_after: int = 60,
        **kwargs,
    ):
        super().__init__(message, security_event_type="rate_limit_exceeded", **kwargs)
        self.user_id = user_id
        self.retry_after = retry_after

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            "Retry-After": str(self.retry_after),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + self.retry_after),
        }


# =============================================================================
# Operational Error Types
# =============================================================================


class ValidationError(PDFToMarkdownError):
    """Input validation errors. Never retried."""



class TransientError(PDFToMarkdownError):
    """Temporary errors that should be retried."""

    def __init__(self, message: str, retry_suggested: bool = True, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_suggested = retry_suggested


class PermanentError(PDFToMarkdownError):
    """Permanent errors that should never be retried."""



class ProcessingError(PDFToMarkdownError):
    """PDF processing errors. May be retryable depending on cause."""

    def __init__(self, message: str, file_path: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path


class OCRError(ProcessingError):
    """OCR processing errors. Usually retryable."""

    def __init__(self, message: str, language: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.language = language


class EmbeddingError(TransientError):
    """Embedding generation errors. Usually retryable."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model


class DatabaseError(TransientError):
    """Database errors. Usually retryable with backoff."""

    def __init__(self, message: str, operation: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation


class ResourceError(TransientError):
    """Resource exhaustion errors. Retryable with longer backoff."""

    def __init__(self, message: str, resource_type: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type


class ConfigurationError(PermanentError):
    """Configuration errors. Not retryable until config is fixed."""



class TimeoutError(TransientError):
    """Operation timeout errors. Usually retryable."""



class CircuitBreakerError(TransientError):
    """Circuit breaker is open. Temporary restriction."""

    def __init__(
        self,
        message: str,
        service_name: str,
        estimated_recovery_time: int = 60,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.estimated_recovery_time = estimated_recovery_time


# =============================================================================
# Error Categorization and Strategy Selection
# =============================================================================


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize errors for appropriate handling strategies."""
    if isinstance(error, (ValidationError, InputValidationError)):
        return ErrorCategory.VALIDATION
    elif isinstance(error, (SecurityError, AuthenticationError, AuthorizationError)):
        return ErrorCategory.SECURITY
    elif isinstance(error, RateLimitError):
        return ErrorCategory.RATE_LIMIT
    elif isinstance(error, TransientError):
        return ErrorCategory.TRANSIENT
    elif isinstance(error, ResourceError):
        return ErrorCategory.RESOURCE
    elif isinstance(error, PermanentError):
        return ErrorCategory.PERMANENT
    elif isinstance(error, ConfigurationError):
        return ErrorCategory.CONFIGURATION
    elif isinstance(error, (EmbeddingError, DatabaseError)):
        return ErrorCategory.TRANSIENT
    else:
        return ErrorCategory.EXTERNAL


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error should be retried based on security and business logic."""
    category = categorize_error(error)

    # Never retry security violations
    non_retryable = {
        ErrorCategory.VALIDATION,
        ErrorCategory.SECURITY,
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.PERMANENT,
        ErrorCategory.CONFIGURATION,
    }

    return category not in non_retryable


# =============================================================================
# Retry Configuration and Strategies
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout: float | None = None


class ExponentialBackoffRetry:
    """Implements exponential backoff retry with jitter."""

    def __init__(self, config: RetryConfig):
        self.config = config

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if attempt >= self.config.max_retries:
            return 0

        delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Add ±25% jitter to prevent thundering herd
            import random

            jitter_factor = 0.75 + (random.random() * 0.5)
            delay *= jitter_factor

        return delay

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if retry should be attempted."""
        return attempt < self.config.max_retries and is_retryable_error(error)


def get_retry_strategy(error: Exception) -> ExponentialBackoffRetry:
    """Get appropriate retry configuration based on error type."""
    category = categorize_error(error)

    if category == ErrorCategory.VALIDATION or category == ErrorCategory.SECURITY:
        return ExponentialBackoffRetry(RetryConfig(max_retries=0))
    elif category == ErrorCategory.RATE_LIMIT:
        # Special handling for rate limits
        if isinstance(error, RateLimitError):
            return ExponentialBackoffRetry(RetryConfig(
                max_retries=1, base_delay=error.retry_after, max_delay=error.retry_after
            ))
        return ExponentialBackoffRetry(RetryConfig(max_retries=1, base_delay=60))
    elif category == ErrorCategory.TRANSIENT:
        return ExponentialBackoffRetry(RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0))
    elif category == ErrorCategory.RESOURCE:
        return ExponentialBackoffRetry(RetryConfig(max_retries=5, base_delay=5.0, max_delay=120.0))
    elif category == ErrorCategory.EXTERNAL:
        return ExponentialBackoffRetry(RetryConfig(max_retries=3, base_delay=2.0, max_delay=60.0))
    else:
        return ExponentialBackoffRetry(RetryConfig(max_retries=0))


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for external service protection."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    def call(self, func: Callable[[], T]) -> T:
        """Execute function through circuit breaker."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN",
                        service_name=self.name,
                        estimated_recovery_time=self.recovery_timeout,
                    )

        try:
            result = func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self._failure_count} failures"
                )

    def record_failure(self):
        """Manually record a failure (for testing)."""
        self._on_failure()


# =============================================================================
# Retry Manager and Execution
# =============================================================================


class RetryManager:
    """Manages retry logic with circuit breakers and error tracking."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.error_tracker = None  # Will be set later to avoid circular imports

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(name=service_name)
        return self.circuit_breakers[service_name]

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str,
        service_name: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> T:
        """Execute operation with retry logic and circuit breaker protection."""
        last_error = None

        for attempt in range(retry_config.max_retries + 1 if retry_config else 4):
            try:
                # Use circuit breaker if service specified
                if service_name:
                    circuit_breaker = self.get_circuit_breaker(service_name)
                    if asyncio.iscoroutinefunction(operation):
                        return await circuit_breaker.call(operation)
                    else:
                        return circuit_breaker.call(operation)
                elif asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()

            except Exception as e:
                last_error = e

                # Track error for monitoring
                if self.error_tracker:
                    self.error_tracker.record_error(
                        error=e, operation=operation_name, component="retry_manager"
                    )

                # Determine if we should retry
                if not retry_config:
                    retry_mechanism = get_retry_strategy(e)
                else:
                    retry_mechanism = ExponentialBackoffRetry(retry_config)

                if not retry_mechanism.should_retry(attempt, e):
                    logger.error(
                        f"Operation {operation_name} failed permanently after {attempt} attempts",
                        extra={"correlation_id": getattr(e, "correlation_id", None)},
                    )
                    raise e

                # Calculate delay and wait
                delay = retry_mechanism.get_delay(attempt)
                if delay > 0:
                    logger.info(
                        f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise last_error or Exception(f"Operation {operation_name} failed")


# =============================================================================
# Error Tracking and Metrics
# =============================================================================


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring and alerting."""

    total_errors: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    errors_by_category: dict[str, int] = field(default_factory=dict)
    errors_by_operation: dict[str, int] = field(default_factory=dict)
    error_rate_by_user: dict[str, float] = field(default_factory=dict)

    def should_rate_limit(self, user_id: str, threshold: float = 0.5) -> bool:
        """Check if user should be rate limited based on error rate."""
        return self.error_rate_by_user.get(user_id, 0.0) > threshold

    def get_error_rate(self, user_id: str) -> float:
        """Get error rate for specific user."""
        return self.error_rate_by_user.get(user_id, 0.0)


class ErrorTracker:
    """Tracks errors for monitoring, alerting, and abuse prevention."""

    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.errors: deque = deque()
        self.user_errors: defaultdict = defaultdict(lambda: deque())
        self._lock = threading.Lock()

    def record_error(
        self,
        error: Exception,
        operation: str,
        component: str = "unknown",
        user_context: dict[str, Any] | None = None,
    ):
        """Record error for tracking and metrics."""
        now = time.time()

        error_record = {
            "timestamp": now,
            "error_type": error.__class__.__name__,
            "error_category": categorize_error(error).name,
            "operation": operation,
            "component": component,
            "correlation_id": getattr(error, "correlation_id", None),
            "user_id": user_context.get("user_id") if user_context else None,
        }

        with self._lock:
            # Clean old errors
            self._clean_old_errors()

            # Record error
            self.errors.append(error_record)

            # Track user-specific errors for rate limiting
            if user_context and "user_id" in user_context:
                user_id = user_context["user_id"]
                self.user_errors[user_id].append(error_record)

    def get_metrics(self) -> ErrorMetrics:
        """Get current error metrics."""
        with self._lock:
            self._clean_old_errors()

            metrics = ErrorMetrics()
            metrics.total_errors = len(self.errors)

            # Aggregate by type, category, operation
            for error_record in self.errors:
                error_type = error_record["error_type"]
                error_category = error_record["error_category"]
                operation = error_record["operation"]

                metrics.errors_by_type[error_type] = (
                    metrics.errors_by_type.get(error_type, 0) + 1
                )
                metrics.errors_by_category[error_category] = (
                    metrics.errors_by_category.get(error_category, 0) + 1
                )
                metrics.errors_by_operation[operation] = (
                    metrics.errors_by_operation.get(operation, 0) + 1
                )

            # Calculate user error rates
            for user_id, user_error_list in self.user_errors.items():
                error_count = len(user_error_list)
                # Simple error rate: errors per minute
                metrics.error_rate_by_user[user_id] = error_count / (
                    self.window_size / 60
                )

            return metrics

    def _clean_old_errors(self):
        """Remove errors outside the tracking window."""
        cutoff_time = time.time() - self.window_size

        # Clean main error list
        while self.errors and self.errors[0]["timestamp"] < cutoff_time:
            self.errors.popleft()

        # Clean user error lists
        for user_id in list(self.user_errors.keys()):
            user_error_list = self.user_errors[user_id]
            while user_error_list and user_error_list[0]["timestamp"] < cutoff_time:
                user_error_list.popleft()

            # Remove empty user lists
            if not user_error_list:
                del self.user_errors[user_id]


# =============================================================================
# Graceful Degradation Support
# =============================================================================


class GracefulDegradationManager:
    """Manages graceful degradation when services fail."""

    async def execute_with_fallback(
        self,
        primary_operation: Callable[[], T],
        fallback_operation: Callable[[], T],
        operation_name: str,
    ) -> T:
        """Execute primary operation with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(primary_operation):
                return await primary_operation()
            else:
                return primary_operation()
        except Exception as e:
            logger.warning(
                f"Primary operation {operation_name} failed, using fallback: {e}"
            )

            try:
                if asyncio.iscoroutinefunction(fallback_operation):
                    return await fallback_operation()
                else:
                    return fallback_operation()
            except Exception as fallback_error:
                logger.error(
                    f"Both primary and fallback operations failed for {operation_name}: {fallback_error}"
                )
                raise e  # Raise original error


class TimeoutManager:
    """Manages operation timeouts with resource cleanup."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    def execute_with_timeout(
        self, operation: Callable[[], T], resource_manager: Any | None = None
    ) -> T:
        """Execute operation with timeout and resource cleanup."""
        import signal

        def timeout_handler(signum, frame):
            if resource_manager and hasattr(resource_manager, "cleanup"):
                resource_manager.cleanup()
            raise TimeoutError(
                f"Operation timed out after {self.timeout_seconds} seconds"
            )

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)

        try:
            result = operation()
            signal.alarm(0)  # Cancel timeout
            return result
        except Exception:
            signal.alarm(0)  # Cancel timeout
            if resource_manager and hasattr(resource_manager, "cleanup"):
                resource_manager.cleanup()
            raise


@dataclass
class PartialSuccessResult:
    """Result of batch operations with partial success."""

    successful_items: list[Any]
    failed_items: list[Any]
    errors: list[Exception]
    success_count: int
    error_count: int
    success_rate: float


class PartialSuccessHandler:
    """Handles operations that may partially succeed."""

    def handle_batch_operation(
        self,
        items: list[Any],
        processor: Callable[[list[Any]], tuple[list[Any], list[Exception]]],
        min_success_rate: float = 0.5,
    ) -> PartialSuccessResult:
        """Handle batch operation with partial success support."""
        successful_items, errors = processor(items)
        failed_items = items[len(successful_items) :]

        success_count = len(successful_items)
        error_count = len(errors)
        total_count = len(items)
        success_rate = success_count / total_count if total_count > 0 else 0

        result = PartialSuccessResult(
            successful_items=successful_items,
            failed_items=failed_items,
            errors=errors,
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
        )

        if success_rate < min_success_rate:
            logger.warning(
                f"Batch operation success rate ({success_rate:.2f}) below threshold ({min_success_rate})"
            )

        return result


# =============================================================================
# Dead Letter Queue and Recovery
# =============================================================================


class DeadLetterQueueManager:
    """Manages failed tasks for later recovery."""

    def __init__(self):
        self.failed_tasks: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_failed_task(self, task_data: dict[str, Any]):
        """Add failed task to dead letter queue."""
        with self._lock:
            task_data["dlq_timestamp"] = datetime.utcnow().isoformat()
            self.failed_tasks.append(task_data)

    def get_failed_tasks(self) -> list[dict[str, Any]]:
        """Get all failed tasks."""
        with self._lock:
            return self.failed_tasks.copy()

    def remove_failed_task(self, task_id: str) -> bool:
        """Remove task from dead letter queue."""
        with self._lock:
            for i, task in enumerate(self.failed_tasks):
                if task.get("task_id") == task_id:
                    del self.failed_tasks[i]
                    return True
            return False


@dataclass
class RecoveryStrategy:
    """Recovery strategy for different error types."""

    strategy_type: str
    retry_delay: int = 0
    max_attempts: int = 0
    recovery_action: str | None = None


class RecoveryStrategyManager:
    """Manages recovery strategies for different error types."""

    def get_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Get appropriate recovery strategy for error."""
        category = categorize_error(error)

        if category == ErrorCategory.TRANSIENT:
            if isinstance(error, DatabaseError):
                return RecoveryStrategy(
                    strategy_type="reconnect_and_retry",
                    retry_delay=5,
                    max_attempts=3,
                    recovery_action="reconnect_database",
                )
            else:
                return RecoveryStrategy(
                    strategy_type="simple_retry", retry_delay=2, max_attempts=3
                )
        elif category == ErrorCategory.RESOURCE:
            return RecoveryStrategy(
                strategy_type="wait_and_retry",
                retry_delay=30,
                max_attempts=5,
                recovery_action="wait_for_resources",
            )
        else:
            return RecoveryStrategy(strategy_type="no_recovery")


# =============================================================================
# Global Error Tracking Instance
# =============================================================================

# Global error tracker instance for application-wide error monitoring
global_error_tracker = ErrorTracker()

# Global retry manager
global_retry_manager = RetryManager()
global_retry_manager.error_tracker = global_error_tracker


# =============================================================================
# Convenience Functions
# =============================================================================


def track_error(
    error: Exception,
    operation: str,
    component: str = "unknown",
    user_context: dict[str, Any] | None = None,
):
    """Convenience function to track errors globally."""
    global_error_tracker.record_error(error, operation, component, user_context)


async def retry_async_operation(
    operation: Callable[[], T],
    operation_name: str,
    service_name: str | None = None,
    retry_config: RetryConfig | None = None,
) -> T:
    """Convenience function for retrying async operations."""
    return await global_retry_manager.execute_with_retry(
        operation, operation_name, service_name, retry_config
    )


# =============================================================================
# Security Validation Helpers
# =============================================================================


def validate_file_path(file_path: str, allowed_base_dirs: list[str] | None = None) -> str:
    """Validate and sanitize file path to prevent directory traversal.

    Args:
        file_path: Path to validate
        allowed_base_dirs: List of allowed base directories for absolute paths.
                          If None, uses watch_directories from settings.

    Returns:
        Normalized path

    Raises:
        InputValidationError: If path is invalid or outside allowed directories
    """
    import os
    from pathlib import Path

    if not file_path:
        raise InputValidationError(
            "File path cannot be empty", validation_rule="path_not_empty"
        )

    # Normalize path first
    normalized = os.path.normpath(file_path)
    resolved_path = Path(normalized).resolve()

    # If it's an absolute path, check if it's within allowed directories
    if os.path.isabs(normalized):
        # Get allowed directories from settings if not provided
        if allowed_base_dirs is None:
            from ..config import settings
            allowed_base_dirs = [
                str(Path(d).resolve()) for d in settings.watcher.watch_directories
            ]
            # Also allow temp directory and output directory
            if hasattr(settings.processing, 'temp_dir'):
                allowed_base_dirs.append(str(Path(settings.processing.temp_dir).resolve()))
            if hasattr(settings.watcher, 'output_directory'):
                allowed_base_dirs.append(str(Path(settings.watcher.output_directory).resolve()))

        # Check if path is within any allowed directory
        path_is_allowed = False
        for base_dir in allowed_base_dirs:
            base_path = Path(base_dir).resolve()
            try:
                resolved_path.relative_to(base_path)
                path_is_allowed = True
                break
            except ValueError:
                # Path is not relative to this base dir
                continue

        if not path_is_allowed:
            raise InputValidationError(
                "File path is outside allowed directories",
                validation_rule="path_outside_allowed_dirs",
            )
    else:
        # For relative paths, check for directory traversal
        if ".." in normalized:
            raise InputValidationError(
                "Directory traversal detected in relative path",
                validation_rule="directory_traversal_check",
            )

    # Additional security checks for dangerous patterns
    dangerous_patterns = ["..\\", "/etc/passwd", "/etc/shadow", "C:\\Windows"]
    for pattern in dangerous_patterns:
        if pattern in normalized:
            raise InputValidationError(
                "Potentially dangerous file path",
                validation_rule="dangerous_path_check",
            )

    return normalized


def validate_input_size(input_data: str, max_size: int = 10000) -> str:
    """Validate input size to prevent resource exhaustion."""
    if len(input_data) > max_size:
        raise InputValidationError(
            f"Input size exceeds maximum allowed ({max_size} characters)",
            validation_rule="input_size_limit",
        )
    return input_data


def sanitize_log_message(message: str) -> str:
    """Sanitize log message to remove sensitive information."""
    sensitive_patterns = [
        r"password[=:]\s*\S+",
        r"token[=:]\s*\S+",
        r"key[=:]\s*\S+",
        r"secret[=:]\s*\S+",
        r"credential[=:]\s*\S+",
    ]

    import re

    sanitized = message
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

    return sanitized


# Convenience functions for creating common errors
def validation_error(message: str, **kwargs) -> ValidationError:
    """Create a ValidationError with automatic sanitization."""
    return ValidationError(message, metadata=kwargs)


def processing_error(message: str, **kwargs) -> ProcessingError:
    """Create a ProcessingError with automatic sanitization."""
    return ProcessingError(message, metadata=kwargs)
