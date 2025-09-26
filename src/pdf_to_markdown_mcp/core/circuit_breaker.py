"""
Circuit Breaker Pattern Implementation for Redis Connection Management.

This module provides a robust circuit breaker to prevent Redis connection pool
saturation and handle Redis connectivity issues gracefully.
"""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..core.errors import (
    CircuitBreakerError,
    create_correlation_id,
    track_error,
)

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for Redis connections with exponential backoff.

    Prevents connection pool exhaustion by failing fast when Redis is
    experiencing issues and providing automatic recovery detection.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: tuple = (Exception,),
        name: str = "redis_circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception types that trigger the circuit
            name: Circuit breaker identifier
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.next_attempt_time: datetime | None = None

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opened_count = 0

        # Correlation tracking
        self.correlation_id = create_correlation_id()

        logger.info(
            f"Circuit breaker '{self.name}' initialized",
            extra={
                "circuit_breaker": self.name,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "correlation_id": self.correlation_id,
            },
        )

    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed through circuit."""
        current_time = datetime.utcnow()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and current_time >= self.next_attempt_time:
                # Transition to half-open to test recovery
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.name}' transitioning to HALF_OPEN",
                    extra={
                        "circuit_breaker": self.name,
                        "correlation_id": self.correlation_id,
                    },
                )
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Allow one request to test recovery
            return True

        return False

    def _record_success(self):
        """Record successful operation."""
        self.successful_calls += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Recovery successful, close circuit
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            self.next_attempt_time = None

            logger.info(
                f"Circuit breaker '{self.name}' recovered, closing circuit",
                extra={
                    "circuit_breaker": self.name,
                    "correlation_id": self.correlation_id,
                    "total_calls": self.total_calls,
                    "success_rate": self.successful_calls / max(self.total_calls, 1),
                },
            )

    def _record_failure(self, exception: Exception):
        """Record failed operation."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        # Track error for monitoring
        track_error(
            exception,
            {
                "circuit_breaker": self.name,
                "failure_count": self.failure_count,
                "state": self.state.value,
                "correlation_id": self.correlation_id,
            },
        )

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery attempt, reopen circuit
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = datetime.utcnow() + timedelta(
                seconds=self.recovery_timeout
            )

        elif self.failure_count >= self.failure_threshold:
            # Open circuit
            self.state = CircuitBreakerState.OPEN
            self.circuit_opened_count += 1
            self.next_attempt_time = datetime.utcnow() + timedelta(
                seconds=self.recovery_timeout
            )

            logger.warning(
                f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures",
                extra={
                    "circuit_breaker": self.name,
                    "failure_count": self.failure_count,
                    "last_failure": str(exception),
                    "next_attempt_time": self.next_attempt_time.isoformat(),
                    "correlation_id": self.correlation_id,
                },
            )

    @contextmanager
    def __call__(self, operation_name: str = "redis_operation"):
        """
        Context manager for protecting operations.

        Args:
            operation_name: Name of the operation being protected

        Raises:
            CircuitBreakerError: When circuit is open
        """
        if not self._should_allow_request():
            error_msg = (
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Next attempt at {self.next_attempt_time.isoformat() if self.next_attempt_time else 'unknown'}"
            )
            raise CircuitBreakerError(error_msg)

        self.total_calls += 1
        operation_start = time.time()

        try:
            logger.debug(
                f"Circuit breaker '{self.name}' allowing {operation_name}",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "state": self.state.value,
                    "correlation_id": self.correlation_id,
                },
            )

            yield

            # Operation successful
            self._record_success()
            operation_duration = time.time() - operation_start

            logger.debug(
                f"Circuit breaker '{self.name}' operation '{operation_name}' succeeded",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "duration_seconds": round(operation_duration, 3),
                    "correlation_id": self.correlation_id,
                },
            )

        except self.expected_exception as e:
            # Expected failure, record and re-raise
            self._record_failure(e)
            operation_duration = time.time() - operation_start

            logger.warning(
                f"Circuit breaker '{self.name}' operation '{operation_name}' failed",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "duration_seconds": round(operation_duration, 3),
                    "error": str(e),
                    "failure_count": self.failure_count,
                    "correlation_id": self.correlation_id,
                },
            )

            raise

    @asynccontextmanager
    async def async_call(self, operation_name: str = "async_redis_operation"):
        """
        Async context manager for protecting async operations.

        Args:
            operation_name: Name of the operation being protected

        Raises:
            CircuitBreakerError: When circuit is open
        """
        if not self._should_allow_request():
            error_msg = (
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Next attempt at {self.next_attempt_time.isoformat() if self.next_attempt_time else 'unknown'}"
            )
            raise CircuitBreakerError(error_msg)

        self.total_calls += 1
        operation_start = time.time()

        try:
            logger.debug(
                f"Circuit breaker '{self.name}' allowing async {operation_name}",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "state": self.state.value,
                    "correlation_id": self.correlation_id,
                },
            )

            yield

            # Operation successful
            self._record_success()
            operation_duration = time.time() - operation_start

            logger.debug(
                f"Circuit breaker '{self.name}' async operation '{operation_name}' succeeded",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "duration_seconds": round(operation_duration, 3),
                    "correlation_id": self.correlation_id,
                },
            )

        except self.expected_exception as e:
            # Expected failure, record and re-raise
            self._record_failure(e)
            operation_duration = time.time() - operation_start

            logger.warning(
                f"Circuit breaker '{self.name}' async operation '{operation_name}' failed",
                extra={
                    "circuit_breaker": self.name,
                    "operation": operation_name,
                    "duration_seconds": round(operation_duration, 3),
                    "error": str(e),
                    "failure_count": self.failure_count,
                    "correlation_id": self.correlation_id,
                },
            )

            raise

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = self.successful_calls / max(self.total_calls, 1)

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(success_rate, 4),
            "circuit_opened_count": self.circuit_opened_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "next_attempt_time": (
                self.next_attempt_time.isoformat() if self.next_attempt_time else None
            ),
            "correlation_id": self.correlation_id,
        }

    def reset(self):
        """Reset circuit breaker to initial state."""
        logger.info(
            f"Resetting circuit breaker '{self.name}'",
            extra={"circuit_breaker": self.name, "correlation_id": self.correlation_id},
        )

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None


class RedisCircuitBreakerManager:
    """Manager for Redis-specific circuit breakers."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(
        self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker for Redis operations.

        Args:
            name: Circuit breaker identifier
            failure_threshold: Failures before opening circuit
            recovery_timeout: Recovery timeout in seconds

        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            # Redis-specific exceptions that should trigger circuit breaker
            redis_exceptions = (
                ConnectionError,
                TimeoutError,
                OSError,  # Covers network errors
            )

            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=redis_exceptions,
                name=name,
            )

        return self.circuit_breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Global circuit breaker manager instance
redis_circuit_breaker_manager = RedisCircuitBreakerManager()


# Convenience functions for common Redis operations
def get_redis_broker_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Redis broker operations."""
    return redis_circuit_breaker_manager.get_circuit_breaker(
        name="redis_broker",
        failure_threshold=3,  # Fail fast for broker
        recovery_timeout=30,  # Quick recovery attempts
    )


def get_redis_result_backend_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Redis result backend operations."""
    return redis_circuit_breaker_manager.get_circuit_breaker(
        name="redis_result_backend",
        failure_threshold=5,  # More tolerant for results
        recovery_timeout=60,  # Longer recovery time
    )


def get_redis_cache_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Redis cache operations."""
    return redis_circuit_breaker_manager.get_circuit_breaker(
        name="redis_cache",
        failure_threshold=10,  # Very tolerant for cache
        recovery_timeout=30,  # Quick recovery for cache
    )
