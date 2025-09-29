"""
Performance monitoring and optimization framework for PDF to Markdown MCP.

This module provides tools for monitoring and optimizing:
- Database query performance
- Memory usage patterns
- Async operation performance
- Vector search optimization
- Connection pool utilization
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import psutil
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    duration_ms: float
    memory_delta_mb: float
    cpu_usage_percent: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "operation_name": self.operation_name,
            "duration_ms": self.duration_ms,
            "memory_delta_mb": self.memory_delta_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class QueryPerformanceMetrics:
    """Container for database query performance metrics."""

    query_hash: str
    sql_statement: str
    execution_time_ms: float
    rows_returned: int | None
    rows_examined: int | None
    plan_cost: float | None
    cache_hit: bool
    timestamp: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert query metrics to dictionary format."""
        return {
            "query_hash": self.query_hash,
            "sql_statement": self.sql_statement,
            "execution_time_ms": self.execution_time_ms,
            "rows_returned": self.rows_returned,
            "rows_examined": self.rows_examined,
            "plan_cost": self.plan_cost,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
        }


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self):
        self.metrics: list[PerformanceMetrics] = []
        self.query_metrics: list[QueryPerformanceMetrics] = []
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 1000.0,
            "query_time_ms": 500.0,
            "memory_delta_mb": 100.0,
        }
        self.query_cache: dict[str, Any] = {}
        self.slow_queries: dict[str, int] = defaultdict(int)

    def set_thresholds(self, **thresholds):
        """Update performance thresholds."""
        self.thresholds.update(thresholds)

    @asynccontextmanager
    async def measure_performance(
        self, operation_name: str, metadata: dict[str, Any] | None = None
    ):
        """Context manager for measuring operation performance."""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_cpu = psutil.cpu_percent()

        try:
            yield
        except Exception as e:
            if metadata is None:
                metadata = {}
            metadata["error"] = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss
            end_cpu = psutil.cpu_percent()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=(end_time - start_time) * 1000,
                memory_delta_mb=(end_memory - start_memory) / 1024 / 1024,
                cpu_usage_percent=end_cpu,
                timestamp=end_time,
                metadata=metadata or {},
            )

            self.record_metrics(metrics)
            await self._check_thresholds(metrics)

    def performance_decorator(
        self, operation_name: str, metadata: dict[str, Any] | None = None
    ):
        """Decorator for automatic performance monitoring."""

        def decorator(func):
            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.measure_performance(operation_name, metadata):
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # Convert sync function to async context
                    async def run_sync():
                        return func(*args, **kwargs)

                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        self.measure_performance(operation_name, metadata).__aenter__()
                    )

                return sync_wrapper

        return decorator

    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics.append(metrics)

        # Log performance issues
        if metrics.duration_ms > self.thresholds["response_time_ms"]:
            logger.warning(
                f"Slow operation detected: {metrics.operation_name} "
                f"took {metrics.duration_ms:.2f}ms"
            )

        if metrics.memory_delta_mb > self.thresholds["memory_delta_mb"]:
            logger.warning(
                f"High memory usage: {metrics.operation_name} "
                f"used {metrics.memory_delta_mb:.2f}MB"
            )

    def record_query_metrics(self, metrics: QueryPerformanceMetrics):
        """Record database query performance metrics."""
        self.query_metrics.append(metrics)

        # Track slow queries
        if metrics.execution_time_ms > self.thresholds["query_time_ms"]:
            self.slow_queries[metrics.query_hash] += 1
            logger.warning(
                f"Slow query detected: {metrics.execution_time_ms:.2f}ms - "
                f"{metrics.sql_statement[:100]}..."
            )

    async def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed performance thresholds."""
        alerts = []

        if metrics.cpu_usage_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")

        if metrics.duration_ms > self.thresholds["response_time_ms"]:
            alerts.append(f"Slow response: {metrics.duration_ms:.1f}ms")

        if metrics.memory_delta_mb > self.thresholds["memory_delta_mb"]:
            alerts.append(f"High memory delta: {metrics.memory_delta_mb:.1f}MB")

        if alerts:
            logger.warning(
                f"Performance alerts for {metrics.operation_name}: {'; '.join(alerts)}"
            )

    def get_performance_summary(self, duration_minutes: int = 5) -> dict[str, Any]:
        """Generate performance summary for specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        recent_queries = [q for q in self.query_metrics if q.timestamp > cutoff_time]

        if not recent_metrics and not recent_queries:
            return {"message": "No recent metrics available"}

        # Calculate operation statistics
        operation_stats = defaultdict(list)
        for metric in recent_metrics:
            operation_stats[metric.operation_name].append(metric.duration_ms)

        operation_summary = {}
        for op_name, durations in operation_stats.items():
            operation_summary[op_name] = {
                "count": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "max_duration_ms": max(durations),
                "min_duration_ms": min(durations),
            }

        # Calculate query statistics
        query_summary = {
            "total_queries": len(recent_queries),
            "slow_queries": len(
                [
                    q
                    for q in recent_queries
                    if q.execution_time_ms > self.thresholds["query_time_ms"]
                ]
            ),
            "avg_query_time_ms": (
                sum(q.execution_time_ms for q in recent_queries) / len(recent_queries)
                if recent_queries
                else 0
            ),
        }

        # Calculate system statistics
        system_summary = {}
        if recent_metrics:
            cpu_values = [m.cpu_usage_percent for m in recent_metrics]
            memory_deltas = [m.memory_delta_mb for m in recent_metrics]

            system_summary = {
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "total_memory_delta_mb": sum(memory_deltas),
                "max_memory_delta_mb": max(memory_deltas),
            }

        return {
            "duration_minutes": duration_minutes,
            "operations": operation_summary,
            "queries": query_summary,
            "system": system_summary,
            "alerts": self._get_recent_alerts(cutoff_time),
        }

    def _get_recent_alerts(self, cutoff_time: float) -> list[str]:
        """Get recent performance alerts."""
        alerts = []

        # Check for patterns in recent data
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        slow_operations = [
            m
            for m in recent_metrics
            if m.duration_ms > self.thresholds["response_time_ms"]
        ]
        if len(slow_operations) > 3:
            alerts.append(
                f"Multiple slow operations detected: {len(slow_operations)} operations > {self.thresholds['response_time_ms']}ms"
            )

        high_memory_ops = [
            m
            for m in recent_metrics
            if m.memory_delta_mb > self.thresholds["memory_delta_mb"]
        ]
        if len(high_memory_ops) > 2:
            alerts.append(
                f"High memory usage pattern: {len(high_memory_ops)} operations > {self.thresholds['memory_delta_mb']}MB"
            )

        return alerts

    def clear_old_metrics(self, max_age_hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        self.query_metrics = [
            q for q in self.query_metrics if q.timestamp > cutoff_time
        ]

        logger.info(f"Cleared metrics older than {max_age_hours} hours")


class DatabasePerformanceOptimizer:
    """Database-specific performance optimization."""

    def __init__(self, engine: Engine, monitor: PerformanceMonitor):
        self.engine = engine
        self.monitor = monitor
        self.query_plans_cache = {}

        # Set up query monitoring
        self._setup_query_monitoring()

    def _setup_query_monitoring(self):
        """Set up SQLAlchemy event listeners for query monitoring."""

        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = time.time()
            context._statement = statement
            context._parameters = parameters

        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if hasattr(context, "_query_start_time"):
                execution_time = (time.time() - context._query_start_time) * 1000

                # Create query hash for tracking
                query_hash = str(hash(statement))

                metrics = QueryPerformanceMetrics(
                    query_hash=query_hash,
                    sql_statement=statement,
                    execution_time_ms=execution_time,
                    rows_returned=(
                        cursor.rowcount if hasattr(cursor, "rowcount") else None
                    ),
                    rows_examined=None,  # Would need EXPLAIN to get this
                    plan_cost=None,  # Would need EXPLAIN to get this
                    cache_hit=query_hash in self.query_plans_cache,
                    timestamp=time.time(),
                    parameters=parameters,
                )

                self.monitor.record_query_metrics(metrics)

    async def analyze_query_performance(
        self, query: str, parameters: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Analyze query performance with EXPLAIN ANALYZE."""
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        async with self.engine.begin() as conn:
            try:
                result = await conn.execute(text(explain_query), parameters or {})
                explain_data = result.fetchone()[0]

                # Parse EXPLAIN ANALYZE output
                plan_data = explain_data[0] if explain_data else {}

                return {
                    "total_cost": plan_data.get("Plan", {}).get("Total Cost", 0),
                    "execution_time": plan_data.get("Execution Time", 0),
                    "planning_time": plan_data.get("Planning Time", 0),
                    "shared_hit": plan_data.get("Shared Hit Blocks", 0),
                    "shared_read": plan_data.get("Shared Read Blocks", 0),
                    "rows": plan_data.get("Plan", {}).get("Actual Rows", 0),
                    "plan": plan_data.get("Plan", {}),
                }
            except Exception as e:
                logger.error(f"Failed to analyze query performance: {e}")
                return {"error": str(e)}

    def optimize_connection_pool(self, pool_size: int = None, max_overflow: int = None):
        """Optimize connection pool settings based on current workload."""
        current_pool = self.engine.pool

        # Get current pool statistics
        pool_stats = {
            "size": current_pool.size(),
            "checked_in": current_pool.checkedin(),
            "checked_out": current_pool.checkedout(),
            "overflow": current_pool.overflow(),
            "invalidated": current_pool.invalidated(),
        }

        logger.info(f"Current pool statistics: {pool_stats}")

        # Suggest optimizations based on usage patterns
        suggestions = []

        if pool_stats["checked_out"] / pool_stats["size"] > 0.8:
            suggestions.append(
                "Consider increasing pool_size - high utilization detected"
            )

        if pool_stats["overflow"] > pool_stats["size"] * 0.5:
            suggestions.append(
                "Consider increasing max_overflow - frequent overflow detected"
            )

        if pool_stats["invalidated"] > 0:
            suggestions.append(
                "Connection invalidations detected - check connection health"
            )

        return {"current_stats": pool_stats, "suggestions": suggestions}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


# Convenience decorators
def monitor_performance(operation_name: str, metadata: dict[str, Any] = None):
    """Decorator for monitoring function performance."""
    return performance_monitor.performance_decorator(operation_name, metadata)


async def profile_async_operation(
    operation_name: str, operation_func: Callable, *args, **kwargs
):
    """Profile an async operation and return results with metrics."""
    async with performance_monitor.measure_performance(operation_name):
        return await operation_func(*args, **kwargs)


def profile_sync_operation(
    operation_name: str, operation_func: Callable, *args, **kwargs
):
    """Profile a sync operation and return results with metrics."""
    start_time = time.time()
    result = operation_func(*args, **kwargs)
    duration_ms = (time.time() - start_time) * 1000

    logger.info(f"Operation '{operation_name}' completed in {duration_ms:.2f}ms")
    return result
