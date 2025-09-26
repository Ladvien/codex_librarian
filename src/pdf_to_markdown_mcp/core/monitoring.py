"""
Comprehensive monitoring and health check infrastructure.

This module provides metrics collection, health monitoring, alerting,
and distributed tracing for the PDF to Markdown MCP server.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def set(self, value):
            pass


# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar(
    "correlation_id", default=None
)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    status: HealthStatus
    last_check: float
    response_time_ms: float | None = None
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    uptime_seconds: float
    components: dict[str, ComponentHealth]
    version: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self):
        self.logger = logger
        self.custom_metrics = {}
        self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning(
                "Prometheus client not available, metrics will be limited"
            )
            return

        # Document processing metrics
        self.document_processing_counter = Counter(
            "documents_processed_total",
            "Total number of documents processed",
            ["status", "file_type"],
        )

        self.processing_duration_histogram = Histogram(
            "document_processing_duration_seconds",
            "Time spent processing documents",
            ["processing_type"],
        )

        # Search query metrics
        self.search_query_counter = Counter(
            "search_queries_total",
            "Total number of search queries",
            ["search_type", "result_count_range"],
        )

        self.search_response_histogram = Histogram(
            "search_response_duration_seconds",
            "Search query response times",
            ["search_type"],
        )

        # System resource metrics
        self.system_resource_gauge = Gauge(
            "system_resource_usage", "System resource usage metrics", ["resource_type"]
        )

        # Celery task metrics
        self.celery_active_tasks_gauge = Gauge(
            "celery_active_tasks", "Number of active Celery tasks"
        )

        self.celery_task_counter = Counter(
            "celery_tasks_total",
            "Total Celery tasks processed",
            ["status", "task_type"],
        )

        # Processing queue metrics
        self.processing_queue_gauge = Gauge(
            "processing_queue_depth",
            "Number of items in processing queue",
            ["queue_type"],
        )

    def record_document_processing(
        self,
        status: str,
        file_type: str,
        duration_seconds: float,
        processing_type: str = "pdf",
    ):
        """Record document processing metrics."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.document_processing_counter.labels(
                    status=status, file_type=file_type
                ).inc()

                self.processing_duration_histogram.labels(
                    processing_type=processing_type
                ).observe(duration_seconds)

            self.logger.info(
                "document_processing_recorded",
                status=status,
                file_type=file_type,
                duration_seconds=duration_seconds,
                processing_type=processing_type,
            )

        except Exception as e:
            self.logger.error(f"Failed to record document processing metrics: {e}")

    def record_search_query(
        self, search_type: str, result_count: int, response_time_ms: float
    ):
        """Record search query metrics."""
        try:
            # Categorize result count for better metrics
            if result_count == 0:
                count_range = "zero"
            elif result_count <= 10:
                count_range = "low"
            elif result_count <= 100:
                count_range = "medium"
            else:
                count_range = "high"

            if PROMETHEUS_AVAILABLE:
                self.search_query_counter.labels(
                    search_type=search_type, result_count_range=count_range
                ).inc()

                self.search_response_histogram.labels(search_type=search_type).observe(
                    response_time_ms / 1000.0
                )  # Convert to seconds

            self.logger.info(
                "search_query_recorded",
                search_type=search_type,
                result_count=result_count,
                result_count_range=count_range,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            self.logger.error(f"Failed to record search query metrics: {e}")

    def record_celery_task(
        self, task_type: str, status: str, duration_seconds: float | None = None
    ):
        """Record Celery task metrics."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.celery_task_counter.labels(
                    status=status, task_type=task_type
                ).inc()

            self.logger.info(
                "celery_task_recorded",
                task_type=task_type,
                status=status,
                duration_seconds=duration_seconds,
            )

        except Exception as e:
            self.logger.error(f"Failed to record Celery task metrics: {e}")

    def update_queue_depth(self, queue_type: str, depth: int):
        """Update processing queue depth metrics."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.processing_queue_gauge.labels(queue_type=queue_type).set(depth)

            self.logger.debug(
                "queue_depth_updated", queue_type=queue_type, depth=depth
            )

        except Exception as e:
            self.logger.error(f"Failed to update queue depth metrics: {e}")

    async def collect_system_metrics(self):
        """Continuously collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning(
                "psutil not available, system metrics collection disabled"
            )
            return

        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if PROMETHEUS_AVAILABLE:
                    self.system_resource_gauge.labels(resource_type="cpu_percent").set(
                        cpu_percent
                    )

                # Memory usage
                memory = psutil.virtual_memory()
                if PROMETHEUS_AVAILABLE:
                    self.system_resource_gauge.labels(
                        resource_type="memory_percent"
                    ).set(memory.percent)
                    self.system_resource_gauge.labels(
                        resource_type="memory_available_gb"
                    ).set(memory.available / (1024**3))

                # Disk usage
                try:
                    disk = psutil.disk_usage("/")
                    disk_percent = (disk.used / disk.total) * 100
                    if PROMETHEUS_AVAILABLE:
                        self.system_resource_gauge.labels(
                            resource_type="disk_percent"
                        ).set(disk_percent)
                        self.system_resource_gauge.labels(
                            resource_type="disk_free_gb"
                        ).set(disk.free / (1024**3))
                except Exception:
                    # Disk metrics might not be available in all environments
                    pass

                self.logger.debug(
                    "system_metrics_collected",
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                )

            except Exception as e:
                self.logger.error(f"Failed to collect system metrics: {e}")

            # Wait 60 seconds before next collection
            await asyncio.sleep(60)

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus metrics not available\n"

        try:
            return generate_latest().decode("utf-8")
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}\n"


class HealthMonitor:
    """Monitors system component health."""

    def __init__(self):
        self.start_time = time.time()
        self.components = {}
        self.health_checks = {}
        self.logger = logger

    async def check_database_health(self) -> ComponentHealth:
        """Check PostgreSQL database connectivity and performance."""
        start_time = time.time()
        try:
            from sqlalchemy import text

            from pdf_to_markdown_mcp.db.session import get_db

            async with get_db() as session:
                # Perform lightweight database query
                await session.execute(text("SELECT 1"))
                response_time = (time.time() - start_time) * 1000

                # Determine status based on response time
                if response_time > 5000:  # 5 seconds
                    status = HealthStatus.UNHEALTHY
                elif response_time > 1000:  # 1 second
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY

                return ComponentHealth(
                    status=status,
                    response_time_ms=response_time,
                    last_check=time.time(),
                    details={"query": "SELECT 1", "connection_status": "connected"},
                )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_celery_health(self) -> ComponentHealth:
        """Check Celery worker status."""
        try:
            from pdf_to_markdown_mcp.worker.celery import celery_app

            inspect = celery_app.control.inspect()
            stats = inspect.stats()

            if stats:
                worker_count = len(stats.keys())
                active_tasks = inspect.active()
                total_active = sum(
                    len(tasks) for tasks in (active_tasks or {}).values()
                )

                # Determine health based on worker availability
                if worker_count == 0:
                    status = HealthStatus.UNHEALTHY
                elif worker_count < 2:  # Prefer multiple workers
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY

                return ComponentHealth(
                    status=status,
                    last_check=time.time(),
                    details={
                        "worker_count": worker_count,
                        "active_tasks": total_active,
                        "workers": list(stats.keys()),
                    },
                )
            else:
                return ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    last_check=time.time(),
                    details={"error": "No Celery workers available"},
                )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_embedding_health(self) -> ComponentHealth:
        """Check embedding service health."""
        try:
            from pdf_to_markdown_mcp.config import settings
            from pdf_to_markdown_mcp.services.embeddings import create_embedding_service

            embedding_service = create_embedding_service(settings.embedding)
            health_result = await embedding_service.health_check()

            if health_result.get("status") == "healthy":
                status = HealthStatus.HEALTHY
            elif health_result.get("status") == "degraded":
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            return ComponentHealth(
                status=status,
                last_check=time.time(),
                details={
                    "provider": settings.embedding.provider,
                    "message": health_result.get("message", ""),
                },
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis connectivity."""
        try:
            from pdf_to_markdown_mcp.worker.celery import celery_app

            # Try to ping Redis through Celery
            result = celery_app.control.ping(timeout=5)
            if result:
                status = HealthStatus.HEALTHY
                details = {"ping_result": "success", "brokers": len(result)}
            else:
                status = HealthStatus.UNHEALTHY
                details = {"ping_result": "failed"}

            return ComponentHealth(
                status=status, last_check=time.time(), details=details
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource availability."""
        try:
            if not PSUTIL_AVAILABLE:
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    last_check=time.time(),
                    details={"error": "psutil not available"},
                )

            # Memory usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Disk usage
            try:
                disk = psutil.disk_usage("/")
                disk_free_gb = disk.free / (1024**3)
            except Exception:
                disk_free_gb = float("inf")  # Skip disk check if not available

            # Determine status based on resource usage
            if memory.percent > 90 or cpu_percent > 95 or disk_free_gb < 1:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 75 or cpu_percent > 80 or disk_free_gb < 5:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return ComponentHealth(
                status=status,
                last_check=time.time(),
                details={
                    "memory_percent": round(memory.percent, 1),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "cpu_percent": round(cpu_percent, 1),
                    "disk_free_gb": (
                        round(disk_free_gb, 1) if disk_free_gb != float("inf") else None
                    ),
                },
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_mineru_health(self) -> ComponentHealth:
        """Check MinerU service availability."""
        try:
            from pdf_to_markdown_mcp.services.mineru import MinerUService

            # Try to initialize MinerU service (this checks if library is available)
            service = MinerUService()
            if hasattr(service, "is_available") and callable(service.is_available):
                available = service.is_available()
            else:
                available = True  # Assume available if check method doesn't exist

            if available:
                status = HealthStatus.HEALTHY
                details = {"library_status": "available"}
            else:
                status = HealthStatus.DEGRADED
                details = {"library_status": "mock mode"}

            return ComponentHealth(
                status=status, last_check=time.time(), details=details
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        components = {}

        # Run all health checks concurrently
        health_checks = {
            "database": self.check_database_health(),
            "celery": self.check_celery_health(),
            "embeddings": self.check_embedding_health(),
            "redis": self.check_redis_health(),
            "system": self.check_system_resources(),
            "mineru": self.check_mineru_health(),
        }

        results = await asyncio.gather(*health_checks.values(), return_exceptions=True)

        # Process results
        overall_status = HealthStatus.HEALTHY
        for name, result in zip(health_checks.keys(), results, strict=False):
            if isinstance(result, Exception):
                components[name] = ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    last_check=time.time(),
                    details={"error": str(result)},
                )
            else:
                components[name] = result

            # Update overall status (worst status wins)
            if components[name].status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif (
                components[name].status == HealthStatus.DEGRADED
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        uptime = time.time() - self.start_time

        return SystemHealth(
            status=overall_status,
            uptime_seconds=uptime,
            components=components,
            version="1.0.0",  # TODO: Get from settings
            timestamp=time.time(),
        )

    async def check_readiness(self) -> dict[str, Any]:
        """Check if system is ready to accept traffic."""
        ready = True
        checks = {}

        # Essential services for readiness
        try:
            db_health = await self.check_database_health()
            checks["database"] = (
                "ready" if db_health.status != HealthStatus.UNHEALTHY else "not_ready"
            )
            if checks["database"] == "not_ready":
                ready = False
        except Exception:
            checks["database"] = "not_ready"
            ready = False

        # Configuration check
        try:
            from pdf_to_markdown_mcp.config import settings

            if hasattr(settings, "database") and hasattr(settings, "embedding"):
                checks["configuration"] = "ready"
            else:
                checks["configuration"] = "not_ready"
                ready = False
        except Exception:
            checks["configuration"] = "not_ready"
            ready = False

        return {"ready": ready, "checks": checks}


class AlertRule:
    """Alert rule definition."""

    def __init__(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        severity: AlertSeverity,
        cooldown_minutes: int = 15,
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None


class AlertingEngine:
    """Manages alerting rules and notifications."""

    def __init__(self):
        self.rules = []
        self.notification_channels = []
        self.logger = logger

    def add_rule(self, rule: AlertRule):
        """Add alerting rule to engine."""
        self.rules.append(rule)

    async def evaluate_alerts(self, metrics: dict[str, Any]):
        """Evaluate all alerting rules against current metrics."""
        for rule in self.rules:
            try:
                if rule.condition(metrics):
                    if self._should_trigger_alert(rule):
                        await self._send_alert(rule, metrics)
                        rule.last_triggered = datetime.utcnow()

            except Exception as e:
                self.logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")

    def _should_trigger_alert(self, rule: AlertRule) -> bool:
        """Check if alert should be triggered based on cooldown."""
        if rule.last_triggered is None:
            return True

        cooldown_elapsed = (
            datetime.utcnow() - rule.last_triggered
        ).total_seconds() / 60

        return cooldown_elapsed >= rule.cooldown_minutes

    async def _send_alert(self, rule: AlertRule, metrics: dict[str, Any]):
        """Send alert notification."""
        self.logger.error(
            "alert_triggered",
            rule_name=rule.name,
            severity=rule.severity,
            metrics=metrics,
            correlation_id=TracingManager.get_correlation_id(),
        )

        # TODO: Implement actual notification channels (email, Slack, etc.)


class TracingManager:
    """Manages distributed tracing and correlation IDs."""

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate unique correlation ID for request tracing."""
        return str(uuid.uuid4())

    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set correlation ID for current context."""
        correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> str | None:
        """Get correlation ID from current context."""
        return correlation_id_var.get()

    @staticmethod
    def trace_operation(operation_name: str):
        """Decorator for operation tracing."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                correlation_id = TracingManager.get_correlation_id()

                logger.info(
                    "operation_started",
                    operation=operation_name,
                    correlation_id=correlation_id,
                )

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(
                        "operation_completed",
                        operation=operation_name,
                        duration_seconds=duration,
                        correlation_id=correlation_id,
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        "operation_failed",
                        operation=operation_name,
                        duration_seconds=duration,
                        error=str(e),
                        correlation_id=correlation_id,
                        exc_info=True,
                    )
                    raise

            return wrapper

        return decorator


# Create standard alerting rules
def create_standard_alerts() -> list[AlertRule]:
    """Create standard alerting rules for common issues."""
    return [
        AlertRule(
            name="High Error Rate",
            condition=lambda m: m.get("error_rate_percent", 0) > 5.0,
            severity=AlertSeverity.ERROR,
            cooldown_minutes=10,
        ),
        AlertRule(
            name="Database Connection Issues",
            condition=lambda m: m.get("database_status") == "unhealthy",
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
        ),
        AlertRule(
            name="High Processing Queue Depth",
            condition=lambda m: m.get("processing_queue_depth", 0) > 1000,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15,
        ),
        AlertRule(
            name="Memory Usage Critical",
            condition=lambda m: m.get("memory_percent", 0) > 90.0,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
        ),
        AlertRule(
            name="No Active Workers",
            condition=lambda m: m.get("celery_workers", 0) == 0,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
        ),
        AlertRule(
            name="Embedding Service Unavailable",
            condition=lambda m: m.get("embeddings_status") == "unhealthy",
            severity=AlertSeverity.ERROR,
            cooldown_minutes=10,
        ),
        AlertRule(
            name="Disk Space Low",
            condition=lambda m: m.get("disk_free_gb", float("inf")) < 5.0,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=30,
        ),
    ]


# Global instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor()
alerting_engine = AlertingEngine()

# Add standard alerts
for rule in create_standard_alerts():
    alerting_engine.add_rule(rule)
