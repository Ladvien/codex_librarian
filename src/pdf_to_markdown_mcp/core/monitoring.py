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
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

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
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


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

        # GPU metrics
        self.gpu_memory_gauge = Gauge(
            "gpu_memory_usage_bytes", "GPU memory usage in bytes", ["device", "memory_type"]
        )

        self.gpu_utilization_gauge = Gauge(
            "gpu_utilization_percent", "GPU utilization percentage", ["device"]
        )

        self.model_loading_counter = Counter(
            "model_loading_total",
            "Total number of model loads",
            ["model_type", "device", "status"],
        )

        self.gpu_fallback_counter = Counter(
            "gpu_fallback_total",
            "Total number of GPU to CPU fallbacks",
            ["reason"],
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

            self.logger.debug("queue_depth_updated", queue_type=queue_type, depth=depth)

        except Exception as e:
            self.logger.error(f"Failed to update queue depth metrics: {e}")

    def record_gpu_usage(self, device_id: int, memory_used: int, memory_total: int, utilization: float):
        """Record GPU usage metrics."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.gpu_memory_gauge.labels(device=f"gpu{device_id}", memory_type="used").set(memory_used)
                self.gpu_memory_gauge.labels(device=f"gpu{device_id}", memory_type="total").set(memory_total)
                self.gpu_utilization_gauge.labels(device=f"gpu{device_id}").set(utilization)

            self.logger.info(
                "gpu_usage_recorded",
                device_id=device_id,
                memory_used_mb=memory_used / (1024**2),
                memory_total_mb=memory_total / (1024**2),
                utilization_percent=utilization,
            )

        except Exception as e:
            self.logger.error(f"Failed to record GPU usage metrics: {e}")

    def record_model_loading(self, model_type: str, device: str, status: str):
        """Record model loading events."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.model_loading_counter.labels(
                    model_type=model_type, device=device, status=status
                ).inc()

            self.logger.info(
                "model_loading_recorded",
                model_type=model_type,
                device=device,
                status=status,
            )

        except Exception as e:
            self.logger.error(f"Failed to record model loading metrics: {e}")

    def record_gpu_fallback(self, reason: str):
        """Record GPU to CPU fallback events."""
        try:
            if PROMETHEUS_AVAILABLE:
                self.gpu_fallback_counter.labels(reason=reason).inc()

            self.logger.warning(
                "gpu_fallback_recorded",
                reason=reason,
            )

        except Exception as e:
            self.logger.error(f"Failed to record GPU fallback metrics: {e}")

    async def collect_gpu_metrics(self):
        """Collect GPU usage metrics if available."""
        if not NVML_AVAILABLE:
            self.logger.debug("NVIDIA ML library not available, skipping GPU metrics")
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Record metrics
                self.record_gpu_usage(
                    device_id=i,
                    memory_used=mem_info.used,
                    memory_total=mem_info.total,
                    utilization=util.gpu
                )

        except Exception as e:
            self.logger.error(f"Failed to collect GPU metrics: {e}")

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

            # Collect GPU metrics if available
            await self.collect_gpu_metrics()

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

            from pdf_to_markdown_mcp.db.session import get_db_session

            with get_db_session() as session:
                # Perform lightweight database query
                session.execute(text("SELECT 1"))
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
            from pdf_to_markdown_mcp.worker.celery import app as celery_app

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

            embedding_service = await create_embedding_service(
                provider=settings.embedding.provider,
                model=settings.embedding.model,
                dimensions=settings.embedding.dimensions,
                batch_size=settings.embedding.batch_size,
            )
            is_healthy = await embedding_service.health_check()

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            return ComponentHealth(
                status=status,
                last_check=time.time(),
                details={
                    "provider": settings.embedding.provider,
                    "model": settings.embedding.model,
                    "status": "healthy" if is_healthy else "unhealthy",
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
            from pdf_to_markdown_mcp.worker.celery import app as celery_app

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

    async def check_gpu_health(self) -> ComponentHealth:
        """Check GPU availability and status."""
        try:
            if not NVML_AVAILABLE:
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    last_check=time.time(),
                    details={"error": "NVIDIA ML library not available"},
                )

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    last_check=time.time(),
                    details={"error": "No GPU devices found"},
                )

            gpu_details = {}
            overall_status = HealthStatus.HEALTHY

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                memory_percent = (mem_info.used / mem_info.total) * 100

                # Determine status based on usage and temperature
                device_status = HealthStatus.HEALTHY
                if temp > 85 or memory_percent > 95:
                    device_status = HealthStatus.UNHEALTHY
                    overall_status = HealthStatus.UNHEALTHY
                elif temp > 75 or memory_percent > 85:
                    device_status = HealthStatus.DEGRADED
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED

                gpu_details[f"gpu_{i}"] = {
                    "name": name,
                    "memory_used_mb": round(mem_info.used / (1024**2), 1),
                    "memory_total_mb": round(mem_info.total / (1024**2), 1),
                    "memory_percent": round(memory_percent, 1),
                    "utilization_percent": util.gpu,
                    "temperature_c": temp,
                    "status": device_status.value,
                }

            return ComponentHealth(
                status=overall_status,
                last_check=time.time(),
                details={
                    "device_count": device_count,
                    "devices": gpu_details,
                },
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_cuda_availability(self) -> ComponentHealth:
        """Check CUDA availability for PyTorch."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                return ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    last_check=time.time(),
                    details={"error": "CUDA not available in PyTorch"},
                )

            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()

            device_details = {}
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory

                device_details[f"cuda_{i}"] = {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_allocated_mb": round(memory_allocated / (1024**2), 1),
                    "memory_reserved_mb": round(memory_reserved / (1024**2), 1),
                    "memory_total_mb": round(memory_total / (1024**2), 1),
                    "memory_utilization_percent": round((memory_allocated / memory_total) * 100, 1),
                }

            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                last_check=time.time(),
                details={
                    "cuda_available": True,
                    "device_count": device_count,
                    "current_device": current_device,
                    "devices": device_details,
                },
            )

        except ImportError:
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                last_check=time.time(),
                details={"error": "PyTorch not available"},
            )
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=time.time(),
                details={"error": str(e)},
            )

    async def check_background_processes(self) -> ComponentHealth:
        """Check health of background processes like file watcher."""
        try:
            if not PSUTIL_AVAILABLE:
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    last_check=time.time(),
                    details={"error": "psutil not available"},
                )

            # Look for key background processes
            target_processes = {
                "file_watcher": ["watch_and_mirror", "file_watcher", "inotify"],
                "worker_manager": ["celery", "worker"],
                "gpu_monitor": ["gpu_monitor", "nvidia-smi"],
            }

            process_status = {}
            overall_status = HealthStatus.HEALTHY

            for process_type, keywords in target_processes.items():
                found_processes = []

                for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent"]):
                    try:
                        cmdline = " ".join(proc.info["cmdline"] or [])
                        name = proc.info["name"] or ""

                        if any(keyword in cmdline.lower() or keyword in name.lower() for keyword in keywords):
                            found_processes.append({
                                "pid": proc.info["pid"],
                                "name": name,
                                "cpu_percent": proc.info["cpu_percent"],
                                "memory_percent": proc.info["memory_percent"],
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                process_status[process_type] = {
                    "count": len(found_processes),
                    "processes": found_processes[:5],  # Limit to 5 for response size
                }

                # Check if critical processes are running
                if process_type == "worker_manager" and len(found_processes) == 0:
                    overall_status = HealthStatus.UNHEALTHY
                elif process_type == "file_watcher" and len(found_processes) == 0:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED

            return ComponentHealth(
                status=overall_status,
                last_check=time.time(),
                details=process_status,
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

            # Check for model singleton usage
            from pdf_to_markdown_mcp.services.mineru import _mineru_model_singleton

            details = {
                "library_status": "available" if available else "unavailable",
                "singleton_initialized": _mineru_model_singleton is not None,
            }

            if available:
                status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.UNHEALTHY

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
            "gpu": self.check_gpu_health(),
            "cuda": self.check_cuda_availability(),
            "background_processes": self.check_background_processes(),
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
        # GPU-specific alerts
        AlertRule(
            name="GPU Memory Critical",
            condition=lambda m: any(
                device.get("memory_percent", 0) > 95
                for device in m.get("gpu_devices", {}).values()
                if isinstance(device, dict)
            ),
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
        ),
        AlertRule(
            name="GPU Temperature High",
            condition=lambda m: any(
                device.get("temperature_c", 0) > 85
                for device in m.get("gpu_devices", {}).values()
                if isinstance(device, dict)
            ),
            severity=AlertSeverity.ERROR,
            cooldown_minutes=10,
        ),
        AlertRule(
            name="CUDA Unavailable",
            condition=lambda m: m.get("cuda_status") == "unhealthy",
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5,
        ),
        AlertRule(
            name="GPU Fallback Detected",
            condition=lambda m: m.get("gpu_fallback_count", 0) > 3,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15,
        ),
        AlertRule(
            name="Multiple Model Loads",
            condition=lambda m: m.get("model_loads_per_hour", 0) > 10,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=30,
        ),
        AlertRule(
            name="Background Process Missing",
            condition=lambda m: m.get("background_processes_status") == "unhealthy",
            severity=AlertSeverity.ERROR,
            cooldown_minutes=10,
        ),
    ]


# Global instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor()
alerting_engine = AlertingEngine()

# Add standard alerts
for rule in create_standard_alerts():
    alerting_engine.add_rule(rule)
