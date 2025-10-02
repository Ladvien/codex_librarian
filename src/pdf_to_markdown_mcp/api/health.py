"""
Health and monitoring API endpoints.

Provides comprehensive health checks, readiness probes, and metrics endpoints
for monitoring the PDF to Markdown MCP server.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse

from pdf_to_markdown_mcp.core.monitoring import (
    HealthStatus,
    TracingManager,
    health_monitor,
    metrics_collector,
)

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/health", response_model=dict[str, Any])
async def get_health(response: Response) -> dict[str, Any]:
    """
    Comprehensive system health check.

    Returns detailed health status for all system components including:
    - Database connectivity and performance
    - Celery worker availability
    - Embedding service status
    - System resource utilization
    - Redis broker connectivity
    - MinerU service availability
    """
    correlation_id = TracingManager.get_correlation_id()

    # Generate a correlation ID if one doesn't exist
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())
        TracingManager.set_correlation_id(correlation_id)

    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id

    try:
        system_health = await health_monitor.get_system_health()

        # Convert to dictionary format for JSON response
        components_dict = {}
        for name, component in system_health.components.items():
            components_dict[name] = {
                "status": component.status.value,
                "last_check": component.last_check,
                "response_time_ms": component.response_time_ms,
                "details": component.details,
            }

        response_data = {
            "status": system_health.status.value,
            "service": "PDF to Markdown MCP Server",
            "version": system_health.version,
            "timestamp": system_health.timestamp,
            "uptime_seconds": system_health.uptime_seconds,
            "components": components_dict,
            "correlation_id": correlation_id,
        }

        logger.info(
            f"health_check_completed: status={system_health.status.value}",
            extra={"component_count": len(components_dict), "correlation_id": correlation_id},
        )

        return response_data

    except Exception as e:
        logger.error(
            f"health_check_failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        return {
            "status": "unhealthy",
            "service": "PDF to Markdown MCP Server",
            "version": "unknown",
            "timestamp": asyncio.get_event_loop().time(),
            "error": "Health check system failure",
            "correlation_id": correlation_id,
        }


@router.get("/health/detailed")
async def get_detailed_health(response: Response) -> dict[str, Any]:
    """
    Detailed health check with additional diagnostic information.

    Includes version information for dependencies and extended diagnostics.
    """
    correlation_id = TracingManager.get_correlation_id()

    # Generate a correlation ID if one doesn't exist
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())
        TracingManager.set_correlation_id(correlation_id)

    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id

    try:
        # Get basic health information
        basic_health = await get_health(response)

        # Add detailed dependency information
        dependencies = {}

        # PostgreSQL version
        try:
            from sqlalchemy import text

            from pdf_to_markdown_mcp.db.session import get_db_session

            with get_db_session() as session:
                result = session.execute(text("SELECT version()"))
                version_info = result.scalar()
                dependencies["postgresql"] = {
                    "version": version_info.split()[1] if version_info else "unknown",
                    "status": "connected",
                }
        except Exception as e:
            dependencies["postgresql"] = {
                "version": "unknown",
                "status": "error",
                "error": str(e),
            }

        # Redis version
        try:
            from pdf_to_markdown_mcp.worker.celery import app as celery_app

            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            if stats:
                # Get Redis info from first worker
                worker_stats = list(stats.values())[0]
                dependencies["redis"] = {
                    "version": worker_stats.get("broker", {}).get(
                        "transport", "unknown"
                    ),
                    "status": "connected",
                }
            else:
                dependencies["redis"] = {"version": "unknown", "status": "no_workers"}
        except Exception as e:
            dependencies["redis"] = {
                "version": "unknown",
                "status": "error",
                "error": str(e),
            }

        # Python version
        import sys

        dependencies["python"] = {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "status": "active",
        }

        # Add dependency information to response
        basic_health["dependencies"] = dependencies
        basic_health["service_version"] = basic_health.get("version", "1.0.0")

        logger.info(
            f"detailed_health_check_completed: {len(dependencies)} dependencies",
            extra={"correlation_id": correlation_id},
        )

        return basic_health

    except Exception as e:
        logger.error(
            f"detailed_health_check_failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "detailed_health_check_failed",
                "message": "Failed to retrieve detailed health information",
                "correlation_id": correlation_id,
            },
        )


@router.get("/ready")
async def get_readiness(response: Response) -> dict[str, Any]:
    """
    Readiness probe for load balancers and orchestrators.

    Checks essential services required for the application to handle traffic.
    Returns HTTP 503 if not ready, HTTP 200 if ready.
    """
    correlation_id = TracingManager.get_correlation_id()

    try:
        readiness_result = await health_monitor.check_readiness()

        if not readiness_result["ready"]:
            response.status_code = 503  # Service Unavailable
            logger.warning(
                "readiness_check_not_ready",
                extra={"checks": readiness_result["checks"], "correlation_id": correlation_id},
            )
        else:
            logger.info(
                "readiness_check_ready",
                extra={"checks": readiness_result["checks"], "correlation_id": correlation_id},
            )

        return {
            "status": "ready" if readiness_result["ready"] else "not_ready",
            "service": "PDF to Markdown MCP Server",
            "checks": readiness_result["checks"],
            "timestamp": asyncio.get_event_loop().time(),
            "correlation_id": correlation_id,
        }

    except Exception as e:
        logger.error(
            f"readiness_check_failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        response.status_code = 503
        return {
            "status": "not_ready",
            "service": "PDF to Markdown MCP Server",
            "error": "Readiness check system failure",
            "timestamp": asyncio.get_event_loop().time(),
            "correlation_id": correlation_id,
        }


@router.get("/metrics")
async def get_prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus text exposition format including:
    - Document processing metrics
    - Search query performance
    - System resource utilization
    - Celery task queue metrics
    - Custom application metrics
    """
    correlation_id = TracingManager.get_correlation_id()

    try:
        # Get Prometheus-formatted metrics from collector
        prometheus_metrics = metrics_collector.get_prometheus_metrics()

        # Add custom metrics that aren't automatically collected
        custom_metrics = await _collect_custom_metrics()

        # Combine all metrics
        all_metrics = prometheus_metrics + custom_metrics

        logger.debug(
            f"prometheus_metrics_generated: size={len(all_metrics)} bytes",
            extra={"correlation_id": correlation_id},
        )

        # Return PlainTextResponse with Prometheus-compatible content-type header
        return PlainTextResponse(
            content=all_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    except Exception as e:
        logger.error(
            f"metrics_collection_failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        # Return minimal metrics with error information
        error_metrics = f"""
# Error collecting metrics
# HELP metrics_collection_errors_total Number of metrics collection errors
# TYPE metrics_collection_errors_total counter
metrics_collection_errors_total{{error_type="collection_failure"}} 1

# HELP metrics_collection_last_error_timestamp Unix timestamp of last metrics collection error
# TYPE metrics_collection_last_error_timestamp gauge
metrics_collection_last_error_timestamp {asyncio.get_event_loop().time()}
"""
        return PlainTextResponse(
            content=error_metrics.strip(),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )


@router.get("/metrics/json")
async def get_json_metrics() -> dict[str, Any]:
    """
    JSON-formatted metrics endpoint for applications that prefer JSON over Prometheus format.

    Returns the same metrics as /metrics but in JSON format for easier consumption.
    """
    correlation_id = TracingManager.get_correlation_id()

    try:
        metrics = {
            "timestamp": asyncio.get_event_loop().time(),
            "service": "PDF to Markdown MCP Server",
            "correlation_id": correlation_id,
        }

        # Collect database metrics
        try:
            from datetime import datetime, timedelta

            from pdf_to_markdown_mcp.db.models import Document, ProcessingQueue
            from pdf_to_markdown_mcp.db.session import get_db_session

            with get_db_session() as session:
                # Processing queue depth
                queue_count = (
                    session.scalar(
                        session.query(ProcessingQueue)
                        .filter(ProcessingQueue.status == "queued")
                        .count()
                    )
                    or 0
                )

                # Total documents
                total_docs = session.scalar(session.query(Document).count()) or 0

                # Documents in last hour
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                recent_docs = (
                    session.scalar(
                        session.query(Document)
                        .filter(Document.created_at >= one_hour_ago)
                        .count()
                    )
                    or 0
                )

                metrics.update(
                    {
                        "database": {
                            "processing_queue_depth": queue_count,
                            "documents_total": total_docs,
                            "documents_last_hour": recent_docs,
                            "processing_rate_per_hour": recent_docs,
                        }
                    }
                )

        except Exception as e:
            metrics["database"] = {"error": str(e)}

        # Collect Celery metrics
        try:
            from pdf_to_markdown_mcp.worker.celery import app as celery_app

            inspect = celery_app.control.inspect()
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}

            active_count = sum(len(tasks) for tasks in active.values())
            reserved_count = sum(len(tasks) for tasks in reserved.values())

            metrics.update(
                {
                    "celery": {
                        "active_tasks": active_count,
                        "reserved_tasks": reserved_count,
                        "workers": len(active.keys()),
                        "worker_names": list(active.keys()),
                    }
                }
            )

        except Exception as e:
            metrics["celery"] = {"error": str(e)}

        # Collect system metrics
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            try:
                disk = psutil.disk_usage("/")
                disk_metrics = {
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "disk_percent": round((disk.used / disk.total) * 100, 1),
                }
            except Exception:
                disk_metrics = {}

            metrics.update(
                {
                    "system": {
                        "memory_percent": round(memory.percent, 1),
                        "memory_available_gb": round(memory.available / (1024**3), 2),
                        "memory_total_gb": round(memory.total / (1024**3), 2),
                        "cpu_percent": round(cpu_percent, 1),
                        **disk_metrics,
                    }
                }
            )

        except ImportError:
            metrics["system"] = {"error": "psutil not available"}
        except Exception as e:
            metrics["system"] = {"error": str(e)}

        # Collect GPU metrics
        try:
            from pdf_to_markdown_mcp.core.monitoring import health_monitor
            gpu_health = await health_monitor.check_gpu_health()
            cuda_health = await health_monitor.check_cuda_availability()

            metrics["gpu"] = {
                "status": gpu_health.status.value,
                "details": gpu_health.details,
                "cuda_status": cuda_health.status.value,
                "cuda_details": cuda_health.details,
            }

        except Exception as e:
            metrics["gpu"] = {"error": str(e)}

        # Collect model loading metrics
        try:
            from pdf_to_markdown_mcp.services.mineru import _mineru_model_singleton
            metrics["model_status"] = {
                "singleton_initialized": _mineru_model_singleton is not None,
                "model_type": "MinerU",
            }
        except Exception as e:
            metrics["model_status"] = {"error": str(e)}

        logger.info(
            f"json_metrics_generated: {len([k for k in metrics if k not in ['timestamp', 'service', 'correlation_id']])} categories",
            extra={"correlation_id": correlation_id},
        )

        return metrics

    except Exception as e:
        logger.error(
            f"json_metrics_collection_failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "metrics_collection_failed",
                "message": "Failed to collect JSON metrics",
                "correlation_id": correlation_id,
            },
        )


async def _collect_custom_metrics() -> str:
    """Collect custom metrics not automatically handled by Prometheus client."""
    custom_metrics = []

    try:
        # System health summary
        system_health = await health_monitor.get_system_health()

        # Add health status as metrics
        healthy_components = sum(
            1
            for comp in system_health.components.values()
            if comp.status == HealthStatus.HEALTHY
        )

        degraded_components = sum(
            1
            for comp in system_health.components.values()
            if comp.status == HealthStatus.DEGRADED
        )

        unhealthy_components = sum(
            1
            for comp in system_health.components.values()
            if comp.status == HealthStatus.UNHEALTHY
        )

        custom_metrics.extend(
            [
                "# HELP system_health_components Number of components by health status",
                "# TYPE system_health_components gauge",
                f'system_health_components{{status="healthy"}} {healthy_components}',
                f'system_health_components{{status="degraded"}} {degraded_components}',
                f'system_health_components{{status="unhealthy"}} {unhealthy_components}',
                "",
                "# HELP system_uptime_seconds System uptime in seconds",
                "# TYPE system_uptime_seconds gauge",
                f"system_uptime_seconds {system_health.uptime_seconds}",
                "",
            ]
        )

        # Add component response times
        for name, component in system_health.components.items():
            if component.response_time_ms is not None:
                custom_metrics.extend(
                    [
                        f'component_response_time_milliseconds{{component="{name}"}} {component.response_time_ms}'
                    ]
                )

        if any("response_time_milliseconds" in line for line in custom_metrics):
            custom_metrics.insert(
                -len(
                    [
                        line
                        for line in custom_metrics
                        if "response_time_milliseconds" in line
                    ]
                ),
                "# HELP component_response_time_milliseconds Component response time in milliseconds",
            )
            custom_metrics.insert(
                -len(
                    [
                        line
                        for line in custom_metrics
                        if "response_time_milliseconds" in line
                    ]
                ),
                "# TYPE component_response_time_milliseconds gauge",
            )

    except Exception as e:
        logger.error(f"Failed to collect custom metrics: {e}")
        custom_metrics.append(f"# Error collecting custom metrics: {e}")

    return "\n".join(custom_metrics) + "\n" if custom_metrics else ""


# Add health endpoints to be included in main application
__all__ = ["router"]
