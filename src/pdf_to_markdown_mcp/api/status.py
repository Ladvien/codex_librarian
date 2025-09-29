"""
Status and monitoring API endpoints.

Implements the get_status MCP tool and additional monitoring endpoints.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.api.streaming import create_sse_response, format_sse_data
from pdf_to_markdown_mcp.core.streaming import get_streaming_stats
from pdf_to_markdown_mcp.db.session import get_db
from pdf_to_markdown_mcp.models.response import (
    ErrorResponse,
    ErrorType,
    JobStatus,
    StatusResponse,
)
from pdf_to_markdown_mcp.worker.celery import create_celery_app

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status", response_model=StatusResponse)
async def get_processing_status(
    job_id: str | None = Query(None, description="Specific job ID to query"),
    include_stats: bool = Query(True, description="Include system statistics"),
    db: Session = Depends(get_db),
) -> StatusResponse:
    """
    Get current processing status and queue information.

    This endpoint implements the get_status MCP tool functionality.
    """
    try:
        logger.info(
            "Status request received",
            extra={"job_id": job_id, "include_stats": include_stats},
        )

        # Get Celery app instance
        celery_app = create_celery_app()

        # Job-specific status
        job_status = None
        progress_percent = None
        current_step = None
        started_at = None
        estimated_completion = None
        completed_at = None
        error_message = None
        retry_count = None

        if job_id:
            try:
                # Get job result from Celery
                job_result = celery_app.AsyncResult(job_id)

                # Map Celery states to our JobStatus enum
                state_mapping = {
                    "PENDING": JobStatus.QUEUED,
                    "STARTED": JobStatus.RUNNING,
                    "SUCCESS": JobStatus.COMPLETED,
                    "FAILURE": JobStatus.FAILED,
                    "REVOKED": JobStatus.CANCELLED,
                }

                job_status = state_mapping.get(job_result.state, JobStatus.QUEUED)

                # Get additional job info
                if job_result.info:
                    if isinstance(job_result.info, dict):
                        progress_percent = job_result.info.get("progress")
                        current_step = job_result.info.get("current_step")
                        started_at = job_result.info.get("started_at")
                        estimated_completion = job_result.info.get(
                            "estimated_completion"
                        )

                if job_result.state == "FAILURE":
                    error_message = str(job_result.info)

            except Exception as e:
                logger.warning(f"Could not retrieve job status for {job_id}: {e}")
                job_status = JobStatus.QUEUED  # Default to queued if we can't find it

        # Queue statistics
        queue_depth = 0
        active_jobs = 0

        try:
            # Get queue information from Celery
            inspect = celery_app.control.inspect()

            # Get active jobs
            active = inspect.active()
            if active:
                active_jobs = sum(len(jobs) for jobs in active.values())

            # Get reserved (queued) jobs
            reserved = inspect.reserved()
            if reserved:
                queue_depth = sum(len(jobs) for jobs in reserved.values())

        except Exception as e:
            logger.warning(f"Could not retrieve Celery queue stats: {e}")

        # System statistics
        total_documents = 0
        processing_rate_per_hour = None

        if include_stats:
            try:
                # Get document statistics from database
                from pdf_to_markdown_mcp.db.queries import DocumentQueries

                doc_stats = DocumentQueries.get_statistics(db)
                total_documents = doc_stats.get("total_documents", 0)

                # Calculate processing rate (simplified - documents processed in last hour)
                from datetime import datetime, timedelta

                from sqlalchemy import and_

                from pdf_to_markdown_mcp.db.models import Document

                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                recent_completed = (
                    db.query(Document)
                    .filter(
                        and_(
                            Document.conversion_status == "completed",
                            Document.updated_at >= one_hour_ago,
                        )
                    )
                    .count()
                )
                processing_rate_per_hour = recent_completed

            except Exception as e:
                logger.warning(f"Could not retrieve system statistics: {e}")

        return StatusResponse(
            job_id=job_id,
            status=job_status,
            progress_percent=progress_percent,
            current_step=current_step,
            started_at=started_at,
            estimated_completion=estimated_completion,
            completed_at=completed_at,
            queue_depth=queue_depth,
            active_jobs=active_jobs,
            total_documents=total_documents,
            processing_rate_per_hour=processing_rate_per_hour,
            error_message=error_message,
            retry_count=retry_count,
        )

    except Exception as e:
        logger.exception("Unexpected error retrieving status")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM, message=f"Failed to retrieve status: {e!s}"
            ).dict(),
        )


@router.get("/queue/stats")
async def get_queue_statistics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Get detailed queue statistics and worker information.
    """
    try:
        celery_app = create_celery_app()
        inspect = celery_app.control.inspect()

        # Worker statistics
        stats = inspect.stats()
        active_queues = inspect.active_queues()
        registered_tasks = inspect.registered()

        # Queue depths by queue name
        queue_stats = {}
        reserved = inspect.reserved()
        if reserved:
            for worker, jobs in reserved.items():
                for job in jobs:
                    queue_name = job.get("delivery_info", {}).get(
                        "routing_key", "default"
                    )
                    queue_stats[queue_name] = queue_stats.get(queue_name, 0) + 1

        return {
            "workers": {
                "total": len(stats) if stats else 0,
                "active": len([w for w in stats.keys()]) if stats else 0,
                "stats": stats or {},
            },
            "queues": {"depths": queue_stats, "active_queues": active_queues or {}},
            "tasks": {"registered": registered_tasks or {}},
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception("Error retrieving queue statistics")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to retrieve queue statistics: {e!s}",
            ).dict(),
        )


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum log entries to return"),
) -> dict[str, Any]:
    """
    Get detailed logs for a specific job.
    """
    try:
        # Get job logs from Celery task result
        celery_app = create_celery_app()
        logs = []
        total_entries = 0

        try:
            job_result = celery_app.AsyncResult(job_id)
            if job_result and job_result.info and isinstance(job_result.info, dict):
                # Extract log entries from job result metadata
                log_entries = job_result.info.get("logs", [])
                total_entries = len(log_entries)

                # Apply limit
                logs = log_entries[-limit:] if len(log_entries) > limit else log_entries

                # Format log entries
                formatted_logs = []
                for entry in logs:
                    if isinstance(entry, dict):
                        formatted_logs.append(entry)
                    else:
                        # Convert string entries to structured format
                        formatted_logs.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "level": "INFO",
                                "message": str(entry),
                                "job_id": job_id,
                            }
                        )
                logs = formatted_logs

        except Exception as e:
            logger.warning(f"Could not retrieve logs for job {job_id}: {e}")

        return {
            "job_id": job_id,
            "logs": logs,
            "total_entries": total_entries,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error retrieving logs for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM, message=f"Failed to retrieve job logs: {e!s}"
            ).dict(),
        )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    """
    Cancel a running or queued job.
    """
    try:
        celery_app = create_celery_app()

        # Revoke the task
        celery_app.control.revoke(job_id, terminate=True)

        logger.info(f"Job cancellation requested for {job_id}")

        return {
            "success": True,
            "job_id": job_id,
            "message": "Job cancellation requested",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error cancelling job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM, message=f"Failed to cancel job: {e!s}"
            ).dict(),
        )


@router.get("/system/performance")
async def get_system_performance(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Get system performance metrics and health indicators.
    """
    try:

        import psutil

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Database connection pool status
        db_metrics = {"connection_pool": "healthy", "active_connections": 0}
        try:
            # Get database engine info
            from pdf_to_markdown_mcp.db.session import engine

            if hasattr(engine.pool, "size"):
                db_metrics.update(
                    {
                        "pool_size": engine.pool.size(),
                        "checked_in": engine.pool.checkedin(),
                        "checked_out": engine.pool.checkedout(),
                        "overflow": engine.pool.overflow(),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not get database pool metrics: {e}")

        # Processing performance metrics
        processing_metrics = {
            "avg_pdf_processing_time_ms": 0,
            "avg_embedding_time_ms": 0,
            "success_rate_percent": 0,
        }
        try:
            # Get processing statistics from database
            from pdf_to_markdown_mcp.db.queries import DocumentQueries

            stats = DocumentQueries.get_statistics(db)

            # Calculate success rate
            by_status = stats.get("by_status", {})
            total = sum(by_status.values())
            completed = by_status.get("completed", 0)
            if total > 0:
                processing_metrics["success_rate_percent"] = (completed / total) * 100

        except Exception as e:
            logger.warning(f"Could not get processing metrics: {e}")

        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "percent": (disk.used / disk.total) * 100,
                },
            },
            "processing": processing_metrics,
            "database": db_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception("Error retrieving system performance metrics")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to retrieve performance metrics: {e!s}",
            ).dict(),
        )


# Server-Sent Events endpoints for real-time progress streaming


@router.get("/stream/{job_id}")
async def stream_job_progress(job_id: str) -> StreamingResponse:
    """
    Stream real-time progress updates for a specific job using Server-Sent Events.

    This endpoint implements the stream_progress MCP tool functionality with
    proper SSE formatting and real-time updates from Celery tasks.
    Enhanced with comprehensive streaming capabilities and memory management.
    """
    try:
        logger.info(f"Starting enhanced progress stream for job {job_id}")

        # Register job with our progress monitor
        progress_monitor.register_job(job_id)

        # Create and return SSE response with enhanced streaming
        return create_sse_response(create_job_progress_stream(job_id))

    except Exception as e:
        logger.exception(f"Error creating progress stream for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to create progress stream: {e!s}",
            ).dict(),
        )


@router.get("/stream/batch/{batch_id}")
async def stream_batch_progress(batch_id: str) -> StreamingResponse:
    """
    Stream progress updates for batch processing operations.

    Similar to job progress streaming but for batch operations that
    may involve multiple jobs.
    """
    try:
        logger.info(f"Starting batch progress stream for batch {batch_id}")

        # For batch operations, we can stream updates about the overall batch
        # as well as individual job updates within the batch
        async def batch_progress_generator():
            # Initial connection event
            yield format_sse_data(
                {
                    "batch_id": batch_id,
                    "event": "connected",
                    "message": "Connected to batch progress stream",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                event_type="batch_status",
            )

            # Stream batch progress updates
            # This would typically monitor multiple jobs within the batch
            try:
                # Get Celery app to check batch status
                celery_app = create_celery_app()

                # Monitor batch progress (simplified implementation)
                last_update_time = datetime.utcnow()
                while True:
                    # Check if batch is complete or failed
                    try:
                        batch_result = celery_app.AsyncResult(batch_id)

                        if batch_result.state == "SUCCESS":
                            yield format_sse_data(
                                {
                                    "batch_id": batch_id,
                                    "status": "completed",
                                    "message": "Batch processing completed",
                                    "timestamp": datetime.utcnow().isoformat(),
                                },
                                event_type="batch_complete",
                            )
                            break
                        elif batch_result.state == "FAILURE":
                            yield format_sse_data(
                                {
                                    "batch_id": batch_id,
                                    "status": "failed",
                                    "error": str(batch_result.info),
                                    "timestamp": datetime.utcnow().isoformat(),
                                },
                                event_type="batch_error",
                            )
                            break
                        else:
                            # Send periodic updates
                            current_time = datetime.utcnow()
                            if (current_time - last_update_time).total_seconds() >= 30:
                                yield format_sse_data(
                                    {
                                        "batch_id": batch_id,
                                        "status": batch_result.state,
                                        "message": f"Batch status: {batch_result.state}",
                                        "timestamp": current_time.isoformat(),
                                    },
                                    event_type="batch_progress",
                                )
                                last_update_time = current_time

                    except Exception as e:
                        logger.warning(f"Error checking batch {batch_id} status: {e}")

                    await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                yield format_sse_data(
                    {
                        "batch_id": batch_id,
                        "error": "stream_error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    event_type="error",
                )

        return create_sse_response(batch_progress_generator())

    except Exception as e:
        logger.exception(f"Error creating batch progress stream for {batch_id}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to create batch progress stream: {e!s}",
            ).dict(),
        )


@router.get("/stream/system")
async def stream_system_metrics(
    update_interval: float = 5.0, include_streaming_stats: bool = True
) -> StreamingResponse:
    """
    Stream real-time system metrics and queue statistics with enhanced capabilities.

    Provides comprehensive system performance monitoring including:
    - CPU and memory usage
    - Disk usage and I/O statistics
    - Celery queue depth and worker status
    - Streaming operation statistics
    - Database connection pool metrics
    - PDF processing performance metrics

    Args:
        update_interval: Seconds between metric updates (default: 5.0)
        include_streaming_stats: Include comprehensive streaming statistics
    """
    try:
        logger.info(
            f"Starting enhanced system metrics stream (interval: {update_interval}s)"
        )

        # Use our enhanced system metrics stream from the streaming module
        return create_sse_response(
            create_system_metrics_stream(
                update_interval=update_interval,
                include_streaming_stats=include_streaming_stats,
            )
        )

    except Exception as e:
        logger.exception("Error creating system metrics stream")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to create system metrics stream: {e!s}",
            ).dict(),
        )


@router.get("/streaming/status")
async def get_streaming_status_endpoint() -> dict[str, Any]:
    """
    Get comprehensive streaming system status and capabilities.

    Returns information about:
    - Active streaming operations
    - Memory usage and backpressure status
    - Concurrency limits and capacity
    - Streaming performance metrics
    - Server-Sent Events statistics
    """
    try:
        # Get comprehensive streaming statistics
        core_streaming_stats = get_streaming_stats()
        api_streaming_stats = get_streaming_status()

        # Combine statistics from both modules
        streaming_status = {
            "streaming_infrastructure": {
                "core_streaming": core_streaming_stats,
                "api_streaming": api_streaming_stats,
                "integration_status": "fully_integrated",
            },
            "capabilities": {
                "large_file_processing": "up_to_500MB",
                "memory_mapped_reading": True,
                "backpressure_handling": True,
                "progress_streaming": True,
                "server_sent_events": True,
                "concurrent_operations": True,
                "memory_monitoring": True,
            },
            "performance_thresholds": {
                "streaming_activation_size_mb": 50,
                "max_file_size_mb": 500,
                "max_concurrent_streams": 5,
                "memory_pressure_threshold_percent": 75.0,
                "chunk_sizes": {
                    "default": "64KB",
                    "large_files": "1MB",
                    "hash_calculation": "64KB",
                },
            },
        }

        return streaming_status

    except Exception as e:
        logger.exception("Error retrieving streaming status")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to retrieve streaming status: {e!s}",
            ).dict(),
        )


@router.post("/streaming/operations/{operation_id}/cancel")
async def cancel_streaming_operation(operation_id: str) -> dict[str, Any]:
    """
    Cancel an active streaming operation.

    This endpoint allows cancelling long-running PDF processing operations
    that are using streaming capabilities.
    """
    try:
        # Import processor here to avoid circular imports
        from pdf_to_markdown_mcp.core.processor import PDFProcessor

        # Create a temporary processor instance to access cancel functionality
        # In a real implementation, this would be managed by a service layer
        processor = PDFProcessor(None, enable_streaming=True)

        cancelled = await processor.cancel_operation(operation_id)

        if cancelled:
            return {
                "operation_id": operation_id,
                "status": "cancelled",
                "message": "Operation successfully cancelled",
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "operation_id": operation_id,
                "status": "not_found",
                "message": "Operation not found or already completed",
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.exception(f"Error cancelling streaming operation {operation_id}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM, message=f"Failed to cancel operation: {e!s}"
            ).dict(),
        )
