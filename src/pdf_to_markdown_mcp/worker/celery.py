"""
Celery application configuration for PDF to Markdown MCP Server.

This module sets up the Celery application with Redis broker, proper task routing,
error handling, monitoring, and coordination with the processing pipeline.
"""

import logging
import os
import socket
from datetime import datetime
from typing import Any

# Set GPU environment variables early for MinerU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MINERU_DEVICE_MODE"] = "cuda"

from celery import Celery, Task
from celery.signals import task_postrun, task_prerun, worker_ready, worker_shutting_down, worker_process_init
from kombu import Exchange, Queue

from ..config import settings
from ..core.circuit_breaker import (
    get_redis_broker_circuit_breaker,
    get_redis_result_backend_circuit_breaker,
    redis_circuit_breaker_manager,
)

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """
    Custom Celery task base class with progress tracking and better error handling.

    Provides standardized progress reporting, error categorization, and resource management.
    """

    def __init__(self):
        super().__init__()
        self._progress_callback = None

    def update_state(self, task_id=None, state=None, meta=None):
        """Enhanced state update with structured progress tracking."""
        if state == "PROGRESS" and meta:
            # Ensure progress metadata has required fields
            meta.setdefault("current", 0)
            meta.setdefault("total", 100)
            meta.setdefault("message", "Processing...")
            meta.setdefault("timestamp", None)

            # Add performance metrics if available
            if hasattr(self, "start_time"):
                import time

                meta["elapsed_seconds"] = time.time() - self.start_time

        return super().update_state(task_id=task_id, state=state, meta=meta)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Enhanced failure handling with error categorization."""
        logger.error(
            f"Task {self.name} [{task_id}] failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "exception": str(exc),
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )

        # Categorize error for better retry logic
        error_category = self._categorize_error(exc)

        self.update_state(
            task_id=task_id,
            state="FAILURE",
            meta={
                "error": str(exc),
                "error_category": error_category,
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Enhanced success handling with metrics."""
        logger.info(
            f"Task {self.name} [{task_id}] completed successfully",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "result": (
                    retval
                    if isinstance(retval, (dict, str, int))
                    else type(retval).__name__
                ),
            },
        )

    def _categorize_error(self, exc) -> str:
        """Categorize errors for appropriate retry strategies."""
        exc_name = type(exc).__name__

        # Transient errors that should be retried
        if exc_name in [
            "ConnectionError",
            "TimeoutError",
            "RedisError",
            "DatabaseError",
        ]:
            return "transient"

        # Resource errors that may resolve with time
        if exc_name in ["MemoryError", "DiskSpaceError"]:
            return "resource"

        # Validation errors that should not be retried
        if exc_name in ["ValidationError", "FileNotFoundError", "PermissionError"]:
            return "validation"

        # Processing errors specific to PDF/content
        if exc_name in ["PDFError", "EmbeddingError", "ProcessingError"]:
            return "processing"

        return "unknown"


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application.

    Returns:
        Celery: Configured Celery application instance
    """
    # Create Celery app with proper naming
    app = Celery("pdf_to_markdown_mcp")

    # Enhanced broker and backend configuration with Redis optimization
    app.conf.update(
        # Enhanced Broker settings with connection pooling
        broker_url=settings.celery.broker_url,
        result_backend=settings.celery.result_backend,
        broker_connection_retry_on_startup=settings.celery.broker_connection_retry_on_startup,
        broker_pool_limit=settings.celery.broker_pool_limit,
        broker_connection_timeout=settings.celery.broker_connection_timeout,
        broker_connection_retry=settings.celery.broker_connection_retry,
        broker_connection_max_retries=settings.celery.broker_connection_max_retries,
        # Enhanced Result backend configuration - Fixed memory leak
        result_expires=300,  # Reduced from 3600 to 5 minutes to prevent memory leak
        result_persistent=False,  # Changed from True to prevent Redis memory buildup
        result_compression=settings.celery.result_compression,
        result_serializer=settings.celery.result_serializer,
        # Result cleanup configuration
        result_backend_always_retry=True,
        result_backend_max_retries=3,
        # Enhanced Task serialization and security
        task_serializer=settings.celery.task_serializer,
        accept_content=settings.celery.accept_content,
        timezone="UTC",
        enable_utc=True,
        # Task routing configuration
        task_routes=_setup_task_routes(),
        task_default_queue=settings.celery.task_default_queue,
        # GPU-Optimized Worker settings for single GPU environment
        worker_concurrency=1,  # Single worker to prevent GPU contention
        worker_max_tasks_per_child=100,  # Increased to reduce model reload overhead
        worker_disable_rate_limits=settings.celery.worker_disable_rate_limits,
        worker_prefetch_multiplier=1,  # Process one task at a time per worker
        # GPU pool configuration - use 'solo' pool for GPU-bound tasks
        worker_pool="solo",  # Single-threaded pool optimal for GPU tasks
        # Enhanced Task execution settings - increased for GPU OCR processing
        task_soft_time_limit=600,  # 10 minutes soft limit for OCR tasks
        task_time_limit=900,  # 15 minutes hard limit for complex PDFs
        task_acks_late=settings.celery.task_acks_late,
        task_reject_on_worker_lost=settings.celery.task_reject_on_worker_lost,
        # Enhanced Retry settings
        task_annotations={
            "*": {
                "max_retries": settings.celery.task_default_max_retries,
                "default_retry_delay": settings.celery.task_default_retry_delay,
                "retry_backoff": settings.celery.task_retry_backoff,
                "retry_jitter": settings.celery.task_retry_jitter,
            },
            "pdf_to_markdown_mcp.worker.tasks.process_pdf_document": {
                "max_retries": 3,
                "default_retry_delay": 120,  # Longer delay for heavy processing
                "priority": 8,  # High priority
                "rate_limit": "10/m",  # Max 10 PDFs per minute to prevent overload
            },
            "pdf_to_markdown_mcp.worker.tasks.process_pdf_batch": {
                "max_retries": 2,
                "default_retry_delay": 180,  # Batch processing gets longer delay
                "priority": 6,  # Medium-high priority
            },
            "pdf_to_markdown_mcp.worker.tasks.generate_embeddings": {
                "max_retries": 5,  # More retries for embedding generation
                "default_retry_delay": 30,
                "priority": 5,  # Medium priority
            },
            "pdf_to_markdown_mcp.worker.tasks.process_document_images": {
                "max_retries": 3,
                "default_retry_delay": 45,
                "priority": 4,  # Medium-low priority
            },
            "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files": {
                "max_retries": 2,
                "default_retry_delay": 300,  # 5 minute delay for cleanup
                "priority": 1,  # Low priority
            },
            "pdf_to_markdown_mcp.worker.tasks.health_check": {
                "max_retries": 1,
                "default_retry_delay": 60,
                "priority": 3,  # Medium-low priority
            },
        },
        # Enhanced Monitoring and events
        worker_send_task_events=settings.celery.worker_send_task_events,
        task_send_sent_event=settings.celery.task_send_sent_event,
        worker_log_format=settings.celery.worker_log_format,
        worker_task_log_format=settings.celery.worker_task_log_format,
        # Enhanced Beat scheduler settings with redundancy
        beat_schedule={
            "cleanup-temp-files": {
                "task": "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files",
                "schedule": 3600.0,  # Every hour
                "options": {"priority": 1},
            },
            "cleanup-task-results": {
                "task": "pdf_to_markdown_mcp.worker.tasks.cleanup_task_results",
                "schedule": 900.0,  # Every 15 minutes - prevent memory leak
                "options": {"priority": 1},
            },
            "health-check": {
                "task": "pdf_to_markdown_mcp.worker.tasks.health_check",
                "schedule": 300.0,  # Every 5 minutes
                "options": {"priority": 3},
            },
            "connection-pool-monitor": {
                "task": "pdf_to_markdown_mcp.worker.tasks.monitor_redis_connections",
                "schedule": 180.0,  # Every 3 minutes
                "options": {"priority": 2},
            },
            "resync-filesystem": {
                "task": "pdf_to_markdown_mcp.worker.indexer.resync_filesystem_to_database",
                "schedule": settings.indexer.resync_interval_minutes * 60.0,  # Configurable interval
                "options": {"priority": 4},
            },
            "queue-pending-documents": {
                "task": "pdf_to_markdown_mcp.worker.indexer.queue_pending_documents",
                "schedule": settings.indexer.queue_pending_interval_minutes * 60.0,  # Configurable interval
                "options": {"priority": 3},
            },
            "gpu-performance-monitor": {
                "task": "pdf_to_markdown_mcp.worker.tasks.monitor_gpu_performance",
                "schedule": 600.0,  # Every 10 minutes
                "options": {"priority": 2},
            },
            "index-document-embeddings": {
                "task": "pdf_to_markdown_mcp.worker.embedding_indexer.index_document_embeddings",
                "schedule": 300.0,  # Every 5 minutes
                "kwargs": {"batch_size": 10},
                "options": {"priority": 3, "queue": "maintenance"},
            },
            "check-embedding-status": {
                "task": "pdf_to_markdown_mcp.worker.embedding_indexer.check_embedding_status",
                "schedule": 1800.0,  # Every 30 minutes
                "options": {"priority": 1, "queue": "maintenance"},
            },
            "reset-stuck-embeddings": {
                "task": "pdf_to_markdown_mcp.worker.embedding_indexer.reset_stuck_processing_status",
                "schedule": 900.0,  # Every 15 minutes
                "kwargs": {"timeout_minutes": 30},
                "options": {"priority": 2, "queue": "maintenance"},
            },
        },
        # Enhanced beat scheduler persistence with backup
        beat_schedule_filename=f"{settings.celery.beat_schedule_filename}.primary",
        beat_max_loop_interval=settings.celery.beat_max_loop_interval,
        # Redis-specific transport options with circuit breaker protection
        broker_transport_options={
            "master_name": "mymaster",
            "visibility_timeout": 3600,
            "retry_policy": {
                "timeout": settings.celery.redis_socket_timeout,
            },
            "fanout_prefix": True,
            "fanout_patterns": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                socket.TCP_KEEPIDLE: 1,
                socket.TCP_KEEPINTVL: 3,
                socket.TCP_KEEPCNT: 5,
            },
            # Enhanced connection pool management
            "max_connections": min(
                settings.celery.redis_max_connections, 15
            ),  # Reduced from 20
            "socket_timeout": settings.celery.redis_socket_timeout,
            "socket_connect_timeout": settings.celery.redis_socket_connect_timeout,
            "retry_on_timeout": settings.celery.redis_retry_on_timeout,
            "health_check_interval": settings.celery.redis_health_check_interval,
            # Connection pool monitoring
            "connection_pool_class_kwargs": {
                "max_connections": min(settings.celery.redis_max_connections, 15),
                "retry_on_timeout": True,
                "socket_keepalive": True,
                "socket_keepalive_options": {
                    socket.TCP_KEEPIDLE: 1,
                    socket.TCP_KEEPINTVL: 3,
                    socket.TCP_KEEPCNT: 5,
                },
            },
        },
        # Result backend transport options with enhanced pool management
        result_backend_transport_options={
            "master_name": "mymaster",
            "retry_policy": {
                "timeout": settings.celery.redis_socket_timeout,
            },
            # Separate connection pool for results to prevent contention
            "max_connections": max(
                5, min(settings.celery.redis_max_connections // 2, 10)
            ),
            "socket_timeout": settings.celery.redis_socket_timeout,
            "socket_connect_timeout": settings.celery.redis_socket_connect_timeout,
            "retry_on_timeout": settings.celery.redis_retry_on_timeout,
            "health_check_interval": settings.celery.redis_health_check_interval,
            "connection_pool_class_kwargs": {
                "max_connections": max(
                    5, min(settings.celery.redis_max_connections // 2, 10)
                ),
                "retry_on_timeout": True,
            },
        },
    )

    # Set up queues and exchanges
    app.conf.task_queues = _setup_queues()

    # Register task base class
    app.Task = CallbackTask

    return app


def _setup_task_routes() -> dict[str, dict[str, str]]:
    """Set up task routing configuration."""
    return {
        # PDF processing tasks - high priority
        "pdf_to_markdown_mcp.worker.tasks.process_pdf_document": {
            "queue": "pdf_processing",
            "routing_key": "pdf_processing",
        },
        # Embedding generation - medium priority
        "pdf_to_markdown_mcp.worker.tasks.generate_embeddings": {
            "queue": "embeddings",
            "routing_key": "embeddings",
        },
        # Embedding indexer tasks
        "pdf_to_markdown_mcp.worker.embedding_indexer.index_document_embeddings": {
            "queue": "embeddings",
            "routing_key": "embeddings",
        },
        "pdf_to_markdown_mcp.worker.embedding_indexer.check_embedding_status": {
            "queue": "maintenance",
            "routing_key": "maintenance",
        },
        # Maintenance tasks - low priority
        "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files": {
            "queue": "maintenance",
            "routing_key": "maintenance",
        },
        "pdf_to_markdown_mcp.worker.tasks.health_check": {
            "queue": "monitoring",
            "routing_key": "monitoring",
        },
    }


def _setup_queues() -> list:
    """Set up Celery queues with proper priorities and routing."""

    # Define exchanges
    default_exchange = Exchange("default", type="direct")

    return [
        # High-priority queue for PDF processing
        Queue(
            "pdf_processing",
            exchange=default_exchange,
            routing_key="pdf_processing",
            queue_arguments={
                "x-max-priority": settings.celery.task_queue_max_priority,
                "x-message-ttl": settings.celery.task_message_ttl
                * 1000,  # Convert to ms
                "x-max-length": 10000,  # Limit queue size to prevent memory issues
                "x-overflow": "reject-publish",  # Reject new messages when queue is full
            },
        ),
        # Medium-priority queue for embeddings and image processing
        Queue(
            "embeddings",
            exchange=default_exchange,
            routing_key="embeddings",
            queue_arguments={
                "x-max-priority": 5,
                "x-message-ttl": (settings.celery.task_message_ttl // 2)
                * 1000,  # 30 minutes TTL
                "x-max-length": 5000,
                "x-overflow": "reject-publish",
            },
        ),
        # Low-priority queue for maintenance tasks
        Queue(
            "maintenance",
            exchange=default_exchange,
            routing_key="maintenance",
            queue_arguments={
                "x-max-priority": 1,
                "x-message-ttl": (settings.celery.task_message_ttl * 2)
                * 1000,  # 2 hours TTL
                "x-max-length": 1000,
                "x-overflow": "drop-head",  # Drop oldest messages when full
            },
        ),
        # Real-time queue for monitoring and health checks
        Queue(
            "monitoring",
            exchange=default_exchange,
            routing_key="monitoring",
            queue_arguments={
                "x-max-priority": 3,
                "x-message-ttl": 300000,  # 5 minutes TTL
                "x-max-length": 100,
                "x-overflow": "drop-head",  # Keep only recent monitoring tasks
            },
        ),
    ]


# Celery signals for monitoring and lifecycle management
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker ready: {sender}")


@worker_process_init.connect
def worker_process_init_handler(sender=None, **kwargs):
    """Handle worker process initialization - runs in each forked worker process."""
    import os
    logger.info(f"Initializing worker process: PID={os.getpid()}")

    # Set GPU environment variables for MinerU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MINERU_DEVICE_MODE"] = "cuda"
    logger.info(f"Worker PID={os.getpid()}: Set CUDA_VISIBLE_DEVICES=0, MINERU_DEVICE_MODE=cuda")

    # Pre-warm GPU models for optimal performance
    try:
        from ..services.mineru import get_shared_mineru_instance
        mineru_instance = get_shared_mineru_instance()
        logger.info(f"Worker PID={os.getpid()}: Pre-warmed shared MinerU instance")
    except Exception as e:
        logger.warning(f"Worker PID={os.getpid()}: Failed to pre-warm MinerU: {e}")

    # Set GPU memory growth to prevent OOM
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear any existing GPU cache
            # Set memory fraction to leave room for other processes
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            logger.info(f"Worker PID={os.getpid()}: Configured GPU memory management")
    except Exception as e:
        logger.warning(f"Worker PID={os.getpid()}: GPU memory config failed: {e}")

    # Load configuration from database in each worker process
    try:
        from ..config import settings
        from ..db.session import SessionLocal
        from ..services.config_service import ConfigurationService

        with SessionLocal() as db:
            config_dict = ConfigurationService.load_from_database(db)

            # Apply watch_directories if present
            if "watch_directories" in config_dict:
                settings.watcher.watch_directories = config_dict["watch_directories"]
                logger.info(f"Worker PID={os.getpid()}: Loaded watch_directories: {settings.watcher.watch_directories}")

            # Apply output_directory if present
            if "output_directory" in config_dict:
                settings.watcher.output_directory = config_dict["output_directory"]
                logger.info(f"Worker PID={os.getpid()}: Loaded output_directory: {settings.watcher.output_directory}")

            logger.info(f"Worker PID={os.getpid()}: Configuration loaded successfully")
    except Exception as e:
        logger.warning(f"Worker PID={os.getpid()}: Failed to load configuration: {e}")
        logger.info(f"Worker PID={os.getpid()}: Using default configuration")


@worker_shutting_down.connect
def worker_shutting_down_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Celery worker shutting down: {sender}")


@task_prerun.connect
def task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds
):
    """Handle task pre-run signal."""
    logger.info(f"Starting task {task.name} [{task_id}]")

    # Set start time for duration tracking
    if hasattr(task, "start_time"):
        import time

        task.start_time = time.time()


@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kwds,
):
    """Handle task post-run signal."""
    duration = None
    if hasattr(task, "start_time"):
        import time

        duration = time.time() - task.start_time

    logger.info(
        f"Completed task {task.name} [{task_id}] in {duration:.2f}s"
        if duration
        else f"Completed task {task.name} [{task_id}]"
    )


# Create the Celery application instance
app = create_celery_app()
celery_app = app  # Alias for compatibility


# Auto-discover tasks in the tasks and indexer modules
app.autodiscover_tasks(["pdf_to_markdown_mcp.worker"], related_name="tasks")
app.autodiscover_tasks(["pdf_to_markdown_mcp.worker"], related_name="indexer")
app.autodiscover_tasks(["pdf_to_markdown_mcp.worker"], related_name="embedding_indexer")


# Enhanced monitoring and management functions
def get_worker_stats() -> dict[str, Any]:
    """Get comprehensive worker statistics for monitoring."""
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()

        return {
            "active_queues": [q.name for q in app.conf.task_queues],
            "registered_tasks": list(app.tasks.keys()),
            "worker_stats": stats,
            "active_tasks": active,
            "scheduled_tasks": scheduled,
            "reserved_tasks": reserved,
            "queue_statistics": get_all_queue_stats(),
            "redis_connection_info": get_redis_connection_info(),
        }
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return {"error": str(e)}


def get_queue_length(queue_name: str) -> int:
    """Get the length of a specific queue with enhanced error handling."""
    try:
        with app.connection() as conn:
            # Try both Celery queue names and direct Redis keys
            redis_client = conn.default_channel.client

            # Standard Celery queue key
            celery_key = f"celery.{queue_name}"
            length = redis_client.llen(celery_key)

            if length == 0:
                # Try direct queue name
                length = redis_client.llen(queue_name)

            return length
    except Exception as e:
        logger.error(f"Failed to get queue length for {queue_name}: {e}")
        return -1


def get_all_queue_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all configured queues."""
    queue_stats = {}

    for queue in app.conf.task_queues:
        queue_name = queue.name
        try:
            with app.connection() as conn:
                redis_client = conn.default_channel.client
                celery_key = f"celery.{queue_name}"

                queue_stats[queue_name] = {
                    "length": redis_client.llen(celery_key),
                    "priority": queue.queue_arguments.get("x-max-priority", 0),
                    "ttl_ms": queue.queue_arguments.get("x-message-ttl", 0),
                    "max_length": queue.queue_arguments.get("x-max-length", -1),
                    "overflow_behavior": queue.queue_arguments.get(
                        "x-overflow", "none"
                    ),
                    "routing_key": queue.routing_key,
                }
        except Exception as e:
            queue_stats[queue_name] = {"error": str(e)}

    return queue_stats


def get_redis_connection_info() -> dict[str, Any]:
    """Get Redis connection information and health status with circuit breaker protection."""
    circuit_breaker = get_redis_broker_circuit_breaker()

    try:
        with circuit_breaker("get_redis_info"), app.connection() as conn:
            redis_client = conn.default_channel.client
            info = redis_client.info()

            connection_stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0"),
                "used_memory": info.get("used_memory", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_connections_received": info.get(
                    "total_connections_received", 0
                ),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "role": info.get("role", "unknown"),
                # Enhanced connection pool metrics
                "max_clients": info.get("maxclients", 0),
                "rejected_connections": info.get("rejected_connections", 0),
                "client_recent_max_input_buffer": info.get(
                    "client_recent_max_input_buffer", 0
                ),
                "client_recent_max_output_buffer": info.get(
                    "client_recent_max_output_buffer", 0
                ),
            }

            # Add circuit breaker statistics
            connection_stats["circuit_breaker"] = circuit_breaker.get_stats()

            # Calculate connection pool utilization
            max_clients = connection_stats["max_clients"]
            connected_clients = connection_stats["connected_clients"]
            if max_clients > 0:
                connection_stats["pool_utilization_percent"] = round(
                    (connected_clients / max_clients) * 100, 2
                )

            return connection_stats

    except Exception as e:
        logger.error(f"Failed to get Redis connection info: {e}")
        return {"error": str(e), "circuit_breaker": circuit_breaker.get_stats()}


def purge_queue(queue_name: str) -> int:
    """Purge all messages from a specific queue."""
    try:
        purged_count = app.control.purge()
        logger.info(f"Purged {purged_count} messages from all queues")
        return purged_count
    except Exception as e:
        logger.error(f"Failed to purge queue {queue_name}: {e}")
        return -1


def get_task_info(task_id: str) -> dict[str, Any]:
    """Get detailed information about a specific task."""
    try:
        result = app.AsyncResult(task_id)

        task_info = {
            "task_id": task_id,
            "state": result.state,
            "ready": result.ready(),
            "successful": result.successful(),
            "failed": result.failed(),
        }

        if result.ready():
            task_info["result"] = result.result
        else:
            # Get task info from result backend
            task_info["info"] = result.info

        return task_info
    except Exception as e:
        logger.error(f"Failed to get task info for {task_id}: {e}")
        return {"error": str(e)}


def cancel_task(task_id: str) -> bool:
    """Cancel a specific task."""
    try:
        app.control.revoke(task_id, terminate=True)
        logger.info(f"Cancelled task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


def get_worker_health() -> dict[str, Any]:
    """Comprehensive worker health check with circuit breaker awareness."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
        "circuit_breakers": redis_circuit_breaker_manager.get_all_stats(),
    }

    # Check Redis broker connectivity with circuit breaker protection
    circuit_breaker = get_redis_broker_circuit_breaker()
    try:
        with circuit_breaker("health_check_broker"), app.connection() as conn:
            conn.default_channel.client.ping()
        health_status["checks"]["redis_broker"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis_broker"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check Redis result backend
    result_circuit_breaker = get_redis_result_backend_circuit_breaker()
    try:
        with result_circuit_breaker("health_check_result_backend"):
            result_backend = app.backend
            # Test result backend connectivity
            test_result = result_backend.get("health_check_test")
        health_status["checks"]["redis_result_backend"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis_result_backend"] = f"unhealthy: {e!s}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    try:
        # Check worker responsiveness
        inspect = app.control.inspect()
        stats = inspect.stats()
        if stats:
            health_status["checks"]["workers"] = "healthy"
            # Add worker count information
            health_status["worker_count"] = len(stats)
        else:
            health_status["checks"]["workers"] = "no_workers_found"
            health_status["status"] = "degraded"
            health_status["worker_count"] = 0
    except Exception as e:
        health_status["checks"]["workers"] = f"unhealthy: {e!s}"
        health_status["status"] = "unhealthy"
        health_status["worker_count"] = 0

    # Check queue depths for potential issues
    try:
        queue_stats = get_all_queue_stats()
        total_queued = sum(
            stat.get("length", 0)
            for stat in queue_stats.values()
            if isinstance(stat, dict) and "length" in stat
        )

        # Enhanced thresholds based on queue priority
        critical_queues = ["pdf_processing", "embeddings"]
        critical_queued = sum(
            stat.get("length", 0)
            for queue_name, stat in queue_stats.items()
            if queue_name in critical_queues
            and isinstance(stat, dict)
            and "length" in stat
        )

        if critical_queued > 100:  # Critical threshold
            health_status["checks"]["queue_depth"] = (
                f"critical: {critical_queued} critical tasks queued"
            )
            health_status["status"] = "unhealthy"
        elif total_queued > 1000:  # General threshold
            health_status["checks"]["queue_depth"] = (
                f"warning: {total_queued} tasks queued"
            )
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        else:
            health_status["checks"]["queue_depth"] = (
                f"healthy: {total_queued} tasks queued"
            )

        health_status["total_queued_tasks"] = total_queued
        health_status["critical_queued_tasks"] = critical_queued
        health_status["queue_breakdown"] = queue_stats
    except Exception as e:
        health_status["checks"]["queue_depth"] = f"error: {e!s}"
        health_status["status"] = "degraded"

    # Check Redis connection pool utilization
    try:
        redis_info = get_redis_connection_info()
        if "pool_utilization_percent" in redis_info:
            utilization = redis_info["pool_utilization_percent"]
            if utilization > 90:
                health_status["checks"]["connection_pool"] = (
                    f"critical: {utilization}% utilized"
                )
                health_status["status"] = "unhealthy"
            elif utilization > 75:
                health_status["checks"]["connection_pool"] = (
                    f"warning: {utilization}% utilized"
                )
                if health_status["status"] == "healthy":
                    health_status["status"] = "degraded"
            else:
                health_status["checks"]["connection_pool"] = (
                    f"healthy: {utilization}% utilized"
                )

        health_status["redis_info"] = redis_info
    except Exception as e:
        health_status["checks"]["connection_pool"] = f"error: {e!s}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    return health_status


if __name__ == "__main__":
    # Allow running worker directly
    app.start()
