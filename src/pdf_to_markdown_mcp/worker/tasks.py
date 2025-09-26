"""
Celery task definitions for PDF processing pipeline.

This module defines all background tasks for PDF processing, embedding generation,
and system maintenance with proper error handling and progress tracking.
"""

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import settings
from ..core.errors import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
    ResourceError,
    SecurityError,
    TransientError,
    ValidationError,
    create_correlation_id,
    get_retry_strategy,
    global_retry_manager,
    sanitize_log_message,
    track_error,
)
from .celery import app

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Helper class for tracking and reporting task progress with persistent state."""

    def __init__(self, task_instance, total_steps: int = 100):
        self.task = task_instance
        self.current_step = 0
        self.total_steps = total_steps
        self.start_time = time.time()
        self.messages = []
        self.correlation_id = create_correlation_id()
        self.operation_name = getattr(task_instance, "name", "unknown_task")

        # Persistent progress tracking with Redis
        self.task_id = getattr(task_instance, "request", {}).get("id", "unknown")
        self.progress_key = f"task_progress:{self.task_id}"

        # Try to recover previous progress state
        self._recover_progress_state()

    def update(self, current: int = None, message: str = "", add_step: bool = True):
        """Update progress with optional message and persistent state."""
        if add_step:
            self.current_step += 1
        if current is not None:
            self.current_step = current

        # Sanitize message before storing
        sanitized_message = sanitize_log_message(message)
        self.messages.append(sanitized_message)

        # Keep only last 10 messages to avoid memory bloat
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]

        elapsed = time.time() - self.start_time
        eta = (
            (elapsed / max(self.current_step, 1))
            * (self.total_steps - self.current_step)
            if self.current_step > 0
            else None
        )

        progress_meta = {
            "current": self.current_step,
            "total": self.total_steps,
            "message": sanitized_message,
            "messages": self.messages[-3:],  # Last 3 messages
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "timestamp": datetime.utcnow().isoformat(),
            "percentage": round((self.current_step / self.total_steps) * 100, 2),
            "correlation_id": self.correlation_id,
            "operation": self.operation_name,
        }

        # Update Celery task state
        self.task.update_state(state="PROGRESS", meta=progress_meta)

        # Persist progress state to Redis for recovery
        self._persist_progress_state(progress_meta)

    def complete(self, final_message: str = "Task completed successfully"):
        """Mark task as completed and clean up persistent state."""
        self.update(current=self.total_steps, message=final_message)
        self._cleanup_progress_state()

    def _recover_progress_state(self):
        """Attempt to recover progress state from Redis on worker restart."""
        try:
            from ..core.circuit_breaker import get_redis_cache_circuit_breaker
            from .celery import app as celery_app

            circuit_breaker = get_redis_cache_circuit_breaker()

            with circuit_breaker("recover_progress_state"):
                with celery_app.connection() as conn:
                    redis_client = conn.default_channel.client

                    progress_data = redis_client.get(self.progress_key)
                    if progress_data:
                        import json

                        saved_progress = json.loads(progress_data)

                        # Restore progress state
                        self.current_step = saved_progress.get("current", 0)
                        self.messages = saved_progress.get("messages", [])

                        # Log recovery
                        logger.info(
                            f"Recovered progress state for task {self.task_id}: "
                            f"{self.current_step}/{self.total_steps}"
                        )

        except Exception as e:
            # Progress recovery is not critical - continue with fresh state
            logger.warning(f"Could not recover progress state for {self.task_id}: {e}")

    def _persist_progress_state(self, progress_meta: dict):
        """Persist progress state to Redis for recovery."""
        try:
            from ..core.circuit_breaker import get_redis_cache_circuit_breaker
            from .celery import app as celery_app

            circuit_breaker = get_redis_cache_circuit_breaker()

            with circuit_breaker("persist_progress_state"):
                with celery_app.connection() as conn:
                    redis_client = conn.default_channel.client

                    # Store essential progress data
                    progress_state = {
                        "current": self.current_step,
                        "total": self.total_steps,
                        "messages": self.messages[-3:],  # Last 3 messages only
                        "timestamp": progress_meta["timestamp"],
                        "correlation_id": self.correlation_id,
                    }

                    import json

                    redis_client.setex(
                        self.progress_key,
                        1800,  # 30 minutes TTL
                        json.dumps(progress_state),
                    )

        except Exception as e:
            # Progress persistence failure is not critical
            logger.debug(f"Could not persist progress state for {self.task_id}: {e}")

    def _cleanup_progress_state(self):
        """Clean up persistent progress state after task completion."""
        try:
            from ..core.circuit_breaker import get_redis_cache_circuit_breaker
            from .celery import app as celery_app

            circuit_breaker = get_redis_cache_circuit_breaker()

            with circuit_breaker("cleanup_progress_state"):
                with celery_app.connection() as conn:
                    redis_client = conn.default_channel.client
                    redis_client.delete(self.progress_key)

        except Exception as e:
            logger.debug(f"Could not cleanup progress state for {self.task_id}: {e}")


@app.task(bind=True, max_retries=5, default_retry_delay=60)
def process_pdf_document(
    self,
    document_id: int,
    file_path: str,
    processing_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process PDF document with MinerU and coordinate the complete pipeline.

    Enhanced with comprehensive error handling, retry mechanisms, and security validation.

    Args:
        document_id: Database ID of the document
        file_path: Path to the PDF file
        processing_options: Processing configuration options

    Returns:
        Dict containing processing results and metadata

    Raises:
        ValidationError: For invalid input or file issues (not retried)
        ProcessingError: For PDF processing failures (retried with backoff)
        DatabaseError: For database operation failures (retried)
        SecurityError: For security violations (not retried, logged)
        ResourceError: For resource exhaustion (retried with longer backoff)
        TimeoutError: For processing timeouts (retried)
    """
    progress = ProgressTracker(self, total_steps=10)
    correlation_id = progress.correlation_id

    # Log start of processing with correlation ID
    logger.info(
        f"Starting PDF processing for document ID {document_id}",
        extra={
            "correlation_id": correlation_id,
            "document_id": document_id,
            "operation": "process_pdf_document",
            "stage": "start",
        },
    )

    progress.update(message=f"Starting PDF processing for document ID {document_id}")

    try:
        # Enhanced input validation with security checks
        if not document_id or not isinstance(document_id, int):
            error = ValidationError(
                "Invalid document_id provided",
                correlation_id=correlation_id,
                error_code="PDF001",
            )
            track_error(error, "process_pdf_document", "input_validation")
            raise error

        # Security: Validate and sanitize file path
        from ..core.errors import validate_file_path

        try:
            validated_path = validate_file_path(str(file_path))
            file_path = Path(validated_path)
        except Exception:
            error = SecurityError(
                "Invalid file path detected",
                correlation_id=correlation_id,
                security_event_type="path_validation_failure",
                error_code="PDF002",
            )
            track_error(error, "process_pdf_document", "security_validation")
            raise error

        if not file_path.exists():
            error = ValidationError(
                f"PDF file not found: {file_path.name}",  # Don't expose full path
                correlation_id=correlation_id,
                error_code="PDF003",
            )
            track_error(error, "process_pdf_document", "file_validation")
            raise error

        if not file_path.suffix.lower() == ".pdf":
            error = ValidationError(
                f"File is not a PDF: {file_path.suffix}",
                correlation_id=correlation_id,
                error_code="PDF004",
            )
            track_error(error, "process_pdf_document", "file_validation")
            raise error

        # Check file size with resource error for oversized files
        file_size = file_path.stat().st_size
        max_size = settings.processing.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            error = ResourceError(
                f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds limit "
                f"({settings.processing.max_file_size_mb}MB)",
                resource_type="file_size",
                correlation_id=correlation_id,
                error_code="PDF005",
            )
            track_error(error, "process_pdf_document", "resource_validation")
            raise error

        progress.update(message="File validation completed")

        # Import services here to avoid circular imports
        from ..services.database import get_db_session
        from ..services.embeddings import EmbeddingService
        from ..services.mineru import MinerUService

        # Initialize services
        progress.update(message="Initializing processing services")
        mineru_service = MinerUService()
        embedding_service = EmbeddingService()

        # Update document status in database
        progress.update(message="Updating database status")
        with get_db_session() as db:
            from ..db.models import Document

            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise DatabaseError(f"Document not found in database: {document_id}")

            document.conversion_status = "processing"
            document.updated_at = datetime.utcnow()
            db.commit()

        # Enhanced PDF processing with MinerU - includes progress tracking
        progress.update(message="Extracting content with MinerU")
        try:
            # Create progress callback for MinerU processing
            def mineru_progress_callback(
                stage: str, current: int, total: int, message: str = ""
            ):
                # Map MinerU progress to our overall progress (steps 4-6)
                overall_current = 3 + int((current / total) * 3) if total > 0 else 4
                progress.update(
                    current=overall_current, message=f"MinerU {stage}: {message}"
                )

            # Use synchronous retry operation in Celery task
            processing_result = global_retry_manager.execute_with_retry(
                lambda: mineru_service.process_pdf(
                    file_path=file_path,
                    options=processing_options or {},
                    progress_callback=mineru_progress_callback,
                ),
                operation_name="mineru_pdf_processing",
                correlation_id=correlation_id,
            )

            if not processing_result or "markdown" not in processing_result:
                error = ProcessingError(
                    "MinerU processing returned invalid result",
                    correlation_id=correlation_id,
                    error_code="PDF006",
                )
                track_error(error, "process_pdf_document", "mineru_processing")
                raise error

        except Exception as e:
            if isinstance(e, (ProcessingError, ValidationError)):
                raise

            # Wrap unknown errors in ProcessingError
            processing_error = ProcessingError(
                "PDF content extraction failed",
                internal_details=str(e),
                correlation_id=correlation_id,
                error_code="PDF007",
            )
            track_error(processing_error, "process_pdf_document", "mineru_processing")
            raise processing_error

        progress.update(
            message="PDF processing completed, validating and saving content"
        )

        # Enhanced content validation
        try:
            _validate_processing_result(processing_result, correlation_id)
        except ValidationError as e:
            track_error(e, "process_pdf_document", "result_validation")
            raise

        # Enhanced database operations with transaction management
        content_record_id = None
        try:
            with get_db_session() as db:
                from ..db.models import DocumentContent

                content_record = DocumentContent(
                    document_id=document_id,
                    markdown_content=processing_result["markdown"],
                    plain_text=processing_result["plain_text"],
                    page_count=processing_result.get("page_count", 0),
                    has_images=processing_result.get("has_images", False),
                    has_tables=processing_result.get("has_tables", False),
                    processing_time_ms=processing_result.get("processing_time_ms", 0),
                )
                db.add(content_record)
                db.flush()  # Get the ID before commit
                content_record_id = content_record.id

                # Update document status with metadata
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.conversion_status = "completed"
                    document.updated_at = datetime.utcnow()
                    if not document.metadata:
                        document.metadata = {}
                    document.metadata.update(
                        {
                            "processing_completed_at": datetime.utcnow().isoformat(),
                            "correlation_id": correlation_id,
                            "content_stats": {
                                "markdown_length": len(processing_result["markdown"]),
                                "plain_text_length": len(
                                    processing_result["plain_text"]
                                ),
                                "chunks_count": len(
                                    processing_result.get("chunks", [])
                                ),
                                "has_tables": processing_result.get(
                                    "has_tables", False
                                ),
                                "has_images": processing_result.get(
                                    "has_images", False
                                ),
                            },
                        }
                    )

                db.commit()

        except Exception as e:
            db_error = DatabaseError(
                "Failed to save processing results to database",
                operation="save_content",
                internal_details=str(e),
                correlation_id=correlation_id,
                error_code="PDF008",
            )
            track_error(db_error, "process_pdf_document", "database_save")
            raise db_error

        progress.update(message="Content saved, coordinating downstream processing")

        # Enhanced task coordination with error handling
        downstream_tasks = []

        # Queue embedding generation as separate task with enhanced options
        if processing_result.get("chunks"):
            try:
                embedding_task = generate_embeddings.apply_async(
                    kwargs={
                        "document_id": document_id,
                        "content": processing_result["plain_text"],
                        "chunks": processing_result["chunks"],
                        "correlation_id": correlation_id,
                        "parent_task_id": self.request.id,
                    },
                    priority=5,  # Medium priority for embeddings
                    retry=True,
                )
                downstream_tasks.append(("embeddings", embedding_task.id))

                logger.info(
                    f"Queued embedding generation task {embedding_task.id}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Failed to queue embedding generation: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    },
                )
                # Continue processing - embeddings can be generated later

        # Queue image processing if images were found
        if processing_result.get("has_images") and processing_result.get("images"):
            try:
                image_task = process_document_images.apply_async(
                    kwargs={
                        "document_id": document_id,
                        "images": processing_result["images"],
                        "correlation_id": correlation_id,
                        "parent_task_id": self.request.id,
                    },
                    priority=4,  # Medium-low priority for images
                    retry=True,
                )
                downstream_tasks.append(("image_processing", image_task.id))

                logger.info(
                    f"Queued image processing task {image_task.id}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Failed to queue image processing: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    },
                )
                # Continue - images can be processed later

        progress.complete("PDF processing completed successfully")

        # Enhanced result with comprehensive processing information
        result = {
            "status": "completed",
            "document_id": document_id,
            "correlation_id": correlation_id,
            "processing_stats": {
                "page_count": processing_result.get("page_count", 0),
                "has_images": processing_result.get("has_images", False),
                "has_tables": processing_result.get("has_tables", False),
                "processing_time_ms": processing_result.get("processing_time_ms", 0),
                "markdown_length": len(processing_result.get("markdown", "")),
                "plain_text_length": len(processing_result.get("plain_text", "")),
                "chunks_count": len(processing_result.get("chunks", [])),
                "images_count": len(processing_result.get("images", [])),
            },
            "downstream_tasks": downstream_tasks,
            "content_record_id": content_record_id,
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Log successful completion with metrics
        logger.info(
            f"PDF processing completed successfully for document {document_id}",
            extra={
                "correlation_id": correlation_id,
                "document_id": document_id,
                "processing_stats": result["processing_stats"],
                "downstream_tasks_count": len(downstream_tasks),
            },
        )

        return result

    except ValidationError as e:
        logger.error(
            f"Validation error in PDF processing: {e.get_user_message()}",
            extra={
                "correlation_id": correlation_id,
                "error_code": e.error_code,
                "document_id": document_id,
            },
        )
        self._update_document_error(
            document_id, e.get_user_message(), "validation_error", correlation_id
        )
        track_error(e, "process_pdf_document", "validation")
        raise

    except SecurityError as e:
        logger.error(
            "Security error in PDF processing",
            extra={
                "correlation_id": correlation_id,
                "security_event": True,
                "security_event_type": e.security_event_type,
                "document_id": document_id,
            },
        )
        self._update_document_error(
            document_id, e.get_user_message(), "security_error", correlation_id
        )
        track_error(e, "process_pdf_document", "security")
        raise  # Never retry security errors

    except (ProcessingError, TransientError, DatabaseError, ResourceError) as e:
        logger.error(
            f"Retryable error in PDF processing: {type(e).__name__}",
            extra={
                "correlation_id": correlation_id,
                "error_code": getattr(e, "error_code", None),
                "document_id": document_id,
                "retry_attempt": self.request.retries,
            },
        )

        # Track error for monitoring
        track_error(e, "process_pdf_document", "processing")

        # Update document with error info
        error_type = (
            "processing_error" if isinstance(e, ProcessingError) else "transient_error"
        )
        self._update_document_error(
            document_id, e.get_user_message(), error_type, correlation_id
        )

        # Use intelligent retry strategy based on error type
        if self.request.retries < self.max_retries:
            retry_config = get_retry_strategy(e)
            retry_delay = retry_config.base_delay * (
                retry_config.backoff_multiplier**self.request.retries
            )
            retry_delay = min(retry_delay, retry_config.max_delay)

            logger.info(
                f"Retrying PDF processing in {retry_delay}s (attempt {self.request.retries + 1}/{self.max_retries})",
                extra={"correlation_id": correlation_id},
            )
            raise self.retry(exc=e, countdown=retry_delay)

        # Max retries exceeded
        final_error = ProcessingError(
            f"PDF processing failed after {self.max_retries} retries",
            correlation_id=correlation_id,
            error_code="PDF099",
        )
        track_error(final_error, "process_pdf_document", "max_retries_exceeded")
        raise final_error

    except Exception as e:
        logger.exception(
            "Unexpected error in PDF processing",
            extra={
                "correlation_id": correlation_id,
                "document_id": document_id,
                "error_type": type(e).__name__,
            },
        )

        # Track unexpected error
        unexpected_error = ProcessingError(
            f"Unexpected system error: {type(e).__name__}",
            correlation_id=correlation_id,
            internal_details=str(e),
            error_code="PDF100",
        )
        track_error(unexpected_error, "process_pdf_document", "unexpected_error")

        self._update_document_error(
            document_id,
            unexpected_error.get_user_message(),
            "system_error",
            correlation_id,
        )

        # Retry unexpected errors with conservative strategy
        if self.request.retries < self.max_retries:
            retry_strategy = get_retry_strategy(e)
            countdown = retry_strategy.get_delay(self.request.retries)
            logger.info(
                f"Retrying after unexpected error in {countdown}s",
                extra={"correlation_id": correlation_id},
            )
            raise self.retry(exc=unexpected_error, countdown=countdown)

        raise unexpected_error

    def _update_document_error(
        self, document_id: int, error_message: str, error_type: str, correlation_id: str
    ):
        """Update document with error information and correlation ID."""
        try:
            from ..services.database import get_db_session

            with get_db_session() as db:
                from ..db.models import Document

                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.conversion_status = "failed"
                    document.error_message = error_message  # Already sanitized
                    document.updated_at = datetime.utcnow()
                    if not document.metadata:
                        document.metadata = {}
                    document.metadata.update(
                        {
                            "error_type": error_type,
                            "correlation_id": correlation_id,
                            "failed_at": datetime.utcnow().isoformat(),
                            "retry_count": self.request.retries,
                        }
                    )
                    db.commit()

                    logger.info(
                        f"Updated document {document_id} with error status",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                            "error_type": error_type,
                        },
                    )
        except Exception as db_error:
            logger.error(
                "Failed to update document error status",
                extra={
                    "correlation_id": correlation_id,
                    "document_id": document_id,
                    "db_error": str(db_error),
                },
            )
            # Track database error for monitoring
            db_tracking_error = DatabaseError(
                "Failed to update document status",
                operation="update_document_error",
                correlation_id=correlation_id,
            )
            track_error(db_tracking_error, "process_pdf_document", "database_update")


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_embeddings(
    self,
    document_id: int,
    content: str,
    chunks: list[dict[str, Any]],
    correlation_id: str | None = None,
    parent_task_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate embeddings for document chunks with enhanced error handling and progress tracking.

    Args:
        document_id: Database ID of the document
        content: Full document content
        chunks: List of text chunks with metadata
        correlation_id: Optional correlation ID for tracking
        parent_task_id: Optional parent task ID for coordination

    Returns:
        Dict containing embedding generation results
    """
    # Initialize correlation ID if not provided
    if not correlation_id:
        correlation_id = create_correlation_id()

    progress = ProgressTracker(self, total_steps=len(chunks) + 3)
    progress.correlation_id = correlation_id

    logger.info(
        f"Starting embedding generation for document {document_id}",
        extra={
            "correlation_id": correlation_id,
            "document_id": document_id,
            "parent_task_id": parent_task_id,
            "chunks_count": len(chunks),
        },
    )

    progress.update(message=f"Starting embedding generation for {len(chunks)} chunks")

    try:
        # Enhanced validation
        if not chunks:
            error = ValidationError(
                "No chunks provided for embedding generation",
                correlation_id=correlation_id,
                error_code="EMB001",
            )
            track_error(error, "generate_embeddings", "input_validation")
            raise error

        # Validate chunk structure
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or "text" not in chunk:
                error = ValidationError(
                    f"Invalid chunk structure at index {i}",
                    correlation_id=correlation_id,
                    error_code="EMB002",
                )
                track_error(error, "generate_embeddings", "input_validation")
                raise error

        progress.update(message="Initializing embedding service")

        # Import services with error handling
        try:
            from ..services.database import get_db_session
            from ..services.embeddings import create_embedding_service
        except ImportError as e:
            error = ProcessingError(
                "Failed to import required services",
                internal_details=str(e),
                correlation_id=correlation_id,
                error_code="EMB003",
            )
            track_error(error, "generate_embeddings", "service_import")
            raise error

        # Initialize embedding service with retry
        embedding_service = global_retry_manager.execute_with_retry(
            lambda: create_embedding_service(),
            operation_name="embedding_service_initialization",
            correlation_id=correlation_id,
        )

        if not embedding_service:
            error = ProcessingError(
                "Failed to initialize embedding service",
                correlation_id=correlation_id,
                error_code="EMB004",
            )
            track_error(error, "generate_embeddings", "service_initialization")
            raise error

        # Enhanced batch processing with error isolation
        batch_size = settings.embedding.batch_size
        embeddings_generated = 0
        failed_chunks = []
        embedding_records = []

        progress.update(message="Processing embeddings in batches")

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            try:
                # Generate embeddings for batch with retry
                embeddings = global_retry_manager.execute_with_retry(
                    lambda: embedding_service.generate_embeddings(batch_texts),
                    operation_name=f"embedding_generation_batch_{i // batch_size + 1}",
                    correlation_id=correlation_id,
                )

                if not embeddings or len(embeddings) != len(batch_texts):
                    raise EmbeddingError(
                        f"Embedding generation returned invalid result for batch {i // batch_size + 1}",
                        correlation_id=correlation_id,
                        error_code="EMB005",
                    )

                # Prepare embedding records
                for chunk, embedding in zip(batch_chunks, embeddings, strict=False):
                    embedding_record = {
                        "document_id": document_id,
                        "page_number": chunk.get("page_number"),
                        "chunk_index": chunk.get("index"),
                        "chunk_text": chunk["text"],
                        "embedding": embedding,
                        "metadata": chunk.get("metadata", {}),
                        "correlation_id": correlation_id,
                    }
                    embedding_records.append(embedding_record)

                embeddings_generated += len(embeddings)

                progress.update(
                    current=1 + (i // batch_size + 1),
                    message=f"Processed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} "
                    f"({embeddings_generated}/{len(chunks)} embeddings)",
                )

            except Exception as batch_error:
                logger.error(
                    f"Error processing embedding batch {i // batch_size + 1}: {batch_error}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                        "batch_number": i // batch_size + 1,
                    },
                )

                # Track failed chunks for retry
                for chunk in batch_chunks:
                    failed_chunks.append(
                        {
                            "chunk": chunk,
                            "batch_number": i // batch_size + 1,
                            "error": str(batch_error),
                        }
                    )

                # Continue with next batch rather than failing entire task
                continue

        progress.update(message="Saving embeddings to database")

        # Enhanced database operations with transaction management
        records_saved = 0
        try:
            with get_db_session() as db:
                from ..db.models import DocumentEmbedding

                # Save all successful embedding records in transaction
                for record_data in embedding_records:
                    embedding_record = DocumentEmbedding(**record_data)
                    db.add(embedding_record)

                db.commit()
                records_saved = len(embedding_records)

                logger.info(
                    f"Saved {records_saved} embedding records to database",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    },
                )

        except Exception as db_error:
            logger.error(
                f"Failed to save embeddings to database: {db_error}",
                extra={
                    "correlation_id": correlation_id,
                    "document_id": document_id,
                    "records_count": len(embedding_records),
                },
            )

            db_error_obj = DatabaseError(
                "Failed to save embeddings to database",
                operation="save_embeddings",
                internal_details=str(db_error),
                correlation_id=correlation_id,
                error_code="EMB006",
            )
            track_error(db_error_obj, "generate_embeddings", "database_save")
            raise db_error_obj

        progress.complete(f"Generated {embeddings_generated} embeddings successfully")

        # Enhanced result with comprehensive information
        result = {
            "status": "completed",
            "document_id": document_id,
            "correlation_id": correlation_id,
            "parent_task_id": parent_task_id,
            "embeddings_generated": embeddings_generated,
            "records_saved": records_saved,
            "total_chunks": len(chunks),
            "failed_chunks_count": len(failed_chunks),
            "success_rate": embeddings_generated / len(chunks) if chunks else 0,
            "batch_stats": {
                "batch_size": batch_size,
                "total_batches": (len(chunks) - 1) // batch_size + 1,
                "successful_batches": (len(chunks) - 1) // batch_size
                + 1
                - len([fc for fc in failed_chunks if fc["batch_number"] not in set()]),
            },
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Include failed chunks info if any
        if failed_chunks:
            result["failed_chunks"] = failed_chunks[
                :10
            ]  # Limit to first 10 for response size

        logger.info(
            f"Embedding generation completed for document {document_id}",
            extra={
                "correlation_id": correlation_id,
                "document_id": document_id,
                "embeddings_generated": embeddings_generated,
                "success_rate": result["success_rate"],
            },
        )

        return result

    except EmbeddingError as e:
        logger.error(f"Embedding service error: {e}")
        if self.request.retries < self.max_retries:
            # Exponential backoff for embedding errors
            retry_strategy = get_retry_strategy(e)
            countdown = retry_strategy.get_delay(self.request.retries)
            raise self.retry(exc=e, countdown=countdown)
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in embedding generation: {e}")
        if self.request.retries < self.max_retries:
            retry_strategy = get_retry_strategy(e)
            countdown = retry_strategy.get_delay(self.request.retries)
            raise self.retry(exc=e, countdown=countdown)
        raise EmbeddingError(f"Embedding generation failed: {e}")


@app.task(bind=True, max_retries=3)
def process_document_images(
    self,
    document_id: int,
    images: list[dict[str, Any]],
    correlation_id: str | None = None,
    parent_task_id: str | None = None,
) -> dict[str, Any]:
    """
    Process and store document images with OCR and embeddings.

    Args:
        document_id: Database ID of the document
        images: List of image data with paths and metadata

    Returns:
        Dict containing image processing results
    """
    progress = ProgressTracker(self, total_steps=len(images) + 1)
    progress.update(message=f"Processing {len(images)} images")

    try:
        from ..services.database import get_db_session
        from ..services.embeddings import EmbeddingService

        embedding_service = EmbeddingService()
        images_processed = 0

        with get_db_session() as db:
            from ..db.models import DocumentImage

            for idx, image_data in enumerate(images):
                try:
                    # Generate image embedding (CLIP-style)
                    image_embedding = embedding_service.generate_image_embedding(
                        image_data["path"]
                    )

                    # Store image record
                    image_record = DocumentImage(
                        document_id=document_id,
                        page_number=image_data.get("page_number"),
                        image_index=idx,
                        image_path=image_data["path"],
                        ocr_text=image_data.get("ocr_text"),
                        ocr_confidence=image_data.get("ocr_confidence"),
                        image_embedding=image_embedding,
                        metadata=image_data.get("metadata", {}),
                    )
                    db.add(image_record)
                    images_processed += 1

                    progress.update(
                        current=idx + 1,
                        message=f"Processed image {idx + 1}/{len(images)}",
                    )

                except Exception as img_error:
                    logger.error(f"Error processing image {idx}: {img_error}")
                    continue

            db.commit()

        progress.complete(f"Processed {images_processed} images successfully")

        return {
            "status": "completed",
            "document_id": document_id,
            "images_processed": images_processed,
            "total_images": len(images),
        }

    except Exception as e:
        logger.exception(f"Error processing document images: {e}")
        if self.request.retries < self.max_retries:
            retry_strategy = get_retry_strategy(e)
            countdown = retry_strategy.get_delay(self.request.retries)
            raise self.retry(exc=e, countdown=countdown)
        raise ProcessingError(f"Image processing failed: {e}")


@app.task(bind=True)
def cleanup_temp_files(self) -> dict[str, Any]:
    """
    Clean up temporary files older than specified threshold.

    Returns:
        Dict containing cleanup statistics
    """
    try:
        temp_dir = settings.processing.temp_dir
        cutoff_time = datetime.now() - timedelta(
            hours=24
        )  # Clean files older than 24 hours

        files_removed = 0
        space_freed_mb = 0

        if temp_dir.exists():
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            space_freed_mb += file_size / (1024 * 1024)
                    except Exception as file_error:
                        logger.warning(
                            f"Failed to remove temp file {file_path}: {file_error}"
                        )

        logger.info(
            f"Cleanup completed: removed {files_removed} files, freed {space_freed_mb:.2f}MB"
        )

        return {
            "status": "completed",
            "files_removed": files_removed,
            "space_freed_mb": round(space_freed_mb, 2),
            "temp_dir": str(temp_dir),
        }

    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}


@app.task(bind=True)
def health_check(self) -> dict[str, Any]:
    """
    Perform health check of worker and connected services.

    Returns:
        Dict containing health status information
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": self.request.id,
            "checks": {},
        }

        # Check database connectivity
        try:
            from ..services.database import get_db_session

            with get_db_session() as db:
                db.execute("SELECT 1")
            health_status["checks"]["database"] = "healthy"
        except Exception as db_error:
            health_status["checks"]["database"] = f"unhealthy: {db_error}"
            health_status["status"] = "degraded"

        # Check embedding service
        try:
            from ..services.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            # Try a simple embedding generation
            test_embedding = embedding_service.generate_embeddings(
                ["health check test"]
            )
            health_status["checks"]["embedding_service"] = "healthy"
        except Exception as embed_error:
            health_status["checks"]["embedding_service"] = f"unhealthy: {embed_error}"
            health_status["status"] = "degraded"

        # Check temp directory
        try:
            temp_dir = settings.processing.temp_dir
            if temp_dir.exists() and os.access(temp_dir, os.W_OK):
                health_status["checks"]["temp_directory"] = "healthy"
            else:
                health_status["checks"]["temp_directory"] = "unhealthy: not writable"
                health_status["status"] = "degraded"
        except Exception as temp_error:
            health_status["checks"]["temp_directory"] = f"unhealthy: {temp_error}"
            health_status["status"] = "degraded"

        # Check Redis connectivity (through Celery)
        try:
            from .celery import app as celery_app

            with celery_app.connection() as conn:
                conn.default_channel.client.ping()
            health_status["checks"]["redis_broker"] = "healthy"
        except Exception as redis_error:
            health_status["checks"]["redis_broker"] = f"unhealthy: {redis_error}"
            health_status["status"] = "unhealthy"

        logger.info(f"Health check completed: {health_status['status']}")
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


# Task for handling batch processing
@app.task(bind=True)
def process_pdf_batch(
    self, file_paths: list[str], processing_options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Process multiple PDF files as a batch.

    Args:
        file_paths: List of PDF file paths to process
        processing_options: Processing configuration options

    Returns:
        Dict containing batch processing results
    """
    progress = ProgressTracker(self, total_steps=len(file_paths))
    progress.update(message=f"Starting batch processing of {len(file_paths)} PDFs")

    results = {
        "total_files": len(file_paths),
        "successful": 0,
        "failed": 0,
        "file_results": [],
        "errors": [],
    }

    for idx, file_path in enumerate(file_paths):
        try:
            # Create document record first
            from ..services.database import get_db_session

            with get_db_session() as db:
                from ..db.models import Document

                file_path_obj = Path(file_path)
                file_hash = _calculate_file_hash(file_path_obj)

                # Check if document already exists
                existing_doc = (
                    db.query(Document).filter(Document.file_hash == file_hash).first()
                )
                if existing_doc:
                    progress.update(
                        message=f"Skipping duplicate file: {file_path_obj.name}"
                    )
                    results["file_results"].append(
                        {
                            "file_path": file_path,
                            "status": "skipped",
                            "reason": "duplicate",
                        }
                    )
                    continue

                # Create new document record
                document = Document(
                    source_path=str(file_path_obj),
                    filename=file_path_obj.name,
                    file_hash=file_hash,
                    file_size_bytes=file_path_obj.stat().st_size,
                    conversion_status="pending",
                )
                db.add(document)
                db.commit()
                document_id = document.id

            # Queue individual processing task
            process_pdf_document.delay(document_id, file_path, processing_options)

            results["successful"] += 1
            results["file_results"].append(
                {"file_path": file_path, "status": "queued", "document_id": document_id}
            )

            progress.update(
                current=idx + 1,
                message=f"Queued {file_path_obj.name} ({idx + 1}/{len(file_paths)})",
            )

        except Exception as e:
            logger.error(f"Failed to queue file {file_path}: {e}")
            results["failed"] += 1
            results["errors"].append({"file_path": file_path, "error": str(e)})

    progress.complete(
        f"Batch queuing completed: {results['successful']} queued, {results['failed']} failed"
    )

    return results


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file for deduplication."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def _validate_processing_result(
    processing_result: dict[str, Any], correlation_id: str
) -> None:
    """
    Validate MinerU processing result to ensure it contains expected data.

    Args:
        processing_result: Result dictionary from MinerU processing
        correlation_id: Correlation ID for error tracking

    Raises:
        ValidationError: If the processing result is invalid or incomplete
    """
    required_fields = ["markdown", "plain_text"]
    missing_fields = []

    for field in required_fields:
        if field not in processing_result:
            missing_fields.append(field)
        elif not processing_result[field] or not isinstance(
            processing_result[field], str
        ):
            missing_fields.append(f"{field} (empty or invalid)")

    if missing_fields:
        error = ValidationError(
            f"Processing result missing required fields: {', '.join(missing_fields)}",
            correlation_id=correlation_id,
            error_code="PDF009",
        )
        raise error

    # Validate content is not just whitespace
    if not processing_result["markdown"].strip():
        error = ValidationError(
            "Extracted markdown content is empty",
            correlation_id=correlation_id,
            error_code="PDF010",
        )
        raise error

    if not processing_result["plain_text"].strip():
        error = ValidationError(
            "Extracted plain text content is empty",
            correlation_id=correlation_id,
            error_code="PDF011",
        )
        raise error

    # Validate chunks if provided
    if "chunks" in processing_result:
        chunks = processing_result["chunks"]
        if not isinstance(chunks, list):
            error = ValidationError(
                "Processing result chunks must be a list",
                correlation_id=correlation_id,
                error_code="PDF012",
            )
            raise error

        # Validate chunk structure
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                error = ValidationError(
                    f"Chunk {i} must be a dictionary",
                    correlation_id=correlation_id,
                    error_code="PDF013",
                )
                raise error

            if "text" not in chunk or not chunk["text"].strip():
                error = ValidationError(
                    f"Chunk {i} missing or has empty text content",
                    correlation_id=correlation_id,
                    error_code="PDF014",
                )
                raise error

    # Validate numeric fields if present
    numeric_fields = ["page_count", "processing_time_ms"]
    for field in numeric_fields:
        if field in processing_result:
            try:
                int(processing_result[field])
            except (ValueError, TypeError):
                error = ValidationError(
                    f"Field '{field}' must be a valid integer",
                    correlation_id=correlation_id,
                    error_code="PDF015",
                )
                raise error

    # Validate boolean fields if present
    boolean_fields = ["has_images", "has_tables"]
    for field in boolean_fields:
        if field in processing_result and not isinstance(
            processing_result[field], bool
        ):
            error = ValidationError(
                f"Field '{field}' must be a boolean value",
                correlation_id=correlation_id,
                error_code="PDF016",
            )
            raise error


@app.task(bind=True, max_retries=2)
def cleanup_task_results(self) -> dict[str, Any]:
    """
    Clean up expired task results to prevent Redis memory leak.

    This task addresses the high-priority memory leak issue where
    persistent task results accumulate in Redis over time.

    Returns:
        Dict containing cleanup statistics
    """
    try:
        from ..core.circuit_breaker import get_redis_result_backend_circuit_breaker
        from .celery import app as celery_app

        circuit_breaker = get_redis_result_backend_circuit_breaker()
        results_cleaned = 0
        memory_freed_mb = 0
        errors_encountered = 0

        with circuit_breaker("cleanup_task_results"):
            # Get Redis client for result backend
            result_backend = celery_app.backend
            redis_client = result_backend.client

            # Pattern for task result keys
            result_key_pattern = "celery-task-meta-*"

            # Get all result keys
            result_keys = redis_client.keys(result_key_pattern)

            # Calculate cutoff time (older than 5 minutes)
            cutoff_timestamp = time.time() - 300  # 5 minutes ago

            logger.info(
                f"Found {len(result_keys)} task result keys for cleanup evaluation"
            )

            for key in result_keys:
                try:
                    # Check if key has TTL
                    ttl = redis_client.ttl(key)
                    if ttl == -1:  # No TTL set, force expiration
                        # Get key size before deletion
                        key_size = redis_client.memory_usage(key) or 0
                        redis_client.expire(key, 1)  # Set to expire in 1 second
                        memory_freed_mb += key_size / (1024 * 1024)
                        results_cleaned += 1

                    elif ttl > 300:  # TTL longer than 5 minutes, reduce it
                        redis_client.expire(key, 300)  # Set to 5 minutes max
                        results_cleaned += 1

                except Exception as key_error:
                    logger.warning(f"Error processing result key {key}: {key_error}")
                    errors_encountered += 1
                    continue

            # Also clean up old worker stats and monitoring data
            monitoring_patterns = [
                "celery-*-stats*",
                "celery-worker-*",
                "_kombu.binding.*",
            ]

            for pattern in monitoring_patterns:
                try:
                    monitoring_keys = redis_client.keys(pattern)
                    for key in monitoring_keys:
                        try:
                            # Check age by trying to get timestamp from data
                            ttl = redis_client.ttl(key)
                            if ttl == -1 or ttl > 1800:  # No TTL or > 30 minutes
                                redis_client.expire(key, 300)  # 5 minutes max
                                results_cleaned += 1
                        except Exception:
                            continue
                except Exception as pattern_error:
                    logger.warning(
                        f"Error cleaning monitoring pattern {pattern}: {pattern_error}"
                    )
                    errors_encountered += 1

        cleanup_stats = {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "results_cleaned": results_cleaned,
            "memory_freed_mb": round(memory_freed_mb, 3),
            "errors_encountered": errors_encountered,
            "circuit_breaker_stats": circuit_breaker.get_stats(),
        }

        logger.info(
            f"Task result cleanup completed: {results_cleaned} results processed, "
            f"{memory_freed_mb:.3f}MB freed, {errors_encountered} errors"
        )

        return cleanup_stats

    except Exception as e:
        logger.error(f"Task result cleanup failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)  # 5 minute delay on retry

        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "retries_exhausted": True,
        }


@app.task(bind=True, max_retries=1)
def monitor_redis_connections(self) -> dict[str, Any]:
    """
    Monitor Redis connection pool utilization and circuit breaker status.

    This task provides proactive monitoring to prevent connection pool
    saturation and worker failures under load.

    Returns:
        Dict containing connection monitoring data
    """
    try:
        from ..core.circuit_breaker import redis_circuit_breaker_manager
        from .celery import (
            get_redis_connection_info,
            get_worker_health,
        )

        monitoring_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "alerts": [],
            "metrics": {},
        }

        # Get Redis connection information
        redis_info = get_redis_connection_info()
        monitoring_data["redis_info"] = redis_info

        # Check connection pool utilization
        if "pool_utilization_percent" in redis_info:
            utilization = redis_info["pool_utilization_percent"]
            monitoring_data["metrics"]["pool_utilization"] = utilization

            if utilization > 90:
                alert = {
                    "level": "critical",
                    "message": f"Redis connection pool at {utilization}% utilization",
                    "recommended_action": "Scale workers or increase connection pool size",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                monitoring_data["status"] = "critical"

            elif utilization > 75:
                alert = {
                    "level": "warning",
                    "message": f"Redis connection pool at {utilization}% utilization",
                    "recommended_action": "Monitor closely, consider scaling soon",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                if monitoring_data["status"] == "healthy":
                    monitoring_data["status"] = "warning"

        # Get circuit breaker statistics
        cb_stats = redis_circuit_breaker_manager.get_all_stats()
        monitoring_data["circuit_breakers"] = cb_stats

        # Check for circuit breaker issues
        for cb_name, stats in cb_stats.items():
            if stats["state"] == "open":
                alert = {
                    "level": "critical",
                    "message": f'Circuit breaker "{cb_name}" is OPEN',
                    "recommended_action": "Investigate Redis connectivity issues",
                    "circuit_breaker": cb_name,
                    "failure_count": stats["failure_count"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                monitoring_data["status"] = "critical"

            elif stats["state"] == "half_open":
                alert = {
                    "level": "warning",
                    "message": f'Circuit breaker "{cb_name}" is testing recovery',
                    "recommended_action": "Monitor for successful recovery",
                    "circuit_breaker": cb_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                if monitoring_data["status"] == "healthy":
                    monitoring_data["status"] = "warning"

            # Check success rate
            success_rate = stats.get("success_rate", 1.0)
            if success_rate < 0.95:  # Less than 95% success rate
                alert = {
                    "level": "warning",
                    "message": f'Circuit breaker "{cb_name}" has low success rate: {success_rate:.1%}',
                    "recommended_action": "Investigate Redis performance issues",
                    "circuit_breaker": cb_name,
                    "success_rate": success_rate,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                if monitoring_data["status"] == "healthy":
                    monitoring_data["status"] = "warning"

        # Get worker health information
        worker_health = get_worker_health()
        monitoring_data["worker_health"] = worker_health

        # Check memory usage
        if "used_memory" in redis_info:
            memory_mb = redis_info["used_memory"] / (1024 * 1024)
            monitoring_data["metrics"]["redis_memory_mb"] = round(memory_mb, 2)

            if memory_mb > 1024:  # 1GB
                alert = {
                    "level": "warning",
                    "message": f"Redis memory usage is high: {memory_mb:.1f}MB",
                    "recommended_action": "Consider result cleanup or memory optimization",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                if monitoring_data["status"] == "healthy":
                    monitoring_data["status"] = "warning"

        # Check queue depths
        if "queue_breakdown" in worker_health:
            total_queued = worker_health.get("total_queued_tasks", 0)
            monitoring_data["metrics"]["total_queued_tasks"] = total_queued

            if total_queued > 500:
                alert = {
                    "level": "warning",
                    "message": f"High queue depth: {total_queued} tasks pending",
                    "recommended_action": "Consider scaling workers or investigating bottlenecks",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                monitoring_data["alerts"].append(alert)
                if monitoring_data["status"] == "healthy":
                    monitoring_data["status"] = "warning"

        # Log critical alerts
        for alert in monitoring_data["alerts"]:
            if alert["level"] == "critical":
                logger.error(f"CRITICAL ALERT: {alert['message']}")
            elif alert["level"] == "warning":
                logger.warning(f"WARNING: {alert['message']}")

        logger.info(
            f"Redis connection monitoring completed: {monitoring_data['status']}, "
            f"{len(monitoring_data['alerts'])} alerts"
        )

        return monitoring_data

    except Exception as e:
        logger.error(f"Redis connection monitoring failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=180)  # 3 minute delay on retry

        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "retries_exhausted": True,
        }
