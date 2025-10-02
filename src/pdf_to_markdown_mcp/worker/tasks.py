"""
Celery task definitions for PDF processing pipeline.

This module defines all background tasks for PDF processing, embedding generation,
and system maintenance with proper error handling and progress tracking.
"""

import asyncio
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


def _run_async_in_sync_context(coro):
    """
    Run async coroutine in sync context safely.

    Critical: Avoids event loop conflicts in Celery workers by creating
    a new event loop for each call. This prevents the "Cannot run event loop
    while another is running" error.

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop: {e}")


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
            self.current_step = int(current)

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

                        # Restore progress state (ensure int type)
                        self.current_step = int(saved_progress.get("current", 0))
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


def _create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Chunk text for embedding generation.

    Creates overlapping text chunks suitable for vector embedding generation.
    Uses character-based chunking with configurable size and overlap.

    Args:
        text: Full document text to chunk
        chunk_size: Maximum characters per chunk (default: 1000)
        overlap: Character overlap between chunks (default: 200)

    Returns:
        List of ChunkData objects with metadata
    """
    from ..models.processing import ChunkData

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:  # Only add non-empty chunks
            chunks.append(ChunkData(
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=start,
                end_char=end,
                page=1  # Default page; will be improved with page tracking in future
            ))
            chunk_index += 1

        start += (chunk_size - overlap)

    return chunks


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

        # Convert to Path object
        file_path = Path(file_path)

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
        from ..db.session import get_db_session
        from ..services.embeddings import EmbeddingService, EmbeddingConfig, EmbeddingProvider

        # Use MinerU client to communicate with standalone GPU service
        progress.update(message="Initializing processing services")
        from ..services.mineru_client import get_mineru_client
        mineru_client = get_mineru_client()

        # Create embedding config from settings
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA if settings.embedding.provider == "ollama" else EmbeddingProvider.OPENAI,
            ollama_model=settings.embedding.model,
            openai_model=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            embedding_dimensions=settings.embedding.dimensions,
            ollama_base_url=settings.embedding.ollama_url,
            openai_api_key=settings.embedding.openai_api_key,
        )
        embedding_service = EmbeddingService(embedding_config)

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

        # Enhanced PDF processing with standalone MinerU GPU service
        progress.update(message="Extracting content with MinerU GPU service")

        try:
            # GPU memory check before processing to prevent OOM errors
            from ..utils.gpu_utils import has_sufficient_gpu_memory

            # Get GPU memory requirement from settings (default: 7.0 GB)
            min_gpu_memory_gb = float(os.getenv("MINERU_MIN_GPU_MEMORY_GB", "7.0"))

            if not has_sufficient_gpu_memory(required_gb=min_gpu_memory_gb):
                # Insufficient GPU memory - retry with delay to wait for memory to free up
                retry_delay = int(os.getenv("GPU_MEMORY_RETRY_DELAY", "15"))

                logger.warning(
                    f"Insufficient GPU memory for PDF processing, retrying in {retry_delay}s",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                        "required_memory_gb": min_gpu_memory_gb,
                        "retry_delay": retry_delay
                    }
                )

                # Raise Retry to requeue the task
                raise self.retry(
                    exc=ResourceError(
                        f"Insufficient GPU memory (need {min_gpu_memory_gb} GB)",
                        resource_type="gpu_memory",
                        correlation_id=correlation_id,
                        error_code="PDF_GPU_MEMORY"
                    ),
                    countdown=retry_delay
                )

            # Process PDF with standalone MinerU service via client
            # The service handles all GPU resource management internally
            result = mineru_client.process_pdf_sync(
                pdf_path=str(file_path),
                extract_formulas=processing_options.get("extract_formulas", True) if processing_options else True,
                extract_tables=processing_options.get("extract_tables", True) if processing_options else True,
                language=processing_options.get("ocr_language", "en") if processing_options else "en"
            )

            # Check if processing was successful
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error in MinerU service")
                raise ProcessingError(
                    f"MinerU processing failed: {error_msg}",
                    correlation_id=correlation_id,
                    error_code="PDF_MINERU"
                )

            # Convert result to ProcessingResult format
            from ..models.processing import ProcessingResult, ProcessingMetadata

            processing_result = ProcessingResult(
                markdown_content=result.get("markdown", ""),
                plain_text=result.get("markdown", ""),  # Use markdown as plain text for now
                processing_metadata=ProcessingMetadata(
                    file_hash="",  # Will be computed separately if needed
                    file_size_bytes=Path(file_path).stat().st_size,
                    pages=max(1, result.get("metadata", {}).get("pages", 1)),
                    processing_time_ms=result.get("metadata", {}).get("processing_time", 0),
                    mineru_version="standalone"
                )
            )

            # Create text chunks for embedding generation
            if processing_result.plain_text and len(processing_result.plain_text.strip()) > 0:
                try:
                    processing_result.chunk_data = _create_text_chunks(
                        processing_result.plain_text,
                        chunk_size=settings.processing.chunk_size,
                        overlap=settings.processing.chunk_overlap
                    )
                    logger.info(
                        f"Created {len(processing_result.chunk_data)} text chunks for embedding generation",
                        extra={"correlation_id": correlation_id, "document_id": document_id}
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create text chunks: {e}. Embeddings will not be generated.",
                        extra={"correlation_id": correlation_id, "document_id": document_id}
                    )
                    # Continue processing - chunking failure shouldn't fail the entire task

            if not processing_result or not processing_result.markdown_content:
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

            # Log the actual exception for debugging
            logger.error(
                f"Unexpected error during PDF processing: {type(e).__name__}: {str(e)}",
                extra={"correlation_id": correlation_id},
                exc_info=True,
            )

            # Wrap unknown errors in ProcessingError
            processing_error = ProcessingError(
                "PDF content extraction failed",
                internal_details=f"{type(e).__name__}: {str(e)}",
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

        # Enhanced database operations with batch writing for better performance
        content_record_id = None
        output_file_path = None

        try:
            # Check if batch writing is enabled via environment
            use_batch_writes = os.getenv("ENABLE_BATCH_WRITES", "false").lower() == "true"

            if use_batch_writes:
                # Use batch writer for better performance (non-blocking)
                from ..services.batch_writer import get_batch_writer

                batch_writer = get_batch_writer()

                # Queue document content write
                batch_writer.queue_document_content(
                    document_id=document_id,
                    markdown_content=processing_result.markdown_content,
                    plain_text=processing_result.plain_text,
                    page_count=processing_result.processing_metadata.pages,
                    has_images=len(processing_result.extracted_images) > 0,
                    has_tables=len(processing_result.extracted_tables) > 0,
                    processing_time_ms=processing_result.processing_metadata.processing_time_ms,
                    correlation_id=correlation_id
                )

                # Queue document status update
                metadata = {
                    "processing_completed_at": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "content_stats": {
                        "markdown_length": len(processing_result.markdown_content),
                        "plain_text_length": len(processing_result.plain_text),
                        "chunks_count": len(processing_result.chunk_data),
                        "has_tables": len(processing_result.extracted_tables) > 0,
                        "has_images": len(processing_result.extracted_images) > 0,
                    }
                }
                batch_writer.queue_document_update(
                    document_id=document_id,
                    status="completed",
                    metadata=metadata,
                    correlation_id=correlation_id
                )

                logger.info(
                    f"Queued database writes for document {document_id} to batch writer",
                    extra={"correlation_id": correlation_id, "document_id": document_id}
                )

            else:
                # Original immediate write path (for compatibility/fallback)
                with get_db_session() as db:
                    from ..db.models import DocumentContent

                    content_record = DocumentContent(
                        document_id=document_id,
                        markdown_content=processing_result.markdown_content,
                        plain_text=processing_result.plain_text,
                        page_count=processing_result.processing_metadata.pages,
                        has_images=len(processing_result.extracted_images) > 0,
                        has_tables=len(processing_result.extracted_tables) > 0,
                        processing_time_ms=processing_result.processing_metadata.processing_time_ms,
                    )
                    db.add(content_record)
                    db.flush()  # Get the ID before commit
                    content_record_id = content_record.id

                    # Update document status with metadata
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        document.conversion_status = "completed"
                        document.updated_at = datetime.utcnow()

                        # Get existing metadata or create new dict
                        metadata = document.meta_data if document.meta_data else {}
                        if not isinstance(metadata, dict):
                            metadata = {}

                        # Update with new data
                        metadata["processing_completed_at"] = datetime.utcnow().isoformat()
                        metadata["correlation_id"] = correlation_id
                        metadata["content_stats"] = {
                            "markdown_length": len(processing_result.markdown_content),
                            "plain_text_length": len(processing_result.plain_text),
                            "chunks_count": len(processing_result.chunk_data),
                            "has_tables": len(processing_result.extracted_tables) > 0,
                            "has_images": len(processing_result.extracted_images) > 0,
                        }

                        # Assign back to trigger SQLAlchemy change detection
                        document.meta_data = metadata

                    db.commit()

            # Get output path for file writing (works for both batch and immediate modes)
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document and document.output_path:
                    output_file_path = Path(document.output_path)

            # Write markdown file to disk (async if enabled)
            if output_file_path:
                try:
                    # Check if async file I/O is enabled
                    use_async_file_io = os.getenv("ENABLE_ASYNC_FILE_IO", "false").lower() == "true"

                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    if use_async_file_io:
                        # Write asynchronously in thread pool (non-blocking)
                        # Use ThreadPoolExecutor instead of asyncio in sync context
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                output_file_path.write_text,
                                processing_result.markdown_content,
                                "utf-8"
                            )
                            future.result()  # Wait for completion
                    else:
                        # Synchronous write (original behavior)
                        output_file_path.write_text(
                            processing_result.markdown_content, encoding="utf-8"
                        )

                    logger.info(
                        f"Wrote markdown file to {output_file_path}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                            "file_size": len(processing_result.markdown_content),
                            "async": use_async_file_io,
                        },
                    )
                except Exception as file_error:
                    logger.error(
                        f"Failed to write markdown file to {output_file_path}: {file_error}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                        },
                        exc_info=True,
                    )

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
        if processing_result.chunk_data:
            try:
                embedding_task = generate_embeddings.apply_async(
                    kwargs={
                        "document_id": document_id,
                        "content": processing_result.plain_text,
                        "chunks": [chunk.model_dump() for chunk in processing_result.chunk_data],
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
        if processing_result.extracted_images:
            try:
                image_task = process_document_images.apply_async(
                    kwargs={
                        "document_id": document_id,
                        "images": [img.model_dump() for img in processing_result.extracted_images],
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
                "page_count": processing_result.processing_metadata.pages,
                "has_images": len(processing_result.extracted_images) > 0,
                "has_tables": len(processing_result.extracted_tables) > 0,
                "processing_time_ms": processing_result.processing_metadata.processing_time_ms,
                "markdown_length": len(processing_result.markdown_content),
                "plain_text_length": len(processing_result.plain_text),
                "chunks_count": len(processing_result.chunk_data),
                "images_count": len(processing_result.extracted_images),
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
        _update_document_error(
            self, document_id, e.get_user_message(), "validation_error", correlation_id
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
        _update_document_error(
            self, document_id, e.get_user_message(), "security_error", correlation_id
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
        _update_document_error(
            self, document_id, e.get_user_message(), error_type, correlation_id
        )

        # Use intelligent retry strategy based on error type
        if self.request.retries < self.max_retries:
            retry_strategy = get_retry_strategy(e)
            retry_delay = retry_strategy.get_delay(self.request.retries)

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

        _update_document_error(
            self,
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

    # Update DocumentContent status to 'processing'
    try:
        from ..db.session import get_db_session
        from ..db.models import DocumentContent

        with get_db_session() as db:
            content_record = db.query(DocumentContent).filter(
                DocumentContent.document_id == document_id
            ).first()

            if content_record:
                content_record.embedding_status = "processing"
                db.commit()
                logger.info(
                    f"Updated document {document_id} embedding status to 'processing'",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    }
                )
    except Exception as status_error:
        logger.warning(
            f"Failed to update embedding status to 'processing': {status_error}",
            extra={
                "correlation_id": correlation_id,
                "document_id": document_id,
            }
        )
        # Continue with embedding generation even if status update fails

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
            from ..db.session import get_db_session
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

        # Initialize embedding service directly
        # Critical: Use helper to avoid event loop conflicts in Celery
        try:
            embedding_service = _run_async_in_sync_context(create_embedding_service())
            if not embedding_service:
                raise ProcessingError("Service initialization returned None")
        except Exception as e:
            error = ProcessingError(
                f"Failed to initialize embedding service: {e}",
                correlation_id=correlation_id,
                error_code="EMB003",
            )
            track_error(error, "generate_embeddings", "service_init")
            raise error

        # Enhanced batch processing with error isolation
        import time

        batch_size = settings.embedding.batch_size
        embeddings_generated = 0
        failed_chunks = []
        embedding_records = []

        # Performance timing
        embedding_start_time = time.time()
        batch_timings = []

        progress.update(message="Processing embeddings in batches")

        # Process batches synchronously using asyncio.run for each batch
        for i in range(0, len(chunks), batch_size):
            batch_start = time.time()
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            try:
                # Generate embeddings for batch synchronously
                # Critical: Use helper to avoid event loop conflicts in Celery
                embeddings_result = _run_async_in_sync_context(
                    embedding_service.generate_embeddings(batch_texts)
                )

                # Extract embeddings from the result
                embeddings = embeddings_result.embeddings if hasattr(embeddings_result, 'embeddings') else embeddings_result

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
                    }
                    embedding_records.append(embedding_record)

                embeddings_generated += len(embeddings)

                # Track batch timing
                batch_elapsed = time.time() - batch_start
                batch_timings.append(batch_elapsed)

                # Calculate running throughput
                total_elapsed = time.time() - embedding_start_time
                embeddings_per_second = embeddings_generated / total_elapsed if total_elapsed > 0 else 0

                progress.update(
                    message=f"Generated {embeddings_generated} embeddings ({i + len(batch_chunks)}/{len(chunks)} chunks) "
                    f"[{embeddings_per_second:.1f} emb/sec]"
                )

                logger.info(
                    f"Batch {i // batch_size + 1}: {len(embeddings)} embeddings in {batch_elapsed:.2f}s "
                    f"({len(embeddings) / batch_elapsed:.1f} emb/sec) | "
                    f"Running avg: {embeddings_per_second:.1f} emb/sec"
                )

            except Exception as e:
                # Log the error but continue with next batch
                error_msg = f"Error processing embedding batch {i // batch_size + 1}: {e}"
                logger.error(error_msg)
                progress.update(
                    message=f"Batch {i // batch_size + 1} failed, continuing with next batch"
                )

                # Add failed chunks to list
                for chunk in batch_chunks:
                    failed_chunks.append({
                        "chunk": chunk,
                        "batch_number": i // batch_size + 1,
                        "error": str(e),
                    })


        # Calculate final throughput metrics
        total_embedding_time = time.time() - embedding_start_time
        final_embeddings_per_second = embeddings_generated / total_embedding_time if total_embedding_time > 0 else 0
        avg_batch_time = sum(batch_timings) / len(batch_timings) if batch_timings else 0

        logger.info(
            f"Embedding generation complete: {embeddings_generated} embeddings in {total_embedding_time:.2f}s | "
            f"Throughput: {final_embeddings_per_second:.1f} emb/sec | "
            f"Avg batch time: {avg_batch_time:.2f}s | "
            f"Batches: {len(batch_timings)}"
        )

        progress.update(message="Saving embeddings to database")

        # Enhanced database operations with transaction management
        records_saved = 0
        try:
            with get_db_session() as db:
                from ..db.models import DocumentEmbedding

                # Save all successful embedding records in transaction
                # PERFORMANCE: Use bulk_insert_mappings for 10-100x faster inserts
                db_write_start = time.time()
                db.bulk_insert_mappings(DocumentEmbedding, embedding_records)
                db.commit()
                db_write_time = time.time() - db_write_start
                records_saved = len(embedding_records)

                logger.info(
                    f"Bulk inserted {records_saved} embeddings in {db_write_time:.2f}s "
                    f"({records_saved / db_write_time:.1f} records/sec)"
                )

                # Update DocumentContent tracking fields after successful embedding save
                content_record = db.query(DocumentContent).filter(
                    DocumentContent.document_id == document_id
                ).first()

                if content_record:
                    content_record.embedding_status = "completed"
                    content_record.embedding_generated_at = datetime.utcnow()
                    content_record.embedding_count = records_saved
                    content_record.embedding_error = None
                    db.commit()

                    logger.info(
                        f"Updated document {document_id} embedding tracking: status='completed', count={records_saved}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                            "embedding_count": records_saved,
                        }
                    )

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

            # Update DocumentContent status to 'failed' on database error
            try:
                with get_db_session() as error_db:
                    error_content = error_db.query(DocumentContent).filter(
                        DocumentContent.document_id == document_id
                    ).first()
                    if error_content:
                        error_content.embedding_status = "failed"
                        error_content.embedding_error = f"Database error: {str(db_error)}"
                        error_db.commit()
                        logger.info(
                            f"Updated document {document_id} embedding status to 'failed' after database error",
                            extra={
                                "correlation_id": correlation_id,
                                "document_id": document_id,
                            }
                        )
            except Exception as update_error:
                logger.warning(
                    f"Failed to update embedding status after database error: {update_error}",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": document_id,
                    }
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

        # Update DocumentContent status to 'failed' on embedding error
        try:
            with get_db_session() as error_db:
                error_content = error_db.query(DocumentContent).filter(
                    DocumentContent.document_id == document_id
                ).first()
                if error_content:
                    error_content.embedding_status = "failed"
                    error_content.embedding_error = f"Embedding error: {str(e)}"
                    error_db.commit()
                    logger.info(
                        f"Updated document {document_id} embedding status to 'failed' after embedding error",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                        }
                    )
        except Exception as update_error:
            logger.warning(
                f"Failed to update embedding status after embedding error: {update_error}",
                extra={
                    "correlation_id": correlation_id,
                    "document_id": document_id,
                }
            )

        if self.request.retries < self.max_retries:
            # Exponential backoff for embedding errors
            retry_strategy = get_retry_strategy(e)
            countdown = retry_strategy.get_delay(self.request.retries)
            raise self.retry(exc=e, countdown=countdown)
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in embedding generation: {e}")

        # Update DocumentContent status to 'failed' on unexpected error
        try:
            with get_db_session() as error_db:
                error_content = error_db.query(DocumentContent).filter(
                    DocumentContent.document_id == document_id
                ).first()
                if error_content:
                    error_content.embedding_status = "failed"
                    error_content.embedding_error = f"Unexpected error: {str(e)}"
                    error_db.commit()
                    logger.info(
                        f"Updated document {document_id} embedding status to 'failed' after unexpected error",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document_id,
                        }
                    )
        except Exception as update_error:
            logger.warning(
                f"Failed to update embedding status after unexpected error: {update_error}",
                extra={
                    "correlation_id": correlation_id,
                    "document_id": document_id,
                }
            )

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
        from ..db.session import get_db_session
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
            from ..db.session import get_db_session

            with get_db_session() as db:
                db.execute("SELECT 1")
            health_status["checks"]["database"] = "healthy"
        except Exception as db_error:
            health_status["checks"]["database"] = f"unhealthy: {db_error}"
            health_status["status"] = "degraded"

        # Check embedding service
        try:
            from ..services.embeddings import create_embedding_service

            # Critical: Use helper to avoid event loop conflicts in Celery
            embedding_service = _run_async_in_sync_context(create_embedding_service())
            # Try a simple embedding generation
            test_embedding = _run_async_in_sync_context(
                embedding_service.generate_embeddings(["health check test"])
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
            from ..db.session import get_db_session

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


def _update_document_error(
    task_self, document_id: int, error_message: str, error_type: str, correlation_id: str
):
    """Update document with error information and correlation ID.

    Args:
        task_self: The Celery task instance (for accessing request.retries)
        document_id: Database ID of the document
        error_message: Error message to store
        error_type: Type of error (validation_error, security_error, etc.)
        correlation_id: Correlation ID for tracking
    """
    try:
        from ..db.session import get_db_session

        with get_db_session() as db:
            from ..db.models import Document

            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.conversion_status = "failed"
                document.error_message = error_message  # Already sanitized
                document.updated_at = datetime.utcnow()
                if not document.meta_data:
                    document.meta_data = {}
                document.meta_data.update(
                    {
                        "error_type": error_type,
                        "correlation_id": correlation_id,
                        "failed_at": datetime.utcnow().isoformat(),
                        "retry_count": task_self.request.retries,
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


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file for deduplication."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def _validate_processing_result(
    processing_result: "ProcessingResult", correlation_id: str
) -> None:
    """
    Validate MinerU processing result to ensure it contains expected data.

    Args:
        processing_result: ProcessingResult Pydantic model from MinerU processing
        correlation_id: Correlation ID for error tracking

    Raises:
        ValidationError: If the processing result is invalid or incomplete
    """
    from ..models.processing import ProcessingResult

    if not isinstance(processing_result, ProcessingResult):
        error = ValidationError(
            "Processing result must be a ProcessingResult instance",
            correlation_id=correlation_id,
            error_code="PDF009",
        )
        raise error

    if not processing_result.markdown_content or not isinstance(
        processing_result.markdown_content, str
    ):
        error = ValidationError(
            "Processing result missing required field: markdown_content",
            correlation_id=correlation_id,
            error_code="PDF009",
        )
        raise error

    if not processing_result.plain_text or not isinstance(
        processing_result.plain_text, str
    ):
        error = ValidationError(
            "Processing result missing required field: plain_text",
            correlation_id=correlation_id,
            error_code="PDF009",
        )
        raise error

    if not processing_result.markdown_content.strip():
        error = ValidationError(
            "Extracted markdown content is empty",
            correlation_id=correlation_id,
            error_code="PDF010",
        )
        raise error

    if not processing_result.plain_text.strip():
        error = ValidationError(
            "Extracted plain text content is empty",
            correlation_id=correlation_id,
            error_code="PDF011",
        )
        raise error

    if processing_result.chunk_data:
        for i, chunk in enumerate(processing_result.chunk_data):
            if not chunk.text or not chunk.text.strip():
                error = ValidationError(
                    f"Chunk {i} missing or has empty text content",
                    correlation_id=correlation_id,
                    error_code="PDF014",
                )
                raise error

    if processing_result.processing_metadata.pages < 1:
        error = ValidationError(
            "Processing result must have at least 1 page",
            correlation_id=correlation_id,
            error_code="PDF015",
        )
        raise error

    if processing_result.processing_metadata.processing_time_ms < 0:
        error = ValidationError(
            "Processing time must be non-negative",
            correlation_id=correlation_id,
            error_code="PDF015",
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


@app.task(bind=True, max_retries=1)
def monitor_gpu_performance(self) -> dict[str, Any]:
    """
    Monitor GPU performance and resource utilization.

    Returns:
        Dict containing GPU performance metrics
    """
    try:
        from .gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()

        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": self.request.id,
            "gpu_status": gpu_manager.get_gpu_status(),
        }

        # Get system GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                performance_data["torch_info"] = {
                    "cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                }
        except Exception as torch_error:
            performance_data["torch_error"] = str(torch_error)

        # Try to get nvidia-smi info
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_stats = result.stdout.strip().split(", ")
                performance_data["nvidia_smi"] = {
                    "memory_used_mb": int(gpu_stats[0]),
                    "memory_total_mb": int(gpu_stats[1]),
                    "gpu_utilization_percent": int(gpu_stats[2]),
                }
        except Exception as smi_error:
            performance_data["nvidia_smi_error"] = str(smi_error)

        logger.info(f"GPU performance monitoring completed")
        return performance_data

    except Exception as e:
        logger.error(f"GPU performance monitoring failed: {e}")
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@app.task(bind=True, max_retries=0)
def export_markdown_to_disk(
    self, document_ids: list[int] | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """
    Export markdown content from database to disk for specified documents.

    This task writes markdown files from the database to their configured output_path.
    Useful for recovering files or bulk export operations.

    Args:
        document_ids: List of document IDs to export. If None, exports all completed documents.
        overwrite: If True, overwrite existing files. If False, skip existing files.

    Returns:
        Dict containing export statistics and results
    """
    try:
        from ..db.models import Document, DocumentContent
        from ..db.session import get_db_session

        logger.info(
            f"Starting markdown export to disk",
            extra={
                "document_ids": document_ids,
                "overwrite": overwrite,
                "task_id": self.request.id,
            },
        )

        results = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "files_written": 0,
            "files_skipped": 0,
            "errors": 0,
            "details": [],
        }

        with get_db_session() as db:
            # Build query
            query = (
                db.query(Document, DocumentContent)
                .join(DocumentContent, Document.id == DocumentContent.document_id)
                .filter(Document.conversion_status == "completed")
                .filter(Document.output_path.isnot(None))
            )

            if document_ids:
                query = query.filter(Document.id.in_(document_ids))

            # Process each document
            for document, content in query.all():
                try:
                    output_file_path = Path(document.output_path)

                    # Check if file exists and overwrite setting
                    if output_file_path.exists() and not overwrite:
                        results["files_skipped"] += 1
                        results["details"].append({
                            "document_id": document.id,
                            "output_path": str(output_file_path),
                            "status": "skipped",
                            "reason": "file exists and overwrite=False",
                        })
                        continue

                    # Create directory if needed
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write markdown content
                    output_file_path.write_text(content.markdown_content, encoding="utf-8")

                    results["files_written"] += 1
                    results["details"].append({
                        "document_id": document.id,
                        "output_path": str(output_file_path),
                        "status": "written",
                        "size_bytes": len(content.markdown_content),
                    })

                    logger.info(
                        f"Exported markdown file: {output_file_path}",
                        extra={
                            "document_id": document.id,
                            "size": len(content.markdown_content),
                        },
                    )

                except Exception as file_error:
                    results["errors"] += 1
                    results["details"].append({
                        "document_id": document.id,
                        "output_path": str(document.output_path) if document.output_path else None,
                        "status": "error",
                        "error": str(file_error),
                    })
                    logger.error(
                        f"Failed to export markdown for document {document.id}: {file_error}",
                        exc_info=True,
                    )

        logger.info(
            f"Markdown export completed: {results['files_written']} written, "
            f"{results['files_skipped']} skipped, {results['errors']} errors"
        )

        return results

    except Exception as e:
        logger.error(f"Markdown export task failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@app.task(bind=True, max_retries=0)
def monitor_gpu_performance(self) -> dict[str, Any]:
    """
    Monitor GPU performance and memory usage for MinerU processing.

    This task tracks GPU utilization, memory usage, and performance metrics
    to help optimize GPU resource allocation for PDF processing.

    Returns:
        Dictionary with GPU performance metrics
    """
    try:
        logger.info("Starting GPU performance monitoring")

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "gpu_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
            "memory_usage": {},
            "utilization": {},
            "temperature": {},
            "power": {},
            "mineru_status": {},
        }

        # Check PyTorch CUDA availability
        try:
            import torch
            metrics["gpu_available"] = torch.cuda.is_available()
            metrics["device_count"] = torch.cuda.device_count()
            metrics["cuda_version"] = torch.version.cuda

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "device_id": i,
                        "name": device_props.name,
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                        "multi_processor_count": device_props.multi_processor_count,
                    }

                    # Get current memory usage
                    if i == torch.cuda.current_device():
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        device_info.update({
                            "memory_allocated_gb": round(allocated, 2),
                            "memory_reserved_gb": round(reserved, 2),
                            "memory_free_gb": round(device_props.total_memory / (1024**3) - reserved, 2),
                            "utilization_percent": round((reserved / (device_props.total_memory / (1024**3))) * 100, 2),
                        })

                    metrics["devices"].append(device_info)

        except Exception as e:
            logger.warning(f"Failed to get PyTorch GPU info: {e}")
            metrics["pytorch_error"] = str(e)

        # Check nvidia-smi if available
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_id = int(parts[0])
                        metrics["temperature"][f"gpu_{gpu_id}"] = parts[2]
                        metrics["utilization"][f"gpu_{gpu_id}"] = parts[3]
                        metrics["memory_usage"][f"gpu_{gpu_id}"] = {
                            "used": parts[4],
                            "total": parts[5]
                        }
                        metrics["power"][f"gpu_{gpu_id}"] = parts[6]

        except Exception as e:
            logger.warning(f"Failed to get nvidia-smi info: {e}")
            metrics["nvidia_smi_error"] = str(e)

        # Check MinerU GPU configuration
        try:
            from ..services.mineru import get_shared_mineru_instance
            import asyncio

            mineru_service = get_shared_mineru_instance()

            if mineru_service:
                mineru_stats = asyncio.run(mineru_service.get_processing_stats())
                metrics["mineru_status"] = {
                    "service_available": True,
                    "device": mineru_stats["gpu_status"]["device"],
                    "cuda_available": mineru_stats["gpu_status"]["cuda_available"],
                    "batch_size": mineru_stats["gpu_status"]["batch_size"],
                    "memory_info": mineru_stats["gpu_status"]["memory_info"],
                    "environment": mineru_stats["gpu_status"]["environment"],
                }
            else:
                metrics["mineru_status"] = {"service_available": False}

        except Exception as e:
            logger.warning(f"Failed to get MinerU status: {e}")
            metrics["mineru_status"] = {
                "service_available": False,
                "error": str(e)
            }

        # Determine overall status
        if not metrics["gpu_available"]:
            metrics["status"] = "no_gpu"
        elif metrics["device_count"] == 0:
            metrics["status"] = "no_devices"
        else:
            # Check for any critical issues
            critical_issues = []

            for device in metrics["devices"]:
                if device.get("utilization_percent", 0) > 95:
                    critical_issues.append(f"GPU {device['device_id']} memory usage > 95%")

            if critical_issues:
                metrics["status"] = "critical"
                metrics["critical_issues"] = critical_issues
            elif any(device.get("utilization_percent", 0) > 80 for device in metrics["devices"]):
                metrics["status"] = "warning"
            else:
                metrics["status"] = "healthy"

        logger.info(
            f"GPU monitoring completed: {metrics['status']} - "
            f"{metrics['device_count']} device(s) available"
        )

        return metrics

    except Exception as e:
        logger.error(f"GPU monitoring failed: {e}", exc_info=True)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error",
            "error": str(e),
            "gpu_available": False,
        }
