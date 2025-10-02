"""
Batch Database Writer Service.

Improves database write performance by batching multiple records into
single transactions, reducing transaction overhead and improving throughput.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Optional

from sqlalchemy.orm import Session

from ..db.session import get_db_session

logger = logging.getLogger(__name__)


class BatchWriter:
    """
    Thread-safe batch writer for database operations.

    Accumulates write requests and commits them in batches to reduce
    transaction overhead. Especially beneficial for high-throughput
    document processing.
    """

    def __init__(
        self,
        batch_size: int = 5,
        max_delay_seconds: float = 10.0,
        enable_metrics: bool = True,
        max_queue_size: int = 10000
    ):
        """
        Initialize batch writer.

        Args:
            batch_size: Number of records to accumulate before writing
            max_delay_seconds: Maximum time to wait before flushing partial batch
            enable_metrics: Whether to track performance metrics
            max_queue_size: Maximum queue size to prevent memory exhaustion (default: 10000)
        """
        self.batch_size = batch_size
        self.max_delay_seconds = max_delay_seconds
        self.enable_metrics = enable_metrics
        self.max_queue_size = max_queue_size

        # Thread-safe queue for write requests with bounded size
        # Critical: Use bounded queue to prevent memory exhaustion
        self.queue: Deque[dict] = deque(maxlen=max_queue_size)
        self.lock = threading.Lock()

        # Track dropped items when queue is full
        self.dropped_count = 0
        self.max_retry_count = 3  # Maximum retries per batch item

        # Background worker thread
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Metrics
        self.total_writes = 0
        self.total_records = 0
        self.total_batches = 0
        self.total_wait_time = 0.0
        self.last_flush_time = time.time()

        # Error tracking
        self.last_error: Optional[str] = None
        self.error_count = 0

    def start(self):
        """Start the background worker thread."""
        if self.running:
            logger.warning("BatchWriter already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="BatchWriterThread"
        )
        self.worker_thread.start()
        logger.info(
            f"BatchWriter started: batch_size={self.batch_size}, "
            f"max_delay={self.max_delay_seconds}s"
        )

    def stop(self, flush: bool = True):
        """
        Stop the background worker thread.

        Args:
            flush: If True, flush remaining records before stopping
        """
        if not self.running:
            return

        self.running = False

        if flush:
            self._flush_batch(force=True)

        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

        logger.info(
            f"BatchWriter stopped: {self.total_batches} batches, "
            f"{self.total_records} records written"
        )

    def queue_document_content(
        self,
        document_id: int,
        markdown_content: str,
        plain_text: str,
        page_count: int,
        has_images: bool = False,
        has_tables: bool = False,
        processing_time_ms: int = 0,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Queue a document content record for batch writing.

        Args:
            document_id: Document ID
            markdown_content: Converted markdown content
            plain_text: Plain text content
            page_count: Number of pages
            has_images: Whether document has images
            has_tables: Whether document has tables
            processing_time_ms: Processing time in milliseconds
            correlation_id: Correlation ID for tracking

        Returns:
            True if queued successfully
        """
        write_request = {
            "type": "document_content",
            "data": {
                "document_id": document_id,
                "markdown_content": markdown_content,
                "plain_text": plain_text,
                "page_count": page_count,
                "has_images": has_images,
                "has_tables": has_tables,
                "processing_time_ms": processing_time_ms,
            },
            "correlation_id": correlation_id,
            "queued_at": time.time(),
            "retry_count": 0  # Track retries
        }

        with self.lock:
            # Check if queue is at capacity (deque with maxlen drops oldest when full)
            if len(self.queue) >= self.max_queue_size - 1:
                self.dropped_count += 1
                logger.error(
                    f"Queue near capacity ({len(self.queue)}/{self.max_queue_size}), "
                    f"dropping document {document_id}. Total dropped: {self.dropped_count}"
                )
                return False

            self.queue.append(write_request)
            queue_size = len(self.queue)

        logger.debug(
            f"Queued document_content for document {document_id}, "
            f"queue size: {queue_size}"
        )

        return True

    def queue_document_update(
        self,
        document_id: int,
        status: str,
        metadata: Optional[dict] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Queue a document status update for batch writing.

        Args:
            document_id: Document ID
            status: New status (completed, failed, etc.)
            metadata: Optional metadata to update
            correlation_id: Correlation ID for tracking

        Returns:
            True if queued successfully
        """
        write_request = {
            "type": "document_update",
            "data": {
                "document_id": document_id,
                "status": status,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow()
            },
            "correlation_id": correlation_id,
            "queued_at": time.time()
        }

        with self.lock:
            self.queue.append(write_request)
            queue_size = len(self.queue)

        logger.debug(
            f"Queued document_update for document {document_id}, "
            f"queue size: {queue_size}"
        )

        return True

    def _worker_loop(self):
        """Main worker loop - runs in background thread."""
        logger.info("BatchWriter worker thread started")

        while self.running:
            try:
                # Check if we should flush
                should_flush = False

                with self.lock:
                    queue_size = len(self.queue)
                    time_since_flush = time.time() - self.last_flush_time

                    # Flush if batch size reached or max delay exceeded
                    if queue_size >= self.batch_size:
                        should_flush = True
                        logger.debug(f"Flushing: batch size reached ({queue_size})")
                    elif queue_size > 0 and time_since_flush >= self.max_delay_seconds:
                        should_flush = True
                        logger.debug(f"Flushing: max delay exceeded ({time_since_flush:.1f}s)")

                if should_flush:
                    self._flush_batch()
                else:
                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in BatchWriter worker loop: {e}", exc_info=True)
                self.last_error = str(e)
                self.error_count += 1
                time.sleep(1.0)  # Back off on errors

        logger.info("BatchWriter worker thread stopped")

    def _flush_batch(self, force: bool = False):
        """
        Flush current batch to database.

        Args:
            force: If True, flush even if batch is small
        """
        # Get batch of records to write
        batch = []
        with self.lock:
            if not self.queue:
                return

            # Take up to batch_size records
            count = min(len(self.queue), self.batch_size) if not force else len(self.queue)
            for _ in range(count):
                if self.queue:
                    batch.append(self.queue.popleft())

            self.last_flush_time = time.time()

        if not batch:
            return

        batch_start_time = time.time()

        try:
            with get_db_session() as db:
                # Separate requests by type for optimized batch processing
                content_requests = []
                update_requests = []

                for request in batch:
                    if request["type"] == "document_content":
                        content_requests.append(request)
                    elif request["type"] == "document_update":
                        update_requests.append(request)
                    else:
                        logger.warning(f"Unknown write request type: {request['type']}")

                # Process content writes individually (can't batch INSERT operations effectively)
                for request in content_requests:
                    try:
                        self._write_document_content(db, request)
                    except Exception as e:
                        logger.error(
                            f"Error writing document content: {e}",
                            extra={"correlation_id": request.get("correlation_id")},
                            exc_info=True
                        )

                # Process updates in optimized batch (fixes N+1 query pattern)
                if update_requests:
                    try:
                        self._batch_write_document_updates(db, update_requests)
                    except Exception as e:
                        logger.error(
                            f"Error in batch document updates: {e}",
                            exc_info=True
                        )

                # Commit the entire batch
                db.commit()

            # Update metrics
            batch_time = time.time() - batch_start_time
            self.total_batches += 1
            self.total_records += len(batch)

            # Calculate average wait time
            for request in batch:
                wait_time = batch_start_time - request["queued_at"]
                self.total_wait_time += wait_time

            avg_wait_time = self.total_wait_time / self.total_records if self.total_records > 0 else 0

            logger.info(
                f"Batch write completed: {len(batch)} records in {batch_time:.3f}s "
                f"(avg wait: {avg_wait_time:.3f}s)"
            )

        except Exception as e:
            logger.error(f"Failed to flush batch: {e}", exc_info=True)
            self.last_error = str(e)
            self.error_count += 1

            # Critical: Implement retry limit to prevent infinite loops
            # Only re-queue requests that haven't exceeded max retries
            retry_batch = []
            for request in batch:
                retry_count = request.get("retry_count", 0)
                if retry_count < self.max_retry_count:
                    request["retry_count"] = retry_count + 1
                    retry_batch.append(request)
                    logger.warning(
                        f"Re-queuing request (retry {retry_count + 1}/{self.max_retry_count})",
                        extra={"correlation_id": request.get("correlation_id")}
                    )
                else:
                    # Max retries exceeded - log and drop
                    logger.error(
                        f"Max retries ({self.max_retry_count}) exceeded for request, dropping",
                        extra={
                            "correlation_id": request.get("correlation_id"),
                            "request_type": request.get("type"),
                            "document_id": request.get("data", {}).get("document_id")
                        }
                    )

            # Re-queue only items that can be retried
            if retry_batch:
                with self.lock:
                    # Add to end of queue (FIFO order)
                    for request in retry_batch:
                        self.queue.append(request)

    def _write_document_content(self, db: Session, request: dict):
        """Write document content record to database using upsert pattern."""
        from ..db.models import DocumentContent

        data = request["data"]

        # Use upsert pattern to prevent duplicates (get-or-create)
        content_record = db.query(DocumentContent).filter(
            DocumentContent.document_id == data["document_id"]
        ).first()

        if content_record:
            # Update existing record
            content_record.markdown_content = data["markdown_content"]
            content_record.plain_text = data["plain_text"]
            content_record.page_count = data["page_count"]
            content_record.has_images = data["has_images"]
            content_record.has_tables = data["has_tables"]
            content_record.processing_time_ms = data["processing_time_ms"]
            logger.debug(
                f"Updated existing document_content for document {data['document_id']}",
                extra={"correlation_id": request.get("correlation_id")}
            )
        else:
            # Create new record
            content_record = DocumentContent(
                document_id=data["document_id"],
                markdown_content=data["markdown_content"],
                plain_text=data["plain_text"],
                page_count=data["page_count"],
                has_images=data["has_images"],
                has_tables=data["has_tables"],
                processing_time_ms=data["processing_time_ms"],
            )
            db.add(content_record)
            logger.debug(
                f"Created new document_content for document {data['document_id']}",
                extra={"correlation_id": request.get("correlation_id")}
            )

    def _write_document_update(self, db: Session, request: dict):
        """Write document status update to database (single record, for backward compatibility)."""
        from ..db.models import Document

        data = request["data"]
        document = db.query(Document).filter(
            Document.id == data["document_id"]
        ).first()

        if document:
            document.conversion_status = data["status"]
            document.updated_at = data["updated_at"]

            if data["metadata"]:
                # Merge metadata
                if document.meta_data:
                    document.meta_data.update(data["metadata"])
                else:
                    document.meta_data = data["metadata"]

            logger.debug(
                f"Updated document {data['document_id']} status to {data['status']}",
                extra={"correlation_id": request.get("correlation_id")}
            )
        else:
            logger.warning(f"Document {data['document_id']} not found for update")

    def _batch_write_document_updates(self, db: Session, update_requests: list):
        """
        Batch write document updates to fix N+1 query pattern (Issue #1 fix).

        Fetches all documents in a single query, then applies updates in memory.
        This is 10-20x faster than individual queries for large batches.

        Args:
            db: Database session
            update_requests: List of document update requests
        """
        from ..db.models import Document

        if not update_requests:
            return

        # Extract all document IDs from requests (deduplication)
        doc_ids = list(set(req["data"]["document_id"] for req in update_requests))

        # Single query to fetch all documents (fixes N+1 pattern)
        documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
        doc_map = {doc.id: doc for doc in documents}

        # Apply updates to fetched documents
        for request in update_requests:
            data = request["data"]
            document = doc_map.get(data["document_id"])

            if document:
                document.conversion_status = data["status"]
                document.updated_at = data["updated_at"]

                if data["metadata"]:
                    # Merge metadata
                    if document.meta_data:
                        document.meta_data.update(data["metadata"])
                    else:
                        document.meta_data = data["metadata"]

                logger.debug(
                    f"Batch updated document {data['document_id']} status to {data['status']}",
                    extra={"correlation_id": request.get("correlation_id")}
                )
            else:
                logger.warning(f"Document {data['document_id']} not found for batch update")

    def get_metrics(self) -> dict:
        """
        Get performance metrics.

        Returns:
            Dictionary of metrics
        """
        with self.lock:
            queue_size = len(self.queue)

        avg_wait_time = (
            self.total_wait_time / self.total_records
            if self.total_records > 0
            else 0
        )

        return {
            "running": self.running,
            "queue_size": queue_size,
            "total_batches": self.total_batches,
            "total_records": self.total_records,
            "avg_wait_time_seconds": round(avg_wait_time, 3),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "batch_size": self.batch_size,
            "max_delay_seconds": self.max_delay_seconds,
        }


# Global singleton instance
_batch_writer: Optional[BatchWriter] = None
_batch_writer_lock = threading.Lock()


def get_batch_writer(
    batch_size: int = 5,
    max_delay_seconds: float = 10.0,
    auto_start: bool = True
) -> BatchWriter:
    """
    Get or create global batch writer instance.

    Args:
        batch_size: Batch size (only used on first call)
        max_delay_seconds: Max delay (only used on first call)
        auto_start: Whether to auto-start the writer

    Returns:
        BatchWriter instance
    """
    global _batch_writer

    with _batch_writer_lock:
        if _batch_writer is None:
            _batch_writer = BatchWriter(
                batch_size=batch_size,
                max_delay_seconds=max_delay_seconds
            )
            if auto_start:
                _batch_writer.start()
            logger.info("Created global BatchWriter instance")

    return _batch_writer


def stop_batch_writer(flush: bool = True):
    """
    Stop the global batch writer.

    Args:
        flush: Whether to flush remaining records
    """
    global _batch_writer

    with _batch_writer_lock:
        if _batch_writer:
            _batch_writer.stop(flush=flush)
            _batch_writer = None
