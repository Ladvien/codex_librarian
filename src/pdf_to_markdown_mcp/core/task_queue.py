"""Task queue interface for integrating file watcher with Celery tasks."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.exceptions import QueueError, ValidationError
from ..db.models import Document, ProcessingQueue
from ..db.queries import DocumentQueries, QueueQueries
from ..db.session import get_db_session
from ..worker.tasks import process_pdf_document

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Interface between file watcher and Celery task queue.

    Handles queueing PDF files for background processing by:
    1. Creating/updating document records in database
    2. Adding entries to processing queue
    3. Triggering appropriate Celery tasks
    """

    def __init__(self, db_session_factory=None):
        """Initialize task queue with database session factory.

        Args:
            db_session_factory: Optional custom session factory for dependency injection
        """
        self.get_session = db_session_factory or get_db_session

    def queue_pdf_processing(
        self,
        file_path: str,
        validation_result: dict[str, Any],
        priority: int = 5,
        processing_options: dict[str, Any] | None = None,
        mirror_info: dict[str, Any] | None = None,
    ) -> int:
        """Queue a PDF file for background processing.

        Args:
            file_path: Path to the PDF file to process
            validation_result: File validation metadata from FileValidator
            priority: Processing priority (1=highest, 10=lowest)
            processing_options: Optional processing configuration
            mirror_info: Optional directory mirroring information

        Returns:
            Document ID for tracking processing status

        Raises:
            QueueError: If queueing operation fails
            DatabaseError: If database operations fail
            ValidationError: If validation result is invalid
        """
        try:
            file_path_obj = Path(file_path)

            # Validate input parameters
            if not validation_result.get("valid"):
                raise ValidationError(
                    f"Cannot queue invalid file: {validation_result.get('error', 'Unknown error')}"
                )

            if not file_path_obj.exists():
                raise ValidationError(f"File does not exist: {file_path}")

            with self.get_session() as session:
                # Check if document already exists (by path or hash)
                existing_doc = DocumentQueries.get_by_path(
                    session, str(file_path_obj.absolute())
                )

                if existing_doc:
                    # Check if already processed successfully
                    if existing_doc.conversion_status == "completed":
                        logger.info(f"File already processed successfully: {file_path}")
                        return existing_doc.id

                    # Update existing document
                    document = existing_doc
                    document.updated_at = datetime.utcnow()
                    document.conversion_status = "pending"
                    document.error_message = None
                else:
                    # Create new document record with mirror information
                    document = Document(
                        source_path=str(file_path_obj.absolute()),
                        filename=file_path_obj.name,
                        file_hash=validation_result.get("hash"),
                        file_size_bytes=validation_result.get("size_bytes"),
                        conversion_status="pending",
                        # Directory mirroring fields
                        source_relative_path=str(mirror_info["source_relative_path"])
                        if mirror_info
                        else None,
                        output_path=str(mirror_info["output_path"])
                        if mirror_info
                        else None,
                        output_relative_path=str(mirror_info["output_relative_path"])
                        if mirror_info
                        else None,
                        directory_depth=mirror_info.get("directory_depth")
                        if mirror_info
                        else None,
                        metadata={
                            "mime_type": validation_result.get("mime_type"),
                            "discovered_at": datetime.utcnow().isoformat(),
                            "mirror_info": mirror_info,  # Store full mirror info in metadata
                            "validator_info": {
                                k: v
                                for k, v in validation_result.items()
                                if k != "hash"
                            },
                        },
                    )
                    session.add(document)
                    session.flush()  # Get the ID without committing

                # Check if already in processing queue
                existing_queue_entry = QueueQueries.get_by_file_path(
                    session, str(file_path_obj.absolute())
                )
                if existing_queue_entry and existing_queue_entry.status in [
                    "queued",
                    "processing",
                ]:
                    logger.info(f"Document already queued for processing: {file_path}")
                    return document.id

                # Add to processing queue
                queue_entry = ProcessingQueue(
                    file_path=str(file_path_obj.absolute()),
                    priority=priority,
                    status="queued",
                )
                session.add(queue_entry)

                session.commit()

                # Trigger Celery task
                try:
                    task = process_pdf_document.delay(
                        document_id=document.id,
                        file_path=str(file_path_obj.absolute()),
                        processing_options=processing_options or {},
                    )

                    # Update queue entry with task ID
                    queue_entry.worker_id = task.id
                    session.commit()

                    logger.info(
                        f"Successfully queued PDF for processing: {file_path} (doc_id={document.id}, task_id={task.id})"
                    )
                    return document.id

                except Exception as task_error:
                    # Update document status to reflect task creation failure
                    document.conversion_status = "failed"
                    document.error_message = (
                        f"Failed to create processing task: {task_error!s}"
                    )
                    session.commit()
                    raise QueueError(f"Failed to create Celery task: {task_error!s}")

        except (ValidationError, QueueError):
            # Re-raise validation and queue errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error queueing PDF {file_path}: {e}")
            raise QueueError(f"Failed to queue PDF processing: {e!s}")

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status and statistics.

        Returns:
            Dictionary with queue statistics and status information
        """
        try:
            with self.get_session() as session:
                stats = QueueQueries.get_queue_statistics(session)
                return {
                    "total_queued": stats.get("queued", 0),
                    "total_processing": stats.get("processing", 0),
                    "total_completed": stats.get("completed", 0),
                    "total_failed": stats.get("failed", 0),
                    "queue_depth": stats.get("queued", 0) + stats.get("processing", 0),
                }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "error": str(e),
                "total_queued": 0,
                "total_processing": 0,
                "total_completed": 0,
                "total_failed": 0,
                "queue_depth": 0,
            }

    def clear_completed_entries(self, older_than_days: int = 7) -> int:
        """Clear completed queue entries older than specified days.

        Args:
            older_than_days: Remove completed entries older than this many days

        Returns:
            Number of entries removed
        """
        try:
            with self.get_session() as session:
                removed_count = QueueQueries.clear_old_entries(session, older_than_days)
                session.commit()
                logger.info(f"Cleared {removed_count} old queue entries")
                return removed_count
        except Exception as e:
            logger.error(f"Error clearing completed entries: {e}")
            return 0

    def retry_failed_processing(self, document_id: int) -> bool:
        """Retry processing for a failed document.

        Args:
            document_id: ID of document to retry

        Returns:
            True if retry was successful, False otherwise
        """
        try:
            with self.get_session() as session:
                document = DocumentQueries.get_by_id(session, document_id)
                if not document:
                    logger.warning(f"Document not found for retry: {document_id}")
                    return False

                if document.conversion_status != "failed":
                    logger.warning(
                        f"Cannot retry document with status: {document.conversion_status}"
                    )
                    return False

                # Reset status
                document.conversion_status = "pending"
                document.error_message = None
                document.updated_at = datetime.utcnow()

                # Re-queue for processing
                task = process_pdf_document.delay(
                    document_id=document.id,
                    file_path=document.source_path,
                    processing_options={},
                )

                session.commit()
                logger.info(
                    f"Successfully retried processing for document {document_id} (task_id={task.id})"
                )
                return True

        except Exception as e:
            logger.error(f"Error retrying document processing {document_id}: {e}")
            return False


# Factory function for easy instantiation
def create_task_queue(db_session_factory=None) -> TaskQueue:
    """Create a TaskQueue instance with optional custom database session factory.

    Args:
        db_session_factory: Optional custom session factory

    Returns:
        Configured TaskQueue instance
    """
    return TaskQueue(db_session_factory)
