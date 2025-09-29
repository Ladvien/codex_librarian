"""
Directory indexer for syncing filesystem state to database.

This module provides tasks for initial directory indexing and periodic
resynchronization to ensure database state matches filesystem state.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from celery import Task

from ..config import settings
from ..core.watcher import FileValidator
from ..db.models import Document
from ..db.session import get_db_session
from .celery import app

logger = logging.getLogger(__name__)


class DirectoryIndexer:
    """
    Indexes PDF files in watch directories and syncs with database.

    Provides functionality for:
    - Initial directory scan and database population
    - Periodic resync to detect filesystem changes
    - Batch processing for efficient database operations
    """

    def __init__(self):
        """Initialize directory indexer."""
        self.validator = FileValidator()

    def scan_directory(
        self, directory: Path, patterns: list[str], recursive: bool = True
    ) -> list[Path]:
        """
        Scan directory for PDF files matching patterns.

        Args:
            directory: Directory to scan
            patterns: File patterns to match (e.g., ['*.pdf'])
            recursive: If True, scan subdirectories

        Returns:
            List of matching file paths
        """
        pdf_files = []

        try:
            if recursive:
                for pattern in patterns:
                    pdf_files.extend(directory.rglob(pattern))
            else:
                for pattern in patterns:
                    pdf_files.extend(directory.glob(pattern))

            # Filter out broken symlinks and inaccessible files
            pdf_files = [f for f in pdf_files if f.is_file() and f.exists()]

            logger.info(
                f"Found {len(pdf_files)} PDF files in {directory} (recursive={recursive})"
            )
            return pdf_files

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return []

    def index_file(
        self,
        file_path: Path,
        output_directory: Path | None = None,
        skip_if_exists: bool = True,
    ) -> dict[str, Any]:
        """
        Index a single PDF file and create/update database record.

        Args:
            file_path: Path to PDF file
            output_directory: Optional output directory for markdown files
            skip_if_exists: If True, skip files already successfully processed

        Returns:
            Dictionary with indexing result
        """
        result = {
            "status": "skipped",
            "file_path": str(file_path),
            "document_id": None,
            "error": None,
        }

        try:
            # Validate PDF file
            validation_result = self.validator.validate_pdf(file_path)

            if not validation_result["valid"]:
                result["status"] = "invalid"
                result["error"] = validation_result.get("error", "Unknown validation error")
                return result

            file_hash = validation_result["hash"]
            file_size = validation_result["size_bytes"]

            with get_db_session() as session:
                # Check if document already exists by hash or path
                existing_doc = (
                    session.query(Document)
                    .filter(
                        (Document.file_hash == file_hash)
                        | (Document.source_path == str(file_path.absolute()))
                    )
                    .first()
                )

                if existing_doc:
                    if skip_if_exists and existing_doc.conversion_status == "completed":
                        result["status"] = "exists_completed"
                        result["document_id"] = existing_doc.id
                        return result

                    # Update existing document
                    existing_doc.source_path = str(file_path.absolute())
                    existing_doc.filename = file_path.name
                    existing_doc.file_hash = file_hash
                    existing_doc.file_size_bytes = file_size
                    existing_doc.updated_at = datetime.utcnow()

                    # Reset status if it was failed
                    if existing_doc.conversion_status == "failed":
                        existing_doc.conversion_status = "pending"
                        existing_doc.error_message = None

                    session.commit()

                    result["status"] = "updated"
                    result["document_id"] = existing_doc.id
                    logger.debug(f"Updated existing document: {file_path.name}")

                else:
                    # Create output path if output directory specified
                    output_path = None
                    if output_directory:
                        output_path = output_directory / f"{file_path.stem}.md"

                    # Create new document record
                    new_doc = Document(
                        source_path=str(file_path.absolute()),
                        filename=file_path.name,
                        file_hash=file_hash,
                        file_size_bytes=file_size,
                        conversion_status="pending",
                        output_path=str(output_path) if output_path else None,
                        metadata={
                            "mime_type": validation_result.get("mime_type"),
                            "indexed_at": datetime.utcnow().isoformat(),
                            "indexer_version": "1.0",
                        },
                    )
                    session.add(new_doc)
                    session.commit()

                    result["status"] = "created"
                    result["document_id"] = new_doc.id
                    logger.debug(f"Created new document: {file_path.name}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Error indexing file {file_path}: {e}", exc_info=True)

        return result

    def index_directory(
        self,
        directory: Path,
        patterns: list[str],
        output_directory: Path | None = None,
        recursive: bool = True,
        batch_size: int = 100,
        skip_existing_completed: bool = True,
    ) -> dict[str, Any]:
        """
        Index all PDF files in a directory.

        Args:
            directory: Directory to index
            patterns: File patterns to match
            output_directory: Optional output directory
            recursive: If True, scan subdirectories
            batch_size: Number of files to process in each batch
            skip_existing_completed: Skip already processed files

        Returns:
            Dictionary with indexing statistics
        """
        stats = {
            "total_found": 0,
            "created": 0,
            "updated": 0,
            "exists_completed": 0,
            "invalid": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat(),
        }

        try:
            # Scan directory for PDF files
            pdf_files = self.scan_directory(directory, patterns, recursive)
            stats["total_found"] = len(pdf_files)

            if not pdf_files:
                logger.info(f"No PDF files found in {directory}")
                stats["end_time"] = datetime.utcnow().isoformat()
                return stats

            # Process files in batches
            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(pdf_files) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

                for file_path in batch:
                    result = self.index_file(
                        file_path,
                        output_directory=output_directory,
                        skip_if_exists=skip_existing_completed,
                    )

                    # Update statistics
                    if result["status"] in stats:
                        stats[result["status"]] += 1
                    elif result["status"] == "error":
                        stats["errors"] += 1

            stats["end_time"] = datetime.utcnow().isoformat()

            logger.info(
                f"Directory indexing completed: {stats['created']} created, "
                f"{stats['updated']} updated, {stats['exists_completed']} skipped (completed), "
                f"{stats['invalid']} invalid, {stats['errors']} errors"
            )

        except Exception as e:
            stats["fatal_error"] = str(e)
            logger.error(f"Fatal error during directory indexing: {e}", exc_info=True)

        return stats

    def resync_database(self, custom_watch_dirs: list[str] = None) -> dict[str, Any]:
        """
        Resync database with filesystem state.

        Performs:
        - Adds new files found in watch directories
        - Updates modified files (by hash comparison)
        - Optionally marks deleted files

        Args:
            custom_watch_dirs: Optional custom watch directories to use instead of settings

        Returns:
            Dictionary with resync statistics
        """
        stats = {
            "new_files": 0,
            "updated_files": 0,
            "deleted_files": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat(),
        }

        try:
            # Get watch directories from parameter or settings
            watch_dirs_list = custom_watch_dirs or settings.watcher.watch_directories
            watch_dirs = [Path(d) for d in watch_dirs_list]
            output_dir = Path(settings.watcher.output_directory)

            logger.info(f"Resyncing with {len(watch_dirs)} watch directories")

            # Scan all directories
            all_files = []
            for watch_dir in watch_dirs:
                if not watch_dir.exists():
                    logger.warning(f"Watch directory does not exist: {watch_dir}")
                    continue

                pdf_files = self.scan_directory(
                    watch_dir, settings.watcher.file_patterns, settings.watcher.recursive
                )
                all_files.extend(pdf_files)

            # Index all found files
            for file_path in all_files:
                result = self.index_file(
                    file_path,
                    output_directory=output_dir,
                    skip_if_exists=False,  # Always check, don't skip
                )

                if result["status"] == "created":
                    stats["new_files"] += 1
                elif result["status"] == "updated":
                    stats["updated_files"] += 1
                elif result["status"] == "error":
                    stats["errors"] += 1

            # Handle deleted files if enabled
            if settings.indexer.handle_deleted_files:
                with get_db_session() as session:
                    all_docs = session.query(Document).filter(
                        Document.conversion_status.in_(["pending", "processing", "completed"])
                    ).all()

                    file_paths_set = {str(f.absolute()) for f in all_files}

                    for doc in all_docs:
                        if doc.source_path not in file_paths_set:
                            # File no longer exists
                            if Path(doc.source_path).exists():
                                continue  # File exists but outside watch directories

                            doc.conversion_status = "failed"
                            doc.error_message = "File deleted from filesystem"
                            doc.updated_at = datetime.utcnow()
                            stats["deleted_files"] += 1

                    session.commit()

            stats["end_time"] = datetime.utcnow().isoformat()

            logger.info(
                f"Resync completed: {stats['new_files']} new, "
                f"{stats['updated_files']} updated, {stats['deleted_files']} deleted"
            )

        except Exception as e:
            stats["fatal_error"] = str(e)
            logger.error(f"Fatal error during resync: {e}", exc_info=True)

        return stats

    def index_new_watch_directories(self, new_watch_dirs: list[str]) -> dict[str, Any]:
        """
        Index files in newly added watch directories.

        Args:
            new_watch_dirs: List of new watch directory paths

        Returns:
            Dictionary with indexing statistics
        """
        stats = {
            "directories_processed": 0,
            "total_found": 0,
            "total_created": 0,
            "total_updated": 0,
            "total_errors": 0,
            "start_time": datetime.utcnow().isoformat(),
        }

        try:
            output_dir = Path(settings.watcher.output_directory)

            for watch_dir_str in new_watch_dirs:
                watch_dir = Path(watch_dir_str)
                if not watch_dir.exists():
                    logger.warning(f"New watch directory does not exist: {watch_dir}")
                    continue

                logger.info(f"Indexing new watch directory: {watch_dir}")

                dir_stats = self.index_directory(
                    directory=watch_dir,
                    patterns=settings.watcher.file_patterns,
                    output_directory=output_dir,
                    recursive=settings.watcher.recursive,
                    batch_size=settings.indexer.index_batch_size,
                    skip_existing_completed=True,  # Skip already processed files
                )

                stats["directories_processed"] += 1
                stats["total_found"] += dir_stats.get("total_found", 0)
                stats["total_created"] += dir_stats.get("created", 0)
                stats["total_updated"] += dir_stats.get("updated", 0)
                stats["total_errors"] += dir_stats.get("errors", 0)

            stats["end_time"] = datetime.utcnow().isoformat()

            logger.info(
                f"New directory indexing completed: {stats['total_found']} files found, "
                f"{stats['total_created']} created, {stats['total_updated']} updated"
            )

        except Exception as e:
            stats["fatal_error"] = str(e)
            logger.error(f"Error indexing new watch directories: {e}", exc_info=True)

        return stats


@app.task(bind=True, base=Task, max_retries=1)
def index_watch_directories(self) -> dict[str, Any]:
    """
    Celery task: Index all PDF files in configured watch directories.

    This task scans all watch directories and creates database records
    for discovered PDF files. Typically run once on startup.

    Returns:
        Dictionary with indexing statistics
    """
    logger.info("Starting directory indexing task")

    try:
        indexer = DirectoryIndexer()
        watch_dirs = [Path(d) for d in settings.watcher.watch_directories]
        output_dir = Path(settings.watcher.output_directory)

        all_stats = {
            "directories_indexed": 0,
            "total_found": 0,
            "total_created": 0,
            "total_updated": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "start_time": datetime.utcnow().isoformat(),
        }

        for watch_dir in watch_dirs:
            if not watch_dir.exists():
                logger.warning(f"Watch directory does not exist: {watch_dir}")
                continue

            logger.info(f"Indexing directory: {watch_dir}")

            dir_stats = indexer.index_directory(
                directory=watch_dir,
                patterns=settings.watcher.file_patterns,
                output_directory=output_dir,
                recursive=settings.watcher.recursive,
                batch_size=settings.indexer.index_batch_size,
                skip_existing_completed=settings.indexer.skip_existing_completed,
            )

            all_stats["directories_indexed"] += 1
            all_stats["total_found"] += dir_stats.get("total_found", 0)
            all_stats["total_created"] += dir_stats.get("created", 0)
            all_stats["total_updated"] += dir_stats.get("updated", 0)
            all_stats["total_skipped"] += dir_stats.get("exists_completed", 0)
            all_stats["total_errors"] += dir_stats.get("errors", 0)

        all_stats["end_time"] = datetime.utcnow().isoformat()

        logger.info(
            f"Directory indexing complete: {all_stats['total_found']} files found, "
            f"{all_stats['total_created']} created, {all_stats['total_updated']} updated, "
            f"{all_stats['total_skipped']} skipped"
        )

        return all_stats

    except Exception as e:
        logger.error(f"Directory indexing task failed: {e}", exc_info=True)
        raise


@app.task(bind=True, base=Task, max_retries=1)
def resync_filesystem_to_database(self) -> dict[str, Any]:
    """
    Celery task: Periodic resync of filesystem state to database.

    This task runs periodically (configured via Celery Beat) to ensure
    the database stays in sync with filesystem changes.

    Returns:
        Dictionary with resync statistics
    """
    logger.info("Starting filesystem resync task")

    try:
        indexer = DirectoryIndexer()
        stats = indexer.resync_database()

        logger.info(
            f"Resync complete: {stats.get('new_files', 0)} new, "
            f"{stats.get('updated_files', 0)} updated, "
            f"{stats.get('deleted_files', 0)} deleted"
        )

        return stats

    except Exception as e:
        logger.error(f"Resync task failed: {e}", exc_info=True)
        raise


@app.task(
    bind=True,
    name="pdf_to_markdown_mcp.worker.indexer.queue_pending_documents",
    soft_time_limit=300,
    time_limit=360,
)
def queue_pending_documents(self, batch_size: int | None = None) -> dict[str, Any]:
    """
    Find documents with pending status and queue them for processing.

    This task runs periodically to ensure all indexed documents are
    eventually processed. It queries the database for pending documents
    and submits them to the processing queue.

    Args:
        batch_size: Number of documents to queue at once (default from settings)

    Returns:
        Dictionary with task results including count of queued documents
    """
    from .tasks import process_pdf_document

    if batch_size is None:
        batch_size = settings.indexer.queue_pending_batch_size

    logger.info(f"Starting queue_pending_documents task (batch_size={batch_size})")

    try:
        queued_count = 0
        failed_count = 0
        already_queued = 0

        with get_db_session() as session:
            # Query for pending documents
            pending_docs = (
                session.query(Document)
                .filter(Document.conversion_status == "pending")
                .limit(batch_size)
                .all()
            )

            if not pending_docs:
                logger.info("No pending documents found to queue")
                return {
                    "status": "completed",
                    "queued": 0,
                    "failed": 0,
                    "already_queued": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            logger.info(f"Found {len(pending_docs)} pending documents to queue")

            for doc in pending_docs:
                try:
                    # Queue the document for processing
                    task = process_pdf_document.delay(doc.id, doc.source_path)

                    # Don't update status here - the task will update it to "processing" when it starts
                    # Just track that we've queued it
                    queued_count += 1
                    logger.debug(
                        f"Queued document {doc.id}: {doc.filename} (task_id={task.id})"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to queue document {doc.id} ({doc.filename}): {e}"
                    )
                    failed_count += 1
                    # Don't mark as failed in DB, let it retry next cycle

            # Commit all status updates
            session.commit()

        logger.info(
            f"Queue pending task completed: {queued_count} queued, "
            f"{failed_count} failed, {already_queued} already queued"
        )

        return {
            "status": "completed",
            "queued": queued_count,
            "failed": failed_count,
            "already_queued": already_queued,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Queue pending documents task failed: {e}", exc_info=True)
        raise