"""
PDF processing core logic.

Orchestrates the entire PDF-to-Markdown conversion pipeline using MinerU.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.core.chunker import TextChunker
from pdf_to_markdown_mcp.core.exceptions import (
    ProcessingError,
    ValidationError,
)
from pdf_to_markdown_mcp.core.streaming import (
    StreamingProgressTracker,
    get_streaming_stats,
    stream_large_file,
)
from pdf_to_markdown_mcp.models.document import (
    Document,
    ProcessingStatus,
)
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.services.database import VectorDatabaseService
from pdf_to_markdown_mcp.services.mineru import MinerUService

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of PDF processing operation."""

    document_id: int
    output_path: Path | None
    processing_time_ms: int
    page_count: int
    chunk_count: int
    has_images: bool
    has_tables: bool
    error_message: str | None = None
    operation_id: str | None = None
    streaming_enabled: bool = False
    memory_usage_mb: float | None = None
    peak_memory_mb: float | None = None


class PDFProcessor:
    """Main PDF processing orchestrator with streaming support."""

    def __init__(self, db_session: Session, enable_streaming: bool = True):
        """Initialize processor with database session."""
        self.db = db_session
        self.mineru_service = MinerUService()
        self.database_service = VectorDatabaseService(db_session)
        self.chunker = TextChunker()
        self.enable_streaming = enable_streaming
        self._active_operations: dict[str, StreamingProgressTracker] = {}

    async def process_pdf(
        self,
        file_path: Path,
        output_dir: Path | None = None,
        options: ProcessingOptions = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        operation_id: str | None = None,
        document_id: int | None = None,
    ) -> ProcessingResult:
        """
        Process a PDF file through the complete pipeline with streaming support.

        Args:
            file_path: Path to PDF file
            output_dir: Optional output directory for markdown files
            options: Processing options and configuration
            progress_callback: Optional callback for progress updates
            operation_id: Unique operation identifier (auto-generated if None)
            document_id: Optional existing document ID (for processing queued docs)

        Returns:
            ProcessingResult with processing statistics

        Raises:
            ValidationError: If file is invalid
            ProcessingError: If processing fails
            ResourceError: If resources are exhausted
        """
        start_time = time.time()

        # Generate operation ID if not provided
        if operation_id is None:
            operation_id = str(uuid.uuid4())

        if options is None:
            options = ProcessingOptions()

        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        use_streaming = (
            self.enable_streaming and file_size_mb > 50
        )  # Use streaming for files > 50MB

        logger.info(
            "Starting PDF processing",
            extra={
                "operation_id": operation_id,
                "file_path": str(file_path),
                "file_size_mb": file_size_mb,
                "streaming_enabled": use_streaming,
                "options": options.dict(),
            },
        )

        # Create progress tracker
        progress_tracker = None
        if use_streaming or progress_callback:
            progress_tracker = StreamingProgressTracker(
                operation_id=operation_id,
                total_size=file_size,
                callback=progress_callback,
            )
            self._active_operations[operation_id] = progress_tracker

        try:
            # Update progress: Starting validation
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0, current_step="Validating PDF file"
                )

            # Validate input file with streaming support
            await self._validate_pdf_file(file_path)

            # Calculate file hash for deduplication
            if use_streaming:
                file_hash = await self._calculate_file_hash_streaming(
                    file_path, progress_tracker
                )
            else:
                file_hash = self._calculate_file_hash(file_path)

            # Get document record (either provided or lookup by hash)
            document = None
            if document_id:
                # Use provided document ID (from task queue)
                document = await self.database_service.get_document_by_id(document_id)
                if not document:
                    raise ValidationError(f"Document with ID {document_id} not found")

                # Check if already processed
                if document.conversion_status == ProcessingStatus.COMPLETED:
                    logger.info(f"PDF already processed: {file_path}")
                    return ProcessingResult(
                        document_id=document.id,
                        output_path=Path(document.output_path)
                        if document.output_path
                        else None,
                        processing_time_ms=0,
                        page_count=0,
                        chunk_count=0,
                        has_images=False,
                        has_tables=False,
                    )
            else:
                # Look up by hash for direct processing (not from queue)
                existing_doc = await self.database_service.get_document_by_hash(
                    file_hash
                )
                if (
                    existing_doc
                    and existing_doc.conversion_status == ProcessingStatus.COMPLETED
                ):
                    logger.info(f"PDF already processed: {file_path}")
                    return ProcessingResult(
                        document_id=existing_doc.id,
                        output_path=Path(existing_doc.output_path)
                        if existing_doc.output_path
                        else None,
                        processing_time_ms=0,
                        page_count=0,
                        chunk_count=0,
                        has_images=False,
                        has_tables=False,
                    )

            # Update progress: Creating document record
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0, current_step="Creating document record"
                )

            # Create or update document record
            if not document:
                document = await self._create_or_update_document(
                    file_path=file_path,
                    file_hash=file_hash,
                    status=ProcessingStatus.PROCESSING,
                )
            else:
                # Update existing document to processing status
                document.conversion_status = ProcessingStatus.PROCESSING
                await self.database_service.save_document(document)

            try:
                # Update progress: Starting PDF processing
                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=0, current_step="Processing PDF with MinerU"
                    )

                # Determine output directory - use mirrored path if available
                actual_output_dir = output_dir
                if document.output_path:
                    # Use the directory from the mirrored output path
                    mirrored_output_path = Path(document.output_path)
                    actual_output_dir = mirrored_output_path.parent
                    logger.info(f"Using mirrored output directory: {actual_output_dir}")

                # Process PDF with MinerU with streaming support
                if use_streaming:
                    mineru_result = await self.mineru_service.process_pdf_streaming(
                        file_path=file_path,
                        output_dir=actual_output_dir,
                        output_filename=mirrored_output_path.name
                        if document.output_path
                        else None,
                        options=options,
                        progress_callback=lambda processed, total, step: (
                            asyncio.create_task(
                                progress_tracker.update_progress(
                                    bytes_processed=0,  # MinerU handles its own progress
                                    current_step=step or "Processing PDF",
                                )
                            )
                            if progress_tracker
                            else None
                        ),
                    )
                else:
                    mineru_result = await self.mineru_service.process_pdf(
                        file_path=file_path,
                        output_dir=actual_output_dir,
                        output_filename=mirrored_output_path.name
                        if document.output_path
                        else None,
                        options=options,
                    )

                # Update progress: Creating chunks
                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=0, current_step="Creating text chunks"
                    )

                # Create chunks for embeddings if requested
                chunks = []
                if options.chunk_for_embeddings:
                    if (
                        use_streaming and len(mineru_result.plain_text) > 100000
                    ):  # > 100KB
                        # Use streaming chunking for large text
                        chunks = await self._create_chunks_streaming(
                            text=mineru_result.plain_text,
                            options=options,
                            document_id=document.id,
                            filename=file_path.name,
                            metadata={
                                "has_images": mineru_result.has_images,
                                "has_tables": mineru_result.has_tables,
                            },
                            progress_tracker=progress_tracker,
                        )
                    else:
                        chunks = await self.chunker.create_chunks(
                            text=mineru_result.plain_text,
                            chunk_size=options.chunk_size,
                            chunk_overlap=options.chunk_overlap,
                            metadata={
                                "document_id": document.id,
                                "filename": file_path.name,
                                "has_images": mineru_result.has_images,
                                "has_tables": mineru_result.has_tables,
                            },
                        )

                # Update progress: Storing content
                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=0, current_step="Storing content in database"
                    )

                # Store content in database
                content_record = await self.database_service.create_document_content(
                    document_id=document.id,
                    markdown_content=mineru_result.markdown_content,
                    plain_text=mineru_result.plain_text,
                    page_count=mineru_result.page_count,
                    has_images=mineru_result.has_images,
                    has_tables=mineru_result.has_tables,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

                # Update progress: Storing chunks
                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=0, current_step="Storing text chunks"
                    )

                # Store chunks in database for embedding generation
                if chunks:
                    if (
                        use_streaming and len(chunks) > 100
                    ):  # Use streaming for many chunks
                        await self._store_chunks_streaming(
                            document_id=document.id,
                            chunks=chunks,
                            progress_tracker=progress_tracker,
                        )
                    else:
                        await self.database_service.store_text_chunks(
                            document_id=document.id, chunks=chunks
                        )

                # Update document status to completed
                await self.database_service.update_document_status(
                    document_id=document.id,
                    status=ProcessingStatus.COMPLETED,
                    error_message=None,
                )

                processing_time_ms = int((time.time() - start_time) * 1000)

                # Final progress update
                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=0,
                        current_step="Processing completed successfully",
                    )
                    await progress_tracker.set_completion(success=True)

                    # Cleanup from active operations
                    self._active_operations.pop(operation_id, None)

                memory_usage = (
                    progress_tracker.metrics.memory_usage_mb
                    if progress_tracker
                    else None
                )
                peak_memory = (
                    progress_tracker.metrics.peak_memory_mb
                    if progress_tracker
                    else None
                )

                logger.info(
                    "PDF processing completed successfully",
                    extra={
                        "operation_id": operation_id,
                        "document_id": document.id,
                        "processing_time_ms": processing_time_ms,
                        "page_count": mineru_result.page_count,
                        "chunk_count": len(chunks),
                        "streaming_enabled": use_streaming,
                        "memory_usage_mb": memory_usage,
                        "peak_memory_mb": peak_memory,
                    },
                )

                # Use mirrored output path if available, otherwise use MinerU result
                final_output_path = (
                    Path(document.output_path)
                    if document.output_path
                    else mineru_result.output_path
                )

                return ProcessingResult(
                    document_id=document.id,
                    output_path=final_output_path,
                    processing_time_ms=processing_time_ms,
                    page_count=mineru_result.page_count,
                    chunk_count=len(chunks),
                    has_images=mineru_result.has_images,
                    has_tables=mineru_result.has_tables,
                    operation_id=operation_id,
                    streaming_enabled=use_streaming,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=peak_memory,
                )

            except Exception as processing_error:
                # Update document status to failed
                await self.database_service.update_document_status(
                    document_id=document.id,
                    status=ProcessingStatus.FAILED,
                    error_message=str(processing_error),
                )

                # Update progress tracker with error
                if progress_tracker:
                    await progress_tracker.set_completion(
                        success=False, error=str(processing_error)
                    )
                    self._active_operations.pop(operation_id, None)

                raise

        except ValidationError:
            raise  # Re-raise validation errors
        except ProcessingError:
            raise  # Re-raise processing errors
        except Exception as e:
            logger.exception("Unexpected error during PDF processing")
            raise ProcessingError(f"Unexpected processing error: {e!s}")

    async def _validate_pdf_file(self, file_path: Path) -> None:
        """
        Validate PDF file for processing.

        Raises:
            ValidationError: If file is invalid
        """
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValidationError(f"File is not a PDF: {file_path}")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size_mb = 500  # TODO: Get from settings

        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
            )

        # Basic PDF header validation
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    raise ValidationError("Invalid PDF file format")
        except OSError as e:
            raise ValidationError(f"Cannot read PDF file: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication."""
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)

            return hash_sha256.hexdigest()

        except OSError as e:
            raise ValidationError(f"Cannot calculate file hash: {e}")

    async def _create_or_update_document(
        self, file_path: Path, file_hash: str, status: ProcessingStatus
    ) -> Document:
        """Create new document record or update existing one."""

        # Check if document already exists
        existing_doc = await self.database_service.get_document_by_hash(file_hash)

        if existing_doc:
            # Update existing document
            existing_doc.conversion_status = status
            existing_doc.source_path = file_path
            existing_doc.filename = file_path.name
            existing_doc.file_size_bytes = file_path.stat().st_size

            await self.database_service.update_document(existing_doc)
            return existing_doc
        else:
            # Create new document
            document = Document(
                source_path=file_path,
                filename=file_path.name,
                file_hash=file_hash,
                file_size_bytes=file_path.stat().st_size,
                conversion_status=status,
                metadata={},
            )

            return await self.database_service.create_document(document)

    async def get_processing_status(self, document_id: int) -> dict[str, Any]:
        """Get processing status for a document."""
        document = await self.database_service.get_document_by_id(document_id)

        if not document:
            raise ValidationError(f"Document not found: {document_id}")

        return {
            "document_id": document.id,
            "status": document.conversion_status,
            "filename": document.filename,
            "file_size_bytes": document.file_size_bytes,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "error_message": document.error_message,
        }

    async def reprocess_failed_document(
        self, document_id: int, options: ProcessingOptions = None
    ) -> ProcessingResult:
        """Retry processing a failed document."""
        document = await self.database_service.get_document_by_id(document_id)

        if not document:
            raise ValidationError(f"Document not found: {document_id}")

        if document.conversion_status != ProcessingStatus.FAILED:
            raise ValidationError(
                f"Document is not in failed status: {document.conversion_status}"
            )

        logger.info(f"Retrying processing for document {document_id}")

        return await self.process_pdf(file_path=document.source_path, options=options)

    async def _calculate_file_hash_streaming(
        self,
        file_path: Path,
        progress_tracker: StreamingProgressTracker | None = None,
    ) -> str:
        """Calculate SHA-256 hash using streaming for large files."""
        hash_sha256 = hashlib.sha256()
        operation_id = f"hash_{uuid.uuid4().hex[:8]}"

        async for chunk in stream_large_file(
            file_path=file_path,
            operation_id=operation_id,
            progress_callback=lambda processed, total, step: (
                asyncio.create_task(
                    progress_tracker.update_progress(
                        bytes_processed=0, current_step="Calculating file hash"
                    )
                )
                if progress_tracker
                else None
            ),
        ):
            hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def _create_chunks_streaming(
        self,
        text: str,
        options: ProcessingOptions,
        document_id: int,
        filename: str,
        metadata: dict[str, Any],
        progress_tracker: StreamingProgressTracker | None = None,
    ) -> list[Any]:
        """Create text chunks using streaming for large text content."""
        logger.info(
            f"Creating chunks with streaming for large text ({len(text)} characters)"
        )

        if progress_tracker:
            await progress_tracker.update_progress(
                bytes_processed=0, current_step="Starting text chunking"
            )

        # Process text in streaming fashion for very large content
        chunks = await self.chunker.create_chunks(
            text=text,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            metadata={"document_id": document_id, "filename": filename, **metadata},
        )

        if progress_tracker:
            await progress_tracker.update_progress(
                bytes_processed=len(text.encode("utf-8")),
                current_step=f"Created {len(chunks)} text chunks",
            )

        return chunks

    async def _store_chunks_streaming(
        self,
        document_id: int,
        chunks: list[Any],
        progress_tracker: StreamingProgressTracker | None = None,
    ) -> None:
        """Store chunks in database using batching for large chunk counts."""
        batch_size = 50  # Process chunks in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        logger.info(f"Storing {len(chunks)} chunks in {total_batches} batches")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            # Store batch
            await self.database_service.store_text_chunks(
                document_id=document_id, chunks=batch_chunks
            )

            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0,
                    current_step=f"Stored batch {batch_idx + 1}/{total_batches} ({len(batch_chunks)} chunks)",
                )

            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.01)

        logger.info(f"Successfully stored all {len(chunks)} chunks")

    def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get status of all active processing operations."""
        return {
            op_id: tracker.metrics.to_dict()
            for op_id, tracker in self._active_operations.items()
        }

    def get_operation_status(self, operation_id: str) -> dict[str, Any] | None:
        """Get status of a specific operation."""
        tracker = self._active_operations.get(operation_id)
        return tracker.metrics.to_dict() if tracker else None

    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an active operation."""
        tracker = self._active_operations.get(operation_id)
        if tracker:
            await tracker.set_completion(
                success=False, error="Operation cancelled by user"
            )
            self._active_operations.pop(operation_id, None)
            logger.info(f"Cancelled operation {operation_id}")
            return True
        return False

    def get_streaming_stats(self) -> dict[str, Any]:
        """Get streaming statistics for this processor."""
        return {
            "processor_stats": {
                "active_operations": len(self._active_operations),
                "streaming_enabled": self.enable_streaming,
                "operations": list(self._active_operations.keys()),
            },
            "global_streaming_stats": get_streaming_stats(),
        }
