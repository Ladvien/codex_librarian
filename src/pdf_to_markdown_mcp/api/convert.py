"""
PDF conversion API endpoints.

Implements the convert_single and batch_convert MCP tools.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from pdf_to_markdown_mcp.auth.security import (
    RequireAuth,
    validate_file_security,
    validate_path_security,
)
from pdf_to_markdown_mcp.core.dependencies import get_database_service
from pdf_to_markdown_mcp.core.processor import PDFProcessor
from pdf_to_markdown_mcp.models.dto import CreateDocumentDTO
from pdf_to_markdown_mcp.models.request import BatchConvertRequest, ConvertSingleRequest
from pdf_to_markdown_mcp.models.response import (
    BatchConvertResponse,
    ConvertSingleResponse,
    ErrorResponse,
    ErrorType,
)
from pdf_to_markdown_mcp.services.database import VectorDatabaseService
from pdf_to_markdown_mcp.worker.tasks import process_pdf_batch, process_pdf_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/convert_single", response_model=ConvertSingleResponse)
async def convert_single_pdf(
    request: ConvertSingleRequest,
    background_tasks: BackgroundTasks,
    db_service: VectorDatabaseService = Depends(get_database_service),
    authenticated: bool = RequireAuth,
) -> ConvertSingleResponse:
    """
    Convert a single PDF file to Markdown and store in database.

    This endpoint implements the convert_single MCP tool functionality.
    """
    try:
        logger.info(
            "Single PDF conversion requested",
            extra={"file_path": str(request.file_path)},
        )

        # Validate file path security and integrity
        validated_path = validate_path_security(request.file_path)
        file_validation = validate_file_security(validated_path)

        logger.info(
            "File validation passed",
            extra={
                "validated_path": str(validated_path),
                "file_size": file_validation["size_bytes"],
            },
        )

        # Calculate file hash for deduplication
        file_hash = _calculate_file_hash(validated_path)

        # Check if file already processed (based on hash)
        existing_doc = await db_service.find_document_by_hash(file_hash)

        if existing_doc:
            logger.info(
                f"Document already exists with hash {file_hash}",
                extra={
                    "document_id": existing_doc.id,
                    "status": existing_doc.processing_status,
                },
            )

            # Return existing document info
            return ConvertSingleResponse(
                success=True,
                document_id=existing_doc.id,
                job_id=None,
                message="PDF already processed",
                source_path=request.file_path,
                output_path=None,  # Would need to look up from content table
                file_size_bytes=existing_doc.file_size_bytes,
            )

        # Create database record using service layer
        create_data = CreateDocumentDTO(
            filename=validated_path.name,
            file_path=str(validated_path),
            file_hash=file_hash,
            size_bytes=file_validation["size_bytes"],
        )
        document = await db_service.create_document(create_data)

        logger.info(
            f"Created document record with ID {document.id}",
            extra={"document_id": document.id},
        )

        # Queue for background processing if store_embeddings is True
        if request.store_embeddings:
            # Use Celery for async processing
            job = process_pdf_document.delay(
                document_id=document.id,
                file_path=str(request.file_path),
                processing_options=request.options.dict(),
            )

            return ConvertSingleResponse(
                success=True,
                document_id=document.id,
                job_id=job.id,
                message="PDF queued for processing",
                source_path=request.file_path,
                output_path=(
                    request.output_dir / f"{request.file_path.stem}.md"
                    if request.output_dir
                    else None
                ),
                file_size_bytes=request.file_path.stat().st_size,
            )

        else:
            # Process synchronously without embeddings
            processor = PDFProcessor(db_service)
            result = await processor.process_pdf(
                file_path=request.file_path,
                output_dir=request.output_dir,
                options=request.options,
            )

            return ConvertSingleResponse(
                success=True,
                document_id=result.document_id,
                job_id=None,
                message="PDF converted successfully",
                source_path=request.file_path,
                output_path=result.output_path,
                processing_time_ms=result.processing_time_ms,
                page_count=result.page_count,
                chunk_count=result.chunk_count,
                embedding_count=0,  # No embeddings for sync processing
                file_size_bytes=request.file_path.stat().st_size,
                has_images=result.has_images,
                has_tables=result.has_tables,
            )

    except FileNotFoundError:
        logger.error(f"File not found: {request.file_path}")
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorType.NOT_FOUND,
                message=f"File not found: {request.file_path}",
            ).dict(),
        )

    except PermissionError:
        logger.error(f"Permission denied: {request.file_path}")
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                error=ErrorType.PERMISSION,
                message=f"Permission denied: {request.file_path}",
            ).dict(),
        )

    except Exception as e:
        logger.exception("Unexpected error in single PDF conversion")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.PROCESSING, message=f"Failed to process PDF: {e!s}"
            ).dict(),
        )


@router.post("/batch_convert", response_model=BatchConvertResponse)
async def batch_convert_pdfs(
    request: BatchConvertRequest,
    background_tasks: BackgroundTasks,
    db_service: VectorDatabaseService = Depends(get_database_service),
    authenticated: bool = RequireAuth,
) -> BatchConvertResponse:
    """
    Batch convert multiple PDF files based on pattern matching.

    This endpoint implements the batch_convert MCP tool functionality.
    """
    try:
        logger.info(
            "Batch PDF conversion requested",
            extra={
                "directory": str(request.directory),
                "pattern": request.pattern,
                "max_files": request.max_files,
            },
        )

        # Validate directory path to prevent traversal attacks
        validated_directory = validate_path_security(request.directory)

        logger.info(
            "Directory validation passed",
            extra={"validated_directory": str(validated_directory)},
        )

        # Find matching PDF files
        if request.recursive:
            found_files = list(validated_directory.rglob(request.pattern))
        else:
            found_files = list(validated_directory.glob(request.pattern))

        # Limit to max_files
        found_files = found_files[: request.max_files]

        # Filter to only PDF files
        pdf_files = [
            f for f in found_files if f.suffix.lower() == ".pdf" and f.is_file()
        ]

        # Filter out already processed files based on file hash
        files_to_process = []
        skipped_files = []

        for file_path in pdf_files:
            try:
                file_hash = _calculate_file_hash(file_path)
                existing_doc = await db_service.find_document_by_hash(file_hash)

                if existing_doc:
                    skipped_files.append(
                        {
                            "file": file_path.name,
                            "reason": f"Already processed (document ID: {existing_doc.id})",
                        }
                    )
                else:
                    files_to_process.append(file_path)
            except Exception as e:
                skipped_files.append(
                    {
                        "file": file_path.name,
                        "reason": f"File validation error: {e!s}",
                    }
                )

        # Queue batch processing job
        if files_to_process:
            job = process_pdf_batch.delay(
                file_paths=[str(f) for f in files_to_process],
                processing_options=request.options.dict(),
            )

            return BatchConvertResponse(
                success=True,
                batch_id=job.id,
                message=f"Batch processing initiated for {len(files_to_process)} files",
                files_found=len(found_files),
                files_queued=len(files_to_process),
                files_skipped=len(skipped_files),
                estimated_time_minutes=len(files_to_process)
                * 2,  # Rough estimate: 2 min per file
                queue_position=None,  # Would need queue depth calculation
                queued_files=[f.name for f in files_to_process],
                skipped_files=skipped_files,
            )

        else:
            return BatchConvertResponse(
                success=False,
                batch_id="",
                message="No files to process",
                files_found=len(found_files),
                files_queued=0,
                files_skipped=len(skipped_files),
                queued_files=[],
                skipped_files=skipped_files,
            )

    except Exception as e:
        logger.exception("Unexpected error in batch PDF conversion")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.PROCESSING,
                message=f"Failed to initiate batch processing: {e!s}",
            ).dict(),
        )


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file for deduplication."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@router.get("/stream_progress")
async def stream_progress(
    job_id: str = None, batch_id: str = None, authenticated: bool = RequireAuth
) -> StreamingResponse:
    """
    Stream real-time progress updates using Server-Sent Events.

    This endpoint implements the stream_progress MCP tool functionality.
    """

    async def generate_progress_events():
        """Generate SSE events for progress updates."""
        try:
            from pdf_to_markdown_mcp.worker.celery import app as celery_app

            target_job_id = job_id or batch_id
            if not target_job_id:
                error_data = {
                    "error": "missing_job_id",
                    "message": "Either job_id or batch_id must be provided",
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # Monitor Celery task progress
            task = celery_app.AsyncResult(target_job_id)
            last_state = None

            while not task.ready():
                current_state = task.state
                task_info = task.info or {}

                if current_state != last_state:
                    progress_data = {
                        "job_id": target_job_id,
                        "state": current_state,
                        "progress_percent": task_info.get("percentage", 0),
                        "current_step": task_info.get(
                            "message", f"Task {current_state.lower()}"
                        ),
                        "timestamp": task_info.get("timestamp", None),
                        "eta_seconds": task_info.get("eta_seconds", None),
                    }

                    yield f"data: {json.dumps(progress_data)}\n\n"
                    last_state = current_state

                await asyncio.sleep(1.0)  # Check every second

            # Task completed - send final status
            final_data = {
                "job_id": target_job_id,
                "state": task.state,
                "status": "completed" if task.successful() else "failed",
                "result": task.result if task.successful() else None,
                "error": str(task.result) if task.failed() else None,
                "timestamp": None,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {"error": "stream_error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    # Import settings for secure CORS configuration
    from pdf_to_markdown_mcp.config import settings

    # Set secure headers for SSE endpoint
    secure_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Security-Policy": "default-src 'self'",
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
    }

    # Add CORS headers only if properly configured (not wildcard in production)
    if settings.cors_origins and settings.cors_origins != ["*"]:
        secure_headers["Access-Control-Allow-Origin"] = settings.cors_origins[0]
        secure_headers["Access-Control-Allow-Headers"] = (
            "Cache-Control, Authorization, X-Correlation-ID"
        )

    return StreamingResponse(
        generate_progress_events(),
        media_type="text/event-stream",
        headers=secure_headers,
    )
