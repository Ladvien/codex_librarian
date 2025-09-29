"""
Embedding indexer task for generating embeddings for documents without them.

This module provides a Celery task that periodically scans for documents with
content but no embeddings, chunks the text, and generates vector embeddings.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from celery import Task
from sqlalchemy import or_, and_, case, func

from ..config import settings
from ..core.chunker import TextChunker, ChunkBoundary
from ..core.errors import DatabaseError, EmbeddingError, create_correlation_id
from ..db.models import Document, DocumentContent, DocumentEmbedding
from ..db.session import get_db_session
from ..core.chunker import TextChunk
from .celery import app
from .tasks import generate_embeddings

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def index_document_embeddings(
    self, batch_size: int = 5, force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Find and process documents that have content but no embeddings.

    This task:
    1. Queries for documents with content but missing embeddings
    2. Chunks the markdown content into appropriate sizes
    3. Calls the existing generate_embeddings task
    4. Updates tracking information

    Args:
        batch_size: Maximum number of documents to process per run
        force_reindex: If True, regenerate embeddings even if they exist

    Returns:
        Dictionary with processing statistics
    """
    correlation_id = create_correlation_id()

    logger.info(
        f"Starting embedding indexer task",
        extra={
            "correlation_id": correlation_id,
            "batch_size": batch_size,
            "force_reindex": force_reindex,
        },
    )

    results = {
        "status": "success",
        "correlation_id": correlation_id,
        "documents_processed": 0,
        "embeddings_generated": 0,
        "errors": [],
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        with get_db_session() as db:
            # Query for documents with content that need embeddings
            # Using tracking fields instead of LEFT JOIN for efficiency
            query = (
                db.query(DocumentContent, Document)
                .join(Document, DocumentContent.document_id == Document.id)
                .filter(Document.conversion_status == "completed")
            )

            if not force_reindex:
                # Get documents with pending or failed embeddings
                # Failed embeddings older than 1 hour can be retried
                from datetime import timedelta

                one_hour_ago = datetime.utcnow() - timedelta(hours=1)

                query = query.filter(
                    or_(
                        DocumentContent.embedding_status == "pending",
                        and_(
                            DocumentContent.embedding_status == "failed",
                            or_(
                                DocumentContent.embedding_generated_at.is_(None),
                                DocumentContent.embedding_generated_at < one_hour_ago
                            )
                        )
                    )
                )
            else:
                # Force reindex all documents except those currently processing
                query = query.filter(DocumentContent.embedding_status != "processing")

            # Order by priority: pending first, then failed
            query = query.order_by(
                case(
                    (DocumentContent.embedding_status == "pending", 0),
                    (DocumentContent.embedding_status == "failed", 1),
                    else_=2
                )
            )
            query = query.limit(batch_size)

            documents_to_process = query.all()

            if not documents_to_process:
                logger.info(
                    "No documents found requiring embeddings",
                    extra={"correlation_id": correlation_id},
                )
                results["message"] = "No documents need embedding generation"
                return results

            logger.info(
                f"Found {len(documents_to_process)} documents to process",
                extra={
                    "correlation_id": correlation_id,
                    "document_count": len(documents_to_process),
                },
            )

            # Initialize text chunker with settings
            chunker = TextChunker(
                chunk_size=1000,  # ~250 tokens for most models
                chunk_overlap=200,  # 20% overlap
                boundary_preference=ChunkBoundary.SENTENCE,
            )

            # Process each document
            for content_record, document in documents_to_process:
                try:
                    logger.info(
                        f"Processing document {document.id}: {document.filename}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document.id,
                            "doc_filename": document.filename,  # Renamed to avoid conflict
                        },
                    )

                    # Skip if content is empty
                    if not content_record.markdown_content:
                        logger.warning(
                            f"Document {document.id} has empty content, skipping",
                            extra={
                                "correlation_id": correlation_id,
                                "document_id": document.id,
                            },
                        )
                        continue

                    # Create chunks from markdown content
                    # Using synchronous version since we're in Celery task
                    chunks = create_chunks_sync(
                        chunker,
                        content_record.markdown_content,
                        document_id=document.id,
                        page_count=content_record.page_count or 1,
                    )

                    if not chunks:
                        logger.warning(
                            f"No chunks created for document {document.id}",
                            extra={
                                "correlation_id": correlation_id,
                                "document_id": document.id,
                            },
                        )
                        continue

                    logger.info(
                        f"Created {len(chunks)} chunks for document {document.id}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document.id,
                            "chunk_count": len(chunks),
                        },
                    )

                    # Convert chunks to format expected by generate_embeddings task
                    chunk_dicts = []
                    for idx, chunk in enumerate(chunks):
                        chunk_dict = {
                            "text": chunk.text,
                            "index": idx,
                            "page_number": chunk.page_number,
                            "metadata": {
                                "start_index": chunk.start_index,
                                "end_index": chunk.end_index,
                                "chunk_index": idx,
                            },
                        }
                        chunk_dicts.append(chunk_dict)

                    # Update status to 'processing' before queuing to prevent double-processing
                    try:
                        content_record.embedding_status = "processing"
                        db.commit()
                        logger.info(
                            f"Set document {document.id} embedding status to 'processing' before queuing",
                            extra={
                                "correlation_id": correlation_id,
                                "document_id": document.id,
                            },
                        )
                    except Exception as status_error:
                        logger.warning(
                            f"Failed to update status before queuing: {status_error}",
                            extra={
                                "correlation_id": correlation_id,
                                "document_id": document.id,
                            },
                        )
                        # Continue anyway - the generate_embeddings task will set it

                    # Queue embedding generation task
                    embedding_task = generate_embeddings.apply_async(
                        kwargs={
                            "document_id": document.id,
                            "content": content_record.plain_text or content_record.markdown_content,
                            "chunks": chunk_dicts,
                            "correlation_id": correlation_id,
                            "parent_task_id": self.request.id,
                        },
                        priority=4,  # Lower priority than PDF processing
                        retry=True,
                    )

                    logger.info(
                        f"Queued embedding task {embedding_task.id} for document {document.id}",
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document.id,
                            "embedding_task_id": embedding_task.id,
                            "chunks_count": len(chunk_dicts),
                        },
                    )

                    results["documents_processed"] += 1
                    results["embeddings_generated"] += len(chunk_dicts)

                    # Update document_content with tracking info (if we add the columns)
                    # For now, just log the successful queueing

                except Exception as doc_error:
                    error_msg = f"Error processing document {document.id}: {str(doc_error)}"
                    logger.error(
                        error_msg,
                        extra={
                            "correlation_id": correlation_id,
                            "document_id": document.id,
                            "error": str(doc_error),
                        },
                        exc_info=True,
                    )
                    results["errors"].append(
                        {
                            "document_id": document.id,
                            "doc_filename": document.filename,
                            "error": str(doc_error),
                        }
                    )
                    continue

        # Log summary
        logger.info(
            f"Embedding indexer completed: {results['documents_processed']} documents, "
            f"{results['embeddings_generated']} embeddings queued",
            extra={
                "correlation_id": correlation_id,
                "results": results,
            },
        )

        return results

    except Exception as e:
        logger.error(
            f"Embedding indexer task failed: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )

        # Retry if we haven't exceeded max retries
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying embedding indexer (attempt {self.request.retries + 1}/{self.max_retries})",
                extra={"correlation_id": correlation_id},
            )
            raise self.retry(exc=e)

        results["status"] = "failed"
        results["error"] = str(e)
        return results


def create_chunks_sync(
    chunker: TextChunker,
    text: str,
    document_id: int,
    page_count: int = 1,
) -> List[TextChunk]:
    """
    Synchronous wrapper for creating text chunks.

    The TextChunker.create_chunks is async, but we're in a Celery task
    which is synchronous, so we need this wrapper.

    Args:
        chunker: TextChunker instance
        text: Text to chunk
        document_id: Document ID for metadata
        page_count: Total pages in document

    Returns:
        List of TextChunk objects
    """
    import asyncio

    # Simple chunking logic since we can't use async in Celery
    # This is a simplified version that doesn't use the async chunker
    chunks = []
    chunk_size = chunker.chunk_size
    chunk_overlap = chunker.chunk_overlap

    # Calculate chunk positions
    start_positions = []
    pos = 0
    while pos < len(text):
        start_positions.append(pos)
        pos += chunk_size - chunk_overlap

    # Create chunks
    for idx, start_pos in enumerate(start_positions):
        end_pos = min(start_pos + chunk_size, len(text))
        chunk_text = text[start_pos:end_pos]

        # Try to break at sentence boundary if possible
        if end_pos < len(text):
            # Look for sentence end
            last_period = chunk_text.rfind('. ')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size // 2:  # Only break if we have at least half chunk
                chunk_text = chunk_text[:break_point + 1]
                end_pos = start_pos + break_point + 1

        # Skip empty chunks
        if not chunk_text.strip():
            continue

        # Estimate page number based on position in text
        page_number = min(
            page_count,
            max(1, int((start_pos / len(text)) * page_count) + 1)
        )

        chunk = TextChunk(
            text=chunk_text.strip(),
            start_index=start_pos,
            end_index=end_pos,
            chunk_index=idx,
            page_number=page_number,
            metadata={
                "document_id": document_id,
                "total_chunks": len(start_positions),
            }
        )
        chunks.append(chunk)

    return chunks


@app.task(bind=True)
def reset_stuck_processing_status(self, timeout_minutes: int = 30) -> Dict[str, Any]:
    """
    Reset documents stuck in 'processing' status back to 'pending'.

    This handles cases where a worker crashes or a task gets stuck,
    leaving documents in a perpetual 'processing' state.

    Args:
        timeout_minutes: Minutes after which a processing status is considered stuck

    Returns:
        Dictionary with reset statistics
    """
    from datetime import timedelta

    correlation_id = create_correlation_id()
    timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)

    logger.info(
        f"Checking for stuck processing statuses older than {timeout_minutes} minutes",
        extra={
            "correlation_id": correlation_id,
            "timeout_minutes": timeout_minutes,
        },
    )

    try:
        with get_db_session() as db:
            # Find documents stuck in 'processing' status
            stuck_documents = (
                db.query(DocumentContent)
                .filter(DocumentContent.embedding_status == "processing")
                .filter(
                    or_(
                        DocumentContent.embedding_generated_at.is_(None),
                        DocumentContent.embedding_generated_at < timeout_threshold
                    )
                )
                .all()
            )

            reset_count = 0
            reset_document_ids = []

            for content_record in stuck_documents:
                content_record.embedding_status = "pending"
                content_record.embedding_error = f"Reset from stuck processing status after {timeout_minutes} minutes"
                reset_count += 1
                reset_document_ids.append(content_record.document_id)

                logger.info(
                    f"Reset stuck document {content_record.document_id} from 'processing' to 'pending'",
                    extra={
                        "correlation_id": correlation_id,
                        "document_id": content_record.document_id,
                    },
                )

            if reset_count > 0:
                db.commit()
                logger.info(
                    f"Reset {reset_count} stuck documents",
                    extra={
                        "correlation_id": correlation_id,
                        "reset_count": reset_count,
                        "document_ids": reset_document_ids[:10],  # Log first 10 IDs
                    },
                )

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "reset_count": reset_count,
                "timeout_minutes": timeout_minutes,
                "document_ids": reset_document_ids,
            }

    except Exception as e:
        logger.error(
            f"Failed to reset stuck processing statuses: {e}",
            extra={"correlation_id": correlation_id},
            exc_info=True,
        )
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@app.task(bind=True)
def check_embedding_status(self) -> Dict[str, Any]:
    """
    Check the current status of embedding generation across all documents.

    Returns:
        Dictionary with statistics about embedding coverage
    """
    try:
        with get_db_session() as db:
            # Count total documents with content
            total_with_content = (
                db.query(DocumentContent)
                .join(Document, DocumentContent.document_id == Document.id)
                .filter(Document.conversion_status == "completed")
                .count()
            )

            # Count documents by embedding status using tracking fields
            status_counts = (
                db.query(
                    DocumentContent.embedding_status,
                    func.count(DocumentContent.id)
                )
                .join(Document, DocumentContent.document_id == Document.id)
                .filter(Document.conversion_status == "completed")
                .group_by(DocumentContent.embedding_status)
                .all()
            )

            status_breakdown = {status: count for status, count in status_counts}

            # Count total embeddings and sum from content records
            total_embeddings = db.query(DocumentEmbedding).count()

            total_expected_embeddings = (
                db.query(func.sum(DocumentContent.embedding_count))
                .join(Document, DocumentContent.document_id == Document.id)
                .filter(Document.conversion_status == "completed")
                .filter(DocumentContent.embedding_status == "completed")
                .scalar()
            ) or 0

            # Get documents by status for detailed reporting
            pending_docs = (
                db.query(Document.id, Document.filename)
                .join(DocumentContent, Document.id == DocumentContent.document_id)
                .filter(Document.conversion_status == "completed")
                .filter(DocumentContent.embedding_status == "pending")
                .limit(10)
                .all()
            )

            processing_docs = (
                db.query(Document.id, Document.filename, DocumentContent.embedding_generated_at)
                .join(DocumentContent, Document.id == DocumentContent.document_id)
                .filter(Document.conversion_status == "completed")
                .filter(DocumentContent.embedding_status == "processing")
                .limit(10)
                .all()
            )

            failed_docs = (
                db.query(
                    Document.id,
                    Document.filename,
                    DocumentContent.embedding_error,
                    DocumentContent.embedding_generated_at
                )
                .join(DocumentContent, Document.id == DocumentContent.document_id)
                .filter(Document.conversion_status == "completed")
                .filter(DocumentContent.embedding_status == "failed")
                .limit(10)
                .all()
            )

            # Calculate statistics
            completed_count = status_breakdown.get("completed", 0)
            pending_count = status_breakdown.get("pending", 0)
            processing_count = status_breakdown.get("processing", 0)
            failed_count = status_breakdown.get("failed", 0)

            coverage_percent = (
                (completed_count / total_with_content * 100)
                if total_with_content > 0
                else 0
            )

            # Build result dictionary
            result = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": {
                    "total_documents_with_content": total_with_content,
                    "status_breakdown": {
                        "completed": completed_count,
                        "pending": pending_count,
                        "processing": processing_count,
                        "failed": failed_count,
                    },
                    "coverage_percent": round(coverage_percent, 2),
                    "total_embeddings_in_db": total_embeddings,
                    "total_expected_embeddings": int(total_expected_embeddings),
                    "embeddings_consistency": (
                        total_embeddings == total_expected_embeddings
                    ),
                },
                "documents": {
                    "pending": [
                        {"id": doc_id, "filename": filename}
                        for doc_id, filename in pending_docs
                    ],
                    "processing": [
                        {
                            "id": doc_id,
                            "filename": filename,
                            "started_at": started_at.isoformat() if started_at else None,
                        }
                        for doc_id, filename, started_at in processing_docs
                    ],
                    "failed": [
                        {
                            "id": doc_id,
                            "filename": filename,
                            "error": error[:100] if error else None,  # Truncate error message
                            "failed_at": failed_at.isoformat() if failed_at else None,
                        }
                        for doc_id, filename, error, failed_at in failed_docs
                    ],
                },
                "recommendations": [],
            }

            # Add recommendations based on status
            if processing_count > 0:
                result["recommendations"].append(
                    f"{processing_count} documents are currently processing. "
                    "If they've been processing for > 30 minutes, run reset_stuck_processing_status."
                )

            if failed_count > 0:
                result["recommendations"].append(
                    f"{failed_count} documents have failed embedding generation. "
                    "They will be automatically retried after 1 hour."
                )

            if pending_count > 10:
                result["recommendations"].append(
                    f"{pending_count} documents are pending embedding generation. "
                    "Consider increasing batch_size in index_document_embeddings."
                )

            return result

    except Exception as e:
        logger.error(f"Failed to check embedding status: {e}", exc_info=True)
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }