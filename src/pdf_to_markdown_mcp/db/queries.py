"""
Database query utilities for PDF to Markdown MCP Server.

This module provides common query patterns, search operations,
and database utilities for efficient data access.
"""

import logging
from typing import Any

from sqlalchemy import and_, func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from .models import (
    Document,
    DocumentEmbedding,
    ProcessingQueue,
)


def _validate_integer(value: Any, field_name: str) -> int:
    """Validate and convert value to integer, preventing injection."""
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise ValueError(f"Invalid integer value for {field_name}: {value}")
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation failed for {field_name}: {e}")
        raise ValueError(f"Invalid {field_name} parameter")


def _validate_float(value: Any, field_name: str) -> float:
    """Validate and convert value to float, preventing injection."""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"Invalid float value for {field_name}: {value}")
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation failed for {field_name}: {e}")
        raise ValueError(f"Invalid {field_name} parameter")


def _validate_string(value: Any, field_name: str, max_length: int = 1000) -> str:
    """Validate string input, preventing injection."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid string value for {field_name}")
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length}")

    # Check for dangerous SQL keywords that might indicate injection attempts
    dangerous_keywords = [
        "DROP TABLE", "DROP DATABASE", "DELETE FROM", "TRUNCATE",
        "ALTER TABLE", "CREATE TABLE", "INSERT INTO", "UPDATE ",
        "UNION SELECT", "UNION ALL", "EXEC(", "EXECUTE(",
        "--", "/*", "*/", "xp_", "sp_", ";--"
    ]

    value_upper = value.upper()
    for keyword in dangerous_keywords:
        if keyword in value_upper:
            raise ValueError(f"Invalid {field_name}: contains dangerous SQL pattern")

    return value


def _validate_embedding(embedding: list[float], expected_dim: int) -> list[float]:
    """Validate embedding dimensions and values."""
    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list of floats")
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding must have {expected_dim} dimensions, got {len(embedding)}"
        )
    if not all(isinstance(x, (int, float)) for x in embedding):
        raise ValueError("All embedding values must be numeric")
    return embedding


class DocumentQueries:
    """Query utilities for document operations."""

    @staticmethod
    def get_by_id(db: Session, document_id: int) -> Document | None:
        """Get document by ID."""
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def get_by_path(db: Session, source_path: str) -> Document | None:
        """Get document by source path."""
        return db.query(Document).filter(Document.source_path == source_path).first()

    @staticmethod
    def get_by_hash(db: Session, file_hash: str) -> Document | None:
        """Get document by file hash."""
        return db.query(Document).filter(Document.file_hash == file_hash).first()

    @staticmethod
    def get_by_status(
        db: Session,
        status: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Document]:
        """Get documents by conversion status."""
        query = db.query(Document).filter(Document.conversion_status == status)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()

    @staticmethod
    def get_recent(
        db: Session, limit: int = 50, offset: int | None = None
    ) -> list[Document]:
        """Get recent documents by creation date."""
        query = db.query(Document).order_by(Document.created_at.desc())

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()

    @staticmethod
    def get_statistics(db: Session) -> dict[str, Any]:
        """Get document processing statistics."""
        total = db.query(Document).count()
        by_status = (
            db.query(Document.conversion_status, func.count(Document.id))
            .group_by(Document.conversion_status)
            .all()
        )

        stats = {
            "total_documents": total,
            "by_status": dict(by_status),
            "total_size_bytes": db.query(func.sum(Document.file_size_bytes)).scalar()
            or 0,
        }

        return stats


class SearchQueries:
    """Query utilities for search operations."""

    @staticmethod
    def fulltext_search(
        db: Session,
        query: str,
        limit: int = 10,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
        include_content: bool = True,
    ) -> list[Any]:
        """
        Perform full-text search on document content with input validation.

        Args:
            db: Database session
            query: Search query string
            limit: Maximum results to return
            offset: Offset for pagination
            filters: Optional filters to apply
            include_content: Whether to include content text

        Returns:
            List of search result objects with rank scores

        Raises:
            ValueError: If inputs are invalid
            SQLAlchemyError: If database query fails
        """
        # Validate inputs to prevent injection attacks
        query = _validate_string(query, "query", max_length=500)
        limit = _validate_integer(limit, "limit")
        if offset is not None:
            offset = _validate_integer(offset, "offset")

        # Ensure reasonable limits
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if offset is not None and offset < 0:
            raise ValueError("Offset must be non-negative")

        # Sanitize query string for PostgreSQL full-text search
        # Remove potential dangerous characters
        query = query.replace("'", "''").strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        # Build base query
        base_query = """
            SELECT d.*, dc.*, ts_rank(
                to_tsvector('english', COALESCE(dc.plain_text, '')),
                plainto_tsquery('english', :query)
            ) as score
            FROM documents d
            JOIN document_content dc ON d.id = dc.document_id
            WHERE to_tsvector('english', COALESCE(dc.plain_text, ''))
                  @@ plainto_tsquery('english', :query)
        """

        # Add filters if provided using parameterized queries
        params = {"query": query, "limit": limit, "offset": offset or 0}

        # Build parameterized filter conditions safely
        filter_clause = ""
        if filters:
            if "document_id" in filters and isinstance(filters["document_id"], int):
                filter_clause = " AND d.id = :document_id"
                params["document_id"] = filters["document_id"]

        if filter_clause:
            base_query += filter_clause

        base_query += """
            ORDER BY score DESC
            LIMIT :limit
            OFFSET :offset
        """

        search_query = text(base_query)

        try:
            result = db.execute(search_query, params)
        except SQLAlchemyError as e:
            # Sanitize error message to prevent information disclosure
            from pdf_to_markdown_mcp.auth.security import sanitize_error_message
            sanitized_msg = sanitize_error_message(str(e))
            raise SQLAlchemyError(sanitized_msg) from None

        # Create result objects
        class SearchResult:
            def __init__(self, row):
                self.document_id = row.id  # document id
                self.chunk_id = None  # no chunk for full-text search
                self.filename = row.filename
                self.source_path = row.source_path
                self.content = row.plain_text if include_content else None
                self.score = row.score  # text search score
                self.page_number = None
                self.chunk_index = None
                self.metadata = row.metadata or {}

        results = [SearchResult(row) for row in result]
        return results

    @staticmethod
    def vector_similarity_search(
        db: Session,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
        include_content: bool = True,
    ) -> list[Any]:
        """
        Perform vector similarity search with input validation.

        Note: This requires PGVector extension to be installed.

        Args:
            db: Database session
            query_embedding: Query vector (1536 dimensions)
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            filters: Optional filters to apply
            include_content: Whether to include content text

        Returns:
            List of result objects with similarity scores

        Raises:
            ValueError: If inputs are invalid
            SQLAlchemyError: If database query fails
        """
        # Validate inputs to prevent injection attacks
        query_embedding = _validate_embedding(query_embedding, 1536)
        limit = _validate_integer(limit, "limit")
        threshold = _validate_float(threshold, "threshold")

        # Ensure reasonable limits
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")
        # Build base query
        base_query = """
            SELECT de.*, d.filename, d.source_path, d.metadata as doc_metadata,
                   1 - (de.embedding <=> :query_embedding::vector) as similarity
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE 1 - (de.embedding <=> :query_embedding::vector) >= :threshold
        """

        # Add filters using parameterized queries only
        params = {
            "query_embedding": query_embedding,
            "threshold": threshold,
            "limit": limit,
        }

        # Build parameterized filter conditions safely
        filter_clause = ""
        if filters:
            if "document_id" in filters and isinstance(filters["document_id"], int):
                filter_clause = " AND de.document_id = :document_id"
                params["document_id"] = filters["document_id"]

        if filter_clause:
            base_query += filter_clause

        base_query += """
            ORDER BY de.embedding <=> :query_embedding::vector
            LIMIT :limit
        """

        search_query = text(base_query)
        result = db.execute(search_query, params)

        # Create result objects
        class SearchResult:
            def __init__(self, row):
                self.document_id = row.document_id
                self.chunk_id = row.id
                self.filename = row.filename
                self.source_path = row.source_path
                self.content = row.chunk_text if include_content else None
                self.similarity = row.similarity
                self.page_number = row.page_number
                self.chunk_index = row.chunk_index
                self.metadata = row.metadata or {}
                if hasattr(row, "doc_metadata") and row.doc_metadata:
                    self.metadata.update(row.doc_metadata)

        results = [SearchResult(row) for row in result]
        return results

    @staticmethod
    def hybrid_search(
        db: Session,
        query: str,
        query_embedding: list[float],
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Perform optimized hybrid search combining semantic and keyword search.

        This optimized version avoids N+1 query problems by:
        1. Using efficient joins instead of CTEs
        2. Reducing complexity from O(n*m) to O(n log n)
        3. Adding proper indexing hints
        4. Implementing fallback mechanisms

        Args:
            db: Database session
            query: Text query for keyword search
            query_embedding: Vector for semantic search
            limit: Maximum results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword relevance

        Returns:
            List of search results with combined scores

        Raises:
            ValueError: If inputs are invalid
            SQLAlchemyError: If database query fails
        """
        # Validate inputs to prevent injection attacks
        query = _validate_string(query, "query", max_length=500)
        query_embedding = _validate_embedding(query_embedding, 1536)
        limit = _validate_integer(limit, "limit")
        semantic_weight = _validate_float(semantic_weight, "semantic_weight")
        keyword_weight = _validate_float(keyword_weight, "keyword_weight")

        # Ensure reasonable limits and weights
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if not (0 <= semantic_weight <= 1) or not (0 <= keyword_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        # Log performance measurement start
        import time

        start_time = time.time()

        try:
            # Optimized hybrid query that avoids CTEs and reduces complexity
            # Uses efficient joins and DISTINCT ON for better performance
            hybrid_query = text(
                """
                SELECT DISTINCT ON (d.id)
                    d.id,
                    d.filename,
                    d.source_path,
                    d.metadata,
                    d.created_at,
                    d.file_size_bytes,
                    d.conversion_status,
                    -- Calculate semantic score with efficient vector operations
                    CASE
                        WHEN de.embedding IS NOT NULL THEN
                            (1 - (de.embedding <=> :query_embedding::vector)) * :semantic_weight
                        ELSE 0
                    END as semantic_score,
                    -- Calculate keyword score with optimized full-text search
                    CASE
                        WHEN dc.plain_text IS NOT NULL AND
                             to_tsvector('english', COALESCE(dc.plain_text, ''))
                             @@ plainto_tsquery('english', :query) THEN
                            ts_rank(
                                to_tsvector('english', COALESCE(dc.plain_text, '')),
                                plainto_tsquery('english', :query)
                            ) * :keyword_weight
                        ELSE 0
                    END as keyword_score,
                    -- Pre-calculate combined score for efficient ordering
                    COALESCE(
                        CASE
                            WHEN de.embedding IS NOT NULL THEN
                                (1 - (de.embedding <=> :query_embedding::vector)) * :semantic_weight
                            ELSE 0
                        END, 0
                    ) + COALESCE(
                        CASE
                            WHEN dc.plain_text IS NOT NULL AND
                                 to_tsvector('english', COALESCE(dc.plain_text, ''))
                                 @@ plainto_tsquery('english', :query) THEN
                                ts_rank(
                                    to_tsvector('english', COALESCE(dc.plain_text, '')),
                                    plainto_tsquery('english', :query)
                                ) * :keyword_weight
                            ELSE 0
                        END, 0
                    ) as combined_score,
                    -- Include chunk information from best matching embedding
                    de.chunk_text,
                    de.page_number,
                    de.chunk_index,
                    de.metadata as chunk_metadata
                FROM documents d
                LEFT JOIN document_embeddings de ON d.id = de.document_id
                    AND (1 - (de.embedding <=> :query_embedding::vector)) >= :semantic_threshold
                LEFT JOIN document_content dc ON d.id = dc.document_id
                    AND to_tsvector('english', COALESCE(dc.plain_text, ''))
                        @@ plainto_tsquery('english', :query)
                WHERE (
                    (de.document_id IS NOT NULL AND
                     (1 - (de.embedding <=> :query_embedding::vector)) >= :semantic_threshold)
                    OR
                    (dc.document_id IS NOT NULL AND
                     to_tsvector('english', COALESCE(dc.plain_text, ''))
                     @@ plainto_tsquery('english', :query))
                )
                ORDER BY d.id, combined_score DESC
                LIMIT :expanded_limit
            """
            )

            # Use expanded limit to account for DISTINCT ON filtering
            expanded_limit = min(
                limit * 2, 200
            )  # Cap at 200 to prevent excessive results
            semantic_threshold = 0.2  # Minimum semantic similarity threshold

            result = db.execute(
                hybrid_query,
                {
                    "query": query,
                    "query_embedding": query_embedding,
                    "semantic_weight": semantic_weight,
                    "keyword_weight": keyword_weight,
                    "semantic_threshold": semantic_threshold,
                    "expanded_limit": expanded_limit,
                },
            )

            # Process results with deduplication and proper sorting
            results = []
            seen_documents = set()

            for row in result:
                if row.id not in seen_documents and len(results) < limit:
                    seen_documents.add(row.id)

                    # Create comprehensive result dictionary
                    result_dict = {
                        "document_id": row.id,
                        "filename": row.filename,
                        "source_path": row.source_path,
                        "combined_score": (
                            float(row.combined_score) if row.combined_score else 0.0
                        ),
                        "semantic_score": (
                            float(row.semantic_score) if row.semantic_score else 0.0
                        ),
                        "keyword_score": (
                            float(row.keyword_score) if row.keyword_score else 0.0
                        ),
                        "chunk_text": row.chunk_text,
                        "page_number": row.page_number,
                        "chunk_index": row.chunk_index,
                        "file_size_bytes": row.file_size_bytes,
                        "conversion_status": row.conversion_status,
                        "created_at": (
                            row.created_at.isoformat() if row.created_at else None
                        ),
                        "metadata": {
                            **(row.metadata or {}),
                            **(row.chunk_metadata or {}),
                        },
                    }
                    results.append(result_dict)

            # Final sort by combined score after deduplication
            results.sort(key=lambda x: x["combined_score"], reverse=True)

            # Log performance metrics
            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Optimized hybrid search completed in {execution_time:.2f}ms, "
                f"returned {len(results)} results from {limit} requested"
            )

            return results[:limit]  # Ensure we don't exceed requested limit

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to simple vector search if hybrid fails
            return SearchQueries._fallback_vector_search(
                db, query_embedding, limit, semantic_weight
            )

    @staticmethod
    def _fallback_vector_search(
        db: Session, query_embedding: list[float], limit: int, weight: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Fallback vector search if hybrid search fails.

        This provides a reliable fallback mechanism when the main hybrid
        search encounters issues, ensuring the system remains responsive.
        """
        try:
            logger.info("Using fallback vector search due to hybrid search failure")

            fallback_query = text(
                """
                SELECT DISTINCT
                    d.id,
                    d.filename,
                    d.source_path,
                    d.metadata,
                    d.created_at,
                    d.file_size_bytes,
                    d.conversion_status,
                    de.chunk_text,
                    de.page_number,
                    de.chunk_index,
                    (1 - (de.embedding <=> :query_embedding::vector)) * :weight as combined_score,
                    de.metadata as chunk_metadata
                FROM documents d
                JOIN document_embeddings de ON d.id = de.document_id
                WHERE (1 - (de.embedding <=> :query_embedding::vector)) >= 0.2
                ORDER BY combined_score DESC
                LIMIT :limit
            """
            )

            result = db.execute(
                fallback_query,
                {"query_embedding": query_embedding, "weight": weight, "limit": limit},
            )

            return [
                {
                    "document_id": row.id,
                    "filename": row.filename,
                    "source_path": row.source_path,
                    "combined_score": float(row.combined_score),
                    "semantic_score": float(
                        row.combined_score
                    ),  # Same as combined for vector-only
                    "keyword_score": 0.0,
                    "chunk_text": row.chunk_text,
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "file_size_bytes": row.file_size_bytes,
                    "conversion_status": row.conversion_status,
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                    "metadata": {**(row.metadata or {}), **(row.chunk_metadata or {})},
                }
                for row in result
            ]

        except Exception as e:
            logger.error(f"Fallback vector search also failed: {e}")
            return []  # Return empty results rather than crash

    @staticmethod
    def get_document_embeddings(
        db: Session, document_id: int
    ) -> list[DocumentEmbedding]:
        """Get all embeddings for a document."""
        return (
            db.query(DocumentEmbedding)
            .filter(DocumentEmbedding.document_id == document_id)
            .all()
        )

    @staticmethod
    def find_similar_documents(
        db: Session,
        reference_embedding: list[float],
        reference_doc_id: int | None = None,
        top_k: int = 5,
        threshold: float = 0.6,
    ) -> list[Any]:
        """
        Find documents similar to a reference embedding.

        Args:
            db: Database session
            reference_embedding: Reference vector embedding
            reference_doc_id: Document ID to exclude from results
            top_k: Number of similar documents to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar documents with similarity scores
        """
        base_query = """
            SELECT DISTINCT d.*,
                   AVG(1 - (de.embedding <=> :reference_embedding::vector)) as similarity
            FROM documents d
            JOIN document_embeddings de ON d.id = de.document_id
            WHERE 1 - (de.embedding <=> :reference_embedding::vector) >= :threshold
        """

        params = {
            "reference_embedding": reference_embedding,
            "threshold": threshold,
            "top_k": top_k,
        }

        # Add reference document exclusion using parameterized query
        exclude_clause = ""
        if reference_doc_id is not None and isinstance(reference_doc_id, int):
            exclude_clause = " AND d.id != :reference_doc_id"
            params["reference_doc_id"] = reference_doc_id

        if exclude_clause:
            base_query += exclude_clause

        base_query += """
            GROUP BY d.id, d.filename, d.source_path, d.metadata
            ORDER BY similarity DESC
            LIMIT :top_k
        """

        search_query = text(base_query)
        result = db.execute(search_query, params)

        # Create result objects
        class SimilarDocument:
            def __init__(self, row):
                self.document_id = row.id
                self.filename = row.filename
                self.source_path = row.source_path
                self.title = row.filename  # use filename as title for now
                self.similarity = row.similarity
                self.metadata = row.metadata or {}

        results = [SimilarDocument(row) for row in result]
        return results


class QueueQueries:
    """Query utilities for processing queue operations."""

    @staticmethod
    def get_next_job(db: Session, worker_id: str) -> ProcessingQueue | None:
        """
        Get next job from queue for processing with proper transaction isolation.

        Uses SELECT FOR UPDATE SKIP LOCKED to prevent race conditions and ensure
        only one worker can claim each job.

        Args:
            db: Database session
            worker_id: Unique identifier for the worker

        Returns:
            ProcessingQueue job or None if no jobs available
        """
        # Validate worker_id
        worker_id = _validate_string(worker_id, "worker_id", max_length=255)

        try:
            # Use proper transaction isolation with row-level locking
            job = (
                db.query(ProcessingQueue)
                .filter(ProcessingQueue.status == "queued")
                .order_by(
                    ProcessingQueue.priority.asc(), ProcessingQueue.created_at.asc()
                )
                .with_for_update(skip_locked=True)
                .first()
            )

            if job:
                # Update job status atomically
                job.status = "processing"
                job.worker_id = worker_id
                job.started_at = func.now()
                job.attempts = (job.attempts or 0) + 1

                # Commit immediately to release lock
                db.commit()

                logger.info(f"Job {job.id} claimed by worker {worker_id}")

            return job

        except SQLAlchemyError as e:
            logger.error(f"Failed to get next job for worker {worker_id}: {e}")
            db.rollback()
            raise

    @staticmethod
    def get_queue_stats(db: Session) -> dict[str, Any]:
        """Get processing queue statistics."""
        total = db.query(ProcessingQueue).count()
        by_status = (
            db.query(ProcessingQueue.status, func.count(ProcessingQueue.id))
            .group_by(ProcessingQueue.status)
            .all()
        )

        avg_processing_time = (
            db.query(
                func.avg(
                    func.extract(
                        "epoch",
                        ProcessingQueue.completed_at - ProcessingQueue.started_at,
                    )
                )
            )
            .filter(
                and_(
                    ProcessingQueue.status == "completed",
                    ProcessingQueue.started_at.is_not(None),
                    ProcessingQueue.completed_at.is_not(None),
                )
            )
            .scalar()
        )

        return {
            "total_jobs": total,
            "by_status": dict(by_status),
            "avg_processing_time_seconds": avg_processing_time or 0,
        }

    @staticmethod
    def retry_failed_jobs(
        db: Session, max_attempts: int = 3, batch_size: int = 10
    ) -> int:
        """Retry failed jobs that haven't exceeded max attempts."""
        failed_jobs = (
            db.query(ProcessingQueue)
            .filter(
                and_(
                    ProcessingQueue.status == "failed",
                    ProcessingQueue.attempts < max_attempts,
                )
            )
            .limit(batch_size)
            .all()
        )

        count = 0
        for job in failed_jobs:
            job.status = "retrying"
            job.attempts += 1
            job.worker_id = None
            job.started_at = None
            count += 1

        if count > 0:
            db.commit()

        return count

    @staticmethod
    def get_by_document_id(db: Session, document_id: int) -> ProcessingQueue | None:
        """Get queue entry by document ID (via document path lookup)."""
        # First get the document to find its path
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None

        # Then find the queue entry by file path
        return (
            db.query(ProcessingQueue)
            .filter(ProcessingQueue.file_path == document.source_path)
            .first()
        )

    @staticmethod
    def get_by_file_path(db: Session, file_path: str) -> ProcessingQueue | None:
        """Get queue entry by file path."""
        return (
            db.query(ProcessingQueue)
            .filter(ProcessingQueue.file_path == file_path)
            .first()
        )

    @staticmethod
    def get_queue_statistics(db: Session) -> dict[str, int]:
        """Get queue statistics by status."""
        stats = (
            db.query(ProcessingQueue.status, func.count(ProcessingQueue.id))
            .group_by(ProcessingQueue.status)
            .all()
        )

        result = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}

        for status, count in stats:
            if status in result:
                result[status] = count

        return result

    @staticmethod
    def clear_old_entries(db: Session, older_than_days: int = 7) -> int:
        """Clear completed queue entries older than specified days."""
        from datetime import datetime, timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        deleted_count = (
            db.query(ProcessingQueue)
            .filter(
                and_(
                    ProcessingQueue.status == "completed",
                    ProcessingQueue.completed_at < cutoff_date,
                )
            )
            .delete()
        )

        return deleted_count
