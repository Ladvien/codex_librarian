"""
Database service for PDF to Markdown MCP Server.

This module provides advanced database operations including PGVector similarity search,
hybrid search combining semantic and full-text search, and database management utilities.
Implements vector operations with proper indexing strategies and query optimization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..core.exceptions import DatabaseError, ValidationError
from ..db.models import (
    Document,
    DocumentEmbedding,
    DocumentImage,
    ProcessingQueue,
)
from ..db.session import db_manager, get_db

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result object for vector similarity search."""

    document_id: int
    chunk_id: int | None
    filename: str
    source_path: str
    content: str | None
    similarity_score: float
    page_number: int | None
    chunk_index: int | None
    metadata: dict[str, Any]
    search_type: str = "vector"


@dataclass
class HybridSearchResult:
    """Result object for hybrid search combining vector and text search."""

    document_id: int
    chunk_id: int | None
    filename: str
    source_path: str
    content: str | None
    combined_score: float
    semantic_score: float
    keyword_score: float
    page_number: int | None
    chunk_index: int | None
    metadata: dict[str, Any]
    search_type: str = "hybrid"


@dataclass
class DatabaseStats:
    """Database statistics and health information."""

    total_documents: int
    processing_stats: dict[str, int]
    total_embeddings: int
    total_images: int
    queue_depth: int
    index_usage: dict[str, Any]
    connection_pool: dict[str, Any]


class VectorDatabaseService:
    """
    Advanced database service with PGVector support.

    Provides vector similarity search, hybrid search, database health monitoring,
    and performance optimization utilities for the PDF to Markdown MCP server.
    """

    def __init__(self, db_session: Session | None = None):
        """
        Initialize database service.

        Args:
            db_session: Optional database session. If None, uses dependency injection.
        """
        self.db_session = db_session
        self._ensure_pgvector_extension()

    def _get_session(self) -> Session:
        """Get database session from dependency injection or instance variable."""
        if self.db_session:
            return self.db_session
        return next(get_db())

    def _ensure_pgvector_extension(self) -> bool:
        """
        Ensure PGVector extension is available and properly configured.

        Returns:
            bool: True if PGVector is available, False otherwise
        """
        try:
            with self._get_session() as db:
                # Check if PGVector extension is available
                result = db.execute(
                    text(
                        """
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """
                    )
                ).scalar()

                if not result:
                    logger.warning("PGVector extension not installed")
                    return False

                # Verify vector operations work
                test_query = text(
                    """
                    SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector as distance
                """
                )
                db.execute(test_query)
                logger.info("PGVector extension verified and working")
                return True

        except Exception as e:
            logger.error(f"PGVector extension check failed: {e}")
            return False

    async def vector_similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        document_filters: dict[str, Any] | None = None,
        distance_metric: str = "cosine",
    ) -> list[VectorSearchResult]:
        """
        Perform async vector similarity search using PGVector.

        This optimized version uses async database operations to prevent
        blocking the event loop and allow concurrent request handling.

        Args:
            query_embedding: Query vector (1536 dimensions for text embeddings)
            top_k: Number of similar results to return
            similarity_threshold: Minimum similarity threshold (0.0 to 1.0)
            document_filters: Optional filters to apply (document_id, etc.)
            distance_metric: Distance metric ("cosine", "euclidean", "inner_product")

        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        from ..core.performance import get_performance_monitor
        from ..db.async_session import async_db_manager

        try:
            # Performance monitoring
            monitor = get_performance_monitor()

            async with monitor.measure_performance("async_vector_similarity_search"):
                # Choose distance operator based on metric
                distance_ops = {
                    "cosine": "<=>",
                    "euclidean": "<->",
                    "inner_product": "<#>",
                }

                if distance_metric not in distance_ops:
                    raise ValidationError(
                        f"Unsupported distance metric: {distance_metric}"
                    )

                distance_op = distance_ops[distance_metric]

                # Build optimized async query with proper PGVector syntax
                base_query = f"""
                    SELECT
                        de.id as embedding_id,
                        de.document_id,
                        de.chunk_text,
                        de.page_number,
                        de.chunk_index,
                        de.metadata as chunk_metadata,
                        d.filename,
                        d.source_path,
                        d.metadata as doc_metadata,
                        CASE
                            WHEN :distance_metric = 'inner_product' THEN
                                (de.embedding <#> :query_embedding::vector)
                            ELSE
                                1 - (de.embedding {distance_op} :query_embedding::vector)
                        END as similarity_score
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id = d.id
                    WHERE d.conversion_status = 'completed'
                """

                params = {
                    "query_embedding": query_embedding,
                    "distance_metric": distance_metric,
                    "top_k": top_k,
                }

                # Add similarity threshold with optimized filtering
                if distance_metric == "inner_product":
                    base_query += (
                        " AND (de.embedding <#> :query_embedding::vector) >= :threshold"
                    )
                    params["threshold"] = similarity_threshold
                else:
                    base_query += f" AND 1 - (de.embedding {distance_op} :query_embedding::vector) >= :threshold"
                    params["threshold"] = similarity_threshold

                # Add document filters with validation
                if document_filters:
                    if "document_id" in document_filters:
                        base_query += " AND de.document_id = :document_id"
                        params["document_id"] = int(document_filters["document_id"])

                    if "exclude_document_id" in document_filters:
                        base_query += " AND de.document_id != :exclude_document_id"
                        params["exclude_document_id"] = int(
                            document_filters["exclude_document_id"]
                        )

                # Optimized ordering and limiting
                if distance_metric == "inner_product":
                    base_query += (
                        " ORDER BY (de.embedding <#> :query_embedding::vector) DESC"
                    )
                else:
                    base_query += f" ORDER BY de.embedding {distance_op} :query_embedding::vector ASC"

                base_query += " LIMIT :top_k"

                # Execute async query
                async with async_db_manager.get_async_session() as db:
                    result = await db.execute(text(base_query), params)

                    results = []
                    for row in result:
                        metadata = {}
                        if row.chunk_metadata:
                            metadata.update(row.chunk_metadata)
                        if row.doc_metadata:
                            metadata.update(row.doc_metadata)

                        results.append(
                            VectorSearchResult(
                                document_id=row.document_id,
                                chunk_id=row.embedding_id,
                                filename=row.filename,
                                source_path=row.source_path,
                                content=row.chunk_text,
                                similarity_score=float(row.similarity_score),
                                page_number=row.page_number,
                                chunk_index=row.chunk_index,
                                metadata=metadata,
                                search_type=f"vector_{distance_metric}",
                            )
                        )

                    logger.info(
                        f"Async vector search returned {len(results)} results in optimized query"
                    )
                    return results

        except Exception as e:
            logger.error(f"Async vector similarity search failed: {e}")
            raise DatabaseError(f"Vector similarity search failed: {e!s}")

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        similarity_threshold: float = 0.5,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        Args:
            query_text: Text query for keyword search
            query_embedding: Vector for semantic search
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword relevance (0.0 to 1.0)
            similarity_threshold: Minimum combined score threshold

        Returns:
            List of HybridSearchResult objects with combined scores
        """
        try:
            # Normalize weights
            total_weight = semantic_weight + keyword_weight
            if total_weight > 0:
                semantic_weight = semantic_weight / total_weight
                keyword_weight = keyword_weight / total_weight

            # Complex hybrid search query
            hybrid_query = text(
                """
                WITH semantic_results AS (
                    SELECT
                        de.id as embedding_id,
                        de.document_id,
                        de.chunk_text,
                        de.page_number,
                        de.chunk_index,
                        de.metadata as chunk_metadata,
                        d.filename,
                        d.source_path,
                        d.metadata as doc_metadata,
                        1 - (de.embedding <=> :query_embedding::vector) as semantic_score
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id = d.id
                    WHERE d.conversion_status = 'completed'
                        AND 1 - (de.embedding <=> :query_embedding::vector) >= 0.3
                    ORDER BY de.embedding <=> :query_embedding::vector
                    LIMIT 100
                ),
                keyword_results AS (
                    SELECT
                        dc.document_id,
                        d.filename,
                        d.source_path,
                        d.metadata as doc_metadata,
                        ts_rank_cd(
                            to_tsvector('english', COALESCE(dc.plain_text, '')),
                            plainto_tsquery('english', :query_text)
                        ) as keyword_score
                    FROM document_content dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.conversion_status = 'completed'
                        AND to_tsvector('english', COALESCE(dc.plain_text, ''))
                            @@ plainto_tsquery('english', :query_text)
                )
                SELECT
                    COALESCE(sr.document_id, kr.document_id) as document_id,
                    sr.embedding_id as chunk_id,
                    COALESCE(sr.filename, kr.filename) as filename,
                    COALESCE(sr.source_path, kr.source_path) as source_path,
                    sr.chunk_text,
                    sr.page_number,
                    sr.chunk_index,
                    COALESCE(sr.chunk_metadata, '{}') as chunk_metadata,
                    COALESCE(sr.doc_metadata, kr.doc_metadata, '{}') as doc_metadata,
                    COALESCE(sr.semantic_score, 0) as semantic_score,
                    COALESCE(kr.keyword_score, 0) as keyword_score,
                    (COALESCE(sr.semantic_score, 0) * :semantic_weight +
                     COALESCE(kr.keyword_score, 0) * :keyword_weight) as combined_score
                FROM semantic_results sr
                FULL OUTER JOIN keyword_results kr ON sr.document_id = kr.document_id
                WHERE (COALESCE(sr.semantic_score, 0) * :semantic_weight +
                       COALESCE(kr.keyword_score, 0) * :keyword_weight) >= :threshold
                ORDER BY combined_score DESC
                LIMIT :top_k
            """
            )

            params = {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "threshold": similarity_threshold,
                "top_k": top_k,
            }

            with self._get_session() as db:
                result = db.execute(hybrid_query, params)

                results = []
                for row in result:
                    metadata = {}
                    if row.chunk_metadata:
                        metadata.update(row.chunk_metadata)
                    if row.doc_metadata:
                        metadata.update(row.doc_metadata)

                    results.append(
                        HybridSearchResult(
                            document_id=row.document_id,
                            chunk_id=row.chunk_id,
                            filename=row.filename,
                            source_path=row.source_path,
                            content=row.chunk_text,
                            combined_score=float(row.combined_score),
                            semantic_score=float(row.semantic_score),
                            keyword_score=float(row.keyword_score),
                            page_number=row.page_number,
                            chunk_index=row.chunk_index,
                            metadata=metadata,
                        )
                    )

                logger.info(f"Hybrid search returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise DatabaseError(f"Hybrid search failed: {e!s}")

    async def find_similar_documents(
        self,
        reference_document_id: int,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        exclude_self: bool = True,
    ) -> list[VectorSearchResult]:
        """
        Find documents similar to a reference document using averaged embeddings.

        Args:
            reference_document_id: ID of reference document
            top_k: Number of similar documents to return
            similarity_threshold: Minimum similarity threshold
            exclude_self: Whether to exclude the reference document from results

        Returns:
            List of similar documents with similarity scores
        """
        try:
            # First, get the average embedding for the reference document
            avg_embedding_query = text(
                """
                SELECT AVG(de.embedding) as avg_embedding
                FROM document_embeddings de
                WHERE de.document_id = :ref_doc_id
                HAVING COUNT(de.id) > 0
            """
            )

            with self._get_session() as db:
                result = db.execute(
                    avg_embedding_query, {"ref_doc_id": reference_document_id}
                )
                row = result.first()

                if not row or not row.avg_embedding:
                    raise ValidationError(
                        f"No embeddings found for document {reference_document_id}"
                    )

                # Convert PostgreSQL vector to list
                reference_embedding = list(row.avg_embedding)

                # Find similar documents using the averaged embedding
                filters = (
                    {"exclude_document_id": reference_document_id}
                    if exclude_self
                    else None
                )

                similar_results = await self.vector_similarity_search(
                    query_embedding=reference_embedding,
                    top_k=top_k * 3,  # Get more results to aggregate by document
                    similarity_threshold=similarity_threshold,
                    document_filters=filters,
                )

                # Group by document and average similarity scores
                doc_similarities = {}
                for result in similar_results:
                    doc_id = result.document_id
                    if doc_id not in doc_similarities:
                        doc_similarities[doc_id] = {
                            "result": result,
                            "similarities": [],
                        }
                    doc_similarities[doc_id]["similarities"].append(
                        result.similarity_score
                    )

                # Create final results with averaged similarities
                final_results = []
                for doc_id, data in doc_similarities.items():
                    avg_similarity = np.mean(data["similarities"])
                    result = data["result"]
                    result.similarity_score = float(avg_similarity)
                    result.search_type = "similar_documents"
                    final_results.append(result)

                # Sort by average similarity and limit
                final_results.sort(key=lambda x: x.similarity_score, reverse=True)
                return final_results[:top_k]

        except Exception as e:
            logger.error(f"Similar documents search failed: {e}")
            raise DatabaseError(f"Similar documents search failed: {e!s}")

    async def get_database_statistics(self) -> DatabaseStats:
        """
        Get comprehensive database statistics and health information.

        Returns:
            DatabaseStats object with comprehensive database information
        """
        try:
            with self._get_session() as db:
                # Basic document statistics
                total_documents = db.query(Document).count()

                # Processing status statistics
                processing_stats = dict(
                    db.query(Document.conversion_status, func.count(Document.id))
                    .group_by(Document.conversion_status)
                    .all()
                )

                # Embedding statistics
                total_embeddings = db.query(DocumentEmbedding).count()
                total_images = db.query(DocumentImage).count()

                # Queue statistics
                queue_depth = (
                    db.query(ProcessingQueue)
                    .filter(ProcessingQueue.status.in_(["queued", "processing"]))
                    .count()
                )

                # Index usage statistics (PostgreSQL specific)
                index_usage_query = text(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                """
                )

                index_stats = db.execute(index_usage_query).fetchall()
                index_usage = {
                    row.indexname: {
                        "scans": row.idx_scan,
                        "tuples_read": row.idx_tup_read,
                        "tuples_fetched": row.idx_tup_fetch,
                    }
                    for row in index_stats
                }

                # Connection pool statistics
                connection_pool = db_manager.get_connection_info()

                return DatabaseStats(
                    total_documents=total_documents,
                    processing_stats=processing_stats,
                    total_embeddings=total_embeddings,
                    total_images=total_images,
                    queue_depth=queue_depth,
                    index_usage=index_usage,
                    connection_pool=connection_pool,
                )

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            raise DatabaseError(f"Failed to get database statistics: {e!s}")

    async def optimize_vector_indexes(self) -> dict[str, Any]:
        """
        Optimize vector indexes for better performance.

        Returns:
            Dictionary with optimization results and statistics
        """
        try:
            with self._get_session() as db:
                # Analyze vector tables for better query planning
                analyze_queries = [
                    "ANALYZE document_embeddings;",
                    "ANALYZE document_images;",
                    "ANALYZE documents;",
                    "ANALYZE document_content;",
                ]

                for query in analyze_queries:
                    db.execute(text(query))

                # Get index statistics
                index_stats_query = text(
                    """
                    SELECT
                        i.relname as index_name,
                        t.relname as table_name,
                        s.idx_scan,
                        s.idx_tup_read,
                        pg_size_pretty(pg_relation_size(i.oid)) as index_size
                    FROM pg_class i
                    JOIN pg_index ix ON i.oid = ix.indexrelid
                    JOIN pg_class t ON t.oid = ix.indrelid
                    JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
                    WHERE i.relkind = 'i'
                        AND t.relname IN ('document_embeddings', 'document_images')
                    ORDER BY s.idx_scan DESC
                """
                )

                index_results = db.execute(index_stats_query).fetchall()

                optimization_results = {
                    "status": "completed",
                    "analyzed_tables": len(analyze_queries),
                    "vector_indexes": [
                        {
                            "index_name": row.index_name,
                            "table_name": row.table_name,
                            "scans": row.idx_scan,
                            "tuples_read": row.idx_tup_read,
                            "size": row.index_size,
                        }
                        for row in index_results
                    ],
                }

                logger.info("Vector indexes optimized successfully")
                return optimization_results

        except Exception as e:
            logger.error(f"Vector index optimization failed: {e}")
            raise DatabaseError(f"Vector index optimization failed: {e!s}")

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive database health check.

        Returns:
            Dictionary with health check results
        """
        try:
            health_status = {
                "database_connection": False,
                "pgvector_available": False,
                "query_performance": None,
                "connection_pool": {},
                "disk_usage": None,
            }

            with self._get_session() as db:
                # Test basic connectivity
                start_time = datetime.now()
                db.execute(text("SELECT 1"))
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                health_status["database_connection"] = True
                health_status["query_performance"] = f"{query_time:.2f}ms"

                # Check PGVector availability
                health_status["pgvector_available"] = self._ensure_pgvector_extension()

                # Connection pool status
                health_status["connection_pool"] = db_manager.get_connection_info()

                # Check disk usage (PostgreSQL specific)
                try:
                    disk_usage_query = text(
                        """
                        SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                    """
                    )
                    result = db.execute(disk_usage_query).scalar()
                    health_status["disk_usage"] = result
                except Exception:
                    pass  # Non-critical

                logger.info("Database health check completed successfully")
                return health_status

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"database_connection": False, "error": str(e)}

    # Document CRUD Operations
    # Following architecture principles: Service layer provides DTO abstraction

    async def find_document_by_hash(self, file_hash: str) -> Optional["DocumentDTO"]:
        """
        Find document by file hash.

        Args:
            file_hash: SHA-256 file hash

        Returns:
            DocumentDTO or None if not found
        """

        try:
            with self._get_session() as db:
                document = (
                    db.query(Document).filter(Document.file_hash == file_hash).first()
                )
                if not document:
                    return None

                return self._convert_to_dto(document)

        except Exception as e:
            logger.error(f"Error finding document by hash {file_hash}: {e}")
            raise DatabaseError(f"Failed to find document: {e}")

    async def create_document(self, create_data: "CreateDocumentDTO") -> "DocumentDTO":
        """
        Create new document record.

        Args:
            create_data: Document creation data

        Returns:
            Created DocumentDTO
        """
        from ..models.dto import ProcessingStatusType

        try:
            with self._get_session() as db:
                # Create new document
                document = Document(
                    filename=create_data.filename,
                    source_path=create_data.file_path,
                    file_hash=create_data.file_hash,
                    file_size_bytes=create_data.size_bytes,
                    conversion_status=ProcessingStatusType.PENDING,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )

                db.add(document)
                db.commit()
                db.refresh(document)

                logger.info(
                    f"Created document with ID {document.id}, hash {document.file_hash}"
                )
                return self._convert_to_dto(document)

        except Exception as e:
            logger.error(f"Error creating document: {e}")
            db.rollback()
            raise DatabaseError(f"Failed to create document: {e}")

    async def update_document(
        self, document_id: int, update_data: "UpdateDocumentDTO"
    ) -> "DocumentDTO":
        """
        Update existing document.

        Args:
            document_id: Document ID
            update_data: Update data

        Returns:
            Updated DocumentDTO
        """

        try:
            with self._get_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise ValidationError(f"Document {document_id} not found")

                # Update fields
                if update_data.processing_status:
                    document.conversion_status = update_data.processing_status
                if update_data.page_count is not None:
                    document.page_count = update_data.page_count
                if update_data.processing_time_seconds is not None:
                    document.processing_duration_seconds = (
                        update_data.processing_time_seconds
                    )
                if update_data.error_message is not None:
                    document.error_details = update_data.error_message
                if update_data.processed_at is not None:
                    document.updated_at = update_data.processed_at
                if update_data.metadata is not None:
                    document.meta_data = update_data.metadata

                document.updated_at = datetime.utcnow()

                db.commit()
                db.refresh(document)

                logger.info(f"Updated document {document_id}")
                return self._convert_to_dto(document)

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            db.rollback()
            raise DatabaseError(f"Failed to update document: {e}")

    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """
        Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document model or None if not found
        """

        try:
            with self._get_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return None

                # Detach from session to prevent issues with session closure
                db.expunge(document)
                return document

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise DatabaseError(f"Failed to get document: {e}")

    async def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """
        Get document by file hash.

        Args:
            file_hash: SHA-256 file hash

        Returns:
            Document model or None if not found
        """

        try:
            with self._get_session() as db:
                document = db.query(Document).filter(Document.file_hash == file_hash).first()
                if not document:
                    return None

                # Detach from session to prevent issues with session closure
                db.expunge(document)
                return document

        except Exception as e:
            logger.error(f"Error getting document by hash {file_hash}: {e}")
            raise DatabaseError(f"Failed to get document: {e}")

    async def save_document(self, document: Document) -> None:
        """
        Save (update) an existing document model to database.

        Args:
            document: Document model to save
        """

        try:
            with self._get_session() as db:
                # Merge the detached object back into the session
                db.merge(document)
                db.commit()
                logger.debug(f"Saved document {document.id}")

        except Exception as e:
            logger.error(f"Error saving document {document.id}: {e}")
            raise DatabaseError(f"Failed to save document: {e}")

    async def update_document_status(
        self, document_id: int, status: str, error_message: str | None = None
    ) -> None:
        """
        Update document processing status.

        Args:
            document_id: Document ID
            status: New processing status
            error_message: Optional error message
        """

        try:
            with self._get_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise DatabaseError(f"Document {document_id} not found")

                document.conversion_status = status
                document.error_message = error_message
                db.commit()
                logger.debug(f"Updated document {document_id} status to {status}")

        except Exception as e:
            logger.error(f"Error updating document status {document_id}: {e}")
            raise DatabaseError(f"Failed to update document status: {e}")

    async def delete_document(self, document_id: int) -> bool:
        """
        Delete document and related records.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        try:
            with self._get_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return False

                # Delete related records first (cascade should handle this)
                db.delete(document)
                db.commit()

                logger.info(f"Deleted document {document_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            db.rollback()
            raise DatabaseError(f"Failed to delete document: {e}")

    def _convert_to_dto(self, document: Document) -> "DocumentDTO":
        """
        Convert SQLAlchemy Document to DocumentDTO.

        Args:
            document: SQLAlchemy Document instance

        Returns:
            DocumentDTO
        """
        from ..models.dto import DocumentDTO, ProcessingStatusType

        # Get page_count and processing_time from DocumentContent if available
        page_count = None
        processing_time_seconds = None
        if document.content:
            page_count = document.content.page_count
            if document.content.processing_time_ms:
                processing_time_seconds = document.content.processing_time_ms / 1000.0

        return DocumentDTO(
            id=document.id,
            filename=document.filename,
            file_path=document.source_path,
            file_hash=document.file_hash,
            size_bytes=document.file_size_bytes,
            processing_status=ProcessingStatusType(document.conversion_status),
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.updated_at if document.conversion_status == "completed" else None,
            page_count=page_count,
            processing_time_seconds=processing_time_seconds,
            error_message=document.error_message,
            metadata=document.meta_data or {},
        )


# Factory function for easy instantiation
def create_database_service(
    db_session: Session | None = None,
) -> VectorDatabaseService:
    """
    Create a VectorDatabaseService instance.

    Args:
        db_session: Optional database session

    Returns:
        Configured VectorDatabaseService instance
    """
    return VectorDatabaseService(db_session=db_session)
