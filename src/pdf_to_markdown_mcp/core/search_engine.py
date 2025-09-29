"""
Search engine for semantic and hybrid search operations.

Provides vector similarity search and combined semantic + keyword search.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.core.exceptions import (
    EmbeddingError,
    SearchError,
    ValidationError,
)
from pdf_to_markdown_mcp.db.queries import SearchQueries
from pdf_to_markdown_mcp.models.response import SearchResult

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search operations."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SIMILAR = "similar"


@dataclass
class SearchOptions:
    """Search configuration options."""

    include_content: bool = True
    include_metadata: bool = True
    highlight_matches: bool = False
    max_content_length: int = 500
    boost_recent: bool = False
    boost_factor: float = 1.1


@dataclass
class SearchResults:
    """Container for search results and metadata."""

    results: list[SearchResult]
    total_count: int
    search_time_ms: int
    search_type: SearchType
    query_embedding: list[float] | None = None


class SearchEngine:
    """Main search engine for document retrieval."""

    def __init__(self, db_session: Session):
        """Initialize search engine with database session."""
        self.db = db_session
        # Import here to avoid circular imports
        from pdf_to_markdown_mcp.services.embeddings import create_embedding_service

        self.embedding_service = create_embedding_service()
        self.search_queries = SearchQueries()

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
        include_content: bool = True,
        options: SearchOptions | None = None,
    ) -> SearchResults:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Additional filtering criteria
            include_content: Whether to include chunk content
            options: Additional search options

        Returns:
            SearchResults with matching documents

        Raises:
            SearchError: If search operation fails
            EmbeddingError: If query embedding generation fails
        """
        start_time = time.time()

        if options is None:
            options = SearchOptions(include_content=include_content)

        logger.info(
            "Semantic search requested",
            extra={
                "query_length": len(query),
                "top_k": top_k,
                "threshold": threshold,
                "filters": filters,
            },
        )

        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)

            if not query_embedding:
                raise EmbeddingError("Failed to generate query embedding")

            # Perform vector similarity search
            raw_results = self.search_queries.vector_similarity_search(
                db=self.db,
                query_embedding=query_embedding,
                limit=top_k,
                threshold=threshold,
                filters=filters,
                include_content=options.include_content,
            )

            # Convert to SearchResult objects
            search_results = []
            for idx, result in enumerate(raw_results):
                search_result = SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    filename=result.filename,
                    source_path=result.source_path,
                    title=(
                        self._extract_title(result.content) if result.content else None
                    ),
                    content=(
                        self._truncate_content(
                            result.content, options.max_content_length
                        )
                        if options.include_content
                        else None
                    ),
                    similarity_score=result.similarity,
                    rank=idx + 1,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    metadata=result.metadata if options.include_metadata else None,
                )

                if options.highlight_matches and search_result.content:
                    search_result.content = self._highlight_matches(
                        search_result.content, query
                    )

                search_results.append(search_result)

            search_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Semantic search completed",
                extra={
                    "results_count": len(search_results),
                    "search_time_ms": search_time_ms,
                },
            )

            return SearchResults(
                results=search_results,
                total_count=len(search_results),
                search_time_ms=search_time_ms,
                search_type=SearchType.SEMANTIC,
                query_embedding=query_embedding,
            )

        except EmbeddingError:
            raise  # Re-raise embedding errors
        except Exception as e:
            logger.exception("Error in semantic search")
            raise SearchError(f"Semantic search failed: {e!s}")

    async def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_content: bool = True,
        options: SearchOptions | None = None,
    ) -> SearchResults:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            top_k: Number of results to return
            filters: Additional filtering criteria
            include_content: Whether to include chunk content
            options: Additional search options

        Returns:
            SearchResults with hybrid rankings

        Raises:
            SearchError: If search operation fails
        """
        start_time = time.time()

        if options is None:
            options = SearchOptions(include_content=include_content)

        # Validate weights
        if abs(semantic_weight + keyword_weight - 1.0) > 0.001:
            raise ValidationError("Semantic and keyword weights must sum to 1.0")

        logger.info(
            "Hybrid search requested",
            extra={
                "query": query,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "top_k": top_k,
            },
        )

        try:
            # Perform semantic search (get more results for reranking)
            semantic_results = await self.semantic_search(
                query=query,
                top_k=top_k * 2,  # Get extra results for reranking
                threshold=0.5,  # Lower threshold for hybrid
                filters=filters,
                include_content=include_content,
                options=options,
            )

            # Perform keyword search
            keyword_results = await self.keyword_search(
                query=query,
                top_k=top_k * 2,  # Get extra results for reranking
                filters=filters,
                include_content=include_content,
                options=options,
            )

            # Combine and rerank results
            combined_results = self._combine_search_results(
                semantic_results=semantic_results.results,
                keyword_results=keyword_results.results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

            # Take top K results
            final_results = combined_results[:top_k]

            # Update ranks
            for idx, result in enumerate(final_results):
                result.rank = idx + 1

            search_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Hybrid search completed",
                extra={
                    "results_count": len(final_results),
                    "search_time_ms": search_time_ms,
                },
            )

            return SearchResults(
                results=final_results,
                total_count=len(final_results),
                search_time_ms=search_time_ms,
                search_type=SearchType.HYBRID,
                query_embedding=semantic_results.query_embedding,
            )

        except Exception as e:
            logger.exception("Error in hybrid search")
            raise SearchError(f"Hybrid search failed: {e!s}")

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_content: bool = True,
        options: SearchOptions | None = None,
    ) -> SearchResults:
        """
        Perform full-text keyword search.

        Args:
            query: Keyword search query
            top_k: Number of results to return
            filters: Additional filtering criteria
            include_content: Whether to include chunk content
            options: Additional search options

        Returns:
            SearchResults with keyword matches
        """
        start_time = time.time()

        if options is None:
            options = SearchOptions(include_content=include_content)

        logger.info("Keyword search requested", extra={"query": query, "top_k": top_k})

        try:
            # Perform full-text search
            raw_results = self.search_queries.fulltext_search(
                db=self.db,
                query=query,
                limit=top_k,
                filters=filters,
                include_content=options.include_content,
            )

            # Convert to SearchResult objects
            search_results = []
            for idx, result in enumerate(raw_results):
                search_result = SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    filename=result.filename,
                    source_path=result.source_path,
                    title=(
                        self._extract_title(result.content) if result.content else None
                    ),
                    content=(
                        self._truncate_content(
                            result.content, options.max_content_length
                        )
                        if options.include_content
                        else None
                    ),
                    similarity_score=result.score,  # Text search score
                    rank=idx + 1,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    metadata=result.metadata if options.include_metadata else None,
                )

                if options.highlight_matches and search_result.content:
                    search_result.content = self._highlight_matches(
                        search_result.content, query
                    )

                search_results.append(search_result)

            search_time_ms = int((time.time() - start_time) * 1000)

            return SearchResults(
                results=search_results,
                total_count=len(search_results),
                search_time_ms=search_time_ms,
                search_type=SearchType.KEYWORD,
            )

        except Exception as e:
            logger.exception("Error in keyword search")
            raise SearchError(f"Keyword search failed: {e!s}")

    async def find_similar(
        self,
        document_id: int,
        top_k: int = 5,
        min_similarity: float = 0.6,
        include_self: bool = False,
        options: SearchOptions | None = None,
    ) -> SearchResults:
        """
        Find documents similar to a reference document.

        Args:
            document_id: Reference document ID
            top_k: Number of similar documents to return
            min_similarity: Minimum similarity threshold
            include_self: Whether to include reference document
            options: Additional search options

        Returns:
            SearchResults with similar documents
        """
        start_time = time.time()

        if options is None:
            options = SearchOptions()

        logger.info(
            "Similar document search requested",
            extra={
                "document_id": document_id,
                "top_k": top_k,
                "min_similarity": min_similarity,
            },
        )

        try:
            # Get document embedding
            doc_embeddings = self.search_queries.get_document_embeddings(
                self.db, document_id
            )

            if not doc_embeddings:
                raise ValidationError(
                    f"Document not found or has no embeddings: {document_id}"
                )

            # Use the first embedding as reference (could be averaged)
            reference_embedding = doc_embeddings[0].embedding

            # Find similar documents
            raw_results = self.search_queries.find_similar_documents(
                db=self.db,
                reference_embedding=reference_embedding,
                reference_doc_id=document_id if not include_self else None,
                top_k=top_k,
                threshold=min_similarity,
            )

            # Convert to SearchResult objects
            search_results = []
            for idx, result in enumerate(raw_results):
                search_result = SearchResult(
                    document_id=result.document_id,
                    chunk_id=None,  # Document-level similarity
                    filename=result.filename,
                    source_path=result.source_path,
                    title=result.title,
                    content=None,  # Not applicable for document similarity
                    similarity_score=result.similarity,
                    rank=idx + 1,
                    page_number=None,  # Not applicable for document similarity
                    chunk_index=None,  # Not applicable for document similarity
                    metadata=result.metadata if options.include_metadata else None,
                )

                search_results.append(search_result)

            search_time_ms = int((time.time() - start_time) * 1000)

            return SearchResults(
                results=search_results,
                total_count=len(search_results),
                search_time_ms=search_time_ms,
                search_type=SearchType.SIMILAR,
            )

        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.exception("Error in similar document search")
            raise SearchError(f"Similar document search failed: {e!s}")

    def _combine_search_results(
        self,
        semantic_results: list[SearchResult],
        keyword_results: list[SearchResult],
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        """
        Combine and rerank semantic and keyword search results.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores

        Returns:
            Combined and reranked results
        """
        # Create lookup for semantic scores
        semantic_scores = {
            (r.document_id, r.chunk_id): r.similarity_score for r in semantic_results
        }

        # Create lookup for keyword scores
        keyword_scores = {
            (r.document_id, r.chunk_id): r.similarity_score for r in keyword_results
        }

        # Get all unique document-chunk pairs
        all_pairs = set(semantic_scores.keys()) | set(keyword_scores.keys())

        combined_results = []
        for doc_id, chunk_id in all_pairs:
            # Get scores (default to 0 if not found)
            semantic_score = semantic_scores.get((doc_id, chunk_id), 0.0)
            keyword_score = keyword_scores.get((doc_id, chunk_id), 0.0)

            # Calculate combined score
            combined_score = (
                semantic_score * semantic_weight + keyword_score * keyword_weight
            )

            # Find the result object (prefer semantic result)
            result_obj = None
            for r in semantic_results:
                if r.document_id == doc_id and r.chunk_id == chunk_id:
                    result_obj = r
                    break

            if not result_obj:
                for r in keyword_results:
                    if r.document_id == doc_id and r.chunk_id == chunk_id:
                        result_obj = r
                        break

            if result_obj:
                # Update similarity score to combined score
                result_obj.similarity_score = combined_score
                combined_results.append(result_obj)

        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)

        return combined_results

    def _extract_title(self, content: str) -> str | None:
        """Extract title from content (first line or heading)."""
        if not content:
            return None

        lines = content.strip().split("\n")
        first_line = lines[0].strip()

        # Check if first line looks like a title
        if len(first_line) < 100 and not first_line.endswith("."):
            return first_line

        return None

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to maximum length with ellipsis."""
        if not content or len(content) <= max_length:
            return content

        # Find last complete word within limit
        truncated = content[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _highlight_matches(self, content: str, query: str) -> str:
        """
        Highlight query terms in content (simple implementation).

        In production, this could use more sophisticated highlighting.
        """
        import re

        # Simple word-based highlighting
        query_words = query.lower().split()
        highlighted_content = content

        for word in query_words:
            if len(word) > 2:  # Only highlight meaningful words
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted_content = pattern.sub(f"**{word}**", highlighted_content)

        return highlighted_content

    async def get_search_suggestions(
        self, partial_query: str, limit: int = 5
    ) -> list[str]:
        """
        Get search query suggestions based on content.

        Args:
            partial_query: Partial query string
            limit: Maximum suggestions to return

        Returns:
            List of query suggestions
        """
        try:
            # TODO: Implement sophisticated query suggestions
            # This could include:
            # - Common terms from documents
            # - Previous successful queries
            # - Document titles and headings
            # - Auto-complete based on partial matches

            # Simple implementation for now
            suggestions = [
                f"{partial_query} algorithm",
                f"{partial_query} method",
                f"{partial_query} analysis",
                f"{partial_query} implementation",
                f"{partial_query} theory",
            ]

            return suggestions[:limit]

        except Exception as e:
            logger.warning(f"Error generating search suggestions: {e}")
            return []
