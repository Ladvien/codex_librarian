"""
Search API endpoints.

Implements the semantic_search, hybrid_search, and find_similar MCP tools.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.core.search_engine import SearchEngine
from pdf_to_markdown_mcp.db.session import get_db
from pdf_to_markdown_mcp.models.request import (
    FindSimilarRequest,
    HybridSearchRequest,
    SemanticSearchRequest,
)
from pdf_to_markdown_mcp.models.response import ErrorResponse, ErrorType, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/semantic_search", response_model=SearchResponse)
async def semantic_search(
    request: SemanticSearchRequest, db: Session = Depends(get_db)
) -> SearchResponse:
    """
    Search documents using natural language queries with vector similarity.

    This endpoint implements the semantic_search MCP tool functionality.
    """
    try:
        logger.info(
            "Semantic search requested",
            extra={
                "query": request.query,
                "top_k": request.top_k,
                "threshold": request.threshold,
            },
        )

        search_engine = SearchEngine(db)

        # Perform semantic search
        results = await search_engine.semantic_search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filter,
            include_content=request.include_content,
        )

        return SearchResponse(
            success=True,
            query=request.query,
            results=results.results,
            total_results=results.total_count,
            search_time_ms=results.search_time_ms,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filter,
        )

    except ValueError as e:
        logger.warning(f"Invalid search query: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorType.VALIDATION,
                message=f"Invalid search parameters: {e!s}",
            ).dict(),
        )

    except Exception as e:
        logger.exception("Unexpected error in semantic search")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.EMBEDDING, message=f"Search failed: {e!s}"
            ).dict(),
        )


@router.post("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest, db: Session = Depends(get_db)
) -> SearchResponse:
    """
    Combine vector semantic search with full-text search for best results.

    This endpoint implements the hybrid_search MCP tool functionality.
    """
    try:
        logger.info(
            "Hybrid search requested",
            extra={
                "query": request.query,
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight,
                "top_k": request.top_k,
            },
        )

        search_engine = SearchEngine(db)

        # Perform hybrid search
        results = await search_engine.hybrid_search(
            query=request.query,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            top_k=request.top_k,
            filters=request.filter,
            include_content=request.include_content,
        )

        return SearchResponse(
            success=True,
            query=request.query,
            results=results.results,
            total_results=results.total_count,
            search_time_ms=results.search_time_ms,
            top_k=request.top_k,
            threshold=None,  # Not applicable for hybrid search
            filters=request.filter,
        )

    except ValueError as e:
        logger.warning(f"Invalid hybrid search parameters: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorType.VALIDATION,
                message=f"Invalid search parameters: {e!s}",
            ).dict(),
        )

    except Exception as e:
        logger.exception("Unexpected error in hybrid search")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.EMBEDDING, message=f"Hybrid search failed: {e!s}"
            ).dict(),
        )


@router.post("/find_similar", response_model=SearchResponse)
async def find_similar_documents(
    request: FindSimilarRequest, db: Session = Depends(get_db)
) -> SearchResponse:
    """
    Find documents similar to a given reference document.

    This endpoint implements the find_similar MCP tool functionality.
    """
    try:
        logger.info(
            "Similar document search requested",
            extra={
                "document_id": request.document_id,
                "top_k": request.top_k,
                "min_similarity": request.min_similarity,
            },
        )

        search_engine = SearchEngine(db)

        # Find similar documents
        results = await search_engine.find_similar(
            document_id=request.document_id,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            include_self=request.include_self,
        )

        return SearchResponse(
            success=True,
            query=f"Similar to document {request.document_id}",
            results=results.results,
            total_results=results.total_count,
            search_time_ms=results.search_time_ms,
            top_k=request.top_k,
            threshold=request.min_similarity,
            filters={"reference_document_id": request.document_id},
        )

    except ValueError as e:
        logger.warning(f"Invalid document similarity request: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorType.VALIDATION,
                message=f"Invalid similarity search parameters: {e!s}",
            ).dict(),
        )

    except Exception as e:
        logger.exception("Unexpected error in document similarity search")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.DATABASE, message=f"Similarity search failed: {e!s}"
            ).dict(),
        )


# Additional search utility endpoints


@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str, limit: int = 5, db: Session = Depends(get_db)
) -> list[str]:
    """
    Get search query suggestions based on existing content.

    This is a utility endpoint to help users formulate better queries.
    """
    try:
        # Get document titles and content for suggestions

        from pdf_to_markdown_mcp.db.models import Document

        suggestions = []

        # Get document titles that match the partial query
        title_matches = (
            db.query(Document.filename)
            .filter(Document.filename.ilike(f"%{query}%"))
            .limit(3)
            .all()
        )

        for match in title_matches:
            # Extract meaningful terms from filename
            filename = (
                match.filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
            )
            if filename not in suggestions:
                suggestions.append(filename)

        # Add some common research-related suggestions
        common_terms = [
            f"{query} algorithm",
            f"{query} method",
            f"{query} analysis",
            f"{query} implementation",
            f"{query} theory",
        ]

        # Combine and deduplicate
        for term in common_terms:
            if term not in suggestions and len(suggestions) < limit:
                suggestions.append(term)

        return suggestions[:limit]

    except Exception as e:
        logger.exception("Error generating search suggestions")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to generate suggestions: {e!s}",
            ).dict(),
        )


@router.get("/search/stats")
async def get_search_stats(db: Session = Depends(get_db)) -> dict:
    """
    Get search-related statistics and corpus information.
    """
    try:
        from sqlalchemy import func, text

        from pdf_to_markdown_mcp.db.models import (
            DocumentContent,
            DocumentEmbedding,
        )
        from pdf_to_markdown_mcp.db.queries import DocumentQueries

        # Get document statistics
        doc_stats = DocumentQueries.get_statistics(db)
        total_documents = doc_stats.get("total_documents", 0)

        # Get embedding count
        total_embeddings = db.query(DocumentEmbedding).count()

        # Get content statistics
        content_stats = db.query(
            func.count(DocumentContent.id).label("total_content"),
            func.avg(DocumentContent.page_count).label("avg_pages"),
            func.sum(DocumentContent.page_count).label("total_pages"),
        ).first()

        # Check index health (simplified)
        index_status = "healthy"
        try:
            # Test a simple vector query to check if PGVector is working
            test_query = text(
                "SELECT COUNT(*) FROM document_embeddings WHERE embedding IS NOT NULL"
            )
            db.execute(test_query)
        except Exception:
            index_status = "degraded"

        return {
            "total_documents": total_documents,
            "total_embeddings": total_embeddings,
            "total_content_chunks": content_stats.total_content if content_stats else 0,
            "total_pages": content_stats.total_pages if content_stats else 0,
            "avg_pages_per_document": (
                float(content_stats.avg_pages)
                if content_stats and content_stats.avg_pages
                else 0.0
            ),
            "index_status": index_status,
            "avg_search_time_ms": 50.0,  # This would need to be tracked over time
            "popular_queries": [],  # This would need query logging to implement
            "documents_by_status": doc_stats.get("by_status", {}),
        }

    except Exception as e:
        logger.exception("Error retrieving search statistics")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.DATABASE,
                message=f"Failed to retrieve search statistics: {e!s}",
            ).dict(),
        )
