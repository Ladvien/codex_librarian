"""
FastMCP server for codex_librarian semantic search.

Provides the `search_library` tool for LLMs to semantically search
the document library using hybrid search (vector + BM25).

All configuration comes from MCP client environment variables.
No .env file required.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Any

from fastmcp import FastMCP, Context

from .config import MCPConfig
from .context import DatabasePool

# Try to import httpx for Ollama API calls
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

# Initialize configuration from environment
try:
    config = MCPConfig.from_env()
    config.validate()
except ValueError as e:
    print(f"Configuration error: {e}", file=sys.stderr)
    print("\nRequired environment variables:", file=sys.stderr)
    print("  DATABASE_URL: PostgreSQL connection string", file=sys.stderr)
    print("  OLLAMA_URL: Ollama API endpoint (default: http://localhost:11434)", file=sys.stderr)
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="codex-librarian",
    version="0.1.0",
)

# Global database pool
db_pool: DatabasePool | None = None


async def generate_query_embedding(query: str) -> list[float]:
    """
    Generate embedding for search query using Ollama.

    Args:
        query: Search query text

    Returns:
        Query embedding vector

    Raises:
        RuntimeError: If embedding generation fails
    """
    if httpx is None:
        raise RuntimeError("httpx package not installed. Install with: uv add httpx")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{config.ollama_url}/api/embeddings",
                json={
                    "model": config.ollama_model,
                    "prompt": query,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    except httpx.HTTPError as e:
        logger.error(f"Ollama API error: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e
    except KeyError as e:
        logger.error(f"Unexpected Ollama response format: {e}")
        raise RuntimeError(f"Invalid embedding response: {e}") from e


async def perform_hybrid_search(
    query_embedding: list[float],
    query_text: str,
    limit: int,
    min_similarity: float,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and full-text search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings.

    Args:
        query_embedding: Query embedding vector
        query_text: Original query text for keyword search
        limit: Maximum number of results
        min_similarity: Minimum similarity threshold
        date_from: Filter documents after this date (ISO 8601)
        date_to: Filter documents before this date (ISO 8601)

    Returns:
        List of search results with metadata
    """
    # Ensure database pool is initialized
    pool = await ensure_db_pool()

    # Build date filter clause
    date_filter = ""
    date_params = []
    param_idx = 2  # Start after query_embedding and limit

    if date_from:
        date_filter += f" AND d.created_at >= ${param_idx}"
        date_params.append(datetime.fromisoformat(date_from.replace("Z", "+00:00")))
        param_idx += 1

    if date_to:
        date_filter += f" AND d.created_at <= ${param_idx}"
        date_params.append(datetime.fromisoformat(date_to.replace("Z", "+00:00")))
        param_idx += 1

    # Vector similarity search - deduplicates by document and fetches full content
    vector_query = f"""
        WITH ranked_chunks AS (
            SELECT
                de.document_id,
                MAX(1 - (de.embedding <=> $1::vector)) as best_similarity,
                (ARRAY_AGG(de.chunk_text ORDER BY (de.embedding <=> $1::vector)))[1] as best_excerpt,
                (ARRAY_AGG(de.page_number ORDER BY (de.embedding <=> $1::vector)))[1] as best_page,
                (ARRAY_AGG(de.chunk_index ORDER BY (de.embedding <=> $1::vector)))[1] as best_chunk_index
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE d.conversion_status = 'completed'
            {date_filter}
            GROUP BY de.document_id
        )
        SELECT
            d.id as document_id,
            d.filename,
            d.source_path,
            d.output_path as markdown_path,
            d.created_at,
            dc.markdown_content as full_content,
            rc.best_similarity as similarity,
            rc.best_excerpt as content,
            rc.best_page as page_number,
            rc.best_chunk_index as chunk_index
        FROM ranked_chunks rc
        JOIN documents d ON d.id = rc.document_id
        JOIN document_content dc ON dc.document_id = d.id
        ORDER BY rc.best_similarity DESC
        LIMIT $2
    """

    # Full-text keyword search - deduplicates by document and fetches full content
    keyword_query = f"""
        WITH ranked_chunks AS (
            SELECT
                de.document_id,
                MAX(ts_rank_cd(
                    to_tsvector('english', de.chunk_text),
                    plainto_tsquery('english', $1)
                )) as best_rank,
                (ARRAY_AGG(de.chunk_text ORDER BY ts_rank_cd(
                    to_tsvector('english', de.chunk_text),
                    plainto_tsquery('english', $1)
                ) DESC))[1] as best_excerpt,
                (ARRAY_AGG(de.page_number ORDER BY ts_rank_cd(
                    to_tsvector('english', de.chunk_text),
                    plainto_tsquery('english', $1)
                ) DESC))[1] as best_page,
                (ARRAY_AGG(de.chunk_index ORDER BY ts_rank_cd(
                    to_tsvector('english', de.chunk_text),
                    plainto_tsquery('english', $1)
                ) DESC))[1] as best_chunk_index
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE
                to_tsvector('english', de.chunk_text) @@
                plainto_tsquery('english', $1)
                AND d.conversion_status = 'completed'
            {date_filter}
            GROUP BY de.document_id
        )
        SELECT
            d.id as document_id,
            d.filename,
            d.source_path,
            d.output_path as markdown_path,
            d.created_at,
            dc.markdown_content as full_content,
            rc.best_rank as rank,
            rc.best_excerpt as content,
            rc.best_page as page_number,
            rc.best_chunk_index as chunk_index
        FROM ranked_chunks rc
        JOIN documents d ON d.id = rc.document_id
        JOIN document_content dc ON dc.document_id = d.id
        ORDER BY rc.best_rank DESC
        LIMIT $2
    """

    # Convert embedding list to string format for pgvector
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    async with pool.acquire() as conn:
        # Execute both searches sequentially (asyncpg doesn't support concurrent ops on same connection)
        vector_results = await conn.fetch(vector_query, embedding_str, limit * 2, *date_params)
        keyword_results = await conn.fetch(keyword_query, query_text, limit * 2, *date_params)

    # Convert to dictionaries
    vector_docs = [dict(row) for row in vector_results]
    keyword_docs = [dict(row) for row in keyword_results]

    # Reciprocal Rank Fusion (RRF)
    rrf_scores: dict[int, dict[str, Any]] = {}
    k = 60  # RRF constant

    # Add vector search scores
    for rank, doc in enumerate(vector_docs, start=1):
        doc_id = doc["document_id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "doc": doc,
                "score": 0.0,
                "vector_rank": rank,
                "keyword_rank": None,
            }
        rrf_scores[doc_id]["score"] += 1 / (k + rank)

    # Add keyword search scores
    for rank, doc in enumerate(keyword_docs, start=1):
        doc_id = doc["document_id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "doc": doc,
                "score": 0.0,
                "vector_rank": None,
                "keyword_rank": rank,
            }
        else:
            rrf_scores[doc_id]["keyword_rank"] = rank
        rrf_scores[doc_id]["score"] += 1 / (k + rank)

    # Sort by RRF score
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    # Filter by similarity threshold and format results
    final_results = []
    for item in ranked_results[:limit]:
        doc = item["doc"]
        similarity = doc.get("similarity", doc.get("rank", 0.0))

        # Apply similarity threshold
        if similarity < min_similarity:
            continue

        # Get full content and best matching excerpt
        full_content = doc.get("full_content", "")
        best_excerpt = doc.get("content", "")
        # Truncate excerpt for context (show which chunk matched)
        excerpt = best_excerpt[:500] + "..." if len(best_excerpt) > 500 else best_excerpt

        final_results.append({
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "source_path": doc["source_path"],
            "markdown_path": doc["markdown_path"],
            "similarity_score": round(float(similarity), 4),
            "full_content": full_content,  # Complete markdown document
            "best_excerpt": excerpt,  # Why it matched (best chunk)
            "page_number": doc.get("page_number"),
            "chunk_index": doc.get("chunk_index"),
            "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
        })

    return final_results


async def _search_library_impl(
    query: str,
    limit: int | None = None,
    min_similarity: float | None = None,
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Core implementation of search_library for testing."""
    start_time = time.time()

    # Apply defaults
    if limit is None:
        limit = config.search_default_limit
    if min_similarity is None:
        min_similarity = config.search_default_similarity

    # Validate parameters
    if not query or not query.strip():
        raise ValueError("Query parameter is required and cannot be empty")

    if limit < 1 or limit > config.search_max_limit:
        raise ValueError(
            f"Limit must be between 1 and {config.search_max_limit}, got {limit}"
        )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(
            f"min_similarity must be between 0.0 and 1.0, got {min_similarity}"
        )

    if tags:
        logger.warning("Tag filtering not yet implemented, ignoring tags parameter")

    # Log search request
    if ctx:
        await ctx.info(f"Searching for: '{query}' (limit={limit}, threshold={min_similarity})")

    logger.info(
        f"Search request: query='{query[:50]}...', limit={limit}, "
        f"min_similarity={min_similarity}"
    )

    try:
        # Generate query embedding
        query_embedding = await generate_query_embedding(query)

        # Perform hybrid search
        results = await perform_hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            limit=limit,
            min_similarity=min_similarity,
            date_from=date_from,
            date_to=date_to,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        response = {
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_duration_ms": duration_ms,
        }

        logger.info(
            f"Search completed: found {len(results)} documents in {duration_ms}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"Search failed: {e}")
        raise


@mcp.tool()
async def search_library(
    query: str,
    limit: int = None,  # type: ignore
    min_similarity: float = None,  # type: ignore
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Semantically search the document library using hybrid search (vector + BM25).

    Combines vector similarity search with keyword search using Reciprocal Rank
    Fusion (RRF) for state-of-the-art retrieval performance.

    Returns complete documents (full markdown content) ranked by relevance.
    Results are deduplicated by document - one result per document with the
    best matching chunk indicated.

    Args:
        query: Natural language search query (required)
        limit: Maximum number of DOCUMENTS to return (default: 10, max: 50)
        min_similarity: Minimum similarity threshold 0.0-1.0 (default: 0.7)
        tags: Filter by document tags - NOT YET IMPLEMENTED (future feature)
        date_from: Filter documents processed after this date (ISO 8601 format)
        date_to: Filter documents processed before this date (ISO 8601 format)
        ctx: FastMCP context for logging

    Returns:
        Dictionary containing:
        - results: List of matching documents with FULL content and metadata
        - query: Original search query
        - total_results: Number of documents returned
        - search_duration_ms: Time taken to execute search

    Example:
        {
            "results": [
                {
                    "document_id": 123,
                    "filename": "paper.pdf",
                    "markdown_path": "/path/to/output/paper.md",
                    "similarity_score": 0.92,
                    "full_content": "# Paper Title\n\n...complete markdown...",
                    "best_excerpt": "...the chunk that matched...",
                    "page_number": 5,
                    "chunk_index": 12,
                    "created_at": "2025-09-29T10:30:00Z"
                }
            ],
            "query": "transformer architectures",
            "total_results": 3,
            "search_duration_ms": 45
        }
    """
    # Delegate to implementation function
    return await _search_library_impl(
        query=query,
        limit=limit,
        min_similarity=min_similarity,
        tags=tags,
        date_from=date_from,
        date_to=date_to,
        ctx=ctx,
    )


async def ensure_db_pool() -> DatabasePool:
    """
    Ensure database pool is initialized.

    Lazy initialization on first use.

    Returns:
        Initialized DatabasePool
    """
    global db_pool

    if db_pool is None:
        logger.info("Initializing database pool")
        logger.info(f"Database: {config.database_url.split('@')[1]}")  # Hide credentials
        logger.info(f"Ollama: {config.ollama_url}")
        logger.info(f"Model: {config.ollama_model}")

        db_pool = DatabasePool(config)
        await db_pool.connect()

        # Verify database connection
        is_healthy = await db_pool.health_check()
        if not is_healthy:
            logger.error("Database health check failed")
            raise RuntimeError("Database connection failed")

        logger.info("Database pool initialized")

    return db_pool


if __name__ == "__main__":
    logger.info("Starting codex-librarian MCP server")

    # Run the MCP server
    mcp.run(transport="stdio")