"""
Tests for MCP server search functionality.

Tests search_library tool, hybrid search algorithm, and error handling.
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set required environment variables before importing server module
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

from src.pdf_to_markdown_mcp.mcp.config import MCPConfig
from src.pdf_to_markdown_mcp.mcp.context import DatabasePool


class TestGenerateQueryEmbedding:
    """Test query embedding generation via Ollama."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self) -> None:
        """Test successful embedding generation."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"embedding": [0.1] * 768})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            embedding = await server.generate_query_embedding("test query")

            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_uses_config(self) -> None:
        """Test embedding generation uses configuration."""
        from src.pdf_to_markdown_mcp.mcp import server

        # Mock config
        with patch.object(server, "config") as mock_config:
            mock_config.ollama_url = "http://test:11434"
            mock_config.ollama_model = "test-model"

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = MagicMock(return_value={"embedding": [0.1] * 768})

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("httpx.AsyncClient", return_value=mock_client):
                await server.generate_query_embedding("test query")

                # Verify correct URL and model used
                call_args = mock_client.post.call_args
                assert call_args[0][0] == "http://test:11434/api/embeddings"
                assert call_args[1]["json"]["model"] == "test-model"
                assert call_args[1]["json"]["prompt"] == "test query"

    @pytest.mark.asyncio
    async def test_generate_embedding_http_error(self) -> None:
        """Test embedding generation handles HTTP errors."""
        from src.pdf_to_markdown_mcp.mcp import server
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("HTTP Error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                await server.generate_query_embedding("test query")

    @pytest.mark.asyncio
    async def test_generate_embedding_missing_httpx(self) -> None:
        """Test embedding generation fails gracefully without httpx."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(server, "httpx", None):
            with pytest.raises(RuntimeError, match="httpx package not installed"):
                await server.generate_query_embedding("test query")

    @pytest.mark.asyncio
    async def test_generate_embedding_invalid_response(self) -> None:
        """Test embedding generation handles invalid response format."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"invalid": "response"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Invalid embedding response"):
                await server.generate_query_embedding("test query")


class TestPerformHybridSearch:
    """Test hybrid search combining vector and keyword search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self) -> None:
        """Test hybrid search combines vector and keyword results."""
        from src.pdf_to_markdown_mcp.mcp import server

        # Mock database pool
        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        # Mock vector search results
        vector_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": "# Document 1\nContent about neural networks",
                "similarity": 0.95,
                "content": "Content about neural networks",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]

        # Mock keyword search results
        keyword_results = [
            {
                "document_id": 2,
                "filename": "doc2.pdf",
                "source_path": "/path/doc2.pdf",
                "markdown_path": "/output/doc2.md",
                "created_at": datetime(2025, 1, 2),
                "full_content": "# Document 2\nKeyword matching content",
                "rank": 0.85,
                "content": "Keyword matching content",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]

        mock_conn.fetch = AsyncMock(side_effect=[vector_results, keyword_results])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            results = await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="neural networks",
                limit=10,
                min_similarity=0.7,
            )

            # Should return both documents
            assert len(results) == 2
            assert any(r["document_id"] == 1 for r in results)
            assert any(r["document_id"] == 2 for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_scoring(self) -> None:
        """Test Reciprocal Rank Fusion scoring."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        # Document appears in both vector and keyword results (should rank higher)
        vector_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": "# Document 1",
                "similarity": 0.95,
                "content": "High similarity",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "document_id": 2,
                "filename": "doc2.pdf",
                "source_path": "/path/doc2.pdf",
                "markdown_path": "/output/doc2.md",
                "created_at": datetime(2025, 1, 2),
                "full_content": "# Document 2",
                "similarity": 0.80,
                "content": "Medium similarity",
                "page_number": 1,
                "chunk_index": 0,
            },
        ]

        keyword_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": "# Document 1",
                "rank": 0.90,
                "content": "High keyword match",
                "page_number": 1,
                "chunk_index": 0,
            },
        ]

        mock_conn.fetch = AsyncMock(side_effect=[vector_results, keyword_results])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            results = await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="test query",
                limit=10,
                min_similarity=0.7,
            )

            # Document 1 should rank first (appears in both searches)
            assert results[0]["document_id"] == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_filters_by_similarity(self) -> None:
        """Test hybrid search filters results by similarity threshold."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        vector_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": "# Document 1",
                "similarity": 0.95,
                "content": "High similarity",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "document_id": 2,
                "filename": "doc2.pdf",
                "source_path": "/path/doc2.pdf",
                "markdown_path": "/output/doc2.md",
                "created_at": datetime(2025, 1, 2),
                "full_content": "# Document 2",
                "similarity": 0.65,  # Below threshold
                "content": "Low similarity",
                "page_number": 1,
                "chunk_index": 0,
            },
        ]

        mock_conn.fetch = AsyncMock(side_effect=[vector_results, []])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            results = await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="test query",
                limit=10,
                min_similarity=0.7,
            )

            # Only doc1 should pass threshold
            assert len(results) == 1
            assert results[0]["document_id"] == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_date_filtering(self) -> None:
        """Test hybrid search applies date filters."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="test query",
                limit=10,
                min_similarity=0.7,
                date_from="2025-01-01T00:00:00Z",
                date_to="2025-12-31T23:59:59Z",
            )

            # Verify fetch was called with date parameters
            fetch_calls = mock_conn.fetch.call_args_list
            assert len(fetch_calls) == 2  # Vector and keyword searches

            # Check that date parameters were passed
            for call in fetch_calls:
                args = call[0]
                # Should have embedding/query + limit + date_from + date_to
                assert len(args) >= 4

    @pytest.mark.asyncio
    async def test_hybrid_search_limits_results(self) -> None:
        """Test hybrid search respects limit parameter."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        # Create 10 mock results
        vector_results = [
            {
                "document_id": i,
                "filename": f"doc{i}.pdf",
                "source_path": f"/path/doc{i}.pdf",
                "markdown_path": f"/output/doc{i}.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": f"# Document {i}",
                "similarity": 0.9 - (i * 0.05),
                "content": f"Content {i}",
                "page_number": 1,
                "chunk_index": 0,
            }
            for i in range(10)
        ]

        mock_conn.fetch = AsyncMock(side_effect=[vector_results, []])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            results = await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="test query",
                limit=3,
                min_similarity=0.0,
            )

            # Should return at most 3 results
            assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_full_content(self) -> None:
        """Test hybrid search returns complete document content."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_conn = AsyncMock()

        full_markdown = "# Complete Document\n\n" + ("Content " * 100)

        vector_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "created_at": datetime(2025, 1, 1),
                "full_content": full_markdown,
                "similarity": 0.95,
                "content": "Excerpt from the document",
                "page_number": 5,
                "chunk_index": 12,
            }
        ]

        mock_conn.fetch = AsyncMock(side_effect=[vector_results, []])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(server, "ensure_db_pool", return_value=mock_pool):
            results = await server.perform_hybrid_search(
                query_embedding=[0.1] * 768,
                query_text="test query",
                limit=10,
                min_similarity=0.7,
            )

            assert len(results) == 1
            # Full content should be included
            assert results[0]["full_content"] == full_markdown
            # Excerpt should be separate and truncated if needed
            assert "best_excerpt" in results[0]
            assert results[0]["page_number"] == 5
            assert results[0]["chunk_index"] == 12


class TestSearchLibraryTool:
    """Test search_library MCP tool function."""

    @pytest.mark.asyncio
    async def test_search_library_basic_query(self) -> None:
        """Test basic search query execution."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_results = [
            {
                "document_id": 1,
                "filename": "doc1.pdf",
                "source_path": "/path/doc1.pdf",
                "markdown_path": "/output/doc1.md",
                "similarity_score": 0.95,
                "full_content": "# Document 1",
                "best_excerpt": "Matching content",
                "page_number": 1,
                "chunk_index": 0,
                "created_at": "2025-01-01T00:00:00",
            }
        ]

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=mock_results):

            result = await server._search_library_impl(query="neural networks")

            assert result["query"] == "neural networks"
            assert result["total_results"] == 1
            assert len(result["results"]) == 1
            assert "search_duration_ms" in result

    @pytest.mark.asyncio
    async def test_search_library_uses_defaults(self) -> None:
        """Test search_library applies default parameters."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]) as mock_search:

            await server._search_library_impl(query="test")

            # Verify defaults were used
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["limit"] == server.config.search_default_limit
            assert call_kwargs["min_similarity"] == server.config.search_default_similarity

    @pytest.mark.asyncio
    async def test_search_library_custom_parameters(self) -> None:
        """Test search_library accepts custom parameters."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]) as mock_search:

            await server._search_library_impl(
                query="test", limit=5, min_similarity=0.85, date_from="2025-01-01T00:00:00Z"
            )

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["limit"] == 5
            assert call_kwargs["min_similarity"] == 0.85
            assert call_kwargs["date_from"] == "2025-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_search_library_validates_empty_query(self) -> None:
        """Test search_library rejects empty queries."""
        from src.pdf_to_markdown_mcp.mcp import server

        with pytest.raises(ValueError, match="Query parameter is required"):
            await server._search_library_impl(query="")

        with pytest.raises(ValueError, match="Query parameter is required"):
            await server._search_library_impl(query="   ")

    @pytest.mark.asyncio
    async def test_search_library_validates_limit_range(self) -> None:
        """Test search_library validates limit parameter."""
        from src.pdf_to_markdown_mcp.mcp import server

        with pytest.raises(ValueError, match="Limit must be between 1 and"):
            await server._search_library_impl(query="test", limit=0)

        with pytest.raises(ValueError, match="Limit must be between 1 and"):
            await server._search_library_impl(query="test", limit=1000)

    @pytest.mark.asyncio
    async def test_search_library_validates_similarity_range(self) -> None:
        """Test search_library validates similarity threshold."""
        from src.pdf_to_markdown_mcp.mcp import server

        with pytest.raises(ValueError, match="min_similarity must be between 0.0 and 1.0"):
            await server._search_library_impl(query="test", min_similarity=-0.1)

        with pytest.raises(ValueError, match="min_similarity must be between 0.0 and 1.0"):
            await server._search_library_impl(query="test", min_similarity=1.5)

    @pytest.mark.asyncio
    async def test_search_library_logs_warning_for_tags(self) -> None:
        """Test search_library logs warning when tags parameter is used."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]), patch.object(
            server.logger, "warning"
        ) as mock_warning:

            await server._search_library_impl(query="test", tags=["tag1", "tag2"])

            # Should log warning about unimplemented feature
            mock_warning.assert_called()
            assert "not yet implemented" in str(mock_warning.call_args)

    @pytest.mark.asyncio
    async def test_search_library_handles_errors(self) -> None:
        """Test search_library handles and re-raises errors."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", side_effect=Exception("Embedding failed")
        ):

            with pytest.raises(Exception, match="Embedding failed"):
                await server._search_library_impl(query="test")

    @pytest.mark.asyncio
    async def test_search_library_logs_to_context(self) -> None:
        """Test search_library logs to MCP context when provided."""
        from src.pdf_to_markdown_mcp.mcp import server

        mock_ctx = AsyncMock()

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]):

            await server._search_library_impl(query="test", ctx=mock_ctx)

            # Verify context logging was called
            mock_ctx.info.assert_called()

    @pytest.mark.asyncio
    async def test_search_library_measures_duration(self) -> None:
        """Test search_library measures and returns search duration."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]):

            result = await server._search_library_impl(query="test")

            assert "search_duration_ms" in result
            assert isinstance(result["search_duration_ms"], (int, float))
            assert result["search_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_search_library_empty_results(self) -> None:
        """Test search_library handles empty results."""
        from src.pdf_to_markdown_mcp.mcp import server

        with patch.object(
            server, "generate_query_embedding", return_value=[0.1] * 768
        ), patch.object(server, "perform_hybrid_search", return_value=[]):

            result = await server._search_library_impl(query="nonexistent query")

            assert result["total_results"] == 0
            assert result["results"] == []
            assert result["query"] == "nonexistent query"


class TestEnsureDbPool:
    """Test database pool initialization."""

    @pytest.mark.asyncio
    async def test_ensure_db_pool_creates_pool(self) -> None:
        """Test ensure_db_pool creates pool on first call."""
        from src.pdf_to_markdown_mcp.mcp import server

        # Reset global pool
        original_pool = server.db_pool
        server.db_pool = None

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_pool.connect = AsyncMock()
        mock_pool.health_check = AsyncMock(return_value=True)

        try:
            with patch.object(server, "DatabasePool", return_value=mock_pool):
                pool = await server.ensure_db_pool()

                assert pool is not None
                mock_pool.connect.assert_called_once()
                mock_pool.health_check.assert_called_once()

        finally:
            # Restore original pool
            server.db_pool = original_pool

    @pytest.mark.asyncio
    async def test_ensure_db_pool_reuses_existing(self) -> None:
        """Test ensure_db_pool reuses existing pool."""
        from src.pdf_to_markdown_mcp.mcp import server

        # Set up existing pool
        original_pool = server.db_pool
        mock_pool = AsyncMock(spec=DatabasePool)
        server.db_pool = mock_pool

        try:
            pool = await server.ensure_db_pool()

            # Should return same pool without creating new one
            assert pool == mock_pool

        finally:
            # Restore original pool
            server.db_pool = original_pool

    @pytest.mark.asyncio
    async def test_ensure_db_pool_fails_on_unhealthy(self) -> None:
        """Test ensure_db_pool raises error when health check fails."""
        from src.pdf_to_markdown_mcp.mcp import server

        # Reset global pool
        original_pool = server.db_pool
        server.db_pool = None

        mock_pool = AsyncMock(spec=DatabasePool)
        mock_pool.connect = AsyncMock()
        mock_pool.health_check = AsyncMock(return_value=False)

        try:
            with patch.object(server, "DatabasePool", return_value=mock_pool):
                with pytest.raises(RuntimeError, match="Database connection failed"):
                    await server.ensure_db_pool()

        finally:
            # Restore original pool
            server.db_pool = original_pool
