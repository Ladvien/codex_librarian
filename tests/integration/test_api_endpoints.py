"""
Integration tests for API endpoints.

This module tests the FastAPI endpoints with real HTTP requests
and service integration following TDD principles.
"""

from unittest.mock import Mock, patch

import pytest
from httpx import AsyncClient

from src.pdf_to_markdown_mcp.db.models import (
    Document,
    DocumentContent,
    DocumentEmbedding,
)
from src.pdf_to_markdown_mcp.main import create_app
from tests.fixtures import (
    DocumentFactory,
    create_mock_embedding_service,
    create_sample_embeddings,
    create_temp_pdf,
)


@pytest.fixture
async def app():
    """Create FastAPI app for testing."""
    app = create_app()
    yield app


@pytest.fixture
async def client(app):
    """Create HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestConvertEndpoints:
    """Test PDF conversion API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convert_single_pdf_success(
        self, client, async_db_session, temp_directory
    ):
        """Test successful single PDF conversion."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        with (
            patch("src.pdf_to_markdown_mcp.api.convert.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.api.convert.process_pdf_document"
            ) as mock_task,
        ):
            mock_get_db.return_value = async_db_session
            mock_task.delay.return_value = Mock(id="task-123")

            request_data = {
                "file_path": str(pdf_path),
                "store_embeddings": True,
                "processing_options": {
                    "language": "en",
                    "chunk_for_embeddings": True,
                    "extract_tables": True,
                },
            }

            # When
            response = await client.post("/convert_single", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "task_id" in response_data
            assert "document_id" in response_data
            assert response_data["task_id"] == "task-123"

            # Verify document was created in database
            document = (
                await async_db_session.query(Document)
                .filter_by(file_path=str(pdf_path))
                .first()
            )
            assert document is not None
            assert document.status == "pending"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convert_single_pdf_file_not_found(self, client, async_db_session):
        """Test conversion with non-existent file."""
        # Given
        request_data = {"file_path": "/nonexistent/file.pdf", "store_embeddings": True}

        with patch("src.pdf_to_markdown_mcp.api.convert.get_db_session") as mock_get_db:
            mock_get_db.return_value = async_db_session

            # When
            response = await client.post("/convert_single", json=request_data)

            # Then
            assert response.status_code == 400
            response_data = response.json()
            assert response_data["success"] is False
            assert "not found" in response_data["error"].lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convert_single_pdf_duplicate_file(
        self, client, async_db_session, temp_directory
    ):
        """Test conversion with duplicate file (already processed)."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        # Create existing document
        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="completed"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        with patch("src.pdf_to_markdown_mcp.api.convert.get_db_session") as mock_get_db:
            mock_get_db.return_value = async_db_session

            request_data = {"file_path": str(pdf_path), "store_embeddings": True}

            # When
            response = await client.post("/convert_single", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert response_data["message"] == "Document already processed"
            assert response_data["document_id"] == document.id

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_convert_success(
        self, client, async_db_session, temp_directory
    ):
        """Test successful batch PDF conversion."""
        # Given
        pdf_paths = [
            create_temp_pdf(content=f"Document {i}", directory=temp_directory)
            for i in range(3)
        ]

        with (
            patch("src.pdf_to_markdown_mcp.api.convert.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.api.convert.batch_process_pdfs"
            ) as mock_task,
        ):
            mock_get_db.return_value = async_db_session
            mock_task.delay.return_value = Mock(id="batch-task-123")

            request_data = {
                "file_paths": [str(path) for path in pdf_paths],
                "store_embeddings": True,
                "processing_options": {"language": "en", "batch_size": 2},
            }

            # When
            response = await client.post("/batch_convert", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert response_data["task_id"] == "batch-task-123"
            assert response_data["total_files"] == 3
            assert len(response_data["document_ids"]) == 3

            # Verify documents were created
            for pdf_path in pdf_paths:
                document = (
                    await async_db_session.query(Document)
                    .filter_by(file_path=str(pdf_path))
                    .first()
                )
                assert document is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_convert_empty_list(self, client, async_db_session):
        """Test batch conversion with empty file list."""
        # Given
        request_data = {"file_paths": [], "store_embeddings": True}

        with patch("src.pdf_to_markdown_mcp.api.convert.get_db_session") as mock_get_db:
            mock_get_db.return_value = async_db_session

            # When
            response = await client.post("/batch_convert", json=request_data)

            # Then
            assert response.status_code == 400
            response_data = response.json()
            assert response_data["success"] is False
            assert "empty" in response_data["error"].lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convert_with_invalid_processing_options(
        self, client, temp_directory
    ):
        """Test conversion with invalid processing options."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        request_data = {
            "file_path": str(pdf_path),
            "processing_options": {
                "language": "invalid_language",  # Invalid language
                "chunk_size": -100,  # Invalid chunk size
            },
        }

        # When
        response = await client.post("/convert_single", json=request_data)

        # Then
        assert response.status_code == 422  # Validation error
        response_data = response.json()
        assert "detail" in response_data


class TestSearchEndpoints:
    """Test search API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_search_success(self, client, async_db_session):
        """Test successful semantic search."""
        # Given - Create documents with embeddings
        document_data = DocumentFactory.create(status="completed")
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Add content
        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test Document\n\nThis is about machine learning.",
            plain_text="Test Document\n\nThis is about machine learning.",
            word_count=10,
            language="en",
        )
        async_db_session.add(content)

        # Add embeddings
        embeddings = create_sample_embeddings(2, 1536)
        for i, embedding in enumerate(embeddings):
            doc_embedding = DocumentEmbedding(
                document_id=document.id,
                chunk_index=i,
                chunk_text=f"Chunk {i} about machine learning",
                embedding=embedding,
                start_char=i * 50,
                end_char=(i + 1) * 50,
                token_count=5,
            )
            async_db_session.add(doc_embedding)

        await async_db_session.commit()

        with (
            patch("src.pdf_to_markdown_mcp.api.search.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.api.search.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_get_db.return_value = async_db_session
            mock_embedding = create_mock_embedding_service()
            mock_embedding.generate_embedding.return_value = embeddings[0]
            mock_embedding_class.return_value = mock_embedding

            request_data = {
                "query": "machine learning algorithms",
                "limit": 10,
                "threshold": 0.7,
            }

            # When
            response = await client.post("/semantic_search", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "results" in response_data
            assert len(response_data["results"]) > 0

            # Verify result structure
            result = response_data["results"][0]
            assert "document_id" in result
            assert "chunk_text" in result
            assert "similarity_score" in result
            assert "document_title" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_search_no_results(self, client, async_db_session):
        """Test semantic search with no matching results."""
        # Given - Empty database
        with (
            patch("src.pdf_to_markdown_mcp.api.search.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.api.search.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_get_db.return_value = async_db_session
            mock_embedding = create_mock_embedding_service()
            mock_embedding_class.return_value = mock_embedding

            request_data = {
                "query": "nonexistent content",
                "limit": 10,
                "threshold": 0.7,
            }

            # When
            response = await client.post("/semantic_search", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert response_data["results"] == []
            assert response_data["total_results"] == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, client, async_db_session):
        """Test successful hybrid search (semantic + keyword)."""
        # Given - Create document with content
        document_data = DocumentFactory.create(status="completed")
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        content = DocumentContent(
            document_id=document.id,
            markdown_content="# AI Research\n\nDeep learning and neural networks",
            plain_text="AI Research\n\nDeep learning and neural networks",
            word_count=8,
            language="en",
        )
        async_db_session.add(content)

        # Add embedding
        embedding = create_sample_embeddings(1, 1536)[0]
        doc_embedding = DocumentEmbedding(
            document_id=document.id,
            chunk_index=0,
            chunk_text="Deep learning and neural networks research",
            embedding=embedding,
            start_char=0,
            end_char=42,
            token_count=6,
        )
        async_db_session.add(doc_embedding)
        await async_db_session.commit()

        with (
            patch("src.pdf_to_markdown_mcp.api.search.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.api.search.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_get_db.return_value = async_db_session
            mock_embedding_service = create_mock_embedding_service()
            mock_embedding_class.return_value = mock_embedding_service

            request_data = {
                "query": "neural networks research",
                "limit": 10,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "threshold": 0.5,
            }

            # When
            response = await client.post("/hybrid_search", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "results" in response_data
            assert len(response_data["results"]) > 0

            # Verify hybrid score calculation
            result = response_data["results"][0]
            assert "hybrid_score" in result
            assert "semantic_score" in result
            assert "keyword_score" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_find_similar_documents(self, client, async_db_session):
        """Test finding similar documents."""
        # Given - Create multiple documents with embeddings
        documents = []
        for i in range(3):
            document_data = DocumentFactory.create(
                file_name=f"document_{i}.pdf", status="completed"
            )
            document = Document(**document_data)
            async_db_session.add(document)
            documents.append(document)

        await async_db_session.commit()

        # Add embeddings
        embeddings = create_sample_embeddings(3, 1536)
        for i, (document, embedding) in enumerate(zip(documents, embeddings, strict=False)):
            doc_embedding = DocumentEmbedding(
                document_id=document.id,
                chunk_index=0,
                chunk_text=f"Content for document {i}",
                embedding=embedding,
                start_char=0,
                end_char=20,
                token_count=4,
            )
            async_db_session.add(doc_embedding)

        await async_db_session.commit()

        with patch("src.pdf_to_markdown_mcp.api.search.get_db_session") as mock_get_db:
            mock_get_db.return_value = async_db_session

            request_data = {
                "document_id": documents[0].id,
                "limit": 5,
                "threshold": 0.1,
            }

            # When
            response = await client.post("/find_similar", json=request_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "results" in response_data

            # Should find other documents (excluding the source document)
            found_document_ids = [r["document_id"] for r in response_data["results"]]
            assert documents[0].id not in found_document_ids


class TestStatusEndpoints:
    """Test status and monitoring API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_status_success(self, client, async_db_session):
        """Test successful status retrieval."""
        # Given - Create some documents and tasks
        documents = []
        for status in ["completed", "processing", "failed"]:
            document_data = DocumentFactory.create(status=status)
            document = Document(**document_data)
            async_db_session.add(document)
            documents.append(document)

        await async_db_session.commit()

        with (
            patch("src.pdf_to_markdown_mcp.api.status.get_db_session") as mock_get_db,
            patch("src.pdf_to_markdown_mcp.api.status.app") as mock_celery,
        ):
            mock_get_db.return_value = async_db_session

            # Mock Celery stats
            mock_celery.control.inspect.return_value.stats.return_value = {
                "worker1": {
                    "total_tasks": 100,
                    "active_tasks": 5,
                    "completed_tasks": 90,
                    "failed_tasks": 5,
                }
            }

            # When
            response = await client.get("/get_status")

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "queue_stats" in response_data
            assert "document_stats" in response_data
            assert "system_stats" in response_data

            # Verify document stats
            doc_stats = response_data["document_stats"]
            assert doc_stats["total_documents"] == 3
            assert doc_stats["completed_documents"] == 1
            assert doc_stats["processing_documents"] == 1
            assert doc_stats["failed_documents"] == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stream_progress_endpoint(self, client):
        """Test progress streaming endpoint."""
        # Given
        task_id = "test-task-123"

        with patch(
            "src.pdf_to_markdown_mcp.api.status.get_task_progress"
        ) as mock_progress:
            mock_progress.return_value = {
                "task_id": task_id,
                "state": "PROGRESS",
                "current": 50,
                "total": 100,
                "status": "Processing chunks...",
            }

            # When
            response = await client.get(f"/stream_progress/{task_id}")

            # Then
            assert response.status_code == 200
            # For streaming responses, we'd typically test the stream content
            # This is a simplified test for the endpoint availability

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_all_services_healthy(self, client):
        """Test health endpoint with all services healthy."""
        # Given
        with (
            patch(
                "src.pdf_to_markdown_mcp.main.check_database_health"
            ) as mock_db_health,
            patch(
                "src.pdf_to_markdown_mcp.main.check_embedding_service_health"
            ) as mock_embedding_health,
            patch(
                "src.pdf_to_markdown_mcp.main.check_celery_health"
            ) as mock_celery_health,
        ):
            mock_db_health.return_value = True
            mock_embedding_health.return_value = True
            mock_celery_health.return_value = True

            # When
            response = await client.get("/health")

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "healthy"
            assert response_data["database"] is True
            assert response_data["embedding_service"] is True
            assert response_data["celery"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_with_service_failures(self, client):
        """Test health endpoint with some service failures."""
        # Given
        with (
            patch(
                "src.pdf_to_markdown_mcp.main.check_database_health"
            ) as mock_db_health,
            patch(
                "src.pdf_to_markdown_mcp.main.check_embedding_service_health"
            ) as mock_embedding_health,
            patch(
                "src.pdf_to_markdown_mcp.main.check_celery_health"
            ) as mock_celery_health,
        ):
            mock_db_health.return_value = True
            mock_embedding_health.return_value = False  # Service down
            mock_celery_health.return_value = True

            # When
            response = await client.get("/health")

            # Then
            assert response.status_code == 503  # Service unavailable
            response_data = response.json()
            assert response_data["status"] == "unhealthy"
            assert response_data["database"] is True
            assert response_data["embedding_service"] is False
            assert response_data["celery"] is True


class TestConfigurationEndpoints:
    """Test configuration API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configure_endpoint_success(self, client):
        """Test successful configuration update."""
        # Given
        config_data = {
            "processing": {"chunk_size": 1200, "chunk_overlap": 250, "language": "en"},
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "batch_size": 50,
            },
        }

        with patch(
            "src.pdf_to_markdown_mcp.api.config.update_configuration"
        ) as mock_update:
            mock_update.return_value = True

            # When
            response = await client.post("/configure", json=config_data)

            # Then
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert response_data["message"] == "Configuration updated successfully"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configure_endpoint_invalid_config(self, client):
        """Test configuration update with invalid data."""
        # Given
        invalid_config = {
            "processing": {
                "chunk_size": -100,  # Invalid negative size
                "language": "invalid_lang",  # Invalid language
            }
        }

        # When
        response = await client.post("/configure", json=invalid_config)

        # Then
        assert response.status_code == 422  # Validation error
        response_data = response.json()
        assert "detail" in response_data
