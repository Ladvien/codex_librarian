"""
Integration tests for embedding service.
Tests actual service functionality with mocked external dependencies.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from pdf_to_markdown_mcp.services.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingService,
    create_embedding_service,
)


class TestEmbeddingServiceIntegration:
    """Integration tests for EmbeddingService functionality."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for integration testing."""
        mock_client = AsyncMock()
        mock_client.embeddings = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for integration testing."""
        mock_client = AsyncMock()
        mock_client.embeddings = AsyncMock()
        mock_client.embeddings.create = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_end_to_end_ollama_embedding_generation(self, mock_ollama_client):
        """Test complete Ollama embedding generation workflow."""
        # Given
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA, batch_size=2, max_retries=1
        )

        texts = ["Document content 1", "Document content 2", "Document content 3"]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        mock_ollama_client.embeddings.side_effect = [
            {"embedding": expected_embeddings[0]},
            {"embedding": expected_embeddings[1]},
            {"embedding": expected_embeddings[2]},
        ]

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            # When
            service = EmbeddingService(config)
            result = await service.generate_embeddings(texts)

            # Then
            assert result.embeddings == expected_embeddings
            assert result.provider == EmbeddingProvider.OLLAMA
            assert result.model == "nomic-embed-text"
            assert result.metadata["total_texts"] == 3
            assert result.metadata["batch_count"] == 2  # 3 texts in batches of 2

    @pytest.mark.asyncio
    async def test_end_to_end_openai_embedding_generation(self, mock_openai_client):
        """Test complete OpenAI embedding generation workflow."""
        # Given
        config = EmbeddingConfig(provider=EmbeddingProvider.OPENAI, batch_size=5)

        texts = ["Document 1", "Document 2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

        mock_response = Mock()
        mock_response.data = []
        for embedding in expected_embeddings:
            mock_item = Mock()
            mock_item.embedding = embedding
            mock_response.data.append(mock_item)

        mock_openai_client.embeddings.create.return_value = mock_response

        with patch("pdf_to_markdown_mcp.services.embeddings.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_openai_client

            # When
            service = EmbeddingService(config)
            result = await service.generate_embeddings(texts)

            # Then
            assert result.embeddings == expected_embeddings
            assert result.provider == EmbeddingProvider.OPENAI
            assert result.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_service_factory_function(self, mock_ollama_client):
        """Test embedding service factory function."""
        # Given
        texts = ["Test document"]
        expected_embedding = [[0.1, 0.2, 0.3]]

        mock_ollama_client.embeddings.return_value = {
            "embedding": expected_embedding[0]
        }

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            # When
            service = await create_embedding_service(
                provider=EmbeddingProvider.OLLAMA, batch_size=5, max_retries=2
            )
            result = await service.generate_embeddings(texts)

            # Then
            assert result.embeddings == expected_embedding
            assert service.config.batch_size == 5
            assert service.config.max_retries == 2

    @pytest.mark.asyncio
    async def test_embedding_normalization_integration(self):
        """Test embedding normalization functionality."""
        # Given
        config = EmbeddingConfig()
        service = EmbeddingService(config)

        # Test vectors with known magnitudes
        embeddings = [
            [3.0, 4.0],  # Magnitude = 5
            [1.0, 1.0],  # Magnitude = sqrt(2) ≈ 1.414
            [0.0, 0.0],  # Zero vector
            [5.0, 0.0],  # Magnitude = 5
        ]

        # When
        normalized = await service.normalize_embeddings(embeddings)

        # Then
        assert len(normalized) == 4

        # Check first vector [3,4] -> [0.6, 0.8]
        assert abs(normalized[0][0] - 0.6) < 1e-6
        assert abs(normalized[0][1] - 0.8) < 1e-6

        # Check second vector normalization
        expected_val = 1.0 / np.sqrt(2)
        assert abs(normalized[1][0] - expected_val) < 1e-6
        assert abs(normalized[1][1] - expected_val) < 1e-6

        # Check zero vector remains unchanged
        assert normalized[2] == [0.0, 0.0]

        # Check unit vector [5,0] -> [1.0, 0.0]
        assert abs(normalized[3][0] - 1.0) < 1e-6
        assert abs(normalized[3][1] - 0.0) < 1e-6

    @pytest.mark.asyncio
    async def test_similarity_search_integration(self):
        """Test vector similarity search functionality."""
        # Given
        config = EmbeddingConfig()
        service = EmbeddingService(config)

        query_vector = [1.0, 0.0, 0.0]
        candidate_vectors = [
            [1.0, 0.0, 0.0],  # Perfect match (cosine = 1.0)
            [0.0, 1.0, 0.0],  # Orthogonal (cosine = 0.0)
            [0.5, 0.5, 0.0],  # Partial match (cosine ≈ 0.707)
            [-1.0, 0.0, 0.0],  # Opposite direction (cosine = -1.0)
        ]

        # When
        results = await service.similarity_search(
            query_vector, candidate_vectors, top_k=3, metric="cosine"
        )

        # Then
        assert len(results) == 3

        # Results should be sorted by similarity (highest first)
        indices = [result[0] for result in results]
        similarities = [result[1] for result in results]

        assert indices[0] == 0  # Perfect match first
        assert abs(similarities[0] - 1.0) < 1e-6

        assert indices[1] == 2  # Partial match second
        assert abs(similarities[1] - (0.5 / np.sqrt(0.5))) < 1e-6

        assert indices[2] == 1  # Orthogonal third
        assert abs(similarities[2] - 0.0) < 1e-6

    @pytest.mark.asyncio
    async def test_error_handling_and_retry_integration(self, mock_ollama_client):
        """Test error handling and retry mechanisms in integration."""
        # Given
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA, max_retries=2)

        texts = ["Test document"]

        # Simulate: first call fails, second call fails, third succeeds
        mock_ollama_client.embeddings.side_effect = [
            Exception("Network error"),
            Exception("Temporary failure"),
            {"embedding": [0.1, 0.2, 0.3]},
        ]

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            # When
            service = EmbeddingService(config)
            result = await service.generate_embeddings(texts)

            # Then
            assert result.embeddings == [[0.1, 0.2, 0.3]]
            assert mock_ollama_client.embeddings.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_integration(self, mock_ollama_client):
        """Test behavior when max retries are exceeded."""
        # Given
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA, max_retries=1)

        texts = ["Test document"]
        mock_ollama_client.embeddings.side_effect = Exception("Persistent failure")

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            # When/Then
            service = EmbeddingService(config)
            with pytest.raises(EmbeddingError) as exc_info:
                await service.generate_embeddings(texts)

            assert "Max retries (1) exceeded" in str(exc_info.value)
            assert mock_ollama_client.embeddings.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_health_check_integration(self, mock_ollama_client):
        """Test health check functionality."""
        # Given
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA)
        mock_ollama_client.embeddings.return_value = {"embedding": [0.1, 0.2]}

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            # When - Healthy service
            service = EmbeddingService(config)
            is_healthy = await service.health_check()

            # Then
            assert is_healthy is True
            mock_ollama_client.embeddings.assert_called_once_with(
                model="nomic-embed-text", prompt="health check"
            )

        # When - Unhealthy service
        mock_ollama_client.embeddings.side_effect = Exception("Service down")

        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_ollama_client

            service = EmbeddingService(config)
            is_healthy = await service.health_check()

            # Then
            assert is_healthy is False
