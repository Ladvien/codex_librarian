"""
Test suite for embedding generation service.
Follows TDD approach with comprehensive coverage.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from pdf_to_markdown_mcp.services.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingService,
    OllamaEmbedder,
    OpenAIEmbedder,
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig model validation and settings."""

    def test_embedding_config_defaults(self):
        """Test that EmbeddingConfig has proper defaults."""
        config = EmbeddingConfig()

        assert config.provider == EmbeddingProvider.OLLAMA
        assert config.ollama_model == "nomic-embed-text"
        assert config.openai_model == "text-embedding-3-small"
        assert config.batch_size == 10
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.embedding_dimensions == 1536

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig validation rules."""
        # Valid configuration
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            batch_size=5,
            timeout=60.0,
            max_retries=5,
            embedding_dimensions=768,
        )

        assert config.provider == EmbeddingProvider.OPENAI
        assert config.batch_size == 5
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.embedding_dimensions == 768

    def test_embedding_config_invalid_batch_size(self):
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=-1)

    def test_embedding_config_invalid_timeout(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError):
            EmbeddingConfig(timeout=0.0)

        with pytest.raises(ValueError):
            EmbeddingConfig(timeout=-1.0)


class TestOllamaEmbedder:
    """Test OllamaEmbedder following TDD approach."""

    @pytest.fixture
    def ollama_embedder(self):
        """Setup OllamaEmbedder with mocked client."""
        with patch("pdf_to_markdown_mcp.services.embeddings.ollama") as mock_ollama:
            mock_client = AsyncMock()
            mock_ollama.AsyncClient.return_value = mock_client

            embedder = OllamaEmbedder(model_name="test-model")
            embedder.client = mock_client
            return embedder

    @pytest.mark.asyncio
    async def test_ollama_embed_single_text(self, ollama_embedder):
        """Test embedding single text with Ollama."""
        # Given
        text = "Test document content"
        expected_embedding = [0.1, 0.2, 0.3, 0.4]

        ollama_embedder.client.embeddings.return_value = {
            "embedding": expected_embedding
        }

        # When
        result = await ollama_embedder.embed_texts([text])

        # Then
        assert len(result) == 1
        assert result[0] == expected_embedding
        ollama_embedder.client.embeddings.assert_called_once_with(
            model="test-model", prompt=text
        )

    @pytest.mark.asyncio
    async def test_ollama_embed_multiple_texts(self, ollama_embedder):
        """Test embedding multiple texts with Ollama."""
        # Given
        texts = ["First text", "Second text", "Third text"]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        # Mock sequential calls
        ollama_embedder.client.embeddings.side_effect = [
            {"embedding": expected_embeddings[0]},
            {"embedding": expected_embeddings[1]},
            {"embedding": expected_embeddings[2]},
        ]

        # When
        result = await ollama_embedder.embed_texts(texts)

        # Then
        assert len(result) == 3
        assert result == expected_embeddings
        assert ollama_embedder.client.embeddings.call_count == 3

    @pytest.mark.asyncio
    async def test_ollama_embed_empty_list(self, ollama_embedder):
        """Test embedding empty text list."""
        # Given
        texts = []

        # When
        result = await ollama_embedder.embed_texts(texts)

        # Then
        assert result == []
        ollama_embedder.client.embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_ollama_embed_with_error(self, ollama_embedder):
        """Test Ollama embedding error handling."""
        # Given
        texts = ["Test text"]
        ollama_embedder.client.embeddings.side_effect = Exception("Connection failed")

        # When/Then
        with pytest.raises(EmbeddingError) as exc_info:
            await ollama_embedder.embed_texts(texts)

        assert "Ollama embedding failed" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ollama_batch_processing(self, ollama_embedder):
        """Test Ollama processes texts individually (no native batching)."""
        # Given
        texts = ["Text 1", "Text 2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

        ollama_embedder.client.embeddings.side_effect = [
            {"embedding": expected_embeddings[0]},
            {"embedding": expected_embeddings[1]},
        ]

        # When
        result = await ollama_embedder.embed_texts(texts)

        # Then
        assert result == expected_embeddings
        assert ollama_embedder.client.embeddings.call_count == 2


class TestOpenAIEmbedder:
    """Test OpenAIEmbedder following TDD approach."""

    @pytest.fixture
    def openai_embedder(self):
        """Setup OpenAIEmbedder with mocked client."""
        with patch("pdf_to_markdown_mcp.services.embeddings.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client

            embedder = OpenAIEmbedder(model_name="test-embedding-model")
            embedder.client = mock_client
            return embedder

    @pytest.mark.asyncio
    async def test_openai_embed_single_text(self, openai_embedder):
        """Test embedding single text with OpenAI."""
        # Given
        text = "Test document content"
        expected_embedding = [0.1, 0.2, 0.3, 0.4]

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = expected_embedding

        openai_embedder.client.embeddings.create.return_value = mock_response

        # When
        result = await openai_embedder.embed_texts([text])

        # Then
        assert len(result) == 1
        assert result[0] == expected_embedding
        openai_embedder.client.embeddings.create.assert_called_once_with(
            model="test-embedding-model", input=[text], dimensions=1536
        )

    @pytest.mark.asyncio
    async def test_openai_embed_multiple_texts(self, openai_embedder):
        """Test embedding multiple texts with OpenAI (native batching)."""
        # Given
        texts = ["First text", "Second text", "Third text"]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        mock_response = Mock()
        mock_response.data = []
        for embedding in expected_embeddings:
            mock_item = Mock()
            mock_item.embedding = embedding
            mock_response.data.append(mock_item)

        openai_embedder.client.embeddings.create.return_value = mock_response

        # When
        result = await openai_embedder.embed_texts(texts)

        # Then
        assert len(result) == 3
        assert result == expected_embeddings
        openai_embedder.client.embeddings.create.assert_called_once_with(
            model="test-embedding-model", input=texts, dimensions=1536
        )

    @pytest.mark.asyncio
    async def test_openai_embed_with_custom_dimensions(self, openai_embedder):
        """Test OpenAI embedding with custom dimensions."""
        # Given
        texts = ["Test text"]
        expected_embedding = [0.1, 0.2]
        custom_dimensions = 768

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = expected_embedding

        openai_embedder.client.embeddings.create.return_value = mock_response

        # When
        result = await openai_embedder.embed_texts(texts, dimensions=custom_dimensions)

        # Then
        assert result[0] == expected_embedding
        openai_embedder.client.embeddings.create.assert_called_once_with(
            model="test-embedding-model", input=texts, dimensions=custom_dimensions
        )

    @pytest.mark.asyncio
    async def test_openai_embed_with_error(self, openai_embedder):
        """Test OpenAI embedding error handling."""
        # Given
        texts = ["Test text"]
        openai_embedder.client.embeddings.create.side_effect = Exception(
            "API quota exceeded"
        )

        # When/Then
        with pytest.raises(EmbeddingError) as exc_info:
            await openai_embedder.embed_texts(texts)

        assert "OpenAI embedding failed" in str(exc_info.value)
        assert "API quota exceeded" in str(exc_info.value)


class TestEmbeddingService:
    """Test main EmbeddingService orchestration."""

    @pytest.fixture
    def mock_ollama_embedder(self):
        """Mock OllamaEmbedder."""
        embedder = AsyncMock(spec=OllamaEmbedder)
        return embedder

    @pytest.fixture
    def mock_openai_embedder(self):
        """Mock OpenAIEmbedder."""
        embedder = AsyncMock(spec=OpenAIEmbedder)
        return embedder

    @pytest.fixture
    def embedding_service(self, mock_ollama_embedder, mock_openai_embedder):
        """Setup EmbeddingService with mocked embedders."""
        config = EmbeddingConfig()

        with (
            patch(
                "pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder"
            ) as mock_ollama_cls,
            patch(
                "pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder"
            ) as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder
            return service

    @pytest.mark.asyncio
    async def test_generate_embeddings_ollama_provider(self, mock_ollama_embedder, mock_openai_embedder):
        """Test embedding generation with Ollama provider."""
        # Given - Create service with Ollama provider (default)
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            texts = ["Test text 1", "Test text 2"]
            expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

            mock_ollama_embedder.embed_texts.return_value = expected_embeddings

            # When
            result = await service.generate_embeddings(texts)

            # Then
            assert isinstance(result, EmbeddingResult)
            assert result.embeddings == expected_embeddings
            assert result.provider == EmbeddingProvider.OLLAMA
            assert result.model == "nomic-embed-text"
            assert len(result.embeddings) == 2

            mock_ollama_embedder.embed_texts.assert_called_once_with(texts)
            mock_openai_embedder.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_openai_provider(self, mock_ollama_embedder, mock_openai_embedder):
        """Test embedding generation with OpenAI provider."""
        # Given - Create service with OpenAI provider
        config = EmbeddingConfig(provider=EmbeddingProvider.OPENAI)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            texts = ["Test text 1", "Test text 2"]
            expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

            mock_openai_embedder.embed_texts.return_value = expected_embeddings

            # When
            result = await service.generate_embeddings(texts)

            # Then
            assert isinstance(result, EmbeddingResult)
            assert result.embeddings == expected_embeddings
            assert result.provider == EmbeddingProvider.OPENAI
            assert result.model == "text-embedding-3-small"
            assert len(result.embeddings) == 2

            mock_openai_embedder.embed_texts.assert_called_once_with(
                texts, dimensions=1536
            )
            mock_ollama_embedder.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self, mock_ollama_embedder, mock_openai_embedder):
        """Test batch processing with large text input."""
        # Given - Create service with batch size 10
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA, batch_size=10)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            texts = [f"Text {i}" for i in range(25)]  # Larger than batch_size=10

            # Mock embedder to return different embeddings for each batch
            mock_ollama_embedder.embed_texts.side_effect = [
                [[0.1, 0.2]] * 10,  # First batch: 10 embeddings
                [[0.3, 0.4]] * 10,  # Second batch: 10 embeddings
                [[0.5, 0.6]] * 5,  # Third batch: 5 embeddings
            ]

            # When
            result = await service.generate_embeddings(texts)

            # Then
            assert len(result.embeddings) == 25
            assert mock_ollama_embedder.embed_texts.call_count == 3

            # Verify batch calls
            calls = mock_ollama_embedder.embed_texts.call_args_list
            assert len(calls[0][0][0]) == 10  # First batch size
            assert len(calls[1][0][0]) == 10  # Second batch size
            assert len(calls[2][0][0]) == 5  # Third batch size

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_input(self, embedding_service):
        """Test embedding generation with empty input."""
        # Given
        texts = []

        # When
        result = await embedding_service.generate_embeddings(texts)

        # Then
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == []
        assert len(result.embeddings) == 0

        embedding_service.ollama_embedder.embed_texts.assert_not_called()
        embedding_service.openai_embedder.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_retry_on_failure(self, mock_ollama_embedder, mock_openai_embedder):
        """Test retry mechanism on embedding failure."""
        # Given - Create service with specific retry config
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA, max_retries=3)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            texts = ["Test text"]

            # First two calls fail, third succeeds
            mock_ollama_embedder.embed_texts.side_effect = [
                EmbeddingError("Temporary failure"),
                EmbeddingError("Another failure"),
                [[0.1, 0.2, 0.3]],  # Success on third try
            ]

            # When
            result = await service.generate_embeddings(texts)

            # Then
            assert result.embeddings == [[0.1, 0.2, 0.3]]
            assert mock_ollama_embedder.embed_texts.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_embeddings_max_retries_exceeded(self, mock_ollama_embedder, mock_openai_embedder):
        """Test failure after max retries exceeded."""
        # Given - Create service with specific retry config
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA, max_retries=2)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            texts = ["Test text"]

            mock_ollama_embedder.embed_texts.side_effect = EmbeddingError(
                "Persistent failure"
            )

            # When/Then
            with pytest.raises(EmbeddingError) as exc_info:
                await service.generate_embeddings(texts)

            assert "Max retries (2) exceeded" in str(exc_info.value)
            assert (
                mock_ollama_embedder.embed_texts.call_count == 3
            )  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_health_check_ollama_healthy(self, mock_ollama_embedder, mock_openai_embedder):
        """Test health check when Ollama is healthy."""
        # Given - Create service with Ollama provider
        config = EmbeddingConfig(provider=EmbeddingProvider.OLLAMA)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            mock_ollama_embedder.embed_texts.return_value = [[0.1, 0.2]]

            # When
            is_healthy = await service.health_check()

            # Then
            assert is_healthy is True
            mock_ollama_embedder.embed_texts.assert_called_once_with(
                ["health check"]
            )

    @pytest.mark.asyncio
    async def test_health_check_service_unhealthy(self, mock_ollama_embedder, mock_openai_embedder):
        """Test health check when service is unhealthy."""
        # Given - Create service with OpenAI provider
        config = EmbeddingConfig(provider=EmbeddingProvider.OPENAI)

        with (
            patch("pdf_to_markdown_mcp.services.embeddings.OllamaEmbedder") as mock_ollama_cls,
            patch("pdf_to_markdown_mcp.services.embeddings.OpenAIEmbedder") as mock_openai_cls,
        ):
            mock_ollama_cls.return_value = mock_ollama_embedder
            mock_openai_cls.return_value = mock_openai_embedder

            service = EmbeddingService(config)
            service.ollama_embedder = mock_ollama_embedder
            service.openai_embedder = mock_openai_embedder

            mock_openai_embedder.embed_texts.side_effect = Exception(
                "Service down"
            )

            # When
            is_healthy = await service.health_check()

            # Then
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_similarity_search_cosine(self, embedding_service):
        """Test cosine similarity search."""
        # Given
        query_embedding = [1.0, 0.0, 0.0]
        candidate_embeddings = [
            [1.0, 0.0, 0.0],  # Perfect match (similarity = 1.0)
            [0.0, 1.0, 0.0],  # Orthogonal (similarity = 0.0)
            [0.5, 0.5, 0.0],  # Partial match
        ]
        top_k = 2

        # When
        results = await embedding_service.similarity_search(
            query_embedding, candidate_embeddings, top_k=top_k, metric="cosine"
        )

        # Then
        assert len(results) == 2
        assert results[0][0] == 0  # Index of perfect match
        assert results[0][1] == pytest.approx(1.0, rel=1e-6)  # Perfect similarity
        assert results[1][0] == 2  # Index of partial match
        assert results[1][1] > 0.0  # Positive similarity

    @pytest.mark.asyncio
    async def test_normalize_embeddings(self, embedding_service):
        """Test embedding normalization."""
        # Given
        embeddings = [
            [3.0, 4.0],  # Magnitude = 5
            [1.0, 1.0],  # Magnitude = sqrt(2)
            [0.0, 0.0],  # Zero vector
        ]

        # When
        normalized = await embedding_service.normalize_embeddings(embeddings)

        # Then
        assert len(normalized) == 3
        # First vector normalized
        assert normalized[0] == pytest.approx([0.6, 0.8], rel=1e-6)
        # Second vector normalized
        assert normalized[1] == pytest.approx(
            [1 / np.sqrt(2), 1 / np.sqrt(2)], rel=1e-6
        )
        # Zero vector remains zero
        assert normalized[2] == [0.0, 0.0]


class TestEmbeddingResult:
    """Test EmbeddingResult model."""

    def test_embedding_result_creation(self):
        """Test EmbeddingResult model creation."""
        # Given
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        provider = EmbeddingProvider.OLLAMA
        model = "test-model"

        # When
        result = EmbeddingResult(embeddings=embeddings, provider=provider, model=model)

        # Then
        assert result.embeddings == embeddings
        assert result.provider == provider
        assert result.model == model
        assert isinstance(result.metadata, dict)

    def test_embedding_result_with_metadata(self):
        """Test EmbeddingResult with custom metadata."""
        # Given
        embeddings = [[0.1, 0.2]]
        metadata = {"processing_time": 1.5, "batch_size": 10}

        # When
        result = EmbeddingResult(
            embeddings=embeddings,
            provider=EmbeddingProvider.OPENAI,
            model="test-model",
            metadata=metadata,
        )

        # Then
        assert result.metadata["processing_time"] == 1.5
        assert result.metadata["batch_size"] == 10
