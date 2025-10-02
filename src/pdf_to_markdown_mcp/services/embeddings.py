"""
Embedding Generation Service with dual provider support (Ollama + OpenAI).

Supports:
- Async embedding generation with batching optimization
- Dual provider strategy: Ollama (local) and OpenAI API
- Comprehensive error handling and retry mechanisms
- Vector similarity search capabilities
- Performance optimization through batching
"""

import asyncio
import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

try:
    import ollama
except ImportError:
    ollama = None

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    # Fall back to standard logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""



class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation service."""

    provider: EmbeddingProvider = EmbeddingProvider.OLLAMA
    ollama_model: str = "nomic-embed-text"
    openai_model: str = "text-embedding-3-small"
    batch_size: int = Field(default=10, gt=0)
    timeout: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    embedding_dimensions: int = Field(default=1536, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_concurrency_limit: int = Field(default=8, ge=1, le=32)
    ollama_batch_size: int = Field(default=16, ge=1)
    openai_api_key: str | None = None
    openai_batch_size: int = Field(default=100, ge=1)

    class Config:
        """Pydantic configuration."""

        frozen = True


class EmbeddingResult(BaseModel):
    """Result from embedding generation."""

    embeddings: list[list[float]]
    provider: EmbeddingProvider
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = True


class OllamaEmbedder:
    """Ollama local embedding provider with concurrent request optimization."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        concurrency_limit: int = 8,
    ):
        """
        Initialize Ollama embedder.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama server URL
            concurrency_limit: Maximum concurrent requests (default: 8)
        """
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self.model_name = model_name
        self.base_url = base_url
        self.concurrency_limit = concurrency_limit
        self.client = ollama.AsyncClient(host=base_url)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for list of texts using Ollama with concurrent requests.

        Uses asyncio.gather() with semaphore-controlled parallelism to process
        multiple texts concurrently, providing 4-10x speedup over sequential processing.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (order preserved)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Limit concurrent requests to avoid overwhelming Ollama
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def embed_single(text: str, index: int) -> tuple[int, list[float] | Exception]:
            """Embed single text with semaphore control."""
            async with semaphore:
                try:
                    response = await self.client.embeddings(
                        model=self.model_name, prompt=text
                    )
                    return (index, response["embedding"])
                except Exception as e:
                    return (index, e)

        try:
            # Execute all requests concurrently
            tasks = [embed_single(text, i) for i, text in enumerate(texts)]
            results = await asyncio.gather(*tasks)

            # Process results maintaining order
            embeddings = [None] * len(texts)
            errors = []

            for index, result in results:
                if isinstance(result, Exception):
                    errors.append(f"Index {index}: {result!s}")
                else:
                    embeddings[index] = result

            # Raise if any errors occurred
            if errors:
                error_msg = "; ".join(errors[:3])  # Show first 3 errors
                if len(errors) > 3:
                    error_msg += f" (and {len(errors) - 3} more)"
                raise EmbeddingError(f"Failed to generate {len(errors)} embeddings: {error_msg}")

            logger.info(
                f"Ollama embeddings generated: count={len(embeddings)}, "
                f"model={self.model_name}, concurrency={self.concurrency_limit}"
            )
            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                f"Ollama embedding failed: error={e!s}, model={self.model_name}, "
                f"text_count={len(texts)}"
            )
            raise EmbeddingError(f"Ollama embedding failed: {e!s}")


class OpenAIEmbedder:
    """OpenAI API embedding provider."""

    def __init__(
        self, model_name: str = "text-embedding-3-small", api_key: str | None = None
    ):
        """Initialize OpenAI embedder."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed_texts(
        self, texts: list[str], dimensions: int = 1536
    ) -> list[list[float]]:
        """
        Generate embeddings for list of texts using OpenAI API.

        Args:
            texts: List of text strings to embed
            dimensions: Embedding dimensions (for compatible models)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                model=self.model_name, input=texts, dimensions=dimensions
            )

            embeddings = [item.embedding for item in response.data]

            logger.info(
                f"OpenAI embeddings generated: count={len(embeddings)}, model={self.model_name}, dimensions={dimensions}"
            )
            return embeddings

        except Exception as e:
            logger.error(
                f"OpenAI embedding failed: error={e!s}, model={self.model_name}, text_count={len(texts)}"
            )
            raise EmbeddingError(f"OpenAI embedding failed: {e!s}")


class EmbeddingService:
    """
    Main embedding service orchestrating multiple providers.

    Provides:
    - Dual provider support (Ollama local + OpenAI API)
    - Automatic batching for performance optimization
    - Retry mechanisms with exponential backoff
    - Vector similarity search capabilities
    - Health checking and monitoring
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding service with configuration."""
        self.config = config

        # Initialize providers based on configuration
        self.ollama_embedder = None
        self.openai_embedder = None

        if config.provider == EmbeddingProvider.OLLAMA:
            self.ollama_embedder = OllamaEmbedder(
                model_name=config.ollama_model,
                base_url=config.ollama_base_url,
                concurrency_limit=config.ollama_concurrency_limit,
            )
        elif config.provider == EmbeddingProvider.OPENAI:
            self.openai_embedder = OpenAIEmbedder(
                model_name=config.openai_model, api_key=config.openai_api_key
            )

        logger.info(
            f"Embedding service initialized: provider={config.provider}, "
            f"batch_size={self._get_batch_size()}, max_retries={config.max_retries}, "
            f"concurrency_limit={config.ollama_concurrency_limit if config.provider == EmbeddingProvider.OLLAMA else 'N/A'}"
        )

    async def generate_embeddings(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for list of texts with batching and retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            EmbeddingError: If embedding generation fails after retries
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                provider=self.config.provider,
                model=self._get_model_name(),
            )

        # Process in batches for performance with provider-specific batch sizes
        all_embeddings = []
        batch_size = self._get_batch_size()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Retry logic for each batch
            for attempt in range(self.config.max_retries + 1):
                try:
                    if self.config.provider == EmbeddingProvider.OLLAMA:
                        batch_embeddings = await self.ollama_embedder.embed_texts(batch)
                    else:  # OpenAI
                        batch_embeddings = await self.openai_embedder.embed_texts(
                            batch, dimensions=self.config.embedding_dimensions
                        )

                    all_embeddings.extend(batch_embeddings)
                    break  # Success, break retry loop

                except EmbeddingError as e:
                    if attempt == self.config.max_retries:
                        raise EmbeddingError(
                            f"Max retries ({self.config.max_retries}) exceeded: {e!s}"
                        )

                    # Exponential backoff
                    await asyncio.sleep(2**attempt)
                    logger.warning(
                        f"Embedding retry: attempt={attempt + 1}, error={e!s}"
                    )

        return EmbeddingResult(
            embeddings=all_embeddings,
            provider=self.config.provider,
            model=self._get_model_name(),
            metadata={
                "batch_count": (len(texts) + batch_size - 1) // batch_size,
                "total_texts": len(texts),
                "batch_size": batch_size,
            },
        )

    async def health_check(self) -> bool:
        """
        Check if embedding service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to generate a test embedding
            await self.generate_embeddings(["health check"])
            return True
        except Exception as e:
            logger.warning(f"Embedding service health check failed: error={e!s}")
            return False

    async def similarity_search(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 10,
        metric: str = "cosine",
    ) -> list[tuple[int, float]]:
        """
        Perform similarity search using embeddings.

        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate vectors
            top_k: Number of top results to return
            metric: Distance metric ("cosine", "euclidean", "dot")

        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        if not candidate_embeddings:
            return []

        query_array = np.array(query_embedding)
        candidate_arrays = np.array(candidate_embeddings)

        if metric == "cosine":
            # Cosine similarity
            query_norm = np.linalg.norm(query_array)
            candidate_norms = np.linalg.norm(candidate_arrays, axis=1)

            if query_norm == 0 or np.any(candidate_norms == 0):
                similarities = np.zeros(len(candidate_arrays))
            else:
                dot_products = np.dot(candidate_arrays, query_array)
                similarities = dot_products / (candidate_norms * query_norm)

        elif metric == "dot":
            # Dot product similarity
            similarities = np.dot(candidate_arrays, query_array)

        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            distances = np.linalg.norm(candidate_arrays - query_array, axis=1)
            similarities = -distances

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    async def normalize_embeddings(
        self, embeddings: list[list[float]]
    ) -> list[list[float]]:
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of normalized embedding vectors
        """
        normalized = []
        for embedding in embeddings:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)

            if norm == 0:
                normalized.append(embedding)  # Keep zero vectors as-is
            else:
                normalized.append((embedding_array / norm).tolist())

        return normalized

    def _get_model_name(self) -> str:
        """Get current model name based on provider."""
        if self.config.provider == EmbeddingProvider.OLLAMA:
            return self.config.ollama_model
        else:
            return self.config.openai_model

    def _get_batch_size(self) -> int:
        """Get provider-optimized batch size."""
        if self.config.provider == EmbeddingProvider.OLLAMA:
            return self.config.ollama_batch_size
        else:
            return self.config.openai_batch_size


# Factory function for easy service creation
async def create_embedding_service(
    provider: EmbeddingProvider = EmbeddingProvider.OLLAMA, **kwargs
) -> EmbeddingService:
    """
    Factory function to create embedding service.

    Args:
        provider: Embedding provider to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured EmbeddingService instance
    """
    config = EmbeddingConfig(provider=provider, **kwargs)
    return EmbeddingService(config)
