"""
Services package for PDF to Markdown MCP Server.

Contains external service integrations:
- embeddings: Dual embedding generation (Ollama + OpenAI)
- mineru: MinerU PDF processing integration
- database: Database operations and vector search
"""

from .embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingService,
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedding_service,
)

# Import other services with graceful fallback
_available_services = []

try:
    from .mineru import MinerUService

    _available_services.append("MinerUService")
except ImportError:
    MinerUService = None

try:
    from .database import DatabaseService

    _available_services.append("DatabaseService")
except ImportError:
    DatabaseService = None

__all__ = [
    # Embedding services (always available)
    "EmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingService",
    "EmbeddingResult",
    "EmbeddingError",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "create_embedding_service",
] + _available_services
