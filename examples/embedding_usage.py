"""
Example usage of the Embedding Generation Service.

This demonstrates how to use the dual embedding strategy
with both Ollama (local) and OpenAI providers.
"""

import asyncio
from pdf_to_markdown_mcp.services.embeddings import (
    create_embedding_service,
    EmbeddingProvider,
    EmbeddingConfig,
)


async def main():
    """Demonstrate embedding service usage."""
    # Sample documents to embed
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Vector embeddings capture semantic meaning of text.",
    ]

    print("🚀 PDF to Markdown MCP - Embedding Service Demo")
    print("=" * 50)

    # Example 1: Using Ollama (local) provider
    print("\n1. Using Ollama (Local) Provider:")
    try:
        ollama_service = await create_embedding_service(
            provider=EmbeddingProvider.OLLAMA, batch_size=2, max_retries=1
        )

        print(f"✅ Service initialized with {ollama_service.config.provider}")
        print(f"   Model: {ollama_service.config.ollama_model}")
        print(f"   Batch size: {ollama_service.config.batch_size}")

        # Generate embeddings
        print("\n   Generating embeddings...")
        result = await ollama_service.generate_embeddings(documents)

        print(f"   ✅ Generated {len(result.embeddings)} embeddings")
        print(f"   Provider: {result.provider}")
        print(f"   Model: {result.model}")
        print(f"   Metadata: {result.metadata}")

        # Health check
        is_healthy = await ollama_service.health_check()
        print(f"   Health status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

    except Exception as e:
        print(f"   ⚠️  Ollama not available: {e}")

    # Example 2: Using OpenAI provider
    print("\n2. Using OpenAI Provider:")
    try:
        openai_service = await create_embedding_service(
            provider=EmbeddingProvider.OPENAI, batch_size=10, embedding_dimensions=1536
        )

        print(f"✅ Service initialized with {openai_service.config.provider}")
        print(f"   Model: {openai_service.config.openai_model}")
        print(f"   Dimensions: {openai_service.config.embedding_dimensions}")

        # Note: This would require actual OpenAI API key
        print("   ⚠️  OpenAI API key required for actual usage")

    except Exception as e:
        print(f"   ⚠️  OpenAI client error: {e}")

    # Example 3: Vector similarity search
    print("\n3. Vector Similarity Search Demo:")

    # Create sample embeddings for demo
    sample_embeddings = [
        [1.0, 0.0, 0.0],  # Document 1
        [0.0, 1.0, 0.0],  # Document 2
        [0.7, 0.7, 0.0],  # Document 3 (similar to both)
        [0.0, 0.0, 1.0],  # Document 4
    ]

    query_vector = [0.8, 0.6, 0.0]  # Query similar to documents 1 and 3

    # Create service for similarity search
    config = EmbeddingConfig()
    service = EmbeddingService(config)

    # Perform similarity search
    search_results = await service.similarity_search(
        query_vector, sample_embeddings, top_k=3, metric="cosine"
    )

    print("   Query vector: [0.8, 0.6, 0.0]")
    print("   Top 3 similar documents:")
    for idx, (doc_idx, similarity) in enumerate(search_results):
        print(f"     {idx+1}. Document {doc_idx}: {similarity:.4f} similarity")

    # Example 4: Embedding normalization
    print("\n4. Embedding Normalization Demo:")

    unnormalized = [
        [3.0, 4.0],  # Magnitude = 5
        [5.0, 12.0],  # Magnitude = 13
        [0.0, 0.0],  # Zero vector
    ]

    normalized = await service.normalize_embeddings(unnormalized)

    print("   Original vectors -> Normalized vectors:")
    for orig, norm in zip(unnormalized, normalized):
        magnitude = (sum(x**2 for x in norm)) ** 0.5 if norm != [0.0, 0.0] else 0.0
        print(f"     {orig} -> {norm} (magnitude: {magnitude:.4f})")

    print("\n✅ Embedding service demo completed!")
    print("🔧 Ready for integration with:")
    print("   - MinerU PDF processing pipeline")
    print("   - PostgreSQL with PGVector storage")
    print("   - Celery background task processing")
    print("   - FastAPI semantic search endpoints")


if __name__ == "__main__":
    asyncio.run(main())
