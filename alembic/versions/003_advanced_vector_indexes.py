"""Advanced PGVector indexes and optimizations

Revision ID: 003
Revises: 002
Create Date: 2025-09-25 14:00:00.000000

This migration adds advanced PGVector indexes and optimizations:
- HNSW indexes for high-performance similarity search
- Additional distance metric indexes (euclidean, inner product)
- Partial indexes for performance optimization
- Covering indexes for common query patterns
- Advanced full-text search indexes for hybrid search
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add advanced PGVector indexes and optimizations."""

    # Create HNSW index for document embeddings (better performance than IVFFlat for high-recall scenarios)
    # HNSW is generally better for smaller datasets or when query performance is more important than index size
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw_cosine
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Create additional distance metric indexes for different similarity search needs
    # Euclidean distance index
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_euclidean
        ON document_embeddings
        USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);
    """)

    # Inner product index (for normalized vectors)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_inner_product
        ON document_embeddings
        USING ivfflat (embedding vector_ip_ops)
        WITH (lists = 100);
    """)

    # HNSW index for image embeddings
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_images_hnsw_cosine
        ON document_images
        USING hnsw (image_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Partial index for completed documents only (most common query pattern)
    # Note: Commented out as PostgreSQL doesn't support subqueries in partial index WHERE clause
    # Would need to be created as a materialized view or use a simpler condition
    # op.execute("""
    #     CREATE INDEX IF NOT EXISTS idx_embeddings_completed_docs
    #     ON document_embeddings
    #     USING ivfflat (embedding vector_cosine_ops)
    #     WITH (lists = 100)
    #     WHERE EXISTS (
    #         SELECT 1 FROM documents d
    #         WHERE d.id = document_embeddings.document_id
    #         AND d.conversion_status = 'completed'
    #     );
    # """)

    # Covering index for common search result fields
    op.create_index(
        'idx_embeddings_search_covering',
        'document_embeddings',
        ['document_id', 'page_number', 'chunk_index'],
        postgresql_include=['chunk_text', 'metadata']
    )

    # Advanced full-text search indexes for hybrid search optimization
    # GIN index with custom text search configuration
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_content_fulltext_gin
        ON document_content
        USING gin(to_tsvector('english', plain_text));
    """)

    # Combined index for document metadata and content search
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
        ON documents
        USING gin(metadata);
    """)

    # Optimize statistics for query planner
    op.execute("ALTER TABLE document_embeddings ALTER COLUMN embedding SET STATISTICS 1000;")
    op.execute("ALTER TABLE document_images ALTER COLUMN image_embedding SET STATISTICS 1000;")

    # Create index for chunk similarity search within documents
    op.create_index(
        'idx_embeddings_chunk_similarity',
        'document_embeddings',
        ['document_id', 'chunk_index', 'page_number']
    )

    # Index for temporal queries (recent documents)
    op.create_index(
        'idx_documents_temporal',
        'documents',
        [sa.text('created_at DESC'), 'conversion_status'],
        postgresql_where=sa.text("conversion_status = 'completed'")
    )

    # Index for file hash lookup (deduplication)
    op.create_index(
        'idx_documents_hash_unique',
        'documents',
        ['file_hash'],
        unique=True,
        postgresql_where=sa.text("file_hash IS NOT NULL")
    )


def downgrade() -> None:
    """Remove advanced PGVector indexes."""

    # Drop unique hash index
    op.drop_index('idx_documents_hash_unique', table_name='documents')

    # Drop temporal index
    op.drop_index('idx_documents_temporal', table_name='documents')

    # Drop chunk similarity index
    op.drop_index('idx_embeddings_chunk_similarity', table_name='document_embeddings')

    # Reset statistics
    op.execute("ALTER TABLE document_embeddings ALTER COLUMN embedding SET STATISTICS -1;")
    op.execute("ALTER TABLE document_images ALTER COLUMN image_embedding SET STATISTICS -1;")

    # Drop full-text search indexes
    op.execute("DROP INDEX IF EXISTS idx_documents_metadata_gin;")
    op.execute("DROP INDEX IF EXISTS idx_content_fulltext_gin;")

    # Drop covering index
    op.drop_index('idx_embeddings_search_covering', table_name='document_embeddings')

    # Drop partial index
    # op.execute("DROP INDEX IF EXISTS idx_embeddings_completed_docs;")

    # Drop advanced distance metric indexes
    op.execute("DROP INDEX IF EXISTS idx_images_hnsw_cosine;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_inner_product;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_euclidean;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_hnsw_cosine;")