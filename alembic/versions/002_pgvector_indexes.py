"""Add PGVector indexes for semantic search

Revision ID: 002
Revises: 001
Create Date: 2025-09-25 12:01:00.000000

This migration adds PGVector-specific indexes and updates vector columns
to use proper vector types instead of arrays. This migration should be
run after PGVector extension and dependencies are properly installed.

PGVector indexes enable efficient similarity search operations on embeddings:
- IVFFlat indexes for document text embeddings (1536 dimensions)
- IVFFlat indexes for image CLIP embeddings (512 dimensions)
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add PGVector indexes and update vector column types."""

    # Ensure PGVector extension is available
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Update embedding column to use vector type (1536 dimensions)
    op.execute("""
        ALTER TABLE document_embeddings
        ALTER COLUMN embedding
        TYPE vector(1536)
        USING embedding::vector(1536);
    """)

    # Update image_embedding column to use vector type (512 dimensions)
    op.execute("""
        ALTER TABLE document_images
        ALTER COLUMN image_embedding
        TYPE vector(512)
        USING image_embedding::vector(512);
    """)

    # Create IVFFlat index for document embeddings (cosine similarity)
    # Using lists=100 as a reasonable default for medium-sized datasets
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector
        ON document_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    # Create IVFFlat index for image embeddings (cosine similarity)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_images_vector
        ON document_images
        USING ivfflat (image_embedding vector_cosine_ops)
        WITH (lists = 50);
    """)

    # Add additional indexes for common query patterns

    # Index for finding embeddings by document and chunk
    op.create_index(
        'idx_embeddings_doc_chunk',
        'document_embeddings',
        ['document_id', 'chunk_index']
    )

    # Index for finding embeddings by page
    op.create_index(
        'idx_embeddings_page',
        'document_embeddings',
        ['document_id', 'page_number']
    )

    # Index for finding images by page
    op.create_index(
        'idx_images_page',
        'document_images',
        ['document_id', 'page_number', 'image_index']
    )


def downgrade() -> None:
    """Remove PGVector indexes and revert to array columns."""

    # Drop PGVector indexes
    op.execute("DROP INDEX IF EXISTS idx_images_vector;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_vector;")

    # Drop additional indexes
    op.drop_index('idx_images_page', table_name='document_images')
    op.drop_index('idx_embeddings_page', table_name='document_embeddings')
    op.drop_index('idx_embeddings_doc_chunk', table_name='document_embeddings')

    # Revert vector columns back to float arrays
    op.execute("""
        ALTER TABLE document_embeddings
        ALTER COLUMN embedding
        TYPE float8[]
        USING embedding::float8[];
    """)

    op.execute("""
        ALTER TABLE document_images
        ALTER COLUMN image_embedding
        TYPE float8[]
        USING image_embedding::float8[];
    """)