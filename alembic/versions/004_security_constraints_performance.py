"""Add security constraints and performance improvements

Revision ID: 004
Revises: 003
Create Date: 2025-09-26 10:25:00.000000

This migration adds critical security constraints and performance improvements:
- Vector dimension validation constraints (CRITICAL security fix)
- Additional check constraints for data integrity
- Performance indexes for filename, created_at, conversion_status
- Unique constraint for file_hash to prevent duplicates
- OCR confidence validation constraint
- Chunk and image index validation constraints
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import CheckConstraint

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add security constraints and performance improvements."""

    # CRITICAL: Add vector dimension validation constraints to prevent data corruption
    # These constraints validate that embeddings have correct dimensions as required by PGVector

    # Text embeddings must be 1536 dimensions (OpenAI text-embedding-3-small)
    op.execute("""
        ALTER TABLE document_embeddings
        ADD CONSTRAINT check_text_embedding_dimensions
        CHECK (vector_dims(embedding) = 1536);
    """)

    # Image embeddings must be 512 dimensions (CLIP model)
    op.execute("""
        ALTER TABLE document_images
        ADD CONSTRAINT check_image_embedding_dimensions
        CHECK (vector_dims(image_embedding) = 512);
    """)

    # Add data validation constraints for better data integrity

    # Ensure OCR confidence is between 0 and 1
    op.execute("""
        ALTER TABLE document_images
        ADD CONSTRAINT check_ocr_confidence_range
        CHECK (ocr_confidence >= 0 AND ocr_confidence <= 1);
    """)

    # Ensure chunk indexes are non-negative
    op.execute("""
        ALTER TABLE document_embeddings
        ADD CONSTRAINT check_chunk_index_positive
        CHECK (chunk_index >= 0);
    """)

    # Ensure image indexes are non-negative
    op.execute("""
        ALTER TABLE document_images
        ADD CONSTRAINT check_image_index_positive
        CHECK (image_index >= 0);
    """)

    # PERFORMANCE: Add indexes for frequently queried columns

    # Index on filename for search operations
    op.create_index(
        'idx_documents_filename',
        'documents',
        ['filename'],
        postgresql_ops={'filename': 'text_pattern_ops'}  # For LIKE queries
    )

    # Index on created_at for temporal queries and sorting
    op.create_index(
        'idx_documents_created_at',
        'documents',
        ['created_at']
    )

    # Index on conversion_status for filtering
    # Note: This index is already created in migration 001
    # op.create_index(
    #     'idx_documents_status',
    #     'documents',
    #     ['conversion_status']
    # )

    # SECURITY: Make file_hash unique to prevent duplicate processing
    # This prevents potential security issues with duplicate hash handling
    op.create_unique_constraint(
        'uq_documents_file_hash',
        'documents',
        ['file_hash']
    )

    # Add composite index for common query patterns
    op.create_index(
        'idx_documents_status_created',
        'documents',
        ['conversion_status', 'created_at']
    )

    # Add index for document embeddings page-based queries
    op.create_index(
        'idx_embeddings_page_chunk',
        'document_embeddings',
        ['document_id', 'page_number', 'chunk_index']
    )


def downgrade() -> None:
    """Remove security constraints and performance improvements."""

    # Drop performance indexes
    op.drop_index('idx_embeddings_page_chunk', table_name='document_embeddings')
    op.drop_index('idx_documents_status_created', table_name='documents')
    op.drop_constraint('uq_documents_file_hash', 'documents', type_='unique')
    # op.drop_index('idx_documents_status', table_name='documents')  # Already handled in migration 001
    op.drop_index('idx_documents_created_at', table_name='documents')
    op.drop_index('idx_documents_filename', table_name='documents')

    # Drop data validation constraints
    op.execute("""
        ALTER TABLE document_images
        DROP CONSTRAINT IF EXISTS check_image_index_positive;
    """)

    op.execute("""
        ALTER TABLE document_embeddings
        DROP CONSTRAINT IF EXISTS check_chunk_index_positive;
    """)

    op.execute("""
        ALTER TABLE document_images
        DROP CONSTRAINT IF EXISTS check_ocr_confidence_range;
    """)

    # Drop critical vector dimension constraints
    op.execute("""
        ALTER TABLE document_images
        DROP CONSTRAINT IF EXISTS check_image_embedding_dimensions;
    """)

    op.execute("""
        ALTER TABLE document_embeddings
        DROP CONSTRAINT IF EXISTS check_text_embedding_dimensions;
    """)