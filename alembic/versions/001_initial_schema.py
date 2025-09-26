"""Initial database schema with PGVector support

Revision ID: 001
Revises:
Create Date: 2025-09-25 12:00:00.000000

This migration creates the complete database schema for the PDF to Markdown MCP Server:
- documents: Main document metadata and processing status
- document_content: Converted Markdown and plain text content
- document_embeddings: Vector embeddings for semantic search (1536 dimensions)
- document_images: Extracted images with CLIP embeddings (512 dimensions)
- processing_queue: Task queue management

All tables include proper indexes for performance, foreign key relationships,
and check constraints for data integrity.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables and indexes for the PDF to Markdown MCP Server."""

    # Enable PGVector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_path', sa.Text(), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('file_hash', sa.Text(), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('conversion_status', sa.String(length=50), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("conversion_status IN ('pending', 'processing', 'completed', 'failed')", name='check_conversion_status'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_path')
    )

    # Create document_content table
    op.create_table(
        'document_content',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('markdown_content', sa.Text(), nullable=True),
        sa.Column('plain_text', sa.Text(), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('has_images', sa.Boolean(), nullable=True, default=False),
        sa.Column('has_tables', sa.Boolean(), nullable=True, default=False),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create document_embeddings table
    op.create_table(
        'document_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),  # Will be vector(1536) with pgvector
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create document_images table
    op.create_table(
        'document_images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('image_index', sa.Integer(), nullable=True),
        sa.Column('image_path', sa.Text(), nullable=True),
        sa.Column('ocr_text', sa.Text(), nullable=True),
        sa.Column('ocr_confidence', sa.Float(), nullable=True),
        sa.Column('image_embedding', postgresql.ARRAY(sa.Float()), nullable=True),  # Will be vector(512) with pgvector
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create processing_queue table
    op.create_table(
        'processing_queue',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=True, default=5),
        sa.Column('status', sa.String(length=50), nullable=True, default='queued'),
        sa.Column('attempts', sa.Integer(), nullable=True, default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('worker_id', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed', 'retrying')", name='check_queue_status'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create performance indexes

    # Documents indexes
    op.create_index('idx_documents_status', 'documents', ['conversion_status'])
    op.create_index('idx_documents_created', 'documents', ['created_at'], postgresql_using='btree')
    op.create_index('idx_documents_source_path', 'documents', ['source_path'])
    op.create_index('idx_documents_file_hash', 'documents', ['file_hash'])

    # Document content indexes
    op.create_index('idx_content_document', 'document_content', ['document_id'])

    # Document embeddings indexes
    op.create_index('idx_embeddings_document', 'document_embeddings', ['document_id'])

    # Document images indexes
    op.create_index('idx_images_document', 'document_images', ['document_id'])

    # Processing queue indexes
    op.create_index('idx_queue_status', 'processing_queue', ['status'])
    op.create_index('idx_queue_created', 'processing_queue', ['created_at'])
    op.create_index('idx_queue_priority', 'processing_queue', ['priority'])

    # Full-text search index on document content (PostgreSQL specific)
    op.execute("""
        CREATE INDEX idx_content_fulltext
        ON document_content
        USING gin(to_tsvector('english', COALESCE(plain_text, '')));
    """)

    # Note: Vector indexes for embeddings will be created after pgvector is properly installed
    # These would be:
    # CREATE INDEX idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
    # CREATE INDEX idx_images_vector ON document_images USING ivfflat (image_embedding vector_cosine_ops);


def downgrade() -> None:
    """Drop all tables and indexes."""

    # Drop indexes first
    op.drop_index('idx_content_fulltext', table_name='document_content')
    op.drop_index('idx_queue_priority', table_name='processing_queue')
    op.drop_index('idx_queue_created', table_name='processing_queue')
    op.drop_index('idx_queue_status', table_name='processing_queue')
    op.drop_index('idx_images_document', table_name='document_images')
    op.drop_index('idx_embeddings_document', table_name='document_embeddings')
    op.drop_index('idx_content_document', table_name='document_content')
    op.drop_index('idx_documents_file_hash', table_name='documents')
    op.drop_index('idx_documents_source_path', table_name='documents')
    op.drop_index('idx_documents_created', table_name='documents')
    op.drop_index('idx_documents_status', table_name='documents')

    # Drop tables
    op.drop_table('processing_queue')
    op.drop_table('document_images')
    op.drop_table('document_embeddings')
    op.drop_table('document_content')
    op.drop_table('documents')

    # Note: We don't drop the vector extension as other databases might use it