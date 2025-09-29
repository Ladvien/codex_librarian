"""Fix embedding dimensions and add tracking fields

Revision ID: 007_fix_embedding_dimensions
Revises: 006_add_server_configuration
Create Date: 2024-01-29

This migration:
1. Drops and recreates the embedding column with correct dimensions (768 for nomic-embed-text)
2. Adds tracking fields to document_content table
3. Updates constraints and indexes
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    """Fix embedding dimensions and add tracking fields."""

    # 1. First, backup any existing embeddings (though we have none)
    op.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings_backup AS
        SELECT * FROM document_embeddings;
    """)

    # 2. Drop the existing constraints and indexes on embedding column
    op.execute("ALTER TABLE document_embeddings DROP CONSTRAINT IF EXISTS check_text_embedding_dimensions")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_euclidean")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_hnsw_cosine")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_inner_product")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_vector")

    # 3. Drop and recreate the embedding column with correct dimensions
    op.drop_column('document_embeddings', 'embedding')
    op.add_column('document_embeddings',
        sa.Column('embedding', Vector(768), nullable=True)
    )

    # 4. Add new constraint for 768 dimensions
    op.execute("""
        ALTER TABLE document_embeddings
        ADD CONSTRAINT check_text_embedding_dimensions
        CHECK (vector_dims(embedding) = 768)
    """)

    # 5. Recreate vector indexes with correct dimensions
    op.execute("""
        CREATE INDEX idx_embeddings_hnsw_cosine
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    op.execute("""
        CREATE INDEX idx_embeddings_ivfflat_cosine
        ON document_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # 6. Add tracking fields to document_content table
    op.add_column('document_content',
        sa.Column('embedding_status', sa.String(50),
                  server_default='pending', nullable=True)
    )
    op.add_column('document_content',
        sa.Column('embedding_generated_at', sa.DateTime(), nullable=True)
    )
    op.add_column('document_content',
        sa.Column('embedding_count', sa.Integer(), server_default='0', nullable=True)
    )
    op.add_column('document_content',
        sa.Column('embedding_error', sa.Text(), nullable=True)
    )

    # 7. Add index on embedding_status for efficient queries
    op.create_index('idx_document_content_embedding_status',
                    'document_content', ['embedding_status'])

    # 8. Update config in server_configuration table if it exists
    op.execute("""
        UPDATE server_configuration
        SET config_value = '{"dimensions": 768}'::jsonb
        WHERE config_key = 'embedding_dimensions'
    """)

    print("Migration complete: Embedding dimensions fixed to 768, tracking fields added")


def downgrade():
    """Revert embedding dimensions and remove tracking fields."""

    # Remove tracking fields from document_content
    op.drop_index('idx_document_content_embedding_status', 'document_content')
    op.drop_column('document_content', 'embedding_status')
    op.drop_column('document_content', 'embedding_generated_at')
    op.drop_column('document_content', 'embedding_count')
    op.drop_column('document_content', 'embedding_error')

    # Drop new indexes
    op.execute("DROP INDEX IF EXISTS idx_embeddings_hnsw_cosine")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_ivfflat_cosine")
    op.execute("ALTER TABLE document_embeddings DROP CONSTRAINT IF EXISTS check_text_embedding_dimensions")

    # Revert embedding column to 1536 dimensions
    op.drop_column('document_embeddings', 'embedding')
    op.add_column('document_embeddings',
        sa.Column('embedding', Vector(1536), nullable=True)
    )

    # Restore old constraint
    op.execute("""
        ALTER TABLE document_embeddings
        ADD CONSTRAINT check_text_embedding_dimensions
        CHECK (vector_dims(embedding) = 1536)
    """)

    # Restore from backup if needed
    op.execute("""
        DROP TABLE IF EXISTS document_embeddings_backup
    """)