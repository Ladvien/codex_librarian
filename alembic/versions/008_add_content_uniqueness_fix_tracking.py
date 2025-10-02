"""Add unique constraint to document_content and fix embedding count tracking

Revision ID: 008
Revises: 007
Create Date: 2025-10-02

This migration:
1. Adds UNIQUE constraint on document_content.document_id to prevent duplicates
2. Adds database trigger to automatically maintain embedding_count
3. Adds indexes for better query performance
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    """Add uniqueness constraint and embedding count tracking."""

    # 1. Add UNIQUE constraint on document_id (prevents duplicate content records)
    op.create_unique_constraint(
        'uq_document_content_document_id',
        'document_content',
        ['document_id']
    )

    # 2. Create function to update embedding_count automatically
    op.execute("""
        CREATE OR REPLACE FUNCTION update_embedding_count()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update embedding_count in document_content when embeddings change
            UPDATE document_content
            SET embedding_count = (
                SELECT COUNT(*)
                FROM document_embeddings
                WHERE document_id = COALESCE(NEW.document_id, OLD.document_id)
            )
            WHERE document_id = COALESCE(NEW.document_id, OLD.document_id);

            RETURN COALESCE(NEW, OLD);
        END;
        $$ LANGUAGE plpgsql;
    """)

    # 3. Create trigger on document_embeddings to maintain count
    op.execute("""
        CREATE TRIGGER trg_update_embedding_count
        AFTER INSERT OR DELETE ON document_embeddings
        FOR EACH ROW
        EXECUTE FUNCTION update_embedding_count();
    """)

    # 4. Initialize embedding_count for existing records
    op.execute("""
        UPDATE document_content dc
        SET embedding_count = (
            SELECT COUNT(*)
            FROM document_embeddings de
            WHERE de.document_id = dc.document_id
        );
    """)

    # 5. Add index for faster upsert operations
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_document_content_lookup
        ON document_content(document_id)
        WHERE document_id IS NOT NULL;
    """)


def downgrade():
    """Revert changes."""

    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trg_update_embedding_count ON document_embeddings;")
    op.execute("DROP FUNCTION IF EXISTS update_embedding_count();")

    # Drop index
    op.execute("DROP INDEX IF EXISTS idx_document_content_lookup;")

    # Drop unique constraint
    op.drop_constraint('uq_document_content_document_id', 'document_content', type_='unique')
