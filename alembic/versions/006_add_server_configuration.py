"""add server_configuration table

Revision ID: 006
Revises: 005
Create Date: 2025-09-27 02:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create server_configuration table for runtime config persistence."""

    # Create server_configuration table
    op.create_table(
        'server_configuration',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('config_key', sa.String(), nullable=False),
        sa.Column('config_value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_by', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('config_key', name='uq_server_configuration_config_key')
    )

    # Create index on config_key for fast lookups
    op.create_index(
        'ix_server_configuration_config_key',
        'server_configuration',
        ['config_key'],
        unique=True
    )

    # Create index on updated_at for audit queries
    op.create_index(
        'ix_server_configuration_updated_at',
        'server_configuration',
        ['updated_at']
    )

    # Seed initial configuration from environment defaults
    # These will be populated at runtime from .env if not already set
    op.execute("""
        INSERT INTO server_configuration (config_key, config_value, description)
        VALUES
            ('watch_directories', '[]'::jsonb, 'List of directories to monitor for PDFs'),
            ('output_directory', '""'::jsonb, 'Output directory for markdown files'),
            ('file_patterns', '["*.pdf", "*.PDF"]'::jsonb, 'File patterns to match')
        ON CONFLICT (config_key) DO NOTHING;
    """)


def downgrade() -> None:
    """Drop server_configuration table."""

    # Drop indexes first
    op.drop_index('ix_server_configuration_updated_at', table_name='server_configuration')
    op.drop_index('ix_server_configuration_config_key', table_name='server_configuration')

    # Drop table
    op.drop_table('server_configuration')