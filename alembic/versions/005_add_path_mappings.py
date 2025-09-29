"""Add path_mappings table and extend documents for directory mirroring

Revision ID: 005
Revises: 004
Create Date: 2025-09-26 00:00:00.000000

This migration adds the path_mappings table for directory structure preservation
and extends the documents table with relative path fields to support mirroring.

New features:
- path_mappings: Directory structure mapping between source and output
- documents: Add relative path fields for structure preservation
- Performance indexes for path-based queries
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add path_mappings table and extend documents with relative paths."""

    # Create path_mappings table for directory structure tracking
    op.create_table(
        'path_mappings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_directory', sa.Text(), nullable=False),
        sa.Column('output_directory', sa.Text(), nullable=False),
        sa.Column('relative_path', sa.Text(), nullable=False),
        sa.Column('directory_level', sa.Integer(), nullable=True),
        sa.Column('files_count', sa.Integer(), nullable=True, default=0),
        sa.Column('last_scanned', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_directory', 'relative_path', name='uq_path_mappings_source_relative')
    )

    # Add relative path columns to documents table
    op.add_column('documents', sa.Column('source_relative_path', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('output_path', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('output_relative_path', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('directory_depth', sa.Integer(), nullable=True))

    # Create indexes for path_mappings table
    op.create_index('idx_path_mappings_source_dir', 'path_mappings', ['source_directory'])
    op.create_index('idx_path_mappings_output_dir', 'path_mappings', ['output_directory'])
    op.create_index('idx_path_mappings_relative_path', 'path_mappings', ['relative_path'])
    op.create_index('idx_path_mappings_directory_level', 'path_mappings', ['directory_level'])
    op.create_index('idx_path_mappings_last_scanned', 'path_mappings', ['last_scanned'])

    # Create indexes for new documents columns
    op.create_index('idx_documents_source_relative_path', 'documents', ['source_relative_path'])
    op.create_index('idx_documents_output_path', 'documents', ['output_path'])
    op.create_index('idx_documents_output_relative_path', 'documents', ['output_relative_path'])
    op.create_index('idx_documents_directory_depth', 'documents', ['directory_depth'])

    # Add check constraints for data integrity
    op.create_check_constraint(
        'check_directory_level_positive',
        'path_mappings',
        'directory_level >= 0'
    )
    op.create_check_constraint(
        'check_files_count_non_negative',
        'path_mappings',
        'files_count >= 0'
    )
    op.create_check_constraint(
        'check_directory_depth_positive',
        'documents',
        'directory_depth >= 0'
    )


def downgrade() -> None:
    """Remove path_mappings table and document relative path columns."""

    # Drop check constraints
    op.drop_constraint('check_directory_depth_positive', 'documents', type_='check')
    op.drop_constraint('check_files_count_non_negative', 'path_mappings', type_='check')
    op.drop_constraint('check_directory_level_positive', 'path_mappings', type_='check')

    # Drop indexes for documents
    op.drop_index('idx_documents_directory_depth', table_name='documents')
    op.drop_index('idx_documents_output_relative_path', table_name='documents')
    op.drop_index('idx_documents_output_path', table_name='documents')
    op.drop_index('idx_documents_source_relative_path', table_name='documents')

    # Drop indexes for path_mappings
    op.drop_index('idx_path_mappings_last_scanned', table_name='path_mappings')
    op.drop_index('idx_path_mappings_directory_level', table_name='path_mappings')
    op.drop_index('idx_path_mappings_relative_path', table_name='path_mappings')
    op.drop_index('idx_path_mappings_output_dir', table_name='path_mappings')
    op.drop_index('idx_path_mappings_source_dir', table_name='path_mappings')

    # Remove columns from documents table
    op.drop_column('documents', 'directory_depth')
    op.drop_column('documents', 'output_relative_path')
    op.drop_column('documents', 'output_path')
    op.drop_column('documents', 'source_relative_path')

    # Drop path_mappings table
    op.drop_table('path_mappings')