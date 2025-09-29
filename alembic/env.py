"""
Alembic environment configuration for PDF to Markdown MCP Server.

This module configures Alembic for database migrations, including:
- SQLAlchemy model integration
- PGVector extension support
- Connection configuration
- Migration context setup
"""

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all models to ensure they are registered with SQLAlchemy
from src.pdf_to_markdown_mcp.db.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get database URL from environment or config.

    Returns:
        str: Database connection URL
    """
    # Try environment variable first
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    # Fall back to config file
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Enable comparison of types including custom ones
        compare_type=True,
        # Include PGVector types
        include_name=lambda name, type_, parent_names: True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Override the sqlalchemy.url in the configuration
    config_section = config.get_section(config.config_ini_section)
    config_section["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Enable PGVector extension
        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            connection.commit()
            logger.info("PGVector extension enabled in migration")
        except Exception as e:
            logger.warning(f"Could not enable PGVector extension: {e}")

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Enable comparison of types including custom ones
            compare_type=True,
            # Include server defaults in migrations
            compare_server_default=True,
            # Include PGVector types
            include_name=lambda name, type_, parent_names: True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()