"""
Real PostgreSQL database fixtures for integration and e2e tests.

These fixtures connect to the actual PostgreSQL + PGVector database
configured in .env, NOT SQLite. Use these for integration/e2e tests only.

For unit tests, use the mocked fixtures from conftest.py.
"""

import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from src.pdf_to_markdown_mcp.db.models import Base


# Real database configuration from .env
REAL_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://codex_librarian:PdfMcp2025Secure@192.168.1.104:5432/codex_librarian"
)

# Convert to async URL for async operations
REAL_DATABASE_URL_ASYNC = REAL_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


def is_postgresql_available() -> bool:
    """Check if PostgreSQL database is available."""
    try:
        engine = create_engine(REAL_DATABASE_URL, poolclass=NullPool)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception as e:
        print(f"PostgreSQL not available: {e}")
        return False


def is_pgvector_available() -> bool:
    """Check if PGVector extension is installed."""
    try:
        engine = create_engine(REAL_DATABASE_URL, poolclass=NullPool)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            exists = result.scalar()
        engine.dispose()
        return exists
    except Exception:
        return False


# Skip integration tests if PostgreSQL not available
require_postgresql = pytest.mark.skipif(
    not is_postgresql_available(),
    reason="PostgreSQL database not available (check DATABASE_URL in .env)",
)

require_pgvector = pytest.mark.skipif(
    not is_pgvector_available(),
    reason="PGVector extension not installed in PostgreSQL",
)


@pytest.fixture(scope="session")
def real_db_url() -> str:
    """Real PostgreSQL database URL from .env."""
    return REAL_DATABASE_URL


@pytest.fixture(scope="session")
def real_db_url_async() -> str:
    """Real PostgreSQL async database URL."""
    return REAL_DATABASE_URL_ASYNC


@pytest.fixture(scope="session")
def real_sync_engine():
    """
    Create synchronous database engine for real PostgreSQL.

    Session-scoped to reuse connection pool across all tests.
    Uses NullPool to avoid connection pool issues in tests.
    """
    if not is_postgresql_available():
        pytest.skip("PostgreSQL database not available")

    engine = create_engine(
        REAL_DATABASE_URL,
        poolclass=NullPool,  # No connection pooling in tests
        echo=False,  # Set to True for SQL debug logging
    )

    yield engine

    engine.dispose()


@pytest.fixture(scope="session")
async def real_async_engine():
    """
    Create asynchronous database engine for real PostgreSQL.

    Session-scoped to reuse connection pool across all tests.
    """
    if not is_postgresql_available():
        pytest.skip("PostgreSQL database not available")

    engine = create_async_engine(
        REAL_DATABASE_URL_ASYNC,
        poolclass=NullPool,
        echo=False,
    )

    yield engine

    await engine.dispose()


@pytest.fixture
def real_db_session(real_sync_engine) -> Generator[Session, None, None]:
    """
    Create real database session for synchronous tests.

    Each test gets its own transaction that is rolled back after the test,
    ensuring test isolation without modifying the real database.
    """
    SessionLocal = sessionmaker(bind=real_sync_engine)
    session = SessionLocal()

    # Begin a transaction
    session.begin()

    try:
        yield session
    finally:
        # Rollback transaction to undo any changes
        session.rollback()
        session.close()


@pytest.fixture
async def real_async_db_session(real_async_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create real async database session for async tests.

    Each test gets its own transaction that is rolled back after the test,
    ensuring test isolation without modifying the real database.
    """
    from sqlalchemy.ext.asyncio import AsyncSession as AsyncSessionClass

    AsyncSessionLocal = sessionmaker(
        bind=real_async_engine,
        class_=AsyncSessionClass,
        expire_on_commit=False,
    )

    async with AsyncSessionLocal() as session:
        # Begin a transaction
        await session.begin()

        try:
            yield session
        finally:
            # Rollback transaction to undo any changes
            await session.rollback()
            await session.close()


@pytest.fixture
def real_db_session_factory(real_sync_engine):
    """
    Factory for creating multiple database sessions in a single test.

    Useful for testing concurrent database operations or
    simulating multiple clients.
    """
    def _create_session() -> Session:
        SessionLocal = sessionmaker(bind=real_sync_engine)
        return SessionLocal()

    return _create_session


@asynccontextmanager
async def get_real_async_session(engine):
    """Context manager for getting real async database sessions."""
    from sqlalchemy.ext.asyncio import AsyncSession as AsyncSessionClass

    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSessionClass,
        expire_on_commit=False,
    )

    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture
def verify_pgvector():
    """
    Verify PGVector extension is available.

    Use this fixture when your test requires vector operations.
    """
    if not is_pgvector_available():
        pytest.skip("PGVector extension not available in PostgreSQL")
    return True


@pytest.fixture
def clean_test_data(real_db_session):
    """
    Clean test data before and after test execution.

    Use this fixture when you need to ensure a clean database state.
    Note: This fixture actually modifies the database and commits changes.
    """
    def _clean():
        # Clean up test data (documents created during tests)
        real_db_session.execute(
            text("DELETE FROM documents WHERE file_name LIKE 'test_%'")
        )
        real_db_session.execute(
            text("DELETE FROM document_content WHERE content LIKE '%TEST_%'")
        )
        real_db_session.commit()

    # Clean before test
    _clean()

    yield

    # Clean after test
    _clean()


@pytest.fixture(scope="session")
def check_database_schema(real_sync_engine):
    """
    Verify database schema is up to date.

    Checks that all required tables exist.
    """
    required_tables = [
        "documents",
        "document_content",
        "document_embeddings",
        "path_mappings",
        "server_configuration",
    ]

    with real_sync_engine.connect() as conn:
        for table in required_tables:
            result = conn.execute(
                text(
                    f"SELECT EXISTS (SELECT FROM information_schema.tables "
                    f"WHERE table_name = '{table}')"
                )
            )
            exists = result.scalar()
            if not exists:
                pytest.fail(f"Required table '{table}' does not exist in database")

    return True


@pytest.fixture
def database_stats(real_db_session):
    """
    Get database statistics for monitoring test performance.

    Returns a dict with table row counts.
    """
    def _get_stats():
        stats = {}
        tables = ["documents", "document_content", "document_embeddings", "path_mappings"]

        for table in tables:
            result = real_db_session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            stats[table] = result.scalar()

        return stats

    return _get_stats


# Example usage in tests:
"""
@pytest.mark.integration
@pytest.mark.database
@require_postgresql
def test_real_database_operation(real_db_session):
    # Test runs against real PostgreSQL
    # Changes are rolled back automatically
    pass

@pytest.mark.integration
@pytest.mark.database
@require_postgresql
@require_pgvector
async def test_real_vector_search(real_async_db_session, verify_pgvector):
    # Test runs against real PostgreSQL with PGVector
    # Changes are rolled back automatically
    pass
"""
