"""
Pytest configuration and fixtures for PDF to Markdown MCP Server tests.

This module provides comprehensive test configuration including:
- Database fixtures with proper setup/teardown
- Mock services for external dependencies
- Test data factories and utilities
- Async test configuration
- Test database isolation
"""

import asyncio
import os
import tempfile
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

# Configure pytest for async testing
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Database Configuration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///./test_pdf_mcp.db")
TEST_DATABASE_URL_ASYNC = os.getenv(
    "TEST_DATABASE_URL_ASYNC", "sqlite+aiosqlite:///./test_pdf_mcp.db"
)


@pytest.fixture(scope="session")
def temp_db_file():
    """Create temporary database file for SQLite tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture(scope="session")
def test_db_url(temp_db_file):
    """Test database URL using temporary file."""
    return f"sqlite:///{temp_db_file}"


@pytest.fixture(scope="session")
def test_db_url_async(temp_db_file):
    """Async test database URL using temporary file."""
    return f"sqlite+aiosqlite:///{temp_db_file}"


@pytest.fixture(scope="session")
def sync_engine(test_db_url):
    """Create synchronous database engine for tests."""
    from src.pdf_to_markdown_mcp.db.models import Base

    engine = create_engine(test_db_url, echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Drop all tables
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="session")
async def async_engine(test_db_url_async):
    """Create asynchronous database engine for tests."""
    from src.pdf_to_markdown_mcp.db.models import Base

    engine = create_async_engine(test_db_url_async, echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def db_session(sync_engine) -> Generator[Session, None, None]:
    """Create database session for synchronous tests."""
    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
async def async_db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for tests."""
    from sqlalchemy.ext.asyncio import AsyncSession as AsyncSessionClass

    AsyncSessionLocal = sessionmaker(
        bind=async_engine, class_=AsyncSessionClass, expire_on_commit=False
    )

    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


# Test Data Factories
@pytest.fixture
def sample_pdf_content() -> bytes:
    """Sample PDF content for testing."""
    # Simple PDF content (not a real PDF, but sufficient for unit tests)
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF"


@pytest.fixture
def sample_pdf_file(tmp_path, sample_pdf_content) -> Path:
    """Create a temporary PDF file for testing."""
    pdf_file = tmp_path / "test_document.pdf"
    pdf_file.write_bytes(sample_pdf_content)
    return pdf_file


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample Markdown content for testing."""
    return """# Test Document

This is a test document with various content types.

## Section 1

Some regular text content.

### Table Example

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

## Section 2

Mathematical formula: $E = mc^2$

More content here...

### Code Example

```python
def hello_world():
    return "Hello, World!"
```

## Conclusion

This is the end of the test document.
"""


@pytest.fixture
def document_factory():
    """Factory for creating test Document instances."""
    from datetime import datetime

    from src.pdf_to_markdown_mcp.db.models import Document

    def _create_document(**kwargs):
        defaults = {
            "file_path": f"/tmp/test_{uuid.uuid4()}.pdf",
            "file_name": f"test_document_{uuid.uuid4()}.pdf",
            "file_size": 12345,
            "file_hash": f"hash_{uuid.uuid4()}",
            "mime_type": "application/pdf",
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Document(**defaults)

    return _create_document


@pytest.fixture
def processing_result_factory():
    """Factory for creating test ProcessingResult instances."""
    from src.pdf_to_markdown_mcp.models.processing import (
        ProcessingMetadata,
        ProcessingResult,
    )

    def _create_processing_result(**kwargs):
        defaults = {
            "success": True,
            "markdown_content": "# Test Document\n\nSample content",
            "plain_text": "Test Document\n\nSample content",
            "chunks": [],
            "tables": [],
            "formulas": [],
            "images": [],
            "metadata": ProcessingMetadata(
                processing_time=1.5,
                page_count=1,
                word_count=100,
                language="en",
                confidence=0.95,
            ),
        }
        defaults.update(kwargs)
        return ProcessingResult(**defaults)

    return _create_processing_result


# Mock Services
@pytest.fixture
def mock_mineru_service():
    """Mock MinerU service for testing."""
    mock = AsyncMock()
    mock.process_pdf.return_value = Mock(
        success=True,
        markdown_content="# Test Document\n\nSample content",
        plain_text="Test Document\n\nSample content",
        chunks=[],
        tables=[],
        formulas=[],
        images=[],
        metadata=Mock(
            processing_time=1.5,
            page_count=1,
            word_count=100,
            language="en",
            confidence=0.95,
        ),
    )
    mock.health_check.return_value = True
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock = AsyncMock()
    mock.generate_embedding.return_value = [0.1] * 1536  # 1536-dimensional vector
    mock.generate_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
    mock.health_check.return_value = True
    return mock


@pytest.fixture
def mock_celery_app():
    """Mock Celery app for testing."""
    mock = Mock()
    mock.send_task.return_value = Mock(id="test-task-id", state="PENDING")
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = Mock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.keys.return_value = []
    return mock


# Test File System
@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for file operations."""
    test_dir = tmp_path / "pdf_test_files"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "database_url": TEST_DATABASE_URL,
        "database_url_async": TEST_DATABASE_URL_ASYNC,
        "redis_url": "redis://localhost:6379/1",
        "celery_broker_url": "redis://localhost:6379/1",
        "celery_result_backend": "redis://localhost:6379/1",
        "mineru_config": {"device": "cpu", "language": "en", "timeout": 300},
        "embedding_config": {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "dimensions": 1536,
        },
    }


# Test Data Collections
@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Sample embedding vectors for testing."""
    return [
        [0.1] * 1536,  # Document 1 embedding
        [0.2] * 1536,  # Document 2 embedding
        [0.3] * 1536,  # Document 3 embedding
    ]


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """Sample text chunks for testing."""
    return [
        {
            "text": "This is the first chunk of text content.",
            "start_char": 0,
            "end_char": 40,
            "token_count": 8,
        },
        {
            "text": "This is the second chunk with different content.",
            "start_char": 41,
            "end_char": 89,
            "token_count": 9,
        },
        {
            "text": "Final chunk containing conclusion and summary.",
            "start_char": 90,
            "end_char": 136,
            "token_count": 7,
        },
    ]


# Performance Testing
@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "memory_usage": [],
        "processing_times": [],
    }
    return metrics


# Integration Test Helpers
@pytest.fixture
def integration_test_setup(
    async_db_session, mock_mineru_service, mock_embedding_service
):
    """Setup for integration tests with all services."""
    return {
        "db_session": async_db_session,
        "mineru_service": mock_mineru_service,
        "embedding_service": mock_embedding_service,
    }


# Error Testing
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "file_not_found": FileNotFoundError("File not found"),
        "permission_denied": PermissionError("Permission denied"),
        "processing_error": Exception("Processing failed"),
        "timeout_error": TimeoutError("Operation timed out"),
        "validation_error": ValueError("Invalid input"),
    }


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "database: mark test as requiring database")
    config.addinivalue_line("markers", "redis: mark test as requiring Redis")
    config.addinivalue_line("markers", "mineru: mark test as requiring MinerU")
    config.addinivalue_line(
        "markers", "embeddings: mark test as requiring embedding service"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests that might be slow
        if any(
            keyword in item.name.lower()
            for keyword in ["batch", "large", "stress", "performance"]
        ):
            item.add_marker(pytest.mark.slow)


# Test Environment Validation
@pytest.fixture(scope="session", autouse=True)
def validate_test_environment():
    """Validate test environment is properly configured."""
    # Check for required environment variables
    required_vars = []
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        pytest.skip(f"Missing required environment variables: {missing_vars}")

    # Validate test database connectivity
    try:
        from sqlalchemy import create_engine

        engine = create_engine(TEST_DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as e:
        pytest.skip(f"Cannot connect to test database: {e}")

    yield  # Run tests

    # Cleanup after all tests
