"""
Test utilities and helper functions for PDF to Markdown MCP Server tests.

This module provides common utility functions for test setup, teardown,
file management, and assertion helpers.
"""

import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

from .test_data import SAMPLE_PDF_CONTENT, create_sample_pdf_content

# Logging setup for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFileManager:
    """Manages temporary files and directories for tests."""

    def __init__(self):
        self.temp_dirs: list[Path] = []
        self.temp_files: list[Path] = []

    def create_temp_dir(self, prefix: str = "pdf_test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_temp_file(
        self,
        content: str | bytes = "",
        suffix: str = ".tmp",
        dir: Path | None = None,
    ) -> Path:
        """Create a temporary file with content."""
        if dir is None:
            dir = self.create_temp_dir()

        with tempfile.NamedTemporaryFile(
            mode="wb" if isinstance(content, bytes) else "w",
            suffix=suffix,
            dir=dir,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        self.temp_files.append(temp_path)
        return temp_path

    def cleanup(self):
        """Clean up all temporary files and directories."""
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")

        for dir_path in self.temp_dirs:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir {dir_path}: {e}")

        self.temp_dirs.clear()
        self.temp_files.clear()


# Global test file manager
_test_file_manager = TestFileManager()


def create_temp_pdf(
    content: str | None = None,
    file_name: str = "test.pdf",
    directory: Path | None = None,
) -> Path:
    """Create a temporary PDF file for testing."""
    pdf_content = create_sample_pdf_content(content) if content else SAMPLE_PDF_CONTENT

    if directory is None:
        directory = _test_file_manager.create_temp_dir()

    pdf_path = directory / file_name
    pdf_path.write_bytes(pdf_content)
    _test_file_manager.temp_files.append(pdf_path)

    return pdf_path


def create_test_directory(
    files: dict[str, str | bytes] | None = None, prefix: str = "pdf_test_"
) -> Path:
    """Create a test directory with optional files."""
    test_dir = _test_file_manager.create_temp_dir(prefix)

    if files:
        for file_name, content in files.items():
            file_path = test_dir / file_name
            if isinstance(content, str):
                file_path.write_text(content)
            else:
                file_path.write_bytes(content)
            _test_file_manager.temp_files.append(file_path)

    return test_dir


def cleanup_test_files():
    """Clean up all test files and directories."""
    _test_file_manager.cleanup()


def wait_for_condition(
    condition: Callable[[], bool], timeout: float = 5.0, check_interval: float = 0.1
) -> bool:
    """Wait for a condition to become true within timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(check_interval)
    return False


def assert_processing_result(result: dict[str, Any], expected: dict[str, Any]):
    """Assert that a processing result matches expected values."""
    assert result["success"] == expected["success"], (
        f"Success mismatch: {result['success']} != {expected['success']}"
    )

    if expected["success"]:
        assert "markdown_content" in result, (
            "Missing markdown_content in successful result"
        )
        assert "plain_text" in result, "Missing plain_text in successful result"
        assert "chunks" in result, "Missing chunks in successful result"
        assert "metadata" in result, "Missing metadata in successful result"

        if expected.get("markdown_content"):
            assert result["markdown_content"] == expected["markdown_content"]

        if expected.get("chunk_count") is not None:
            assert len(result["chunks"]) == expected["chunk_count"]

        if expected.get("table_count") is not None:
            assert len(result.get("tables", [])) == expected["table_count"]

        if expected.get("formula_count") is not None:
            assert len(result.get("formulas", [])) == expected["formula_count"]

        # Validate metadata
        metadata = result["metadata"]
        assert "processing_time" in metadata, "Missing processing_time in metadata"
        assert "page_count" in metadata, "Missing page_count in metadata"
        assert "word_count" in metadata, "Missing word_count in metadata"
        assert "language" in metadata, "Missing language in metadata"
        assert "confidence" in metadata, "Missing confidence in metadata"

        if expected.get("min_confidence"):
            assert metadata["confidence"] >= expected["min_confidence"]

    # Failed result checks
    elif expected.get("error_message"):
        assert result.get("error_message") == expected["error_message"]


def assert_database_state(
    session, expected_counts: dict[str, int], table_models: dict[str, Any]
):
    """Assert database contains expected number of records."""
    for table_name, expected_count in expected_counts.items():
        if table_name in table_models:
            model = table_models[table_name]
            actual_count = session.query(model).count()
            assert actual_count == expected_count, (
                f"Table {table_name}: expected {expected_count}, got {actual_count}"
            )


def create_mock_service(
    service_name: str,
    methods: dict[str, Any] | None = None,
    async_methods: list[str] | None = None,
) -> Mock | AsyncMock:
    """Create a mock service with specified methods."""
    if async_methods is None:
        async_methods = []

    if any(method in async_methods for method in (methods or {}).keys()):
        mock = AsyncMock()
    else:
        mock = Mock()

    if methods:
        for method_name, return_value in methods.items():
            if method_name in async_methods:
                getattr(mock, method_name).return_value = return_value
            else:
                setattr(mock, method_name, Mock(return_value=return_value))

    return mock


def create_mock_mineru_service(success: bool = True) -> AsyncMock:
    """Create a mock MinerU service for testing."""
    mock = AsyncMock()

    if success:
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
    else:
        mock.process_pdf.side_effect = Exception("MinerU processing failed")

    mock.health_check.return_value = success
    return mock


def create_mock_embedding_service(dimensions: int = 1536) -> AsyncMock:
    """Create a mock embedding service for testing."""
    mock = AsyncMock()

    # Generate deterministic embeddings
    mock.generate_embedding.return_value = [0.1 + i * 0.001 for i in range(dimensions)]
    mock.generate_batch.return_value = [
        [0.1 + i * 0.001 for i in range(dimensions)],
        [0.2 + i * 0.001 for i in range(dimensions)],
    ]
    mock.health_check.return_value = True

    return mock


def create_mock_database_session(
    query_results: dict[str, list[Any]] | None = None,
) -> Mock:
    """Create a mock database session with configurable query results."""
    mock_session = Mock()

    if query_results:

        def mock_query(model):
            model_name = model.__name__
            results = query_results.get(model_name, [])
            mock_query_obj = Mock()
            mock_query_obj.all.return_value = results
            mock_query_obj.first.return_value = results[0] if results else None
            mock_query_obj.count.return_value = len(results)
            return mock_query_obj

        mock_session.query = mock_query

    mock_session.add = Mock()
    mock_session.commit = Mock()
    mock_session.rollback = Mock()
    mock_session.close = Mock()

    return mock_session


def create_mock_celery_task(task_id: str = "test-task-123") -> Mock:
    """Create a mock Celery task result."""
    mock_task = Mock()
    mock_task.id = task_id
    mock_task.state = "PENDING"
    mock_task.result = None
    mock_task.info = {}

    def get_task_info():
        return {
            "task_id": task_id,
            "state": mock_task.state,
            "result": mock_task.result,
            "info": mock_task.info,
        }

    mock_task.get = Mock(return_value=get_task_info())
    return mock_task


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}
        self._is_connected = True

    def get(self, key: str) -> bytes | None:
        """Get value from mock Redis."""
        value = self._data.get(key)
        return value.encode() if isinstance(value, str) else value

    def set(self, key: str, value: str | bytes, ex: int | None = None) -> bool:
        """Set value in mock Redis."""
        self._data[key] = value
        return True

    def delete(self, *keys: str) -> int:
        """Delete keys from mock Redis."""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    def keys(self, pattern: str = "*") -> list[bytes]:
        """Get keys matching pattern."""
        import fnmatch

        matching_keys = []
        for key in self._data.keys():
            if fnmatch.fnmatch(key, pattern):
                matching_keys.append(key.encode())
        return matching_keys

    def exists(self, key: str) -> int:
        """Check if key exists."""
        return 1 if key in self._data else 0

    def ping(self) -> bool:
        """Ping mock Redis."""
        return self._is_connected

    def flushdb(self):
        """Clear all data."""
        self._data.clear()


def create_performance_monitor():
    """Create a performance monitoring context manager."""
    from contextlib import contextmanager

    import psutil

    @contextmanager
    def monitor():
        """Monitor memory and CPU usage during test execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu = process.cpu_percent()

        start_time = time.time()

        try:
            yield {
                "start_memory": initial_memory,
                "start_cpu": initial_cpu,
                "start_time": start_time,
            }
        finally:
            end_time = time.time()
            final_memory = process.memory_info().rss
            final_cpu = process.cpu_percent()

            logger.info("Performance metrics:")
            logger.info(f"  Duration: {end_time - start_time:.2f}s")
            logger.info(
                f"  Memory change: {(final_memory - initial_memory) / 1024 / 1024:.2f} MB"
            )
            logger.info(f"  CPU usage: {final_cpu:.1f}%")

    return monitor


def assert_similar_vectors(
    vec1: list[float], vec2: list[float], threshold: float = 0.9
):
    """Assert that two vectors are similar (cosine similarity above threshold)."""
    import numpy as np

    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    # Calculate cosine similarity
    dot_product = np.dot(vec1_np, vec2_np)
    norms = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
    cosine_sim = dot_product / norms if norms > 0 else 0

    assert cosine_sim >= threshold, (
        f"Vector similarity {cosine_sim:.3f} below threshold {threshold}"
    )


def setup_test_logging(level: int = logging.INFO):
    """Setup logging for tests with proper formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Export cleanup function to be called in test teardown
__all__ = [
    "MockRedisClient",
    "TestFileManager",
    "assert_database_state",
    "assert_processing_result",
    "assert_similar_vectors",
    "cleanup_test_files",
    "create_mock_celery_task",
    "create_mock_database_session",
    "create_mock_embedding_service",
    "create_mock_mineru_service",
    "create_mock_service",
    "create_performance_monitor",
    "create_temp_pdf",
    "create_test_directory",
    "setup_test_logging",
    "wait_for_condition",
]
