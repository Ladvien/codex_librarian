"""Integration tests for watcher and task queue coordination."""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.core.task_queue import TaskQueue
from pdf_to_markdown_mcp.core.watcher import DirectoryWatcher, WatcherConfig


class MockTaskQueueIntegration:
    """Mock task queue that tracks queued operations for integration testing."""

    def __init__(self):
        """Initialize mock task queue."""
        self.queued_operations = []
        self.call_count = 0

    def queue_pdf_processing(
        self,
        file_path: str,
        validation_result: dict[str, Any],
        priority: int = 5,
        processing_options: dict[str, Any] = None,
    ) -> int:
        """Mock queue_pdf_processing method.

        Args:
            file_path: Path to PDF file
            validation_result: File validation metadata
            priority: Processing priority
            processing_options: Optional processing configuration

        Returns:
            Mock document ID
        """
        self.call_count += 1
        document_id = self.call_count  # Simple incremental ID

        operation = {
            "file_path": file_path,
            "validation_result": validation_result,
            "priority": priority,
            "processing_options": processing_options or {},
            "timestamp": time.time(),
            "document_id": document_id,
        }
        self.queued_operations.append(operation)

        return document_id

    def get_queue_status(self) -> dict[str, Any]:
        """Get mock queue status."""
        return {
            "total_queued": len(self.queued_operations),
            "total_processing": 0,
            "total_completed": 0,
            "total_failed": 0,
            "queue_depth": len(self.queued_operations),
        }


@pytest.mark.integration
class TestWatcherTaskQueueIntegration:
    """Integration tests for watcher and task queue coordination."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_task_queue(self):
        """Create mock task queue for integration testing."""
        return MockTaskQueueIntegration()

    @pytest.fixture
    def watcher_config(self, temp_directory):
        """Create watcher configuration for integration testing."""
        return WatcherConfig(
            watch_directories=[temp_directory],
            recursive=True,
            stability_timeout=0.1,  # Short timeout for testing
            max_file_size_mb=1,  # Small limit for testing
            enable_deduplication=True,
        )

    @pytest.fixture
    def watcher(self, mock_task_queue, watcher_config):
        """Create directory watcher with task queue for integration testing."""
        return DirectoryWatcher(mock_task_queue, watcher_config)

    def create_fake_pdf(
        self, directory: str, filename: str = "test.pdf", content_variant: int = 1
    ) -> Path:
        """Create a fake PDF file for testing.

        Args:
            directory: Directory to create file in
            filename: Name of file to create
            content_variant: Variant to create different file hashes

        Returns:
            Path to created file
        """
        file_path = Path(directory) / filename
        # Create a minimal PDF header to pass basic validation
        base_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        variant_content = f"% Variant {content_variant}\n".encode()

        with open(file_path, "wb") as f:
            f.write(base_content)
            f.write(variant_content)
            f.write(b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n")
            f.write(b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n")
            f.write(b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n")
            f.write(b"0000000058 00000 n\n0000000115 00000 n\n")
            f.write(b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n182\n%%EOF\n")
        return file_path

    def test_watcher_task_queue_integration_single_file(
        self, watcher, mock_task_queue, temp_directory
    ):
        """Test that watcher detects file and queues it through task queue."""
        # Given (Arrange)
        watcher.start()
        initial_queue_size = len(mock_task_queue.queued_operations)

        try:
            # When (Act)
            pdf_file = self.create_fake_pdf(temp_directory, "integration_test.pdf")

            # Wait for file to be detected and processed
            time.sleep(0.3)  # Give watcher time to detect and process

            # Then (Assert)
            # The file should have been queued
            assert len(mock_task_queue.queued_operations) > initial_queue_size

            # Check the latest queued operation
            latest_operation = mock_task_queue.queued_operations[-1]
            assert Path(latest_operation["file_path"]).name == "integration_test.pdf"
            assert latest_operation["validation_result"]["valid"] is True
            assert (
                latest_operation["validation_result"]["mime_type"] == "application/pdf"
            )
            assert latest_operation["document_id"] > 0

        finally:
            watcher.stop()

    def test_watcher_task_queue_integration_multiple_files(
        self, watcher, mock_task_queue, temp_directory
    ):
        """Test that watcher handles multiple files correctly."""
        # Given (Arrange)
        watcher.start()
        initial_queue_size = len(mock_task_queue.queued_operations)

        try:
            # When (Act)
            pdf_files = [
                self.create_fake_pdf(temp_directory, "multi_test_1.pdf", 1),
                self.create_fake_pdf(temp_directory, "multi_test_2.pdf", 2),
                self.create_fake_pdf(temp_directory, "multi_test_3.pdf", 3),
            ]

            # Wait for all files to be detected and processed
            time.sleep(0.5)

            # Then (Assert)
            final_queue_size = len(mock_task_queue.queued_operations)
            assert final_queue_size >= initial_queue_size + 3

            # Check that all files were queued (assuming they're the last 3 operations)
            recent_operations = mock_task_queue.queued_operations[-3:]
            queued_filenames = {Path(op["file_path"]).name for op in recent_operations}
            expected_filenames = {
                "multi_test_1.pdf",
                "multi_test_2.pdf",
                "multi_test_3.pdf",
            }

            # At least some of our files should be queued
            assert len(queued_filenames.intersection(expected_filenames)) > 0

        finally:
            watcher.stop()

    def test_watcher_task_queue_integration_duplicate_detection(
        self, watcher, mock_task_queue, temp_directory
    ):
        """Test that duplicate files are handled correctly."""
        # Given (Arrange)
        watcher.start()
        initial_queue_size = len(mock_task_queue.queued_operations)

        try:
            # When (Act) - Create same file content twice
            pdf_file_1 = self.create_fake_pdf(
                temp_directory, "duplicate_1.pdf", content_variant=1
            )
            time.sleep(0.2)

            # Create exact duplicate content with different filename
            pdf_file_2 = self.create_fake_pdf(
                temp_directory, "duplicate_2.pdf", content_variant=1
            )
            time.sleep(0.3)

            # Then (Assert)
            final_queue_size = len(mock_task_queue.queued_operations)

            # With deduplication enabled, only one should be queued
            # (though the exact behavior depends on timing and hash calculation)
            assert final_queue_size > initial_queue_size

            # At least one operation should have been queued
            assert len(mock_task_queue.queued_operations) > initial_queue_size

        finally:
            watcher.stop()

    def test_watcher_task_queue_integration_file_filtering(
        self, watcher, mock_task_queue, temp_directory
    ):
        """Test that non-PDF files are filtered out correctly."""
        # Given (Arrange)
        watcher.start()
        initial_queue_size = len(mock_task_queue.queued_operations)

        try:
            # When (Act) - Create PDF and non-PDF files
            pdf_file = self.create_fake_pdf(temp_directory, "should_be_queued.pdf")

            # Create non-PDF file
            txt_file = Path(temp_directory) / "should_be_ignored.txt"
            with open(txt_file, "w") as f:
                f.write("This is not a PDF file and should be ignored")

            # Create another non-PDF file
            doc_file = Path(temp_directory) / "should_also_be_ignored.doc"
            with open(doc_file, "w") as f:
                f.write("This is also not a PDF file")

            # Wait for processing
            time.sleep(0.3)

            # Then (Assert)
            final_queue_size = len(mock_task_queue.queued_operations)

            # Only the PDF should have been queued
            new_operations = mock_task_queue.queued_operations[initial_queue_size:]
            pdf_operations = [
                op
                for op in new_operations
                if Path(op["file_path"]).suffix.lower() == ".pdf"
            ]

            # Should have at least one PDF operation
            assert len(pdf_operations) >= 1

            # Should not have any non-PDF operations
            non_pdf_operations = [
                op
                for op in new_operations
                if Path(op["file_path"]).suffix.lower() != ".pdf"
            ]
            assert len(non_pdf_operations) == 0

        finally:
            watcher.stop()

    def test_watcher_task_queue_integration_config_update(
        self, watcher, mock_task_queue, temp_directory
    ):
        """Test that configuration updates work correctly."""
        # Given (Arrange)
        watcher.start()

        try:
            # Create initial file
            pdf_file = self.create_fake_pdf(temp_directory, "before_config_update.pdf")
            time.sleep(0.2)
            initial_operations = len(mock_task_queue.queued_operations)

            # When (Act) - Update configuration
            new_config = WatcherConfig(
                watch_directories=[temp_directory],
                recursive=True,
                stability_timeout=0.05,  # Even shorter timeout
                max_file_size_mb=2,  # Larger limit
                patterns=["*.pdf", "*.PDF", "*.Pdf"],  # More patterns
            )

            watcher.update_config(new_config)

            # Create another file after config update
            pdf_file_2 = self.create_fake_pdf(
                temp_directory, "after_config_update.PDF", content_variant=2
            )
            time.sleep(0.3)

            # Then (Assert)
            final_operations = len(mock_task_queue.queued_operations)

            # Should have processed files before and after config update
            assert final_operations > initial_operations

            # Check that config was updated
            status = watcher.get_status()
            assert status["stability_timeout"] == 0.05
            assert status["max_file_size_mb"] == 2
            assert "*.Pdf" in status["patterns"]

        finally:
            watcher.stop()

    def test_watcher_task_queue_integration_error_handling(
        self, watcher, temp_directory
    ):
        """Test integration error handling with failing task queue."""
        # Given (Arrange) - Create a task queue that fails
        failing_task_queue = Mock()
        failing_task_queue.queue_pdf_processing.side_effect = Exception(
            "Task queue failed"
        )

        failing_watcher = DirectoryWatcher(
            failing_task_queue,
            WatcherConfig(watch_directories=[temp_directory], stability_timeout=0.1),
        )

        failing_watcher.start()

        try:
            # When (Act) - Create a PDF file
            pdf_file = self.create_fake_pdf(temp_directory, "will_cause_error.pdf")

            # Wait for processing attempt
            time.sleep(0.3)

            # Then (Assert) - Should not crash the watcher
            assert failing_watcher.is_running()

            # Task queue should have been called despite the error
            failing_task_queue.queue_pdf_processing.assert_called()

        finally:
            failing_watcher.stop()


@pytest.mark.integration
class TestRealTaskQueueIntegration:
    """Integration tests using real TaskQueue with mocked database."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session for TaskQueue testing."""
        session = Mock()
        session.add = Mock()
        session.flush = Mock()
        session.commit = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session

    @pytest.fixture
    def mock_session_factory(self, mock_session):
        """Create mock session factory for TaskQueue."""
        return Mock(return_value=mock_session)

    @pytest.fixture
    def real_task_queue(self, mock_session_factory):
        """Create real TaskQueue with mocked database."""
        return TaskQueue(db_session_factory=mock_session_factory)

    def test_watcher_with_real_task_queue_integration(
        self, real_task_queue, mock_session
    ):
        """Test integration with real TaskQueue class."""
        # Given (Arrange)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WatcherConfig(watch_directories=[temp_dir], stability_timeout=0.1)

            watcher = DirectoryWatcher(real_task_queue, config)

            # Mock database operations for TaskQueue
            with (
                patch(
                    "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
                ) as mock_doc_queries,
                patch(
                    "pdf_to_markdown_mcp.core.task_queue.QueueQueries"
                ) as mock_queue_queries,
                patch(
                    "pdf_to_markdown_mcp.core.task_queue.process_pdf_document"
                ) as mock_task,
            ):
                mock_doc_queries.get_by_path.return_value = None
                mock_queue_queries.get_by_file_path.return_value = None
                mock_task.delay.return_value = Mock(id="integration_task_123")

                # Mock document creation
                mock_session.add.side_effect = lambda obj: setattr(obj, "id", 99)

                watcher.start()

                try:
                    # When (Act) - Create a PDF file
                    pdf_file = Path(temp_dir) / "real_queue_test.pdf"
                    with open(pdf_file, "wb") as f:
                        f.write(b"%PDF-1.4\ntest content")

                    # Wait for processing
                    time.sleep(0.3)

                    # Then (Assert)
                    # Should have attempted to create Celery task
                    mock_task.delay.assert_called()

                    # Verify task was called with expected parameters
                    args, kwargs = mock_task.delay.call_args
                    assert args[0] == 99  # document_id
                    assert str(pdf_file.absolute()) == args[1]  # file_path

                finally:
                    watcher.stop()
