"""Unit tests for TaskQueue interface."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.core.exceptions import (
    QueueError,
    ValidationError,
)
from pdf_to_markdown_mcp.core.task_queue import TaskQueue, create_task_queue


class TestTaskQueue:
    """Test TaskQueue following TDD"""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        session = Mock()
        session.add = Mock()
        session.flush = Mock()
        session.commit = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session

    @pytest.fixture
    def mock_session_factory(self, mock_session):
        """Create mock session factory"""
        return Mock(return_value=mock_session)

    @pytest.fixture
    def task_queue(self, mock_session_factory):
        """Create TaskQueue with mocked dependencies"""
        return TaskQueue(db_session_factory=mock_session_factory)

    @pytest.fixture
    def valid_validation_result(self):
        """Create valid validation result"""
        return {
            "valid": True,
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "hash": "abc123def456",
            "error": None,
        }

    def test_queue_pdf_processing_with_valid_file(
        self, task_queue, mock_session, valid_validation_result
    ):
        """Test that valid PDF files are queued successfully"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.flush()
            file_path = temp_file.name

            # Mock database queries
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
                mock_task.delay.return_value = Mock(id="task_123")

                # Mock document creation
                mock_document = Mock()
                mock_document.id = 42
                mock_session.add.side_effect = lambda obj: setattr(obj, "id", 42)

                # When (Act)
                result = task_queue.queue_pdf_processing(
                    file_path, valid_validation_result
                )

                # Then (Assert)
                assert result == 42
                mock_session.add.assert_called()
                mock_task.delay.assert_called_once()

                # Verify task was called with correct parameters
                args, kwargs = mock_task.delay.call_args
                assert args[0] == 42  # document_id
                assert Path(file_path).absolute() == Path(args[1])  # file_path

    def test_queue_pdf_processing_with_invalid_validation_result(
        self, task_queue, valid_validation_result
    ):
        """Test that invalid validation results raise ValidationError"""
        # Given (Arrange)
        invalid_result = valid_validation_result.copy()
        invalid_result["valid"] = False
        invalid_result["error"] = "Invalid MIME type"

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            file_path = temp_file.name

            # When (Act) & Then (Assert)
            with pytest.raises(ValidationError) as exc_info:
                task_queue.queue_pdf_processing(file_path, invalid_result)

            assert "Cannot queue invalid file" in str(exc_info.value)
            assert "Invalid MIME type" in str(exc_info.value)

    def test_queue_pdf_processing_with_nonexistent_file(
        self, task_queue, valid_validation_result
    ):
        """Test that nonexistent files raise ValidationError"""
        # Given (Arrange)
        nonexistent_path = "/nonexistent/file.pdf"

        # When (Act) & Then (Assert)
        with pytest.raises(ValidationError) as exc_info:
            task_queue.queue_pdf_processing(nonexistent_path, valid_validation_result)

        assert "File does not exist" in str(exc_info.value)

    def test_queue_pdf_processing_with_existing_completed_document(
        self, task_queue, mock_session, valid_validation_result
    ):
        """Test that already processed documents are not requeued"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.flush()
            file_path = temp_file.name

            # Mock existing completed document
            existing_doc = Mock()
            existing_doc.id = 42
            existing_doc.conversion_status = "completed"

            with patch(
                "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
            ) as mock_doc_queries:
                mock_doc_queries.get_by_path.return_value = existing_doc

                # When (Act)
                result = task_queue.queue_pdf_processing(
                    file_path, valid_validation_result
                )

                # Then (Assert)
                assert result == 42
                mock_session.add.assert_not_called()

    def test_queue_pdf_processing_with_existing_queued_document(
        self, task_queue, mock_session, valid_validation_result
    ):
        """Test that already queued documents are not requeued"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.flush()
            file_path = temp_file.name

            # Mock existing document
            existing_doc = Mock()
            existing_doc.id = 42
            existing_doc.conversion_status = "pending"

            # Mock existing queue entry
            existing_queue = Mock()
            existing_queue.status = "queued"

            with (
                patch(
                    "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
                ) as mock_doc_queries,
                patch(
                    "pdf_to_markdown_mcp.core.task_queue.QueueQueries"
                ) as mock_queue_queries,
            ):
                mock_doc_queries.get_by_path.return_value = existing_doc
                mock_queue_queries.get_by_file_path.return_value = existing_queue

                # When (Act)
                result = task_queue.queue_pdf_processing(
                    file_path, valid_validation_result
                )

                # Then (Assert)
                assert result == 42
                # Should not add new queue entry
                mock_session.add.assert_not_called()

    def test_queue_pdf_processing_with_celery_task_failure(
        self, task_queue, mock_session, valid_validation_result
    ):
        """Test that Celery task creation failures are handled"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.flush()
            file_path = temp_file.name

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
                mock_task.delay.side_effect = Exception("Celery connection failed")

                # Mock document creation
                mock_document = Mock()
                mock_document.id = 42
                mock_session.add.side_effect = lambda obj: setattr(obj, "id", 42)

                # When (Act) & Then (Assert)
                with pytest.raises(QueueError) as exc_info:
                    task_queue.queue_pdf_processing(file_path, valid_validation_result)

                assert "Failed to create Celery task" in str(exc_info.value)

    def test_get_queue_status_success(self, task_queue, mock_session):
        """Test that queue status is retrieved successfully"""
        # Given (Arrange)
        mock_stats = {"queued": 5, "processing": 2, "completed": 10, "failed": 1}

        with patch(
            "pdf_to_markdown_mcp.core.task_queue.QueueQueries"
        ) as mock_queue_queries:
            mock_queue_queries.get_queue_statistics.return_value = mock_stats

            # When (Act)
            result = task_queue.get_queue_status()

            # Then (Assert)
            assert result["total_queued"] == 5
            assert result["total_processing"] == 2
            assert result["total_completed"] == 10
            assert result["total_failed"] == 1
            assert result["queue_depth"] == 7  # queued + processing

    def test_get_queue_status_with_database_error(self, task_queue, mock_session):
        """Test that database errors in get_queue_status are handled gracefully"""
        # Given (Arrange)
        with patch(
            "pdf_to_markdown_mcp.core.task_queue.QueueQueries"
        ) as mock_queue_queries:
            mock_queue_queries.get_queue_statistics.side_effect = Exception(
                "Database connection failed"
            )

            # When (Act)
            result = task_queue.get_queue_status()

            # Then (Assert)
            assert "error" in result
            assert result["total_queued"] == 0
            assert result["queue_depth"] == 0

    def test_clear_completed_entries(self, task_queue, mock_session):
        """Test clearing completed entries"""
        # Given (Arrange)
        with patch(
            "pdf_to_markdown_mcp.core.task_queue.QueueQueries"
        ) as mock_queue_queries:
            mock_queue_queries.clear_old_entries.return_value = 5

            # When (Act)
            result = task_queue.clear_completed_entries(older_than_days=7)

            # Then (Assert)
            assert result == 5
            mock_queue_queries.clear_old_entries.assert_called_once_with(
                mock_session, 7
            )

    def test_retry_failed_processing_success(self, task_queue, mock_session):
        """Test successful retry of failed processing"""
        # Given (Arrange)
        mock_document = Mock()
        mock_document.id = 42
        mock_document.conversion_status = "failed"
        mock_document.source_path = "/path/to/failed.pdf"

        with (
            patch(
                "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
            ) as mock_doc_queries,
            patch(
                "pdf_to_markdown_mcp.core.task_queue.process_pdf_document"
            ) as mock_task,
        ):
            mock_doc_queries.get_by_id.return_value = mock_document
            mock_task.delay.return_value = Mock(id="retry_task_123")

            # When (Act)
            result = task_queue.retry_failed_processing(42)

            # Then (Assert)
            assert result is True
            assert mock_document.conversion_status == "pending"
            assert mock_document.error_message is None
            mock_task.delay.assert_called_once_with(
                document_id=42, file_path="/path/to/failed.pdf", processing_options={}
            )

    def test_retry_failed_processing_with_nonexistent_document(
        self, task_queue, mock_session
    ):
        """Test retry with nonexistent document"""
        # Given (Arrange)
        with patch(
            "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
        ) as mock_doc_queries:
            mock_doc_queries.get_by_id.return_value = None

            # When (Act)
            result = task_queue.retry_failed_processing(999)

            # Then (Assert)
            assert result is False

    def test_retry_failed_processing_with_non_failed_status(
        self, task_queue, mock_session
    ):
        """Test retry with document that hasn't failed"""
        # Given (Arrange)
        mock_document = Mock()
        mock_document.id = 42
        mock_document.conversion_status = "completed"

        with patch(
            "pdf_to_markdown_mcp.core.task_queue.DocumentQueries"
        ) as mock_doc_queries:
            mock_doc_queries.get_by_id.return_value = mock_document

            # When (Act)
            result = task_queue.retry_failed_processing(42)

            # Then (Assert)
            assert result is False


class TestCreateTaskQueue:
    """Test create_task_queue factory function"""

    def test_create_task_queue_with_default_session(self):
        """Test factory function with default session factory"""
        # When (Act)
        queue = create_task_queue()

        # Then (Assert)
        assert isinstance(queue, TaskQueue)
        assert queue.get_session is not None

    def test_create_task_queue_with_custom_session(self):
        """Test factory function with custom session factory"""
        # Given (Arrange)
        custom_session_factory = Mock()

        # When (Act)
        queue = create_task_queue(custom_session_factory)

        # Then (Assert)
        assert isinstance(queue, TaskQueue)
        assert queue.get_session is custom_session_factory
