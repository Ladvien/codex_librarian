"""
Unit tests for Celery task definitions.

Tests all task functions with proper mocking and error handling scenarios.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.core.exceptions import (
    ProcessingError,
    ValidationError,
)
from pdf_to_markdown_mcp.worker.tasks import (
    ProgressTracker,
    _calculate_file_hash,
    cleanup_temp_files,
    generate_embeddings,
    health_check,
    process_pdf_batch,
    process_pdf_document,
)


class TestProgressTracker:
    """Test the ProgressTracker helper class."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = Mock()
        task.update_state = Mock()
        return task

    def test_progress_tracker_initialization(self, mock_task):
        """Test ProgressTracker initialization."""
        # When
        tracker = ProgressTracker(mock_task, total_steps=50)

        # Then
        assert tracker.task == mock_task
        assert tracker.current_step == 0
        assert tracker.total_steps == 50
        assert tracker.messages == []

    def test_progress_tracker_update(self, mock_task):
        """Test progress tracking updates."""
        # Given
        tracker = ProgressTracker(mock_task, total_steps=10)

        # When
        tracker.update(current=5, message="Processing step 5")

        # Then
        assert tracker.current_step == 5
        assert "Processing step 5" in tracker.messages
        mock_task.update_state.assert_called_once()

        call_args = mock_task.update_state.call_args[1]
        assert call_args["state"] == "PROGRESS"
        meta = call_args["meta"]
        assert meta["current"] == 5
        assert meta["total"] == 10
        assert meta["message"] == "Processing step 5"
        assert meta["percentage"] == 50.0

    def test_progress_tracker_complete(self, mock_task):
        """Test progress completion."""
        # Given
        tracker = ProgressTracker(mock_task, total_steps=10)

        # When
        tracker.complete("All done!")

        # Then
        assert tracker.current_step == 10
        mock_task.update_state.assert_called()
        call_args = mock_task.update_state.call_args[1]
        assert call_args["meta"]["current"] == 10
        assert call_args["meta"]["message"] == "All done!"


class TestProcessPdfDocument:
    """Test the process_pdf_document task."""

    @pytest.fixture
    def mock_task_instance(self):
        """Create a mock task instance."""
        task = Mock()
        task.update_state = Mock()
        task.request = Mock()
        task.request.retries = 0
        task.max_retries = 3
        task.retry = Mock()
        task._update_document_error = Mock()
        return task

    @pytest.fixture
    def valid_pdf_path(self, tmp_path):
        """Create a valid PDF file path for testing."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")
        return str(pdf_file)

    def test_process_pdf_document_validation_invalid_document_id(
        self, mock_task_instance
    ):
        """Test validation error for invalid document ID."""
        # When/Then
        with pytest.raises(ValidationError, match="Invalid document_id"):
            process_pdf_document.apply(
                args=[None, "/path/to/file.pdf"], throw=True
            ).get()

    def test_process_pdf_document_validation_file_not_found(self, mock_task_instance):
        """Test validation error for non-existent file."""
        # When/Then
        with pytest.raises(ValidationError, match="PDF file not found"):
            process_pdf_document.apply(
                args=[1, "/nonexistent/file.pdf"], throw=True
            ).get()

    def test_process_pdf_document_validation_not_pdf(
        self, tmp_path, mock_task_instance
    ):
        """Test validation error for non-PDF file."""
        # Given
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not a PDF")

        # When/Then
        with pytest.raises(ValidationError, match="File is not a PDF"):
            process_pdf_document.apply(args=[1, str(txt_file)], throw=True).get()

    @patch("pdf_to_markdown_mcp.worker.tasks.settings")
    def test_process_pdf_document_file_too_large(
        self, mock_settings, tmp_path, mock_task_instance
    ):
        """Test validation error for oversized file."""
        # Given
        mock_settings.processing.max_file_size_mb = 1  # 1MB limit

        large_pdf = tmp_path / "large.pdf"
        # Create a file larger than 1MB
        large_pdf.write_bytes(b"%PDF-1.4" + b"x" * (2 * 1024 * 1024))

        # When/Then
        with pytest.raises(ValidationError, match="File size.*exceeds limit"):
            process_pdf_document.apply(args=[1, str(large_pdf)], throw=True).get()

    @patch("pdf_to_markdown_mcp.worker.tasks.MinerUService")
    @patch("pdf_to_markdown_mcp.worker.tasks.EmbeddingService")
    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    @patch("pdf_to_markdown_mcp.worker.tasks.generate_embeddings")
    def test_process_pdf_document_success(
        self,
        mock_generate_embeddings,
        mock_get_db_session,
        mock_embedding_service_class,
        mock_mineru_service_class,
        valid_pdf_path,
    ):
        """Test successful PDF processing."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        mock_document = Mock()
        mock_document.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        mock_mineru = Mock()
        mock_mineru_service_class.return_value = mock_mineru
        mock_mineru.process_pdf.return_value = {
            "markdown": "# Test Document",
            "plain_text": "Test Document content",
            "page_count": 1,
            "has_images": False,
            "has_tables": False,
            "processing_time_ms": 1000,
            "chunks": [{"text": "chunk1", "index": 0}],
        }

        mock_generate_embeddings.delay = Mock()

        # When
        result = process_pdf_document.apply(
            args=[1, valid_pdf_path, {}], throw=True
        ).get()

        # Then
        assert result["status"] == "completed"
        assert result["document_id"] == 1
        assert result["page_count"] == 1
        assert result["markdown_length"] == 15  # len('# Test Document')

        # Verify database operations
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

        # Verify embedding task was queued
        mock_generate_embeddings.delay.assert_called_once()

    @patch("pdf_to_markdown_mcp.worker.tasks.MinerUService")
    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    def test_process_pdf_document_processing_error_retry(
        self, mock_get_db_session, mock_mineru_service_class, valid_pdf_path
    ):
        """Test retry logic for processing errors."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        mock_document = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        mock_mineru = Mock()
        mock_mineru_service_class.return_value = mock_mineru
        mock_mineru.process_pdf.side_effect = ProcessingError("PDF processing failed")

        # When/Then
        with pytest.raises(ProcessingError):
            # This will fail after max retries
            process_pdf_document.apply(args=[1, valid_pdf_path], throw=True).get()


class TestGenerateEmbeddings:
    """Test the generate_embeddings task."""

    @patch("pdf_to_markdown_mcp.worker.tasks.EmbeddingService")
    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    def test_generate_embeddings_success(
        self, mock_get_db_session, mock_embedding_service_class
    ):
        """Test successful embedding generation."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],  # First embedding
            [0.4, 0.5, 0.6],  # Second embedding
        ]

        chunks = [
            {"text": "First chunk", "index": 0, "page_number": 1},
            {"text": "Second chunk", "index": 1, "page_number": 1},
        ]

        # When
        result = generate_embeddings.apply(
            args=[1, "Full content", chunks], throw=True
        ).get()

        # Then
        assert result["status"] == "completed"
        assert result["document_id"] == 1
        assert result["embeddings_generated"] == 2
        assert result["total_chunks"] == 2
        assert result["success_rate"] == 1.0

        # Verify database operations
        assert mock_db.add.call_count == 2  # Two embedding records
        mock_db.commit.assert_called()

    @patch("pdf_to_markdown_mcp.worker.tasks.EmbeddingService")
    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    def test_generate_embeddings_partial_failure(
        self, mock_get_db_session, mock_embedding_service_class
    ):
        """Test embedding generation with partial batch failures."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        # First batch succeeds, second batch fails
        mock_embedding_service.generate_embeddings.side_effect = [
            [[0.1, 0.2, 0.3]],  # First batch success
            Exception("API failure"),  # Second batch failure
        ]

        chunks = [
            {"text": "First chunk", "index": 0},
            {"text": "Second chunk", "index": 1},
        ]

        # When
        result = generate_embeddings.apply(
            args=[1, "Full content", chunks], throw=True
        ).get()

        # Then
        assert result["status"] == "completed"
        assert result["embeddings_generated"] == 1  # Only first batch succeeded
        assert result["total_chunks"] == 2
        assert result["success_rate"] == 0.5


class TestCleanupTempFiles:
    """Test the cleanup_temp_files task."""

    @patch("pdf_to_markdown_mcp.worker.tasks.settings")
    def test_cleanup_temp_files_success(self, mock_settings, tmp_path):
        """Test successful temp file cleanup."""
        # Given
        mock_settings.processing.temp_dir = tmp_path

        # Create some old and new files
        old_file = tmp_path / "old_file.txt"
        new_file = tmp_path / "new_file.txt"

        old_file.write_text("old content")
        new_file.write_text("new content")

        # Make old_file actually old by modifying its timestamp
        import os
        import time

        old_timestamp = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_file, (old_timestamp, old_timestamp))

        # When
        result = cleanup_temp_files.apply(throw=True).get()

        # Then
        assert result["status"] == "completed"
        assert result["files_removed"] == 1
        assert result["space_freed_mb"] > 0
        assert not old_file.exists()  # Old file should be removed
        assert new_file.exists()  # New file should remain

    @patch("pdf_to_markdown_mcp.worker.tasks.settings")
    def test_cleanup_temp_files_no_directory(self, mock_settings, tmp_path):
        """Test cleanup when temp directory doesn't exist."""
        # Given
        mock_settings.processing.temp_dir = tmp_path / "nonexistent"

        # When
        result = cleanup_temp_files.apply(throw=True).get()

        # Then
        assert result["status"] == "completed"
        assert result["files_removed"] == 0
        assert result["space_freed_mb"] == 0


class TestHealthCheck:
    """Test the health_check task."""

    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    @patch("pdf_to_markdown_mcp.worker.tasks.EmbeddingService")
    @patch("pdf_to_markdown_mcp.worker.tasks.settings")
    def test_health_check_all_healthy(
        self, mock_settings, mock_embedding_service_class, mock_get_db_session
    ):
        """Test health check when all services are healthy."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db
        mock_db.execute = Mock()

        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service
        mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2]]

        mock_settings.processing.temp_dir = Path("/tmp")

        # When
        result = health_check.apply(throw=True).get()

        # Then
        assert result["status"] == "healthy"
        assert result["checks"]["database"] == "healthy"
        assert result["checks"]["embedding_service"] == "healthy"
        assert result["checks"]["temp_directory"] == "healthy"

    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    def test_health_check_database_failure(self, mock_get_db_session):
        """Test health check with database failure."""
        # Given
        mock_get_db_session.side_effect = Exception("DB connection failed")

        # When
        result = health_check.apply(throw=True).get()

        # Then
        assert result["status"] == "degraded"
        assert "unhealthy: DB connection failed" in result["checks"]["database"]


class TestUtilityFunctions:
    """Test utility functions used by tasks."""

    def test_calculate_file_hash(self, tmp_path):
        """Test file hash calculation."""
        # Given
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        # When
        hash1 = _calculate_file_hash(test_file)
        hash2 = _calculate_file_hash(test_file)

        # Then
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hash length
        assert hash1 == hash2  # Same file should produce same hash

        # Different content should produce different hash
        test_file.write_text("Different content")
        hash3 = _calculate_file_hash(test_file)
        assert hash3 != hash1


class TestProcessPdfBatch:
    """Test the process_pdf_batch task."""

    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    @patch("pdf_to_markdown_mcp.worker.tasks.process_pdf_document")
    def test_process_pdf_batch_success(
        self, mock_process_task, mock_get_db_session, tmp_path
    ):
        """Test successful batch processing."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        # Mock document creation
        mock_document = Mock()
        mock_document.id = 1
        mock_db.add = Mock()
        mock_db.commit = Mock()

        # Create test files
        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.write_bytes(b"%PDF-1.4 content1")
        pdf2.write_bytes(b"%PDF-1.4 content2")

        mock_process_task.delay = Mock()

        # When
        result = process_pdf_batch.apply(
            args=[[str(pdf1), str(pdf2)]], throw=True
        ).get()

        # Then
        assert result["total_files"] == 2
        assert result["successful"] == 2
        assert result["failed"] == 0
        assert len(result["file_results"]) == 2

        # Verify tasks were queued
        assert mock_process_task.delay.call_count == 2

    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    def test_process_pdf_batch_with_duplicates(self, mock_get_db_session, tmp_path):
        """Test batch processing with duplicate files."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        # Mock existing document (duplicate)
        mock_existing_doc = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_existing_doc
        )

        pdf1 = tmp_path / "duplicate.pdf"
        pdf1.write_bytes(b"%PDF-1.4 duplicate content")

        # When
        result = process_pdf_batch.apply(args=[[str(pdf1)]], throw=True).get()

        # Then
        assert result["total_files"] == 1
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["file_results"][0]["status"] == "skipped"
        assert result["file_results"][0]["reason"] == "duplicate"
