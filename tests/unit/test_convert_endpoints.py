"""
Unit tests for convert API endpoints.

Tests convert_single and batch_convert MCP tools following TDD principles.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.models.request import (
    BatchConvertRequest,
    ConvertSingleRequest,
)


class TestConvertSingleEndpoint:
    """Test convert_single endpoint following TDD principles."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_pdf_file(self, tmp_path):
        """Create a mock PDF file for testing."""
        pdf_file = tmp_path / "test.pdf"
        # Create a valid PDF header
        with open(pdf_file, "wb") as f:
            f.write(b"%PDF-1.4\n")
            f.write(b"%" + b"\xe2\xe3\xcf\xd3" + b"\n")  # Binary comment
            f.write(b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
            f.write(b"xref\n0 3\n")
            f.write(b"%%EOF\n")
        return pdf_file

    @pytest.fixture
    def valid_request_data(self, mock_pdf_file):
        """Valid request data for testing."""
        return {
            "file_path": str(mock_pdf_file),
            "output_dir": None,
            "store_embeddings": True,
            "options": {
                "ocr_language": "eng",
                "preserve_layout": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "extract_images": True,
                "extract_tables": True,
                "extract_formulas": True,
                "chunk_for_embeddings": True,
            },
        }

    @pytest.mark.asyncio
    async def test_convert_single_with_embeddings_queues_background_task(
        self, mock_db_session, valid_request_data
    ):
        """Test that convert_single queues background task when store_embeddings=True."""
        # Given
        request = ConvertSingleRequest(**valid_request_data)

        # Mock Celery task
        mock_job = Mock()
        mock_job.id = "job_123"

        with patch("pdf_to_markdown_mcp.api.convert.process_pdf_document") as mock_task:
            mock_task.delay.return_value = mock_job

            # When
            from pdf_to_markdown_mcp.api.convert import convert_single_pdf

            response = await convert_single_pdf(request, Mock(), mock_db_session)

            # Then
            assert response.success is True
            assert response.job_id == "job_123"
            assert response.message == "PDF queued for processing"
            assert response.source_path == Path(valid_request_data["file_path"])
            assert response.file_size_bytes > 0

            # Verify task was called with correct parameters
            mock_task.delay.assert_called_once()
            call_args = mock_task.delay.call_args
            assert call_args[1]["file_path"] == str(request.file_path)
            assert call_args[1]["options"]["ocr_language"] == "eng"

    @pytest.mark.asyncio
    async def test_convert_single_without_embeddings_processes_synchronously(
        self, mock_db_session, valid_request_data
    ):
        """Test that convert_single processes synchronously when store_embeddings=False."""
        # Given
        valid_request_data["store_embeddings"] = False
        request = ConvertSingleRequest(**valid_request_data)

        # Mock processor and result
        mock_result = Mock()
        mock_result.document_id = 42
        mock_result.output_path = Path("/output/test.md")
        mock_result.processing_time_ms = 5000
        mock_result.page_count = 10
        mock_result.chunk_count = 50
        mock_result.has_images = True
        mock_result.has_tables = False

        with patch(
            "pdf_to_markdown_mcp.api.convert.PDFProcessor"
        ) as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_pdf.return_value = mock_result
            mock_processor_class.return_value = mock_processor

            # When
            from pdf_to_markdown_mcp.api.convert import convert_single_pdf

            response = await convert_single_pdf(request, Mock(), mock_db_session)

            # Then
            assert response.success is True
            assert response.document_id == 42
            assert response.job_id is None
            assert response.message == "PDF converted successfully"
            assert response.processing_time_ms == 5000
            assert response.page_count == 10
            assert response.chunk_count == 50
            assert response.embedding_count == 0  # No embeddings for sync processing
            assert response.has_images is True
            assert response.has_tables is False

            # Verify processor was called correctly
            mock_processor.process_pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_single_handles_file_not_found_error(self, mock_db_session):
        """Test that convert_single handles FileNotFoundError properly."""
        # Given
        request_data = {"file_path": "/nonexistent/file.pdf", "store_embeddings": True}

        # When/Then - ValidationError should be raised during request validation
        with pytest.raises(ValueError, match="File does not exist"):
            ConvertSingleRequest(**request_data)

    @pytest.mark.asyncio
    async def test_convert_single_handles_invalid_pdf_file(
        self, mock_db_session, tmp_path
    ):
        """Test that convert_single handles invalid PDF files."""
        # Given - create a non-PDF file
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is not a PDF file")

        request_data = {"file_path": str(fake_pdf), "store_embeddings": True}

        # When/Then - ValidationError should be raised during request validation
        with pytest.raises(ValueError, match="does not appear to be a valid PDF"):
            ConvertSingleRequest(**request_data)

    @pytest.mark.asyncio
    async def test_convert_single_handles_large_pdf_file(
        self, mock_db_session, tmp_path
    ):
        """Test that convert_single rejects files larger than 500MB."""
        # Given - create a large file mock
        large_pdf = tmp_path / "large.pdf"
        # Create small file but mock stat to return large size
        large_pdf.write_bytes(b"%PDF-1.4\n%test")

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600MB

            request_data = {"file_path": str(large_pdf), "store_embeddings": True}

            # When/Then
            with pytest.raises(
                ValueError, match="File too large.*Maximum allowed: 500MB"
            ):
                ConvertSingleRequest(**request_data)


class TestBatchConvertEndpoint:
    """Test batch_convert endpoint following TDD principles."""

    @pytest.fixture
    def mock_pdf_directory(self, tmp_path):
        """Create a directory with mock PDF files."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create multiple PDF files
        for i in range(3):
            pdf_file = pdf_dir / f"doc{i}.pdf"
            with open(pdf_file, "wb") as f:
                f.write(b"%PDF-1.4\n")
                f.write(b"%test")

        return pdf_dir

    @pytest.fixture
    def valid_batch_request_data(self, mock_pdf_directory):
        """Valid batch request data."""
        return {
            "directory": str(mock_pdf_directory),
            "pattern": "*.pdf",
            "recursive": False,
            "output_base": None,
            "store_embeddings": True,
            "max_files": 10,
            "priority": 5,
            "options": {
                "ocr_language": "eng",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

    @pytest.mark.asyncio
    async def test_batch_convert_queues_background_task(
        self, mock_db_session, valid_batch_request_data
    ):
        """Test that batch_convert queues background processing task."""
        # Given
        request = BatchConvertRequest(**valid_batch_request_data)

        # Mock Celery task
        mock_job = Mock()
        mock_job.id = "batch_456"

        with patch("pdf_to_markdown_mcp.api.convert.process_pdf_batch") as mock_task:
            mock_task.delay.return_value = mock_job

            # When
            from pdf_to_markdown_mcp.api.convert import batch_convert_pdfs

            response = await batch_convert_pdfs(request, Mock(), mock_db_session)

            # Then
            assert response.success is True
            assert response.batch_id == "batch_456"
            assert response.files_found == 3  # Three PDF files created in fixture
            assert response.files_queued == 3
            assert response.files_skipped == 0
            assert "Batch processing initiated for 3 files" in response.message
            assert response.estimated_time_minutes == 6  # 3 files * 2 minutes estimate

            # Verify task was called
            mock_task.delay.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_convert_handles_no_files_found(
        self, mock_db_session, tmp_path
    ):
        """Test batch_convert when no PDF files are found."""
        # Given - empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        request_data = {
            "directory": str(empty_dir),
            "pattern": "*.pdf",
            "max_files": 10,
            "priority": 5,
        }
        request = BatchConvertRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.convert import batch_convert_pdfs

        response = await batch_convert_pdfs(request, Mock(), mock_db_session)

        # Then
        assert response.success is False
        assert response.batch_id == ""
        assert response.message == "No files to process"
        assert response.files_found == 0
        assert response.files_queued == 0
        assert response.files_skipped == 0

    @pytest.mark.asyncio
    async def test_batch_convert_respects_max_files_limit(
        self, mock_db_session, tmp_path
    ):
        """Test that batch_convert respects max_files limit."""
        # Given - create more files than max_files
        pdf_dir = tmp_path / "many_pdfs"
        pdf_dir.mkdir()

        for i in range(5):
            pdf_file = pdf_dir / f"doc{i}.pdf"
            with open(pdf_file, "wb") as f:
                f.write(b"%PDF-1.4\n")

        request_data = {
            "directory": str(pdf_dir),
            "pattern": "*.pdf",
            "max_files": 3,  # Limit to 3 files
            "priority": 5,
        }
        request = BatchConvertRequest(**request_data)

        mock_job = Mock()
        mock_job.id = "batch_789"

        with patch("pdf_to_markdown_mcp.api.convert.process_pdf_batch") as mock_task:
            mock_task.delay.return_value = mock_job

            # When
            from pdf_to_markdown_mcp.api.convert import batch_convert_pdfs

            response = await batch_convert_pdfs(request, Mock(), mock_db_session)

            # Then
            assert response.success is True
            assert response.files_found <= 5  # Should find all files
            assert response.files_queued == 3  # But only process max_files
            assert len(response.queued_files) == 3

    @pytest.mark.asyncio
    async def test_batch_convert_handles_invalid_directory(self, mock_db_session):
        """Test batch_convert with invalid directory."""
        # Given
        request_data = {
            "directory": "/nonexistent/directory",
            "pattern": "*.pdf",
            "max_files": 10,
            "priority": 5,
        }

        # When/Then
        with pytest.raises(ValueError, match="Directory does not exist"):
            BatchConvertRequest(**request_data)


class TestStreamProgressEndpoint:
    """Test stream_progress endpoint."""

    @pytest.mark.asyncio
    async def test_stream_progress_generates_sse_events(self):
        """Test that stream_progress generates proper SSE events."""
        # Given
        job_id = "test_job_123"

        # When
        from pdf_to_markdown_mcp.api.convert import stream_progress

        response = await stream_progress(job_id=job_id)

        # Then
        assert response.media_type == "text/event-stream"
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"

        # Test the generator (first few events)
        generator = response.body_iterator
        events = []

        # Collect first few events
        for _ in range(3):
            try:
                event = await generator.__anext__()
                events.append(event)
            except StopAsyncIteration:
                break

        # Verify SSE format
        assert len(events) > 0
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

        # Parse first event and verify structure
        import json

        first_event_data = events[0][6:-2]  # Remove "data: " and "\n\n"
        parsed_data = json.loads(first_event_data)

        assert "job_id" in parsed_data
        assert "progress_percent" in parsed_data
        assert "current_step" in parsed_data
        assert "timestamp" in parsed_data
        assert parsed_data["job_id"] == job_id
