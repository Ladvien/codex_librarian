"""
Comprehensive unit tests for Celery tasks.

This module tests all Celery task definitions with proper mocking
and error scenario handling following TDD principles.
"""

from unittest.mock import Mock, patch

import pytest
from celery.exceptions import Retry

from src.pdf_to_markdown_mcp.core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
    ValidationError,
)
from src.pdf_to_markdown_mcp.worker.tasks import (
    ProgressTracker,
    batch_process_pdfs,
    cleanup_old_files,
    generate_embeddings,
    monitor_worker_health,
    process_pdf_document,
)
from tests.fixtures import (
    ProcessingResultFactory,
    create_mock_embedding_service,
    create_mock_mineru_service,
    create_sample_embeddings,
    create_temp_pdf,
)


class TestProgressTracker:
    """Test ProgressTracker helper class."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initializes correctly."""
        # Given
        mock_task = Mock()
        total_steps = 50

        # When
        tracker = ProgressTracker(mock_task, total_steps)

        # Then
        assert tracker.task == mock_task
        assert tracker.current_step == 0
        assert tracker.total_steps == 50
        assert tracker.start_time is not None
        assert tracker.messages == []

    def test_progress_tracker_update_with_message(self):
        """Test progress tracker updates with messages."""
        # Given
        mock_task = Mock()
        tracker = ProgressTracker(mock_task, 100)

        # When
        tracker.update(message="Processing started")
        tracker.update(current=25, message="25% complete")

        # Then
        assert tracker.current_step == 25
        assert len(tracker.messages) == 2
        assert tracker.messages[0] == "Processing started"
        assert tracker.messages[1] == "25% complete"

    def test_progress_tracker_auto_increment(self):
        """Test progress tracker auto-increments steps."""
        # Given
        mock_task = Mock()
        tracker = ProgressTracker(mock_task, 100)

        # When
        tracker.update(add_step=True)
        tracker.update(add_step=True)
        tracker.update(add_step=True)

        # Then
        assert tracker.current_step == 3

    def test_progress_tracker_update_task_state(self):
        """Test progress tracker updates Celery task state."""
        # Given
        mock_task = Mock()
        tracker = ProgressTracker(mock_task, 100)

        # When
        tracker.update(current=50, message="Half way done")

        # Then
        mock_task.update_state.assert_called()
        call_args = mock_task.update_state.call_args
        assert call_args[1]["state"] == "PROGRESS"
        assert call_args[1]["meta"]["current"] == 50
        assert call_args[1]["meta"]["total"] == 100
        assert call_args[1]["meta"]["status"] == "Half way done"

    def test_progress_tracker_message_limit(self):
        """Test progress tracker limits message history."""
        # Given
        mock_task = Mock()
        tracker = ProgressTracker(mock_task, 100)

        # When - Add more than 10 messages
        for i in range(15):
            tracker.update(message=f"Message {i}")

        # Then - Should only keep last 10
        assert len(tracker.messages) == 10
        assert tracker.messages[0] == "Message 5"  # First kept message
        assert tracker.messages[-1] == "Message 14"  # Last message

    def test_progress_tracker_eta_calculation(self):
        """Test progress tracker calculates ETA correctly."""
        # Given
        mock_task = Mock()
        tracker = ProgressTracker(mock_task, 100)

        # When
        with patch("time.time", side_effect=[0, 10]):  # 10 seconds elapsed
            tracker.start_time = 0
            tracker.update(current=25)  # 25% complete

        # Then
        call_args = mock_task.update_state.call_args
        eta = call_args[1]["meta"].get("eta")
        # At 25% in 10 seconds, should take ~40 seconds total, so ~30 seconds remaining
        assert eta is not None
        assert 25 <= eta <= 35  # Allow some tolerance


class TestProcessPdfDocument:
    """Test PDF document processing task."""

    @pytest.mark.asyncio
    async def test_process_pdf_document_success(self):
        """Test successful PDF processing."""
        # Given
        document_id = 1
        pdf_path = create_temp_pdf()
        processing_result = ProcessingResultFactory.create(success=True)

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_class,
        ):
            # Setup mocks
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_document = Mock()
            mock_document.file_path = str(pdf_path)
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document

            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = create_mock_embedding_service()
            mock_embedding_class.return_value = mock_embedding

        # When
        result = await process_pdf_document(document_id)

        # Then
        assert result["success"] is True
        assert result["document_id"] == document_id
        assert "processing_time" in result
        assert "chunks_processed" in result
        mock_mineru.process_pdf.assert_called_once()
        mock_document.status = "completed"  # Should be updated

    @pytest.mark.asyncio
    async def test_process_pdf_document_file_not_found(self):
        """Test PDF processing with missing file."""
        # Given
        document_id = 1

        with patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_document = Mock()
            mock_document.file_path = "/nonexistent/file.pdf"
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document

        # When/Then
        with pytest.raises(ValidationError):
            await process_pdf_document(document_id)

    @pytest.mark.asyncio
    async def test_process_pdf_document_processing_error(self):
        """Test PDF processing with MinerU error."""
        # Given
        document_id = 1
        pdf_path = create_temp_pdf()

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_document = Mock()
            mock_document.file_path = str(pdf_path)
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document

            mock_mineru = Mock()
            mock_mineru.process_pdf.side_effect = ProcessingError("MinerU failed")
            mock_mineru_class.return_value = mock_mineru

        # When/Then
        with pytest.raises(ProcessingError):
            await process_pdf_document(document_id)

    @pytest.mark.asyncio
    async def test_process_pdf_document_with_retry(self):
        """Test PDF processing with retry mechanism."""
        # Given
        document_id = 1
        pdf_path = create_temp_pdf()

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
            patch("src.pdf_to_markdown_mcp.worker.tasks.current_task") as mock_task,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_document = Mock()
            mock_document.file_path = str(pdf_path)
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document

            # First call fails, second succeeds
            mock_mineru = Mock()
            mock_mineru.process_pdf.side_effect = [
                ProcessingError("Temporary failure"),
                ProcessingResultFactory.create(success=True),
            ]
            mock_mineru_class.return_value = mock_mineru

            mock_task.retry.side_effect = Retry("Retrying task")

        # When/Then
        with pytest.raises(Retry):
            await process_pdf_document(document_id)

        # Verify retry was called
        mock_task.retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pdf_document_database_error(self):
        """Test PDF processing with database error."""
        # Given
        document_id = 1

        with patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db:
            mock_db.side_effect = DatabaseError("Database connection failed")

        # When/Then
        with pytest.raises(DatabaseError):
            await process_pdf_document(document_id)

    @pytest.mark.asyncio
    async def test_process_pdf_document_progress_tracking(self):
        """Test PDF processing updates progress correctly."""
        # Given
        document_id = 1
        pdf_path = create_temp_pdf()

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
            patch("src.pdf_to_markdown_mcp.worker.tasks.current_task") as mock_task,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_document = Mock()
            mock_document.file_path = str(pdf_path)
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document

            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru_class.return_value = mock_mineru

        # When
        result = await process_pdf_document(document_id)

        # Then
        assert mock_task.update_state.call_count > 0
        # Verify progress updates were made
        progress_calls = [
            call
            for call in mock_task.update_state.call_args_list
            if call[1].get("state") == "PROGRESS"
        ]
        assert len(progress_calls) > 0


class TestGenerateEmbeddings:
    """Test embedding generation task."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        # Given
        document_id = 1
        chunks = [
            {"text": "First chunk", "chunk_index": 0},
            {"text": "Second chunk", "chunk_index": 1},
        ]
        embeddings = create_sample_embeddings(2, 1536)

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_content = Mock()
            mock_content.chunks = chunks
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_content

            mock_embedding_service = create_mock_embedding_service()
            mock_embedding_service.generate_batch.return_value = embeddings
            mock_embedding_class.return_value = mock_embedding_service

        # When
        result = await generate_embeddings(document_id)

        # Then
        assert result["success"] is True
        assert result["document_id"] == document_id
        assert result["embeddings_generated"] == 2
        mock_embedding_service.generate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_document_not_found(self):
        """Test embedding generation with missing document."""
        # Given
        document_id = 999

        with patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # When/Then
        with pytest.raises(ValidationError, match="Document content not found"):
            await generate_embeddings(document_id)

    @pytest.mark.asyncio
    async def test_generate_embeddings_service_error(self):
        """Test embedding generation with service error."""
        # Given
        document_id = 1
        chunks = [{"text": "Test chunk", "chunk_index": 0}]

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_content = Mock()
            mock_content.chunks = chunks
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_content

            mock_embedding_service = Mock()
            mock_embedding_service.generate_batch.side_effect = EmbeddingError(
                "API failed"
            )
            mock_embedding_class.return_value = mock_embedding_service

        # When/Then
        with pytest.raises(EmbeddingError):
            await generate_embeddings(document_id)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self):
        """Test embedding generation processes chunks in batches."""
        # Given
        document_id = 1
        # Create 15 chunks (should be processed in batches)
        chunks = [{"text": f"Chunk {i}", "chunk_index": i} for i in range(15)]
        batch_size = 10

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            mock_content = Mock()
            mock_content.chunks = chunks
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_content

            mock_embedding_service = create_mock_embedding_service()
            # Return embeddings for each batch
            mock_embedding_service.generate_batch.side_effect = [
                create_sample_embeddings(10, 1536),  # First batch of 10
                create_sample_embeddings(5, 1536),  # Second batch of 5
            ]
            mock_embedding_class.return_value = mock_embedding_service

        # When
        result = await generate_embeddings(document_id, batch_size=batch_size)

        # Then
        assert result["success"] is True
        assert result["embeddings_generated"] == 15
        # Should be called twice (two batches)
        assert mock_embedding_service.generate_batch.call_count == 2


class TestBatchProcessPdfs:
    """Test batch PDF processing task."""

    @pytest.mark.asyncio
    async def test_batch_process_pdfs_success(self):
        """Test successful batch processing of multiple PDFs."""
        # Given
        document_ids = [1, 2, 3]
        pdf_paths = [create_temp_pdf() for _ in document_ids]

        with patch(
            "src.pdf_to_markdown_mcp.worker.tasks.process_pdf_document"
        ) as mock_process:
            # Mock successful processing for all documents
            mock_process.side_effect = [
                {"success": True, "document_id": doc_id} for doc_id in document_ids
            ]

        # When
        result = await batch_process_pdfs(document_ids)

        # Then
        assert result["success"] is True
        assert result["total_documents"] == 3
        assert result["successful_documents"] == 3
        assert result["failed_documents"] == 0
        assert len(result["results"]) == 3
        assert mock_process.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_process_pdfs_partial_failure(self):
        """Test batch processing with some failures."""
        # Given
        document_ids = [1, 2, 3]

        with patch(
            "src.pdf_to_markdown_mcp.worker.tasks.process_pdf_document"
        ) as mock_process:
            # Mock mixed results
            mock_process.side_effect = [
                {"success": True, "document_id": 1},
                ProcessingError("Processing failed"),
                {"success": True, "document_id": 3},
            ]

        # When
        result = await batch_process_pdfs(document_ids)

        # Then
        assert result["success"] is True  # Overall success if any completed
        assert result["total_documents"] == 3
        assert result["successful_documents"] == 2
        assert result["failed_documents"] == 1
        assert len(result["results"]) == 3
        assert result["results"][1]["success"] is False

    @pytest.mark.asyncio
    async def test_batch_process_pdfs_empty_list(self):
        """Test batch processing with empty document list."""
        # Given
        document_ids = []

        # When
        result = await batch_process_pdfs(document_ids)

        # Then
        assert result["success"] is True
        assert result["total_documents"] == 0
        assert result["successful_documents"] == 0
        assert result["failed_documents"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_batch_process_pdfs_duplicate_detection(self):
        """Test batch processing handles duplicate document IDs."""
        # Given
        document_ids = [1, 2, 1, 3, 2]  # Duplicates

        with patch(
            "src.pdf_to_markdown_mcp.worker.tasks.process_pdf_document"
        ) as mock_process:
            mock_process.side_effect = [
                {"success": True, "document_id": doc_id}
                for doc_id in [1, 2, 3]  # Only unique IDs processed
            ]

        # When
        result = await batch_process_pdfs(document_ids)

        # Then
        assert result["total_documents"] == 3  # Duplicates removed
        assert mock_process.call_count == 3  # Only called for unique IDs


class TestCleanupOldFiles:
    """Test file cleanup maintenance task."""

    @patch("os.listdir")
    @patch("os.path.getmtime")
    @patch("os.remove")
    @patch("os.path.isfile")
    def test_cleanup_old_files_success(
        self, mock_isfile, mock_remove, mock_getmtime, mock_listdir
    ):
        """Test successful cleanup of old files."""
        # Given
        cleanup_dir = "/tmp/test_cleanup"
        max_age_days = 7
        current_time = 1000000
        old_time = current_time - (8 * 24 * 3600)  # 8 days old
        recent_time = current_time - (5 * 24 * 3600)  # 5 days old

        mock_listdir.return_value = ["old_file.tmp", "recent_file.tmp", "very_old.tmp"]
        mock_isfile.return_value = True
        mock_getmtime.side_effect = [old_time, recent_time, old_time]

        with patch("time.time", return_value=current_time):
            # When
            result = cleanup_old_files(cleanup_dir, max_age_days)

        # Then
        assert result["success"] is True
        assert result["files_removed"] == 2
        assert result["files_kept"] == 1
        assert mock_remove.call_count == 2

    @patch("os.listdir")
    def test_cleanup_old_files_directory_not_found(self, mock_listdir):
        """Test cleanup with non-existent directory."""
        # Given
        mock_listdir.side_effect = FileNotFoundError()

        # When
        result = cleanup_old_files("/nonexistent/dir", 7)

        # Then
        assert result["success"] is False
        assert "error" in result

    @patch("os.listdir")
    @patch("os.remove")
    @patch("os.path.isfile")
    @patch("os.path.getmtime")
    def test_cleanup_old_files_permission_error(
        self, mock_getmtime, mock_isfile, mock_remove, mock_listdir
    ):
        """Test cleanup with permission errors."""
        # Given
        mock_listdir.return_value = ["protected_file.tmp"]
        mock_isfile.return_value = True
        mock_getmtime.return_value = 1000000 - (10 * 24 * 3600)  # Old file
        mock_remove.side_effect = PermissionError("Permission denied")

        with patch("time.time", return_value=1000000):
            # When
            result = cleanup_old_files("/tmp/test", 7)

        # Then
        assert result["success"] is True  # Partial success
        assert result["files_removed"] == 0
        assert result["files_kept"] == 0
        assert "errors" in result


class TestMonitorWorkerHealth:
    """Test worker health monitoring task."""

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_monitor_worker_health_success(self, mock_disk, mock_memory, mock_cpu):
        """Test successful health monitoring."""
        # Given
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=60.0, available=4000000000)
        mock_disk.return_value = Mock(percent=70.0, free=5000000000)

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch("src.pdf_to_markdown_mcp.worker.tasks.app") as mock_app,
        ):
            mock_session = Mock()
            mock_db.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.__exit__ = Mock(return_value=None)

            # Mock database connectivity test
            mock_session.execute.return_value = None

            # Mock Celery stats
            mock_stats = {
                "total_tasks": 100,
                "active_tasks": 5,
                "failed_tasks": 2,
            }
            mock_app.control.inspect.return_value.stats.return_value = {
                "worker1": mock_stats
            }

        # When
        result = monitor_worker_health()

        # Then
        assert result["success"] is True
        assert result["cpu_percent"] == 45.0
        assert result["memory_percent"] == 60.0
        assert result["disk_percent"] == 70.0
        assert result["database_connected"] is True
        assert result["celery_stats"] == {"worker1": mock_stats}

    @patch("psutil.cpu_percent")
    def test_monitor_worker_health_high_resource_usage(self, mock_cpu):
        """Test health monitoring with high resource usage warnings."""
        # Given
        mock_cpu.return_value = 95.0  # High CPU

        with (
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):
            mock_memory.return_value = Mock(
                percent=90.0, available=1000000
            )  # High memory
            mock_disk.return_value = Mock(percent=95.0, free=1000000)  # High disk

        # When
        result = monitor_worker_health()

        # Then
        assert result["success"] is True
        assert len(result["warnings"]) >= 3  # CPU, memory, and disk warnings
        assert any("CPU" in warning for warning in result["warnings"])
        assert any("memory" in warning for warning in result["warnings"])
        assert any("disk" in warning for warning in result["warnings"])

    def test_monitor_worker_health_database_error(self):
        """Test health monitoring with database connection issues."""
        # Given
        with patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db:
            mock_db.side_effect = DatabaseError("Connection failed")

        # When
        result = monitor_worker_health()

        # Then
        assert result["success"] is True  # Still reports success but with warnings
        assert result["database_connected"] is False
        assert any(
            "database" in warning.lower() for warning in result.get("warnings", [])
        )
