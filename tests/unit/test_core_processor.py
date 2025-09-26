"""
Unit tests for core PDF processor.

This module tests the PDFProcessor orchestrator class that coordinates
PDF processing workflows following TDD principles.
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.pdf_to_markdown_mcp.core.exceptions import (
    EmbeddingError,
    ProcessingError,
    ValidationError,
)
from src.pdf_to_markdown_mcp.core.processor import PDFProcessor
from tests.fixtures import (
    DocumentFactory,
    ProcessingResultFactory,
    create_mock_embedding_service,
    create_mock_mineru_service,
    create_sample_embeddings,
    create_temp_pdf,
)


class TestPDFProcessor:
    """Test PDFProcessor initialization and configuration."""

    def test_pdf_processor_initialization(self):
        """Test PDFProcessor initializes with default configuration."""
        # When
        processor = PDFProcessor()

        # Then
        assert processor is not None
        assert hasattr(processor, "config")
        assert hasattr(processor, "mineru_service")
        assert hasattr(processor, "embedding_service")
        assert hasattr(processor, "database_service")

    def test_pdf_processor_custom_config(self):
        """Test PDFProcessor with custom configuration."""
        # Given
        custom_config = {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "language": "fr",
            "extract_tables": False,
        }

        # When
        processor = PDFProcessor(config=custom_config)

        # Then
        assert processor.config["chunk_size"] == 1500
        assert processor.config["chunk_overlap"] == 300
        assert processor.config["language"] == "fr"
        assert processor.config["extract_tables"] is False

    @patch("src.pdf_to_markdown_mcp.core.processor.MinerUService")
    @patch("src.pdf_to_markdown_mcp.core.processor.EmbeddingService")
    def test_pdf_processor_service_initialization(
        self, mock_embedding_class, mock_mineru_class
    ):
        """Test PDFProcessor initializes services correctly."""
        # Given
        mock_mineru = create_mock_mineru_service()
        mock_embedding = create_mock_embedding_service()
        mock_mineru_class.return_value = mock_mineru
        mock_embedding_class.return_value = mock_embedding

        # When
        processor = PDFProcessor()

        # Then
        mock_mineru_class.assert_called_once()
        mock_embedding_class.assert_called_once()
        assert processor.mineru_service == mock_mineru
        assert processor.embedding_service == mock_embedding


class TestPDFProcessorValidation:
    """Test PDFProcessor input validation."""

    def test_validate_file_path_success(self, temp_directory):
        """Test successful file path validation."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # When/Then - Should not raise exception
        result = processor._validate_file_path(str(pdf_path))
        assert result == Path(pdf_path)

    def test_validate_file_path_not_exists(self):
        """Test file path validation with non-existent file."""
        # Given
        processor = PDFProcessor()
        non_existent_path = "/nonexistent/file.pdf"

        # When/Then
        with pytest.raises(ValidationError, match="File not found"):
            processor._validate_file_path(non_existent_path)

    def test_validate_file_path_not_pdf(self, temp_directory):
        """Test file path validation with non-PDF file."""
        # Given
        text_file = temp_directory / "test.txt"
        text_file.write_text("Not a PDF file")
        processor = PDFProcessor()

        # When/Then
        with pytest.raises(ValidationError, match="PDF file"):
            processor._validate_file_path(str(text_file))

    def test_validate_file_path_empty_file(self, temp_directory):
        """Test file path validation with empty PDF file."""
        # Given
        empty_pdf = temp_directory / "empty.pdf"
        empty_pdf.write_bytes(b"")  # Empty file
        processor = PDFProcessor()

        # When/Then
        with pytest.raises(ValidationError, match="empty"):
            processor._validate_file_path(str(empty_pdf))

    def test_validate_file_path_too_large(self, temp_directory):
        """Test file path validation with oversized file."""
        # Given
        processor = PDFProcessor()
        large_pdf = temp_directory / "large.pdf"

        with patch("pathlib.Path.stat") as mock_stat:
            # Mock file size to be larger than limit (500MB)
            mock_stat.return_value.st_size = 600 * 1024 * 1024
            large_pdf.write_bytes(b"%PDF-1.4\nsmall content")

            # When/Then
            with pytest.raises(ValidationError, match="too large"):
                processor._validate_file_path(str(large_pdf))


class TestPDFProcessorProcessing:
    """Test PDFProcessor document processing workflows."""

    @pytest.mark.asyncio
    async def test_process_document_success(self, temp_directory):
        """Test successful document processing."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # Mock services
        processing_result = ProcessingResultFactory.create(
            success=True, chunk_count=3, include_tables=True
        )

        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result

        embeddings = create_sample_embeddings(3, 1536)
        mock_embedding = create_mock_embedding_service()
        mock_embedding.generate_batch.return_value = embeddings

        processor.mineru_service = mock_mineru
        processor.embedding_service = mock_embedding

        # When
        result = await processor.process_document(str(pdf_path))

        # Then
        assert result["success"] is True
        assert result["document_path"] == str(pdf_path)
        assert "processing_time" in result
        assert "chunks_processed" in result
        assert result["chunks_processed"] == 3

        # Verify service calls
        mock_mineru.process_pdf.assert_called_once_with(Path(pdf_path))
        mock_embedding.generate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_mineru_failure(self, temp_directory):
        """Test document processing with MinerU failure."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        mock_mineru = Mock()
        mock_mineru.process_pdf.side_effect = ProcessingError(
            "MinerU processing failed"
        )
        processor.mineru_service = mock_mineru

        # When/Then
        with pytest.raises(ProcessingError, match="MinerU processing failed"):
            await processor.process_document(str(pdf_path))

    @pytest.mark.asyncio
    async def test_process_document_embedding_failure(self, temp_directory):
        """Test document processing with embedding failure."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # Mock successful MinerU processing
        processing_result = ProcessingResultFactory.create(success=True, chunk_count=2)
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        # Mock embedding failure
        mock_embedding = Mock()
        mock_embedding.generate_batch.side_effect = EmbeddingError(
            "Embedding service failed"
        )
        processor.embedding_service = mock_embedding

        # When/Then
        with pytest.raises(EmbeddingError, match="Embedding service failed"):
            await processor.process_document(str(pdf_path))

    @pytest.mark.asyncio
    async def test_process_document_no_embeddings_requested(self, temp_directory):
        """Test document processing without embedding generation."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        processing_result = ProcessingResultFactory.create(success=True, chunk_count=2)
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        mock_embedding = create_mock_embedding_service()
        processor.embedding_service = mock_embedding

        # When
        result = await processor.process_document(
            str(pdf_path), generate_embeddings=False
        )

        # Then
        assert result["success"] is True
        assert result["chunks_processed"] == 2

        # Embedding service should not be called
        mock_embedding.generate_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_document_empty_chunks(self, temp_directory):
        """Test document processing with no text chunks."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # Mock processing result with no chunks
        processing_result = ProcessingResultFactory.create(
            success=True, chunk_count=0, chunks=[]
        )
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        mock_embedding = create_mock_embedding_service()
        processor.embedding_service = mock_embedding

        # When
        result = await processor.process_document(str(pdf_path))

        # Then
        assert result["success"] is True
        assert result["chunks_processed"] == 0

        # Embedding service should not be called with empty chunks
        mock_embedding.generate_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_document_progress_callback(self, temp_directory):
        """Test document processing with progress callback."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        progress_updates = []

        def progress_callback(stage: str, progress: float, message: str = ""):
            progress_updates.append(
                {"stage": stage, "progress": progress, "message": message}
            )

        processing_result = ProcessingResultFactory.create(success=True, chunk_count=2)
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        embeddings = create_sample_embeddings(2, 1536)
        mock_embedding = create_mock_embedding_service()
        mock_embedding.generate_batch.return_value = embeddings
        processor.embedding_service = mock_embedding

        # When
        result = await processor.process_document(
            str(pdf_path), progress_callback=progress_callback
        )

        # Then
        assert result["success"] is True
        assert len(progress_updates) > 0

        # Verify progress stages
        stages = [update["stage"] for update in progress_updates]
        assert "validation" in stages
        assert "processing" in stages
        assert "embeddings" in stages


class TestPDFProcessorBatchProcessing:
    """Test PDFProcessor batch processing capabilities."""

    @pytest.mark.asyncio
    async def test_batch_process_documents_success(self, temp_directory):
        """Test successful batch processing of multiple documents."""
        # Given
        pdf_paths = [
            create_temp_pdf(content=f"Document {i} content", directory=temp_directory)
            for i in range(3)
        ]
        processor = PDFProcessor()

        # Mock services for successful processing
        processing_result = ProcessingResultFactory.create(success=True, chunk_count=1)
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        embeddings = create_sample_embeddings(1, 1536)
        mock_embedding = create_mock_embedding_service()
        mock_embedding.generate_batch.return_value = embeddings
        processor.embedding_service = mock_embedding

        # When
        results = await processor.batch_process_documents(
            [str(path) for path in pdf_paths]
        )

        # Then
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert mock_mineru.process_pdf.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_process_documents_partial_failure(self, temp_directory):
        """Test batch processing with some failures."""
        # Given
        pdf_paths = [
            create_temp_pdf(content=f"Document {i} content", directory=temp_directory)
            for i in range(3)
        ]
        processor = PDFProcessor()

        # Mock mixed results
        def side_effect_process(path):
            if "1" in str(path):  # Second document fails
                raise ProcessingError("Processing failed")
            return ProcessingResultFactory.create(success=True, chunk_count=1)

        mock_mineru = Mock()
        mock_mineru.process_pdf.side_effect = side_effect_process
        processor.mineru_service = mock_mineru

        embeddings = create_sample_embeddings(1, 1536)
        mock_embedding = create_mock_embedding_service()
        mock_embedding.generate_batch.return_value = embeddings
        processor.embedding_service = mock_embedding

        # When
        results = await processor.batch_process_documents(
            [str(path) for path in pdf_paths]
        )

        # Then
        assert len(results) == 3
        assert sum(1 for r in results if r["success"]) == 2
        assert sum(1 for r in results if not r["success"]) == 1

        # Find the failed result
        failed_result = next(r for r in results if not r["success"])
        assert "error" in failed_result
        assert "Processing failed" in failed_result["error"]

    @pytest.mark.asyncio
    async def test_batch_process_documents_concurrent_limit(self, temp_directory):
        """Test batch processing respects concurrency limits."""
        # Given
        pdf_paths = [
            create_temp_pdf(content=f"Document {i} content", directory=temp_directory)
            for i in range(10)  # More documents than default concurrency limit
        ]
        processor = PDFProcessor()

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_process_with_tracking(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.01)  # Simulate processing time

            concurrent_count -= 1
            return ProcessingResultFactory.create(success=True, chunk_count=1)

        with patch.object(processor, "process_document", mock_process_with_tracking):
            # When
            results = await processor.batch_process_documents(
                [str(path) for path in pdf_paths], max_concurrent=3
            )

            # Then
            assert len(results) == 10
            assert all(result["success"] for result in results)
            assert max_concurrent <= 3  # Should not exceed concurrency limit


class TestPDFProcessorDatabaseIntegration:
    """Test PDFProcessor database integration methods."""

    @pytest.mark.asyncio
    async def test_process_document_with_storage_success(
        self, async_db_session, temp_directory
    ):
        """Test processing document with database storage."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # Create document record
        from src.pdf_to_markdown_mcp.db.models import Document

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock services
        processing_result = ProcessingResultFactory.create(success=True, chunk_count=2)
        mock_mineru = create_mock_mineru_service(success=True)
        mock_mineru.process_pdf.return_value = processing_result
        processor.mineru_service = mock_mineru

        embeddings = create_sample_embeddings(2, 1536)
        mock_embedding = create_mock_embedding_service()
        mock_embedding.generate_batch.return_value = embeddings
        processor.embedding_service = mock_embedding

        # When
        result = await processor.process_document_with_storage(
            document.id, async_db_session
        )

        # Then
        assert result["success"] is True
        assert result["document_id"] == document.id

        # Verify document status updated
        await async_db_session.refresh(document)
        assert document.status == "completed"

    @pytest.mark.asyncio
    async def test_process_document_with_storage_document_not_found(
        self, async_db_session
    ):
        """Test processing with non-existent document ID."""
        # Given
        processor = PDFProcessor()
        non_existent_id = 999999

        # When/Then
        with pytest.raises(ValidationError, match="Document not found"):
            await processor.process_document_with_storage(
                non_existent_id, async_db_session
            )

    @pytest.mark.asyncio
    async def test_process_document_with_storage_processing_failure(
        self, async_db_session, temp_directory
    ):
        """Test storage with processing failure updates document status."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        from src.pdf_to_markdown_mcp.db.models import Document

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock processing failure
        mock_mineru = Mock()
        mock_mineru.process_pdf.side_effect = ProcessingError("Processing failed")
        processor.mineru_service = mock_mineru

        # When/Then
        with pytest.raises(ProcessingError):
            await processor.process_document_with_storage(document.id, async_db_session)

        # Verify document status updated to failed
        await async_db_session.refresh(document)
        assert document.status == "failed"


class TestPDFProcessorErrorHandling:
    """Test PDFProcessor error handling and recovery."""

    @pytest.mark.asyncio
    async def test_process_document_timeout_handling(self, temp_directory):
        """Test document processing with timeout."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        # Mock slow processing that times out
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return ProcessingResultFactory.create(success=True)

        mock_mineru = Mock()
        mock_mineru.process_pdf = slow_process
        processor.mineru_service = mock_mineru

        # When/Then
        with pytest.raises(asyncio.TimeoutError):
            await processor.process_document(str(pdf_path), timeout=1.0)

    @pytest.mark.asyncio
    async def test_process_document_memory_error_handling(self, temp_directory):
        """Test document processing with memory error."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        mock_mineru = Mock()
        mock_mineru.process_pdf.side_effect = MemoryError("Out of memory")
        processor.mineru_service = mock_mineru

        # When/Then
        with pytest.raises(ProcessingError, match="memory"):
            await processor.process_document(str(pdf_path))

    @pytest.mark.asyncio
    async def test_process_document_cleanup_on_error(self, temp_directory):
        """Test document processing cleans up resources on error."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)
        processor = PDFProcessor()

        cleanup_called = []

        def mock_cleanup():
            cleanup_called.append(True)

        processor._cleanup_resources = mock_cleanup

        mock_mineru = Mock()
        mock_mineru.process_pdf.side_effect = ProcessingError("Processing failed")
        processor.mineru_service = mock_mineru

        # When
        with pytest.raises(ProcessingError):
            await processor.process_document(str(pdf_path))

        # Then
        assert len(cleanup_called) > 0  # Cleanup should be called
