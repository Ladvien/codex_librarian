"""
Integration tests for the complete PDF processing pipeline.

This module tests the end-to-end workflow from PDF ingestion
to final storage with embeddings, following TDD principles.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.pdf_to_markdown_mcp.core.processor import PDFProcessor
from src.pdf_to_markdown_mcp.db.models import (
    Document,
    DocumentContent,
    DocumentEmbedding,
)
from src.pdf_to_markdown_mcp.worker.tasks import process_pdf_document
from tests.fixtures import (
    DocumentFactory,
    ProcessingResultFactory,
    create_mock_embedding_service,
    create_mock_mineru_service,
    create_temp_pdf,
)


class TestPDFProcessingPipeline:
    """Test complete PDF processing pipeline integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pdf_pipeline_success(
        self, async_db_session, temp_directory
    ):
        """Test successful end-to-end PDF processing pipeline."""
        # Given
        pdf_path = create_temp_pdf(
            content="Test document content", directory=temp_directory
        )

        # Setup services
        processor = PDFProcessor()
        mineru_service = create_mock_mineru_service(success=True)
        embedding_service = create_mock_embedding_service()
        database_service = Mock()

        # Mock successful processing result
        processing_result = ProcessingResultFactory.create(
            success=True, chunk_count=3, include_tables=True, include_formulas=True
        )
        mineru_service.process_pdf.return_value = processing_result

        # Mock embedding generation
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        embedding_service.generate_batch.return_value = embeddings

        with (
            patch.object(processor, "mineru_service", mineru_service),
            patch.object(processor, "embedding_service", embedding_service),
            patch.object(processor, "database_service", database_service),
        ):
            # When
            result = await processor.process_document(str(pdf_path))

            # Then
            assert result["success"] is True
            assert result["document_path"] == str(pdf_path)
            assert "processing_time" in result
            assert "chunks_processed" in result
            assert result["chunks_processed"] == 3

            # Verify service calls
            mineru_service.process_pdf.assert_called_once_with(pdf_path)
            embedding_service.generate_batch.assert_called_once()
            database_service.store_processing_result.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pdf_pipeline_with_database_storage(
        self, async_db_session, temp_directory
    ):
        """Test PDF pipeline with actual database storage."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        # Create document record
        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock services
        processing_result = ProcessingResultFactory.create(
            success=True,
            chunk_count=2,
            markdown_content="# Test\n\nContent here",
            plain_text="Test\n\nContent here",
        )

        with (
            patch(
                "src.pdf_to_markdown_mcp.core.processor.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.core.processor.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru.process_pdf.return_value = processing_result
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = create_mock_embedding_service()
            mock_embedding.generate_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_embedding_class.return_value = mock_embedding

            processor = PDFProcessor()

            # When
            result = await processor.process_document_with_storage(
                document.id, async_db_session
            )

            # Then
            assert result["success"] is True

            # Verify database updates
            await async_db_session.refresh(document)
            assert document.status == "completed"

            # Verify content was stored
            content = (
                await async_db_session.query(DocumentContent)
                .filter_by(document_id=document.id)
                .first()
            )
            assert content is not None
            assert content.markdown_content == "# Test\n\nContent here"
            assert content.plain_text == "Test\n\nContent here"

            # Verify embeddings were stored
            embeddings = (
                await async_db_session.query(DocumentEmbedding)
                .filter_by(document_id=document.id)
                .all()
            )
            assert len(embeddings) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pdf_pipeline_error_handling(self, async_db_session, temp_directory):
        """Test PDF pipeline error handling and rollback."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock processing failure
        with patch(
            "src.pdf_to_markdown_mcp.core.processor.MinerUService"
        ) as mock_mineru_class:
            mock_mineru = Mock()
            mock_mineru.process_pdf.side_effect = Exception("Processing failed")
            mock_mineru_class.return_value = mock_mineru

            processor = PDFProcessor()

            # When/Then
            with pytest.raises(Exception, match="Processing failed"):
                await processor.process_document_with_storage(
                    document.id, async_db_session
                )

            # Verify document status updated to failed
            await async_db_session.refresh(document)
            assert document.status == "failed"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_large_document(self, async_db_session, temp_directory):
        """Test pipeline handling of large documents with many chunks."""
        # Given
        # Create a larger PDF with more content
        large_content = "This is a test document. " * 200  # Create larger content
        pdf_path = create_temp_pdf(content=large_content, directory=temp_directory)

        # Mock processing result with many chunks
        chunks = [
            {
                "text": f"Chunk {i} content",
                "chunk_index": i,
                "start_char": i * 50,
                "end_char": (i + 1) * 50,
            }
            for i in range(20)  # 20 chunks
        ]
        processing_result = ProcessingResultFactory.create(
            success=True, chunks=chunks, chunk_count=20
        )

        # Mock embeddings for all chunks
        embeddings = [[0.1 + i * 0.01] * 1536 for i in range(20)]

        with (
            patch(
                "src.pdf_to_markdown_mcp.core.processor.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.core.processor.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru.process_pdf.return_value = processing_result
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = create_mock_embedding_service()
            mock_embedding.generate_batch.return_value = embeddings
            mock_embedding_class.return_value = mock_embedding

            processor = PDFProcessor()

            # When
            result = await processor.process_document(str(pdf_path))

            # Then
            assert result["success"] is True
            assert result["chunks_processed"] == 20

            # Verify batch processing was used for embeddings
            mock_embedding.generate_batch.assert_called_once()
            call_args = mock_embedding.generate_batch.call_args[0]
            assert len(call_args[0]) == 20  # All chunks passed for batch embedding

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_performance_benchmarks(self, temp_directory):
        """Test pipeline performance with timing benchmarks."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        # Setup mocked services with realistic timing
        async def mock_slow_processing(path):
            await asyncio.sleep(0.1)  # Simulate 100ms processing
            return ProcessingResultFactory.create(success=True)

        async def mock_slow_embedding(texts):
            await asyncio.sleep(0.05 * len(texts))  # 50ms per text
            return [[0.1] * 1536] * len(texts)

        with (
            patch(
                "src.pdf_to_markdown_mcp.core.processor.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.core.processor.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_mineru = Mock()
            mock_mineru.process_pdf = mock_slow_processing
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = Mock()
            mock_embedding.generate_batch = mock_slow_embedding
            mock_embedding_class.return_value = mock_embedding

            processor = PDFProcessor()

            # When
            import time

            start_time = time.time()
            result = await processor.process_document(str(pdf_path))
            total_time = time.time() - start_time

            # Then
            assert result["success"] is True
            assert total_time < 1.0  # Should complete within 1 second
            assert "processing_time" in result

    @pytest.mark.integration
    async def test_pipeline_concurrent_processing(
        self, async_db_session, temp_directory
    ):
        """Test pipeline handling of concurrent document processing."""
        # Given
        pdf_paths = [
            create_temp_pdf(content=f"Document {i} content", directory=temp_directory)
            for i in range(3)
        ]

        # Create document records
        documents = []
        for i, pdf_path in enumerate(pdf_paths):
            document_data = DocumentFactory.create(
                file_path=str(pdf_path), status="processing"
            )
            document = Document(**document_data)
            documents.append(document)
            async_db_session.add(document)

        await async_db_session.commit()

        # Mock services
        with (
            patch(
                "src.pdf_to_markdown_mcp.core.processor.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.core.processor.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = create_mock_embedding_service()
            mock_embedding_class.return_value = mock_embedding

            processor = PDFProcessor()

            # When - Process documents concurrently
            tasks = [
                processor.process_document_with_storage(doc.id, async_db_session)
                for doc in documents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Then
            assert len(results) == 3
            assert all(
                result["success"] for result in results if isinstance(result, dict)
            )

            # Verify all documents completed
            for doc in documents:
                await async_db_session.refresh(doc)
                assert doc.status == "completed"

    @pytest.mark.integration
    async def test_pipeline_partial_failure_recovery(
        self, async_db_session, temp_directory
    ):
        """Test pipeline recovery from partial failures."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock successful PDF processing but failed embedding
        processing_result = ProcessingResultFactory.create(success=True)

        with (
            patch(
                "src.pdf_to_markdown_mcp.core.processor.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.core.processor.EmbeddingService"
            ) as mock_embedding_class,
        ):
            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru.process_pdf.return_value = processing_result
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = Mock()
            mock_embedding.generate_batch.side_effect = Exception(
                "Embedding service failed"
            )
            mock_embedding_class.return_value = mock_embedding

            processor = PDFProcessor()

            # When
            with pytest.raises(Exception, match="Embedding service failed"):
                await processor.process_document_with_storage(
                    document.id, async_db_session
                )

            # Then - Document content should still be stored even if embeddings failed
            await async_db_session.refresh(document)
            assert document.status == "failed"

            # Content should be stored
            content = (
                await async_db_session.query(DocumentContent)
                .filter_by(document_id=document.id)
                .first()
            )
            assert content is not None

            # But no embeddings should be stored
            embeddings = (
                await async_db_session.query(DocumentEmbedding)
                .filter_by(document_id=document.id)
                .all()
            )
            assert len(embeddings) == 0


class TestCeleryTaskIntegration:
    """Test integration with Celery task system."""

    @pytest.mark.integration
    @pytest.mark.celery
    async def test_celery_pdf_processing_task(self, async_db_session, temp_directory):
        """Test PDF processing through Celery task system."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="pending"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock the Celery app and task
        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_class,
        ):
            # Setup database session mock
            mock_get_db.return_value.__enter__.return_value = async_db_session
            mock_get_db.return_value.__exit__.return_value = None

            # Setup service mocks
            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru_class.return_value = mock_mineru

            mock_embedding = create_mock_embedding_service()
            mock_embedding_class.return_value = mock_embedding

            # When
            result = await process_pdf_document(document.id)

            # Then
            assert result["success"] is True
            assert result["document_id"] == document.id

    @pytest.mark.integration
    @pytest.mark.celery
    async def test_celery_task_progress_tracking(self, temp_directory):
        """Test Celery task progress tracking integration."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        with (
            patch("src.pdf_to_markdown_mcp.worker.tasks.current_task") as mock_task,
            patch("src.pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_get_db,
            patch(
                "src.pdf_to_markdown_mcp.worker.tasks.MinerUService"
            ) as mock_mineru_class,
        ):
            # Setup mocks
            mock_task.update_state = Mock()

            mock_session = Mock()
            mock_document = Mock()
            mock_document.file_path = str(pdf_path)
            mock_document.status = "processing"
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_document
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_get_db.return_value.__exit__.return_value = None

            mock_mineru = create_mock_mineru_service(success=True)
            mock_mineru_class.return_value = mock_mineru

            # When
            result = await process_pdf_document(1)

            # Then
            assert result["success"] is True

            # Verify progress updates were made
            progress_calls = [
                call
                for call in mock_task.update_state.call_args_list
                if call[1].get("state") == "PROGRESS"
            ]
            assert len(progress_calls) > 0

            # Verify final state update
            final_calls = [
                call
                for call in mock_task.update_state.call_args_list
                if call[1].get("state") == "SUCCESS"
            ]
            assert len(final_calls) > 0


class TestDatabaseIntegration:
    """Test database integration with the processing pipeline."""

    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_transaction_handling(
        self, async_db_session, temp_directory
    ):
        """Test proper database transaction handling during processing."""
        # Given
        pdf_path = create_temp_pdf(directory=temp_directory)

        document_data = DocumentFactory.create(
            file_path=str(pdf_path), status="processing"
        )
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Mock partial failure scenario
        with patch(
            "src.pdf_to_markdown_mcp.core.processor.MinerUService"
        ) as mock_mineru_class:
            mock_mineru = create_mock_mineru_service(success=True)

            # Simulate database error during storage
            original_commit = async_db_session.commit
            async_db_session.commit = AsyncMock(side_effect=Exception("Database error"))

            mock_mineru_class.return_value = mock_mineru
            processor = PDFProcessor()

            # When/Then
            with pytest.raises(Exception, match="Database error"):
                await processor.process_document_with_storage(
                    document.id, async_db_session
                )

            # Verify transaction was rolled back
            async_db_session.commit = original_commit
            await async_db_session.rollback()

            # Document should not have been updated
            await async_db_session.refresh(document)
            assert document.status == "processing"  # Still in original state

    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_constraint_validation(self, async_db_session):
        """Test database constraint validation during processing."""
        # Given - Create document with duplicate file_hash
        document1_data = DocumentFactory.create(file_hash="duplicate_hash")
        document1 = Document(**document1_data)
        async_db_session.add(document1)
        await async_db_session.commit()

        document2_data = DocumentFactory.create(file_hash="duplicate_hash")
        document2 = Document(**document2_data)
        async_db_session.add(document2)

        # When/Then
        with pytest.raises(Exception):  # Should raise integrity error
            await async_db_session.commit()

    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_cascade_operations(self, async_db_session):
        """Test database cascade delete operations."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        async_db_session.add(document)
        await async_db_session.commit()

        # Add related records
        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test",
            plain_text="Test",
            word_count=1,
            language="en",
        )
        async_db_session.add(content)

        embedding = DocumentEmbedding(
            document_id=document.id,
            chunk_index=0,
            chunk_text="Test chunk",
            embedding=[0.1] * 1536,
            start_char=0,
            end_char=10,
            token_count=2,
        )
        async_db_session.add(embedding)
        await async_db_session.commit()

        content_id = content.id
        embedding_id = embedding.id

        # When - Delete document
        await async_db_session.delete(document)
        await async_db_session.commit()

        # Then - Related records should be deleted
        assert (
            await async_db_session.query(DocumentContent)
            .filter_by(id=content_id)
            .first()
            is None
        )
        assert (
            await async_db_session.query(DocumentEmbedding)
            .filter_by(id=embedding_id)
            .first()
            is None
        )
