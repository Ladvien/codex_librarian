"""
Integration tests for MinerU PDF processing service.

Tests the integration between MinerU service and the core processing pipeline.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from pdf_to_markdown_mcp.core.exceptions import ValidationError
from pdf_to_markdown_mcp.models.processing import ProcessingMetadata, ProcessingResult
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.services.mineru import MinerUService


@pytest.fixture
def sample_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write minimal PDF header
        f.write(b"%PDF-1.4\n%EOF\n")
        pdf_path = Path(f.name)

    yield pdf_path

    # Clean up
    if pdf_path.exists():
        os.unlink(pdf_path)


@pytest.fixture
def processing_options():
    """Default processing options for testing."""
    return ProcessingOptions(
        ocr_language="eng",
        preserve_layout=True,
        extract_tables=True,
        extract_formulas=True,
        extract_images=True,
        chunk_for_embeddings=True,
        chunk_size=500,
        chunk_overlap=100,
    )


@pytest.fixture
def mineru_service():
    """MinerU service instance for testing."""
    return MinerUService()


class TestMinerUIntegration:
    """Integration tests for MinerU service within the processing pipeline."""

    @pytest.mark.asyncio
    async def test_complete_processing_pipeline(
        self, mineru_service, sample_pdf_file, processing_options
    ):
        """Test complete processing pipeline from PDF to searchable content."""
        # Given (Arrange)
        # This test uses the mock implementation since MinerU library isn't installed

        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, processing_options)

        # Then (Assert)
        assert isinstance(result, ProcessingResult)
        assert result.markdown_content
        assert result.plain_text
        assert isinstance(result.processing_metadata, ProcessingMetadata)
        assert result.processing_metadata.pages >= 1
        assert result.processing_metadata.processing_time_ms > 0

        # Check that chunking was performed
        if processing_options.chunk_for_embeddings:
            assert len(result.chunk_data) > 0
            for chunk in result.chunk_data:
                assert chunk.text
                assert chunk.start_char >= 0
                assert chunk.end_char > chunk.start_char
                assert chunk.chunk_index >= 0

    @pytest.mark.asyncio
    async def test_processing_with_table_extraction(
        self, mineru_service, sample_pdf_file
    ):
        """Test processing with table extraction enabled."""
        # Given (Arrange)
        options = ProcessingOptions(
            extract_tables=True,
            extract_formulas=False,
            extract_images=False,
            chunk_for_embeddings=False,
        )

        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, options)

        # Then (Assert)
        assert isinstance(result, ProcessingResult)
        # Mock implementation returns empty tables, but structure should be correct
        assert isinstance(result.extracted_tables, list)

    @pytest.mark.asyncio
    async def test_processing_with_formula_extraction(
        self, mineru_service, sample_pdf_file
    ):
        """Test processing with formula extraction enabled."""
        # Given (Arrange)
        options = ProcessingOptions(
            extract_tables=False,
            extract_formulas=True,
            extract_images=False,
            chunk_for_embeddings=False,
        )

        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, options)

        # Then (Assert)
        assert isinstance(result, ProcessingResult)
        # Mock implementation returns empty formulas, but structure should be correct
        assert isinstance(result.extracted_formulas, list)

    @pytest.mark.asyncio
    async def test_processing_with_image_extraction(
        self, mineru_service, sample_pdf_file
    ):
        """Test processing with image extraction enabled."""
        # Given (Arrange)
        options = ProcessingOptions(
            extract_tables=False,
            extract_formulas=False,
            extract_images=True,
            chunk_for_embeddings=False,
        )

        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, options)

        # Then (Assert)
        assert isinstance(result, ProcessingResult)
        # Mock implementation returns empty images, but structure should be correct
        assert isinstance(result.extracted_images, list)

    @pytest.mark.asyncio
    async def test_processing_performance_metrics(
        self, mineru_service, sample_pdf_file, processing_options
    ):
        """Test that processing generates performance metrics."""
        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, processing_options)

        # Then (Assert)
        metadata = result.processing_metadata

        # Check required metadata fields
        assert metadata.pages > 0
        assert metadata.processing_time_ms > 0
        assert metadata.file_size_bytes is not None
        assert metadata.file_hash is not None

        # Check optional quality metrics
        assert metadata.ocr_confidence is not None
        assert 0.0 <= metadata.ocr_confidence <= 1.0

        # Check processing statistics
        assert metadata.tables_found is not None
        assert metadata.formulas_found is not None
        assert metadata.images_found is not None

    @pytest.mark.asyncio
    async def test_chunking_integration(self, mineru_service, sample_pdf_file):
        """Test that chunking integrates properly with processing."""
        # Given (Arrange)
        options = ProcessingOptions(
            chunk_for_embeddings=True,
            chunk_size=200,  # Small chunks for testing
            chunk_overlap=50,
        )

        # When (Act)
        result = await mineru_service.process_pdf(sample_pdf_file, options)

        # Then (Assert)
        assert len(result.chunk_data) > 0

        # Check chunk properties
        for i, chunk in enumerate(result.chunk_data):
            assert chunk.chunk_index == i
            assert chunk.text
            assert (
                len(chunk.text) <= options.chunk_size + options.chunk_overlap
            )  # Allow for overlap
            assert chunk.end_char > chunk.start_char
            assert chunk.token_count > 0

        # Check overlap between adjacent chunks
        if len(result.chunk_data) > 1:
            chunk1 = result.chunk_data[0]
            chunk2 = result.chunk_data[1]

            # Second chunk should start before first chunk ends (overlap)
            overlap = chunk1.end_char - chunk2.start_char
            assert overlap > 0, "Chunks should overlap"
            assert overlap <= options.chunk_overlap, (
                "Overlap should not exceed configured amount"
            )

    @pytest.mark.asyncio
    async def test_file_validation_integration(
        self, mineru_service, processing_options
    ):
        """Test file validation as part of processing pipeline."""
        # Test non-existent file
        with pytest.raises(ValidationError) as exc_info:
            await mineru_service.process_pdf(
                Path("/nonexistent/file.pdf"), processing_options
            )
        assert "File not found" in str(exc_info.value)

        # Test non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not a PDF file")
            txt_path = Path(f.name)

        try:
            with pytest.raises(ValidationError) as exc_info:
                await mineru_service.process_pdf(txt_path, processing_options)
            assert "Invalid file type" in str(exc_info.value)
        finally:
            os.unlink(txt_path)

    @pytest.mark.asyncio
    async def test_service_statistics(self, mineru_service):
        """Test service statistics and configuration reporting."""
        # When (Act)
        stats = await mineru_service.get_processing_stats()

        # Then (Assert)
        assert isinstance(stats, dict)
        assert "service" in stats
        assert "version" in stats
        assert "max_file_size_mb" in stats
        assert "timeout_seconds" in stats
        assert "supported_languages" in stats
        assert "features" in stats

        # Check supported languages
        supported_languages = stats["supported_languages"]
        assert "eng" in supported_languages
        assert "fra" in supported_languages
        assert "deu" in supported_languages

        # Check features
        features = stats["features"]
        assert "layout_aware_extraction" in features
        assert "table_detection" in features
        assert "formula_recognition" in features
        assert "built_in_ocr" in features

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mineru_service):
        """Test error handling throughout the processing pipeline."""
        # Test with invalid processing options
        invalid_options = ProcessingOptions(
            chunk_size=50,
            chunk_overlap=100,  # Overlap > chunk_size should cause validation error
        )

        # This should be caught during validation
        with pytest.raises(ValidationError):
            invalid_options.chunk_overlap = 100  # This triggers Pydantic validation

    @pytest.mark.asyncio
    async def test_concurrent_processing(
        self, mineru_service, sample_pdf_file, processing_options
    ):
        """Test that service handles concurrent processing requests."""
        # Given (Arrange)
        # Create multiple processing tasks
        tasks = [
            mineru_service.process_pdf(sample_pdf_file, processing_options)
            for _ in range(3)
        ]

        # When (Act)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Then (Assert)
        # All tasks should complete successfully
        for result in results:
            assert not isinstance(result, Exception), f"Processing failed: {result}"
            assert isinstance(result, ProcessingResult)

    def test_service_representation(self, mineru_service):
        """Test service string representations."""
        # Test __str__
        str_repr = str(mineru_service)
        assert "MinerUService" in str_repr

        # Test __repr__
        repr_str = repr(mineru_service)
        assert "MinerUService" in repr_str
        assert "mineru_available" in repr_str

    @pytest.mark.asyncio
    async def test_mock_vs_real_mineru_behavior(
        self, sample_pdf_file, processing_options
    ):
        """Test that mock and real MinerU behavior are consistent."""
        # This test verifies that the mock implementation provides
        # consistent interface with real MinerU (when available)

        service = MinerUService()

        # Since MinerU library is not installed, this uses mock implementation
        result = await service.process_pdf(sample_pdf_file, processing_options)

        # Verify mock implementation provides complete interface
        assert hasattr(result, "markdown_content")
        assert hasattr(result, "plain_text")
        assert hasattr(result, "extracted_tables")
        assert hasattr(result, "extracted_formulas")
        assert hasattr(result, "extracted_images")
        assert hasattr(result, "chunk_data")
        assert hasattr(result, "processing_metadata")

        # Verify metadata completeness
        metadata = result.processing_metadata
        assert hasattr(metadata, "pages")
        assert hasattr(metadata, "processing_time_ms")
        assert hasattr(metadata, "ocr_confidence")
        assert hasattr(metadata, "file_size_bytes")
        assert hasattr(metadata, "file_hash")
