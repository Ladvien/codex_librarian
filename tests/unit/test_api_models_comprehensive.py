"""
Comprehensive unit tests for enhanced Pydantic API models.

Following TDD principles, this module tests all the enhanced validation,
serialization, and error handling features for API-002 implementation.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pdf_to_markdown_mcp.models.request import (
    BatchConvertRequest,
    ConfigurationRequest,
    ConvertSingleRequest,
    HybridSearchRequest,
    ProcessingOptions,
    SemanticSearchRequest,
)
from src.pdf_to_markdown_mcp.models.response import (
    ConfigurationResponse,
    ConvertSingleResponse,
    ErrorResponse,
    ErrorType,
    HealthResponse,
    JobStatus,
    SearchResponse,
    SearchResult,
    StatusResponse,
)


class TestProcessingOptionsEnhanced:
    """Test enhanced ProcessingOptions validation."""

    def test_processing_options_valid_languages(self):
        """Test ProcessingOptions accepts all supported OCR languages."""
        # Given - all supported languages
        supported_languages = [
            "eng",
            "chi_sim",
            "chi_tra",
            "fra",
            "deu",
            "spa",
            "jpn",
            "kor",
        ]

        # When/Then - all should be valid
        for lang in supported_languages:
            options = ProcessingOptions(ocr_language=lang)
            assert options.ocr_language == lang

    def test_processing_options_invalid_language(self):
        """Test ProcessingOptions rejects invalid languages."""
        # Given
        invalid_lang = "xyz"

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ProcessingOptions(ocr_language=invalid_lang)

        assert "ocr_language" in str(exc_info.value)

    def test_chunk_overlap_validation(self):
        """Test chunk_overlap must be less than chunk_size."""
        # Given - overlap >= chunk_size
        invalid_data = {
            "chunk_size": 1000,
            "chunk_overlap": 1000,  # Equal to chunk_size
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ProcessingOptions(**invalid_data)

        assert "overlap must be less than chunk size" in str(exc_info.value).lower()

    def test_chunk_size_boundaries(self):
        """Test chunk_size accepts valid range."""
        # Given - valid boundaries
        valid_sizes = [100, 1000, 5000]  # Min, default, max

        # When/Then
        for size in valid_sizes:
            options = ProcessingOptions(chunk_size=size)
            assert options.chunk_size == size

    def test_chunk_size_invalid_boundaries(self):
        """Test chunk_size rejects invalid values."""
        # Given - invalid values
        invalid_sizes = [50, 5001]  # Below min, above max

        # When/Then
        for size in invalid_sizes:
            with pytest.raises(ValidationError):
                ProcessingOptions(chunk_size=size)

    def test_serialization_roundtrip(self):
        """Test ProcessingOptions can be serialized and deserialized."""
        # Given
        original = ProcessingOptions(
            ocr_language="fra", chunk_size=1500, extract_tables=False
        )

        # When
        serialized = original.dict()
        restored = ProcessingOptions(**serialized)

        # Then
        assert restored == original
        assert restored.ocr_language == "fra"
        assert restored.chunk_size == 1500
        assert restored.extract_tables is False


class TestConvertSingleRequestEnhanced:
    """Test enhanced ConvertSingleRequest validation."""

    def test_valid_pdf_file_path(self):
        """Test ConvertSingleRequest accepts valid PDF path."""
        # Given - create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"%PDF-1.4\n")  # Valid PDF header
            tmp_path = Path(tmp_file.name)

        try:
            # When
            request = ConvertSingleRequest(file_path=tmp_path)

            # Then
            assert request.file_path == tmp_path
            assert request.store_embeddings is True  # Default

        finally:
            # Cleanup
            tmp_path.unlink()

    def test_nonexistent_file_path(self):
        """Test ConvertSingleRequest rejects non-existent files."""
        # Given
        fake_path = Path("/path/that/does/not/exist.pdf")

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ConvertSingleRequest(file_path=fake_path)

        assert "does not exist" in str(exc_info.value)

    def test_non_pdf_file(self):
        """Test ConvertSingleRequest rejects non-PDF files."""
        # Given - create temporary non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not a PDF")
            tmp_path = Path(tmp_file.name)

        try:
            # When/Then
            with pytest.raises(ValidationError) as exc_info:
                ConvertSingleRequest(file_path=tmp_path)

            assert "must be a PDF" in str(exc_info.value)

        finally:
            # Cleanup
            tmp_path.unlink()

    def test_output_dir_validation(self):
        """Test output directory validation."""
        # Given - create valid input file and output directory
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"%PDF-1.4\n")
            input_path = Path(tmp_file.name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            try:
                # When
                request = ConvertSingleRequest(
                    file_path=input_path, output_dir=output_dir
                )

                # Then
                assert request.output_dir == output_dir

            finally:
                # Cleanup
                input_path.unlink()

    def test_invalid_output_dir(self):
        """Test output directory validation with file instead of directory."""
        # Given - create valid input file and invalid output (file instead of dir)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as input_file:
            input_file.write(b"%PDF-1.4\n")
            input_path = Path(input_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            # When/Then
            with pytest.raises(ValidationError) as exc_info:
                ConvertSingleRequest(
                    file_path=input_path,
                    output_dir=output_path,  # File, not directory
                )

            assert "not a directory" in str(exc_info.value)

        finally:
            # Cleanup
            input_path.unlink()
            output_path.unlink()


class TestSearchRequestValidation:
    """Test enhanced search request validation."""

    def test_semantic_search_valid_query(self):
        """Test SemanticSearchRequest with valid parameters."""
        # Given
        valid_data = {
            "query": "machine learning algorithms",
            "top_k": 10,
            "threshold": 0.8,
            "include_content": True,
        }

        # When
        request = SemanticSearchRequest(**valid_data)

        # Then
        assert request.query == "machine learning algorithms"
        assert request.top_k == 10
        assert request.threshold == 0.8
        assert request.include_content is True

    def test_semantic_search_empty_query(self):
        """Test SemanticSearchRequest rejects empty queries."""
        # Given
        invalid_data = {"query": "   "}  # Whitespace only

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(**invalid_data)

        assert "cannot be empty" in str(exc_info.value)

    def test_semantic_search_query_length_limits(self):
        """Test SemanticSearchRequest query length validation."""
        # Given - query too long
        long_query = "x" * 1001  # Exceeds max length

        # When/Then
        with pytest.raises(ValidationError):
            SemanticSearchRequest(query=long_query)

    def test_hybrid_search_weight_validation(self):
        """Test HybridSearchRequest weight validation."""
        # Given - weights that don't sum to 1.0
        invalid_data = {
            "query": "test query",
            "semantic_weight": 0.8,
            "keyword_weight": 0.3,  # Sum = 1.1
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            HybridSearchRequest(**invalid_data)

        assert "must sum to 1.0" in str(exc_info.value)

    def test_hybrid_search_valid_weights(self):
        """Test HybridSearchRequest with valid weights."""
        # Given
        valid_data = {
            "query": "neural networks",
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
        }

        # When
        request = HybridSearchRequest(**valid_data)

        # Then
        assert request.semantic_weight == 0.7
        assert request.keyword_weight == 0.3


class TestResponseModelsEnhanced:
    """Test enhanced response model features."""

    def test_error_response_structure(self):
        """Test ErrorResponse includes all required fields."""
        # Given
        error_data = {
            "error": ErrorType.VALIDATION,
            "message": "Invalid input parameters",
            "details": {"field": "chunk_size", "value": -1},
            "correlation_id": "req_123456789",
        }

        # When
        error_response = ErrorResponse(**error_data)

        # Then
        assert error_response.error == ErrorType.VALIDATION
        assert error_response.message == "Invalid input parameters"
        assert error_response.details["field"] == "chunk_size"
        assert error_response.correlation_id == "req_123456789"
        assert isinstance(error_response.timestamp, datetime)

    def test_search_result_validation(self):
        """Test SearchResult includes all required fields."""
        # Given
        result_data = {
            "document_id": 42,
            "chunk_id": 156,
            "filename": "test.pdf",
            "content": "Sample content",
            "similarity_score": 0.89,
            "rank": 1,
        }

        # When
        result = SearchResult(**result_data)

        # Then
        assert result.document_id == 42
        assert result.chunk_id == 156
        assert result.similarity_score == 0.89
        assert result.rank == 1

    def test_search_result_score_validation(self):
        """Test SearchResult validates similarity score bounds."""
        # Given - invalid score
        invalid_data = {
            "document_id": 1,
            "filename": "test.pdf",
            "similarity_score": 1.5,  # > 1.0
            "rank": 1,
        }

        # When/Then
        with pytest.raises(ValidationError):
            SearchResult(**invalid_data)

    def test_status_response_job_progress(self):
        """Test StatusResponse with job progress information."""
        # Given
        status_data = {
            "job_id": "pdf_proc_123",
            "status": JobStatus.RUNNING,
            "progress_percent": 65.5,
            "current_step": "Generating embeddings",
            "queue_depth": 5,
            "active_jobs": 3,
            "total_documents": 1247,
        }

        # When
        status = StatusResponse(**status_data)

        # Then
        assert status.job_id == "pdf_proc_123"
        assert status.status == JobStatus.RUNNING
        assert status.progress_percent == 65.5
        assert status.current_step == "Generating embeddings"

    def test_health_response_structure(self):
        """Test HealthResponse includes all health check components."""
        # Given
        health_data = {
            "status": "healthy",
            "service": "PDF to Markdown MCP Server",
            "version": "0.1.0",
            "checks": {
                "database": "healthy",
                "celery": "healthy",
                "embeddings": "degraded",
                "storage": "healthy",
            },
            "uptime_seconds": 3600,
            "memory_usage_mb": 256.5,
        }

        # When
        health = HealthResponse(**health_data)

        # Then
        assert health.status == "healthy"
        assert health.checks["embeddings"] == "degraded"
        assert health.uptime_seconds == 3600


class TestConfigurationValidation:
    """Test configuration request and response validation."""

    def test_configuration_request_valid_directories(self):
        """Test ConfigurationRequest with valid watch directories."""
        # Given - create temporary directories
        with (
            tempfile.TemporaryDirectory() as tmp_dir1,
            tempfile.TemporaryDirectory() as tmp_dir2,
        ):
            config_data = {
                "watch_directories": [tmp_dir1, tmp_dir2],
                "restart_watcher": True,
            }

            # When
            config_request = ConfigurationRequest(**config_data)

            # Then
            assert len(config_request.watch_directories) == 2
            assert config_request.restart_watcher is True

    def test_configuration_request_invalid_directory(self):
        """Test ConfigurationRequest rejects non-existent directories."""
        # Given
        invalid_data = {"watch_directories": ["/path/that/does/not/exist"]}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ConfigurationRequest(**invalid_data)

        assert "does not exist" in str(exc_info.value)

    def test_configuration_response_validation_errors(self):
        """Test ConfigurationResponse includes validation errors."""
        # Given
        config_data = {
            "success": False,
            "message": "Configuration update failed",
            "validation_errors": [
                "Directory does not exist: /invalid/path",
                "Invalid embedding provider: unknown",
            ],
        }

        # When
        response = ConfigurationResponse(**config_data)

        # Then
        assert response.success is False
        assert len(response.validation_errors) == 2
        assert "does not exist" in response.validation_errors[0]


class TestBatchConversionValidation:
    """Test batch conversion request validation."""

    def test_batch_request_valid_directory(self):
        """Test BatchConvertRequest with valid directory."""
        # Given - create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_data = {"directory": tmp_dir, "pattern": "**/*.pdf", "max_files": 50}

            # When
            request = BatchConvertRequest(**batch_data)

            # Then
            assert request.directory == Path(tmp_dir)
            assert request.pattern == "**/*.pdf"
            assert request.max_files == 50

    def test_batch_request_max_files_validation(self):
        """Test BatchConvertRequest max_files validation."""
        # Given - create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test valid range
            for max_files in [1, 100, 1000]:
                request = BatchConvertRequest(directory=tmp_dir, max_files=max_files)
                assert request.max_files == max_files

            # Test invalid values
            for invalid_max in [0, 1001]:
                with pytest.raises(ValidationError):
                    BatchConvertRequest(directory=tmp_dir, max_files=invalid_max)


class TestSerializationCompatibility:
    """Test JSON serialization compatibility for API responses."""

    def test_convert_response_json_serializable(self):
        """Test ConvertSingleResponse is JSON serializable."""
        # Given
        response = ConvertSingleResponse(
            success=True,
            document_id=42,
            message="Success",
            source_path=Path("/test.pdf"),
            file_size_bytes=1024,
        )

        # When
        json_str = response.json()

        # Then
        assert json_str is not None
        data = json.loads(json_str)
        assert data["success"] is True
        assert data["document_id"] == 42

    def test_search_response_json_serializable(self):
        """Test SearchResponse with results is JSON serializable."""
        # Given
        results = [
            SearchResult(
                document_id=1, filename="test.pdf", similarity_score=0.9, rank=1
            )
        ]

        response = SearchResponse(
            success=True,
            query="test",
            results=results,
            total_results=1,
            search_time_ms=50,
            top_k=10,
        )

        # When
        json_str = response.json()

        # Then
        assert json_str is not None
        data = json.loads(json_str)
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity_score"] == 0.9

    def test_error_response_json_serializable(self):
        """Test ErrorResponse is JSON serializable."""
        # Given
        error = ErrorResponse(
            error=ErrorType.PROCESSING,
            message="Processing failed",
            details={"file": "test.pdf"},
            correlation_id="req_123",
        )

        # When
        json_str = error.json()

        # Then
        assert json_str is not None
        data = json.loads(json_str)
        assert data["error"] == "processing_error"  # Enum value
        assert data["correlation_id"] == "req_123"
