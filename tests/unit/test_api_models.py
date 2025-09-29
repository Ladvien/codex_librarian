"""
Unit tests for API models (Pydantic models).

This module tests all Pydantic models for validation,
serialization, and edge cases following TDD principles.
"""

import json

import pytest
from pydantic import ValidationError

from src.pdf_to_markdown_mcp.models.document import (
    DocumentContentModel,
    DocumentEmbeddingModel,
    DocumentModel,
)
from src.pdf_to_markdown_mcp.models.request import (
    BatchConversionRequest,
    ConversionRequest,
    HybridSearchRequest,
    ProcessingOptions,
    SearchRequest,
)
from src.pdf_to_markdown_mcp.models.response import (
    ConversionResponse,
    SearchResponse,
    SearchResult,
)
from tests.fixtures import (
    DocumentFactory,
    create_sample_embeddings,
)


class TestProcessingOptions:
    """Test ProcessingOptions model validation."""

    def test_processing_options_default_values(self):
        """Test ProcessingOptions with default values."""
        # When
        options = ProcessingOptions()

        # Then
        assert options.language == "en"
        assert options.chunk_for_embeddings is True
        assert options.chunk_size == 1000
        assert options.chunk_overlap == 200
        assert options.extract_tables is True
        assert options.extract_formulas is True
        assert options.extract_images is True
        assert options.ocr_enabled is True

    def test_processing_options_custom_values(self):
        """Test ProcessingOptions with custom values."""
        # Given
        custom_options = {
            "language": "fr",
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "extract_tables": False,
            "ocr_enabled": False,
        }

        # When
        options = ProcessingOptions(**custom_options)

        # Then
        assert options.language == "fr"
        assert options.chunk_size == 1500
        assert options.chunk_overlap == 300
        assert options.extract_tables is False
        assert options.ocr_enabled is False

    def test_processing_options_invalid_language(self):
        """Test ProcessingOptions with invalid language."""
        # Given
        invalid_data = {"language": "invalid_lang"}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ProcessingOptions(**invalid_data)

        assert "language" in str(exc_info.value)

    def test_processing_options_negative_chunk_size(self):
        """Test ProcessingOptions with negative chunk size."""
        # Given
        invalid_data = {"chunk_size": -100}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ProcessingOptions(**invalid_data)

        assert "chunk_size" in str(exc_info.value)

    def test_processing_options_overlap_larger_than_size(self):
        """Test ProcessingOptions with overlap larger than chunk size."""
        # Given
        invalid_data = {
            "chunk_size": 500,
            "chunk_overlap": 600,  # Larger than chunk_size
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ProcessingOptions(**invalid_data)

        assert "overlap" in str(exc_info.value).lower()


class TestConversionRequest:
    """Test ConversionRequest model validation."""

    def test_conversion_request_minimal(self, temp_directory):
        """Test ConversionRequest with minimal required fields."""
        # Given
        pdf_path = temp_directory / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        request_data = {"file_path": str(pdf_path)}

        # When
        request = ConversionRequest(**request_data)

        # Then
        assert request.file_path == str(pdf_path)
        assert request.store_embeddings is True  # Default value
        assert isinstance(request.processing_options, ProcessingOptions)

    def test_conversion_request_full_options(self, temp_directory):
        """Test ConversionRequest with all options."""
        # Given
        pdf_path = temp_directory / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        request_data = {
            "file_path": str(pdf_path),
            "store_embeddings": False,
            "processing_options": {
                "language": "fr",
                "chunk_size": 1200,
                "extract_tables": False,
            },
        }

        # When
        request = ConversionRequest(**request_data)

        # Then
        assert request.file_path == str(pdf_path)
        assert request.store_embeddings is False
        assert request.processing_options.language == "fr"
        assert request.processing_options.chunk_size == 1200
        assert request.processing_options.extract_tables is False

    def test_conversion_request_nonexistent_file(self):
        """Test ConversionRequest with non-existent file."""
        # Given
        request_data = {"file_path": "/nonexistent/file.pdf"}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ConversionRequest(**request_data)

        assert "file not found" in str(exc_info.value).lower()

    def test_conversion_request_invalid_file_extension(self, temp_directory):
        """Test ConversionRequest with invalid file extension."""
        # Given
        txt_path = temp_directory / "test.txt"
        txt_path.write_text("not a pdf")

        request_data = {"file_path": str(txt_path)}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            ConversionRequest(**request_data)

        assert "pdf" in str(exc_info.value).lower()

    def test_conversion_request_json_serialization(self, temp_directory):
        """Test ConversionRequest JSON serialization."""
        # Given
        pdf_path = temp_directory / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        request_data = {"file_path": str(pdf_path), "store_embeddings": True}

        # When
        request = ConversionRequest(**request_data)
        json_str = request.json()
        parsed_data = json.loads(json_str)

        # Then
        assert parsed_data["file_path"] == str(pdf_path)
        assert parsed_data["store_embeddings"] is True
        assert "processing_options" in parsed_data


class TestBatchConversionRequest:
    """Test BatchConversionRequest model validation."""

    def test_batch_conversion_request_valid(self, temp_directory):
        """Test BatchConversionRequest with valid file paths."""
        # Given
        pdf_paths = []
        for i in range(3):
            pdf_path = temp_directory / f"test_{i}.pdf"
            pdf_path.write_bytes(b"dummy pdf content")
            pdf_paths.append(str(pdf_path))

        request_data = {"file_paths": pdf_paths, "store_embeddings": True}

        # When
        request = BatchConversionRequest(**request_data)

        # Then
        assert len(request.file_paths) == 3
        assert all(str(path) in request.file_paths for path in pdf_paths)
        assert request.store_embeddings is True

    def test_batch_conversion_request_empty_list(self):
        """Test BatchConversionRequest with empty file list."""
        # Given
        request_data = {"file_paths": []}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            BatchConversionRequest(**request_data)

        assert "empty" in str(exc_info.value).lower()

    def test_batch_conversion_request_duplicate_files(self, temp_directory):
        """Test BatchConversionRequest removes duplicate file paths."""
        # Given
        pdf_path = temp_directory / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        request_data = {"file_paths": [str(pdf_path), str(pdf_path), str(pdf_path)]}

        # When
        request = BatchConversionRequest(**request_data)

        # Then
        assert len(request.file_paths) == 1
        assert request.file_paths[0] == str(pdf_path)

    def test_batch_conversion_request_mixed_valid_invalid(self, temp_directory):
        """Test BatchConversionRequest with mix of valid and invalid files."""
        # Given
        valid_pdf = temp_directory / "valid.pdf"
        valid_pdf.write_bytes(b"dummy pdf content")

        request_data = {
            "file_paths": [
                str(valid_pdf),
                "/nonexistent/file.pdf",  # Invalid
                str(temp_directory / "another.pdf"),  # Invalid (doesn't exist)
            ]
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            BatchConversionRequest(**request_data)

        assert "file not found" in str(exc_info.value).lower()


class TestSearchRequest:
    """Test SearchRequest model validation."""

    def test_search_request_minimal(self):
        """Test SearchRequest with minimal required fields."""
        # Given
        request_data = {"query": "machine learning algorithms"}

        # When
        request = SearchRequest(**request_data)

        # Then
        assert request.query == "machine learning algorithms"
        assert request.limit == 10  # Default value
        assert request.threshold == 0.7  # Default value

    def test_search_request_full_options(self):
        """Test SearchRequest with all options."""
        # Given
        request_data = {
            "query": "neural networks",
            "limit": 20,
            "threshold": 0.8,
            "document_ids": [1, 2, 3],
            "language_filter": "en",
        }

        # When
        request = SearchRequest(**request_data)

        # Then
        assert request.query == "neural networks"
        assert request.limit == 20
        assert request.threshold == 0.8
        assert request.document_ids == [1, 2, 3]
        assert request.language_filter == "en"

    def test_search_request_empty_query(self):
        """Test SearchRequest with empty query."""
        # Given
        request_data = {"query": ""}

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(**request_data)

        assert "query" in str(exc_info.value).lower()

    def test_search_request_invalid_limit(self):
        """Test SearchRequest with invalid limit values."""
        # Test negative limit
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=-1)

        # Test zero limit
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)

        # Test too large limit
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=1001)

    def test_search_request_invalid_threshold(self):
        """Test SearchRequest with invalid threshold values."""
        # Test negative threshold
        with pytest.raises(ValidationError):
            SearchRequest(query="test", threshold=-0.1)

        # Test threshold > 1
        with pytest.raises(ValidationError):
            SearchRequest(query="test", threshold=1.1)


class TestHybridSearchRequest:
    """Test HybridSearchRequest model validation."""

    def test_hybrid_search_request_default_weights(self):
        """Test HybridSearchRequest with default weights."""
        # Given
        request_data = {"query": "artificial intelligence"}

        # When
        request = HybridSearchRequest(**request_data)

        # Then
        assert request.query == "artificial intelligence"
        assert request.semantic_weight == 0.7
        assert request.keyword_weight == 0.3
        assert abs(request.semantic_weight + request.keyword_weight - 1.0) < 1e-6

    def test_hybrid_search_request_custom_weights(self):
        """Test HybridSearchRequest with custom weights."""
        # Given
        request_data = {
            "query": "deep learning",
            "semantic_weight": 0.8,
            "keyword_weight": 0.2,
        }

        # When
        request = HybridSearchRequest(**request_data)

        # Then
        assert request.semantic_weight == 0.8
        assert request.keyword_weight == 0.2

    def test_hybrid_search_request_weights_not_sum_to_one(self):
        """Test HybridSearchRequest with weights that don't sum to 1."""
        # Given
        request_data = {
            "query": "test",
            "semantic_weight": 0.6,
            "keyword_weight": 0.5,  # Total = 1.1
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            HybridSearchRequest(**request_data)

        assert "sum to 1" in str(exc_info.value).lower()


class TestConversionResponse:
    """Test ConversionResponse model."""

    def test_conversion_response_success(self):
        """Test successful ConversionResponse."""
        # Given
        response_data = {
            "success": True,
            "task_id": "task-123",
            "document_id": 1,
            "message": "Processing started",
        }

        # When
        response = ConversionResponse(**response_data)

        # Then
        assert response.success is True
        assert response.task_id == "task-123"
        assert response.document_id == 1
        assert response.message == "Processing started"
        assert response.error is None

    def test_conversion_response_failure(self):
        """Test failed ConversionResponse."""
        # Given
        response_data = {"success": False, "error": "File not found"}

        # When
        response = ConversionResponse(**response_data)

        # Then
        assert response.success is False
        assert response.error == "File not found"
        assert response.task_id is None
        assert response.document_id is None

    def test_conversion_response_with_metadata(self):
        """Test ConversionResponse with processing metadata."""
        # Given
        metadata = {
            "processing_time": 2.5,
            "page_count": 5,
            "word_count": 1000,
            "language": "en",
            "confidence": 0.95,
        }

        response_data = {
            "success": True,
            "task_id": "task-456",
            "document_id": 2,
            "processing_metadata": metadata,
        }

        # When
        response = ConversionResponse(**response_data)

        # Then
        assert response.processing_metadata is not None
        assert response.processing_metadata.processing_time == 2.5
        assert response.processing_metadata.page_count == 5
        assert response.processing_metadata.confidence == 0.95


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_search_response_with_results(self):
        """Test SearchResponse with search results."""
        # Given
        results = [
            {
                "document_id": 1,
                "chunk_text": "Machine learning is a subset of AI",
                "similarity_score": 0.92,
                "document_title": "AI Introduction",
                "chunk_index": 0,
            },
            {
                "document_id": 2,
                "chunk_text": "Neural networks are powerful models",
                "similarity_score": 0.85,
                "document_title": "Neural Networks Guide",
                "chunk_index": 1,
            },
        ]

        response_data = {
            "success": True,
            "results": results,
            "total_results": 2,
            "query": "machine learning",
            "search_time": 0.045,
        }

        # When
        response = SearchResponse(**response_data)

        # Then
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.query == "machine learning"
        assert response.search_time == 0.045

        # Check individual result
        result = response.results[0]
        assert result.document_id == 1
        assert result.similarity_score == 0.92
        assert result.document_title == "AI Introduction"

    def test_search_response_empty_results(self):
        """Test SearchResponse with no results."""
        # Given
        response_data = {
            "success": True,
            "results": [],
            "total_results": 0,
            "query": "nonexistent topic",
        }

        # When
        response = SearchResponse(**response_data)

        # Then
        assert response.success is True
        assert len(response.results) == 0
        assert response.total_results == 0

    def test_search_response_failure(self):
        """Test failed SearchResponse."""
        # Given
        response_data = {
            "success": False,
            "results": [],
            "total_results": 0,
            "error": "Embedding service unavailable",
        }

        # When
        response = SearchResponse(**response_data)

        # Then
        assert response.success is False
        assert response.error == "Embedding service unavailable"
        assert len(response.results) == 0


class TestDocumentModels:
    """Test document-related Pydantic models."""

    def test_document_model_creation(self):
        """Test DocumentModel creation and validation."""
        # Given
        document_data = DocumentFactory.create(file_name="test.pdf", status="completed")

        # When
        document_model = DocumentModel(**document_data)

        # Then
        assert document_model.file_name == "test.pdf"
        assert document_model.status == "completed"
        assert document_model.mime_type == "application/pdf"

    def test_document_model_invalid_status(self):
        """Test DocumentModel with invalid status."""
        # Given
        document_data = DocumentFactory.create(status="invalid_status")

        # When/Then
        with pytest.raises(ValidationError):
            DocumentModel(**document_data)

    def test_document_content_model_creation(self):
        """Test DocumentContentModel creation."""
        # Given
        content_data = {
            "id": 1,
            "document_id": 1,
            "markdown_content": "# Test Document\n\nContent here",
            "plain_text": "Test Document\n\nContent here",
            "word_count": 4,
            "language": "en",
        }

        # When
        content_model = DocumentContentModel(**content_data)

        # Then
        assert content_model.document_id == 1
        assert content_model.word_count == 4
        assert content_model.language == "en"
        assert "Test Document" in content_model.markdown_content

    def test_document_embedding_model_creation(self):
        """Test DocumentEmbeddingModel creation."""
        # Given
        embedding_vector = create_sample_embeddings(1, 1536)[0]
        embedding_data = {
            "id": 1,
            "document_id": 1,
            "chunk_index": 0,
            "chunk_text": "Sample text chunk",
            "embedding": embedding_vector,
            "start_char": 0,
            "end_char": 17,
            "token_count": 3,
        }

        # When
        embedding_model = DocumentEmbeddingModel(**embedding_data)

        # Then
        assert embedding_model.document_id == 1
        assert embedding_model.chunk_index == 0
        assert embedding_model.chunk_text == "Sample text chunk"
        assert len(embedding_model.embedding) == 1536
        assert embedding_model.token_count == 3

    def test_document_embedding_model_invalid_dimension(self):
        """Test DocumentEmbeddingModel with invalid embedding dimensions."""
        # Given
        embedding_data = {
            "id": 1,
            "document_id": 1,
            "chunk_index": 0,
            "chunk_text": "Sample text",
            "embedding": [0.1] * 100,  # Wrong dimension (should be 1536)
            "start_char": 0,
            "end_char": 11,
            "token_count": 2,
        }

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            DocumentEmbeddingModel(**embedding_data)

        assert "embedding" in str(exc_info.value).lower()


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_conversion_request_roundtrip(self, temp_directory):
        """Test ConversionRequest serialization roundtrip."""
        # Given
        pdf_path = temp_directory / "test.pdf"
        pdf_path.write_bytes(b"dummy content")

        original_request = ConversionRequest(
            file_path=str(pdf_path),
            store_embeddings=True,
            processing_options=ProcessingOptions(language="fr", chunk_size=1200),
        )

        # When - Serialize to JSON and back
        json_str = original_request.json()
        request_dict = json.loads(json_str)
        reconstructed_request = ConversionRequest(**request_dict)

        # Then
        assert reconstructed_request.file_path == original_request.file_path
        assert (
            reconstructed_request.store_embeddings == original_request.store_embeddings
        )
        assert (
            reconstructed_request.processing_options.language
            == original_request.processing_options.language
        )
        assert (
            reconstructed_request.processing_options.chunk_size
            == original_request.processing_options.chunk_size
        )

    def test_search_response_roundtrip(self):
        """Test SearchResponse serialization roundtrip."""
        # Given
        original_response = SearchResponse(
            success=True,
            results=[
                SearchResult(
                    document_id=1,
                    chunk_text="Test content",
                    similarity_score=0.95,
                    document_title="Test Doc",
                    chunk_index=0,
                )
            ],
            total_results=1,
            query="test query",
            search_time=0.123,
        )

        # When - Serialize to JSON and back
        json_str = original_response.json()
        response_dict = json.loads(json_str)
        reconstructed_response = SearchResponse(**response_dict)

        # Then
        assert reconstructed_response.success == original_response.success
        assert len(reconstructed_response.results) == len(original_response.results)
        assert (
            reconstructed_response.results[0].similarity_score
            == original_response.results[0].similarity_score
        )
        assert reconstructed_response.query == original_response.query
