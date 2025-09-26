"""
Tests for MinerU PDF processing service.

Following TDD approach - tests first, then implementation.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdf_to_markdown_mcp.core.exceptions import ProcessingError, ValidationError
from pdf_to_markdown_mcp.models.processing import ProcessingResult
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.services.mineru import MinerUService


class TestMinerUService:
    """Test MinerU PDF processing service following TDD"""

    @pytest.fixture
    def mineru_service(self) -> MinerUService:
        """Setup MinerU service with mocked dependencies"""
        return MinerUService()

    @pytest.fixture
    def sample_pdf_path(self) -> Path:
        """Sample PDF path for testing"""
        return Path("/tmp/test_document.pdf")

    @pytest.fixture
    def processing_options(self) -> ProcessingOptions:
        """Default processing options"""
        return ProcessingOptions(
            ocr_language="eng",
            preserve_layout=True,
            extract_tables=True,
            extract_formulas=True,
            extract_images=True,
            chunk_for_embeddings=True,
            chunk_size=1000,
            chunk_overlap=200,
        )

    @pytest.mark.asyncio
    async def test_process_pdf_success(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test successful PDF processing with MinerU"""
        # Given (Arrange)
        expected_result = ProcessingResult(
            markdown_content="# Test Document\n\nThis is test content.",
            plain_text="Test Document\n\nThis is test content.",
            extracted_tables=[],
            extracted_formulas=[],
            extracted_images=[],
            chunk_data=[],
            processing_metadata={
                "pages": 1,
                "processing_time_ms": 1500,
                "ocr_confidence": 0.95,
            },
        )

        # Mock MinerU API calls
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(return_value=expected_result)

            # When (Act)
            result = await mineru_service.process_pdf(
                sample_pdf_path, processing_options
            )

            # Then (Assert)
            assert result == expected_result
            assert result.markdown_content.startswith("# Test Document")
            assert result.processing_metadata["pages"] == 1
            mock_instance.process_pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, mineru_service, processing_options):
        """Test error handling when PDF file doesn't exist"""
        # Given (Arrange)
        non_existent_path = Path("/tmp/non_existent.pdf")

        # When/Then (Act/Assert)
        with pytest.raises(ValidationError) as exc_info:
            await mineru_service.process_pdf(non_existent_path, processing_options)

        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_pdf_invalid_file_type(
        self, mineru_service, processing_options
    ):
        """Test error handling for non-PDF files"""
        # Given (Arrange)
        text_file_path = Path("/tmp/test.txt")

        # When/Then (Act/Assert)
        with pytest.raises(ValidationError) as exc_info:
            await mineru_service.process_pdf(text_file_path, processing_options)

        assert "Invalid file type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_pdf_with_tables(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test PDF processing with table extraction"""
        # Given (Arrange)
        expected_tables = [
            {
                "page": 1,
                "table_index": 0,
                "headers": ["Column 1", "Column 2"],
                "rows": [["Data 1", "Data 2"], ["Data 3", "Data 4"]],
                "confidence": 0.92,
            }
        ]

        expected_result = ProcessingResult(
            markdown_content="# Document\n\n| Column 1 | Column 2 |\n|----------|----------|\n| Data 1 | Data 2 |\n| Data 3 | Data 4 |",
            plain_text="Document\n\nColumn 1 Column 2\nData 1 Data 2\nData 3 Data 4",
            extracted_tables=expected_tables,
            extracted_formulas=[],
            extracted_images=[],
            chunk_data=[],
            processing_metadata={"pages": 1, "processing_time_ms": 2000},
        )

        # Mock MinerU API
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(return_value=expected_result)

            # When (Act)
            result = await mineru_service.process_pdf(
                sample_pdf_path, processing_options
            )

            # Then (Assert)
            assert len(result.extracted_tables) == 1
            assert result.extracted_tables[0]["headers"] == ["Column 1", "Column 2"]
            assert "Column 1" in result.markdown_content

    @pytest.mark.asyncio
    async def test_process_pdf_with_formulas(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test PDF processing with formula extraction"""
        # Given (Arrange)
        expected_formulas = [
            {
                "page": 1,
                "formula_index": 0,
                "latex": "E = mc^2",
                "confidence": 0.98,
                "bbox": [100, 200, 150, 220],
            }
        ]

        expected_result = ProcessingResult(
            markdown_content="# Physics\n\n$$E = mc^2$$",
            plain_text="Physics\n\nE = mc^2",
            extracted_tables=[],
            extracted_formulas=expected_formulas,
            extracted_images=[],
            chunk_data=[],
            processing_metadata={"pages": 1, "processing_time_ms": 1800},
        )

        # Mock MinerU API
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(return_value=expected_result)

            # When (Act)
            result = await mineru_service.process_pdf(
                sample_pdf_path, processing_options
            )

            # Then (Assert)
            assert len(result.extracted_formulas) == 1
            assert result.extracted_formulas[0]["latex"] == "E = mc^2"
            assert "$$E = mc^2$$" in result.markdown_content

    @pytest.mark.asyncio
    async def test_process_pdf_with_images(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test PDF processing with image extraction"""
        # Given (Arrange)
        expected_images = [
            {
                "page": 1,
                "image_index": 0,
                "path": "/tmp/extracted_image_0.png",
                "ocr_text": "Extracted text from image",
                "confidence": 0.87,
                "bbox": [50, 100, 200, 300],
            }
        ]

        expected_result = ProcessingResult(
            markdown_content="# Document\n\n![Image](extracted_image_0.png)",
            plain_text="Document\n\nExtracted text from image",
            extracted_tables=[],
            extracted_formulas=[],
            extracted_images=expected_images,
            chunk_data=[],
            processing_metadata={"pages": 1, "processing_time_ms": 2500},
        )

        # Mock MinerU API
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(return_value=expected_result)

            # When (Act)
            result = await mineru_service.process_pdf(
                sample_pdf_path, processing_options
            )

            # Then (Assert)
            assert len(result.extracted_images) == 1
            assert result.extracted_images[0]["ocr_text"] == "Extracted text from image"
            assert "![Image]" in result.markdown_content

    @pytest.mark.asyncio
    async def test_process_pdf_chunking_enabled(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test PDF processing with content chunking for embeddings"""
        # Given (Arrange)
        long_content = "This is a very long document. " * 100  # Simulate long content

        expected_chunks = [
            {
                "chunk_index": 0,
                "text": long_content[:1000],
                "start_char": 0,
                "end_char": 1000,
                "page": 1,
            },
            {
                "chunk_index": 1,
                "text": long_content[800:1800],  # 200 char overlap
                "start_char": 800,
                "end_char": 1800,
                "page": 1,
            },
        ]

        expected_result = ProcessingResult(
            markdown_content=f"# Long Document\n\n{long_content}",
            plain_text=f"Long Document\n\n{long_content}",
            extracted_tables=[],
            extracted_formulas=[],
            extracted_images=[],
            chunk_data=expected_chunks,
            processing_metadata={"pages": 1, "processing_time_ms": 3000},
        )

        # Mock MinerU API
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(return_value=expected_result)

            # When (Act)
            result = await mineru_service.process_pdf(
                sample_pdf_path, processing_options
            )

            # Then (Assert)
            assert len(result.chunk_data) >= 2
            assert result.chunk_data[0]["chunk_index"] == 0
            assert result.chunk_data[1]["start_char"] == 800  # Overlap test

    @pytest.mark.asyncio
    async def test_process_pdf_timeout_error(
        self, mineru_service, sample_pdf_path, processing_options
    ):
        """Test timeout handling for large PDF processing"""
        # Given (Arrange)
        # Mock MinerU to simulate timeout
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            mock_instance.process_pdf = AsyncMock(side_effect=TimeoutError())

            # When/Then (Act/Assert)
            with pytest.raises(ProcessingError) as exc_info:
                await mineru_service.process_pdf(sample_pdf_path, processing_options)

            assert "Processing timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_valid(self, mineru_service):
        """Test PDF file validation for valid files"""
        # Given (Arrange)
        valid_pdf_path = Path("/tmp/valid.pdf")

        # Mock file existence and type checking
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(Path, "suffix", ".pdf"),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            # Mock file size
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB

            # When (Act)
            result = await mineru_service.validate_pdf_file(valid_pdf_path)

            # Then (Assert)
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_pdf_file_too_large(self, mineru_service):
        """Test PDF file validation for files exceeding size limit"""
        # Given (Arrange)
        large_pdf_path = Path("/tmp/large.pdf")

        # Mock file existence but large size
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(Path, "suffix", ".pdf"),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            # Mock file size > 500MB
            mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600MB

            # When/Then (Act/Assert)
            with pytest.raises(ValidationError) as exc_info:
                await mineru_service.validate_pdf_file(large_pdf_path)

            assert "File too large" in str(exc_info.value)

    def test_get_mineru_config(self, mineru_service, processing_options):
        """Test MinerU configuration generation"""
        # Given (Arrange)
        # When (Act)
        config = mineru_service._get_mineru_config(processing_options)

        # Then (Assert)
        assert config is not None
        assert hasattr(config, "layout_mode")
        assert hasattr(config, "ocr_language")
        assert hasattr(config, "extract_tables")
        assert hasattr(config, "extract_formulas")
        assert hasattr(config, "extract_images")

    # Tests for critical production safety fixes
    @pytest.mark.asyncio
    async def test_production_mode_fails_when_mineru_unavailable(
        self, sample_pdf_path, processing_options
    ):
        """Test that production mode fails fast when MinerU library is unavailable"""
        # Given (Arrange) - Production mode, no mock services, MinerU unavailable
        with patch("pdf_to_markdown_mcp.services.mineru.settings") as mock_settings:
            mock_settings.environment = "production"
            mock_settings.mock_services = False

            # Create service that will fail to import MinerU
            with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI", None):
                service = MinerUService()

                # When/Then (Act/Assert)
                with pytest.raises(ProcessingError) as exc_info:
                    await service.process_pdf(sample_pdf_path, processing_options)

                assert "MinerU library not available in production" in str(
                    exc_info.value
                )
                assert "DEPENDENCY_MISSING" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_development_mode_allows_mock_processing(
        self, sample_pdf_path, processing_options
    ):
        """Test that development mode allows mock processing when MinerU unavailable"""
        # Given (Arrange) - Development mode allows mocking
        with patch("pdf_to_markdown_mcp.services.mineru.settings") as mock_settings:
            mock_settings.environment = "development"
            mock_settings.mock_services = True

            # Create service that will fall back to mock
            with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI", None):
                service = MinerUService()

                # Mock file validation to pass
                with (
                    patch.object(
                        service,
                        "validate_pdf_file",
                        new_callable=AsyncMock,
                        return_value=True,
                    ),
                    patch("pathlib.Path.stat") as mock_stat,
                ):
                    mock_stat.return_value.st_size = 1024 * 1024  # 1MB

                    # When (Act)
                    result = await service.process_pdf(
                        sample_pdf_path, processing_options
                    )

                    # Then (Assert)
                    assert result is not None
                    assert (
                        "mock"
                        in result.processing_metadata.get("mineru_version", "").lower()
                    )
                    assert result.markdown_content is not None

    @pytest.mark.asyncio
    async def test_file_size_streaming_threshold_enforced(self, processing_options):
        """Test that streaming is enforced for files larger than 25MB (reduced threshold)"""
        # Given (Arrange) - Large file requiring streaming
        large_file_path = Path("/tmp/large_file.pdf")

        with patch("pdf_to_markdown_mcp.services.mineru.settings") as mock_settings:
            mock_settings.environment = "production"

            service = MinerUService()

            # Mock large file (30MB)
            with (
                patch.object(
                    service,
                    "validate_pdf_file",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch("pathlib.Path.stat") as mock_stat,
                patch.object(
                    service, "_process_with_mineru_streaming", new_callable=AsyncMock
                ) as mock_streaming,
            ):
                mock_stat.return_value.st_size = 30 * 1024 * 1024  # 30MB
                mock_streaming.return_value = ProcessingResult(
                    markdown_content="# Test",
                    plain_text="Test",
                    extracted_tables=[],
                    extracted_formulas=[],
                    extracted_images=[],
                    chunk_data=[],
                    processing_metadata={},
                )

                # When (Act)
                await service.process_pdf_streaming(large_file_path, processing_options)

                # Then (Assert) - Streaming method should be called for files > 25MB
                mock_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_safe_hash_calculation(self, processing_options):
        """Test that file hash calculation is memory-safe for large files"""
        # Given (Arrange) - Large file requiring memory-safe hashing
        large_file_path = Path("/tmp/very_large.pdf")

        with patch("pdf_to_markdown_mcp.services.mineru.settings") as mock_settings:
            mock_settings.environment = "production"

            service = MinerUService()

            # Mock very large file (200MB)
            with (
                patch("pathlib.Path.stat") as mock_stat,
                patch("builtins.open", create=True) as mock_open,
                patch("hashlib.sha256") as mock_hasher,
            ):
                mock_stat.return_value.st_size = 200 * 1024 * 1024  # 200MB

                # Mock file read chunks - simulate reading in small chunks
                mock_file = mock_open.return_value.__enter__.return_value
                mock_file.read.side_effect = [
                    b"chunk1",
                    b"chunk2",
                    b"",
                ]  # End with empty to stop iteration

                mock_hasher_instance = Mock()
                mock_hasher.return_value = mock_hasher_instance
                mock_hasher_instance.hexdigest.return_value = "safe_hash_123"

                # When (Act)
                result_hash = service._calculate_file_hash(large_file_path)

                # Then (Assert) - Should read in small chunks, not load entire file
                assert mock_file.read.call_count >= 2  # Multiple chunk reads
                assert result_hash == "safe_hash_123"
                # Verify chunks are reasonable size (not larger than 64KB)
                for call in mock_file.read.call_args_list:
                    chunk_size = call[0][0] if call[0] else 4096
                    assert chunk_size <= 65536  # 64KB max chunk size

    @pytest.mark.asyncio
    async def test_explicit_dependency_validation(self):
        """Test explicit validation of MinerU dependency with clear error messages"""
        # Given (Arrange) - Service without MinerU
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI", None):
            service = MinerUService()

            # When (Act)
            is_available = service.validate_mineru_dependency()

            # Then (Assert)
            assert is_available is False

        # Given (Arrange) - Service with MinerU
        with patch("pdf_to_markdown_mcp.services.mineru.MinerUAPI", Mock()):
            service = MinerUService()

            # When (Act)
            is_available = service.validate_mineru_dependency()

            # Then (Assert)
            assert is_available is True
