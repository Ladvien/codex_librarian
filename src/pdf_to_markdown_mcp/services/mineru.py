"""
MinerU PDF processing service.

This module provides advanced PDF processing capabilities using the MinerU library,
including layout-aware text extraction, table detection, formula recognition, and OCR.
"""

import asyncio
import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import time

from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.core.streaming import (
    stream_large_file,
    StreamingProgressTracker,
    MemoryMappedFileReader,
    stream_processing_with_backpressure
)
from pdf_to_markdown_mcp.models.processing import (
    ProcessingResult,
    ProcessingMetadata,
    TableData,
    FormulaData,
    ImageData,
    ChunkData
)
from pdf_to_markdown_mcp.core.errors import (
    ValidationError,
    ProcessingError,
    OCRError,
    ResourceError,
    validation_error,
    processing_error
)
from pdf_to_markdown_mcp.config import settings

logger = logging.getLogger(__name__)

# File size limit (500MB as per architecture)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Processing timeout (5 minutes as per architecture)
PROCESSING_TIMEOUT_SECONDS = 5 * 60

# Streaming threshold - use streaming for files > 25MB (reduced from 50MB for memory safety)
STREAMING_THRESHOLD_BYTES = 25 * 1024 * 1024

# Memory-safe chunk size for hash calculation
HASH_CHUNK_SIZE = 64 * 1024  # 64KB chunks


class MinerUService:
    """
    MinerU PDF processing service with advanced features.

    Provides layout-aware text extraction, table detection, formula recognition,
    built-in OCR, and automatic content chunking for embeddings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MinerU service.

        Args:
            config: Optional configuration dictionary for MinerU
        """
        self.config = config or {}
        self.logger = logger.getChild(self.__class__.__name__)

        # Import MinerU components (delayed import to handle missing dependencies)
        try:
            from mineru.api import MinerUAPI
            from mineru.config import MinerUConfig
            from mineru.data_types import LayoutMode, OCRLanguage

            self.MinerUAPI = MinerUAPI
            self.MinerUConfig = MinerUConfig
            self.LayoutMode = LayoutMode
            self.OCRLanguage = OCRLanguage
            self._mineru_available = True

        except ImportError as e:
            self.logger.error("MinerU library not available: %s", e)
            self.MinerUAPI = None
            self.MinerUConfig = None
            self.LayoutMode = None
            self.OCRLanguage = None
            self._mineru_available = False

            # Fail fast in production mode
            if settings.environment == "production" and not settings.mock_services:
                if not self._mineru_available:
                    raise processing_error(
                        "MinerU library not available in production environment. "
                        "Please install MinerU or enable mock services for development.",
                        "dependency_validation",
                        "DEPENDENCY_MISSING"
                    )

    def validate_mineru_dependency(self) -> bool:
        """
        Explicitly validate MinerU dependency availability.

        Returns:
            True if MinerU is available, False otherwise
        """
        return self._mineru_available

    def _validate_production_requirements(self) -> None:
        """
        Validate that production requirements are met.

        Raises:
            ProcessingError: If MinerU is not available in production
        """
        if settings.environment == "production" and not settings.mock_services:
            if not self._mineru_available:
                raise processing_error(
                    "MinerU library not available in production environment. "
                    "Please install MinerU or enable mock services for development.",
                    "dependency_validation",
                    "DEPENDENCY_MISSING"
                )

    async def process_pdf(
        self,
        pdf_path: Path,
        options: ProcessingOptions
    ) -> ProcessingResult:
        """Process PDF without streaming (compatibility method)."""
        return await self.process_pdf_streaming(
            pdf_path=pdf_path,
            options=options
        )

    async def process_pdf_streaming(
        self,
        pdf_path: Path,
        options: ProcessingOptions,
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
        output_dir: Optional[Path] = None
    ) -> ProcessingResult:
        """
        Process PDF file with MinerU library using streaming support.

        Args:
            pdf_path: Path to PDF file
            options: Processing options
            progress_callback: Optional callback for progress updates
            output_dir: Optional output directory for files

        Returns:
            ProcessingResult with extracted content and metadata

        Raises:
            ValidationError: If file validation fails
            ProcessingError: If PDF processing fails
        """
        start_time = time.time()
        operation_id = f"mineru_{uuid.uuid4().hex[:8]}"
        file_size = pdf_path.stat().st_size
        use_streaming = file_size > STREAMING_THRESHOLD_BYTES  # Use streaming for files > 25MB

        # Create progress tracker if callback provided
        progress_tracker = None
        if progress_callback:
            progress_tracker = StreamingProgressTracker(
                operation_id=operation_id,
                total_size=file_size,
                callback=progress_callback
            )

        try:
            # Update progress: Starting validation
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0,
                    current_step="Validating PDF file"
                )

            # Validate input file with streaming support
            if use_streaming:
                await self._validate_pdf_file_streaming(pdf_path, progress_tracker)
            else:
                await self.validate_pdf_file(pdf_path)

            self.logger.info(
                "Starting PDF processing: %s (streaming: %s)",
                pdf_path,
                use_streaming
            )

            # Update progress: Starting processing
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0,
                    current_step="Configuring MinerU"
                )

            # Get MinerU configuration
            mineru_config = self._get_mineru_config(options)

            # Update progress: Processing with MinerU
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0,
                    current_step="Processing PDF with MinerU"
                )

            # Process PDF with timeout and streaming support
            if use_streaming:
                result = await asyncio.wait_for(
                    self._process_with_mineru_streaming(
                        pdf_path, mineru_config, output_dir, progress_tracker
                    ),
                    timeout=PROCESSING_TIMEOUT_SECONDS
                )
            else:
                result = await asyncio.wait_for(
                    self._process_with_mineru(pdf_path, mineru_config),
                    timeout=PROCESSING_TIMEOUT_SECONDS
                )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update metadata with actual processing time
            result.processing_metadata.processing_time_ms = processing_time_ms

            # Generate chunks if requested
            if options.chunk_for_embeddings:
                chunks = self._generate_chunks(
                    result.plain_text,
                    options.chunk_size,
                    options.chunk_overlap
                )
                result.chunk_data = chunks

            # Final progress update
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=file_size,
                    current_step="PDF processing completed"
                )
                await progress_tracker.set_completion(success=True)

            self.logger.info(
                "PDF processing completed: %s (%.2fs, streaming: %s)",
                pdf_path,
                processing_time_ms / 1000,
                use_streaming
            )

            return result

        except asyncio.TimeoutError:
            if progress_tracker:
                await progress_tracker.set_completion(
                    success=False,
                    error=f"Processing timeout exceeded {PROCESSING_TIMEOUT_SECONDS}s"
                )
            self.logger.error("PDF processing timeout: %s", pdf_path)
            raise ProcessingError(
                f"Processing timeout exceeded {PROCESSING_TIMEOUT_SECONDS}s",
                pdf_path=str(pdf_path),
                error_code="TIMEOUT"
            )
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            if progress_tracker:
                await progress_tracker.set_completion(
                    success=False,
                    error=str(e)
                )
            self.logger.exception("PDF processing failed: %s", pdf_path)
            raise processing_error(
                f"PDF processing failed: {str(e)}",
                str(pdf_path),
                "mineru_processing"
            )

    async def validate_pdf_file(self, pdf_path: Path) -> bool:
        """
        Validate PDF file for processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check if file exists
        if not pdf_path.exists():
            raise validation_error(
                f"File not found: {pdf_path}",
                "pdf_path",
                str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}",
                "pdf_path",
                str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != '.pdf':
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                "file_type",
                pdf_path.suffix
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                "file_size",
                file_size
            )

        # Check if file is readable
        try:
            with open(pdf_path, 'rb') as f:
                # Read first few bytes to verify it's a valid PDF
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    raise validation_error(
                        "File does not appear to be a valid PDF",
                        "pdf_header",
                        header.decode('ascii', errors='ignore')
                    )
        except IOError as e:
            raise validation_error(
                f"Cannot read PDF file: {str(e)}",
                "file_access",
                str(e)
            )

        return True

    def _get_mineru_config(self, options: ProcessingOptions) -> Any:
        """
        Generate MinerU configuration from processing options.

        Args:
            options: Processing options

        Returns:
            MinerU configuration object
        """
        if not self.MinerUConfig:
            # Mock configuration for testing
            return type('MockConfig', (), {
                'layout_mode': 'preserve' if options.preserve_layout else 'auto',
                'ocr_language': options.ocr_language,
                'extract_tables': options.extract_tables,
                'extract_formulas': options.extract_formulas,
                'extract_images': options.extract_images,
                'chunk_for_embeddings': options.chunk_for_embeddings
            })()

        # Real MinerU configuration
        layout_mode = self.LayoutMode.PRESERVE if options.preserve_layout else self.LayoutMode.AUTO

        # Map language codes to MinerU OCR languages
        ocr_language_map = {
            'eng': self.OCRLanguage.ENGLISH,
            'chi_sim': self.OCRLanguage.CHINESE_SIMPLIFIED,
            'chi_tra': self.OCRLanguage.CHINESE_TRADITIONAL,
            'fra': self.OCRLanguage.FRENCH,
            'deu': self.OCRLanguage.GERMAN,
            'spa': self.OCRLanguage.SPANISH,
            'jpn': self.OCRLanguage.JAPANESE,
            'kor': self.OCRLanguage.KOREAN
        }

        ocr_language = ocr_language_map.get(options.ocr_language, self.OCRLanguage.ENGLISH)

        return self.MinerUConfig(
            layout_mode=layout_mode,
            ocr_language=ocr_language,
            extract_tables=options.extract_tables,
            extract_formulas=options.extract_formulas,
            extract_images=options.extract_images,
            chunk_for_embeddings=options.chunk_for_embeddings
        )

    async def _process_with_mineru(self, pdf_path: Path, config: Any) -> ProcessingResult:
        """
        Internal method to process PDF with MinerU.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration

        Returns:
            ProcessingResult
        """
        if not self.MinerUAPI:
            # Validate production requirements before mock processing
            if settings.environment == "production" and not settings.mock_services:
                raise processing_error(
                    "MinerU library not available in production environment",
                    "dependency_validation",
                    "DEPENDENCY_MISSING"
                )
            # Mock processing for testing/development only
            return self._mock_mineru_processing(pdf_path, config)

        # Real MinerU processing
        api = self.MinerUAPI(config)

        # Process the PDF
        result = await api.process_pdf(str(pdf_path))

        return result

    def _mock_mineru_processing(self, pdf_path: Path, config: Any) -> ProcessingResult:
        """
        Mock MinerU processing for testing/development.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration

        Returns:
            Mock ProcessingResult
        """
        self.logger.warning(
            "Using mock MinerU processing (MinerU library not available) - "
            "Environment: %s, Mock Services: %s",
            settings.environment,
            settings.mock_services
        )

        # Generate file hash
        file_hash = self._calculate_file_hash(pdf_path)
        file_size = pdf_path.stat().st_size

        # Mock processing metadata
        metadata = ProcessingMetadata(
            pages=1,
            processing_time_ms=1500,
            ocr_confidence=0.95,
            file_size_bytes=file_size,
            file_hash=file_hash,
            tables_found=0,
            formulas_found=0,
            images_found=0,
            chunks_created=0,
            text_extraction_quality=0.98,
            layout_preservation_quality=0.92,
            language_detected="en",
            mineru_version="mock-0.1.0",
            processing_options={
                "layout_mode": config.layout_mode,
                "ocr_language": config.ocr_language,
                "extract_tables": config.extract_tables,
                "extract_formulas": config.extract_formulas,
                "extract_images": config.extract_images
            }
        )

        # Mock content
        filename = pdf_path.name
        markdown_content = f"# {filename}\n\nThis is mock content extracted from {filename}."
        plain_text = f"{filename}\n\nThis is mock content extracted from {filename}."

        return ProcessingResult(
            markdown_content=markdown_content,
            plain_text=plain_text,
            extracted_tables=[],
            extracted_formulas=[],
            extracted_images=[],
            chunk_data=[],
            processing_metadata=metadata
        )

    def _generate_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[ChunkData]:
        """
        Generate text chunks for embeddings.

        Args:
            text: Full text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks

        Returns:
            List of ChunkData objects
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))

            # Extract chunk text
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = ChunkData(
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    page=1,  # Simplified for now
                    token_count=self._estimate_token_count(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position (with overlap)
            if end >= len(text):
                break

            start = end - overlap
            if start <= chunks[-1].start_char if chunks else 0:
                start = (chunks[-1].start_char if chunks else 0) + 1

        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters (GPT-style tokenization)
        return max(1, len(text) // 4)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()

        # Use memory-safe chunk size for large files
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "service": "MinerU PDF Processing",
            "version": "0.1.0",
            "max_file_size_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
            "timeout_seconds": PROCESSING_TIMEOUT_SECONDS,
            "supported_languages": [
                "eng", "chi_sim", "chi_tra", "fra", "deu",
                "spa", "jpn", "kor"
            ],
            "features": [
                "layout_aware_extraction",
                "table_detection",
                "formula_recognition",
                "built_in_ocr",
                "automatic_chunking",
                "multi_language_support"
            ]
        }

    def __str__(self) -> str:
        """String representation of service."""
        return f"MinerUService(config={self.config})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MinerUService(config={self.config}, mineru_available={self.MinerUAPI is not None})"

    async def _validate_pdf_file_streaming(
        self,
        pdf_path: Path,
        progress_tracker: Optional[StreamingProgressTracker] = None
    ) -> bool:
        """
        Validate PDF file using streaming for large files.

        Args:
            pdf_path: Path to PDF file
            progress_tracker: Progress tracker for updates

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check if file exists
        if not pdf_path.exists():
            raise validation_error(
                f"File not found: {pdf_path}",
                "pdf_path",
                str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}",
                "pdf_path",
                str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != '.pdf':
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                "file_type",
                pdf_path.suffix
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                "file_size",
                file_size
            )

        # Stream validation of PDF header for large files
        try:
            async with MemoryMappedFileReader(pdf_path, chunk_size=8192) as reader:
                # Read first chunk to verify PDF header
                header_chunk = await reader.read_chunk(8)

                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=len(header_chunk),
                        current_step="Validating PDF header"
                    )

                if not header_chunk.startswith(b'%PDF-'):
                    raise validation_error(
                        "File does not appear to be a valid PDF",
                        "pdf_header",
                        header_chunk.decode('ascii', errors='ignore')
                    )

                # For very large files, do additional integrity checks
                if file_size > 100 * 1024 * 1024:  # > 100MB
                    # Check for PDF EOF marker by reading from the end
                    end_chunk = await reader.read_range(
                        max(0, file_size - 1024), file_size
                    )

                    if progress_tracker:
                        await progress_tracker.update_progress(
                            bytes_processed=len(end_chunk),
                            current_step="Validating PDF footer"
                        )

                    # Look for PDF end marker
                    if b'%%EOF' not in end_chunk:
                        self.logger.warning(
                            f"PDF {pdf_path} may be incomplete (no EOF marker found)"
                        )

        except IOError as e:
            raise validation_error(
                f"Cannot read PDF file: {str(e)}",
                "file_access",
                str(e)
            )

        return True

    async def _process_with_mineru_streaming(
        self,
        pdf_path: Path,
        config: Any,
        output_dir: Optional[Path] = None,
        progress_tracker: Optional[StreamingProgressTracker] = None
    ) -> ProcessingResult:
        """
        Internal method to process PDF with MinerU using streaming.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration
            output_dir: Optional output directory
            progress_tracker: Progress tracker for updates

        Returns:
            ProcessingResult
        """
        if not self.MinerUAPI:
            # Validate production requirements before mock processing
            if settings.environment == "production" and not settings.mock_services:
                raise processing_error(
                    "MinerU library not available in production environment",
                    "dependency_validation",
                    "DEPENDENCY_MISSING"
                )
            # Use mock processing with streaming simulation for development only
            return await self._mock_mineru_processing_streaming(
                pdf_path, config, progress_tracker
            )

        # Real MinerU processing with streaming
        api = self.MinerUAPI(config)

        if progress_tracker:
            await progress_tracker.update_progress(
                bytes_processed=0,
                current_step="Initializing MinerU processing"
            )

        # Process the PDF with progress tracking
        # Note: Real MinerU integration would provide progress callbacks
        result = await api.process_pdf(str(pdf_path))

        return result

    async def _mock_mineru_processing_streaming(
        self,
        pdf_path: Path,
        config: Any,
        progress_tracker: Optional[StreamingProgressTracker] = None
    ) -> ProcessingResult:
        """
        Mock MinerU processing with streaming simulation.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration
            progress_tracker: Progress tracker for updates

        Returns:
            Mock ProcessingResult
        """
        self.logger.warning(
            "Using mock MinerU processing with streaming simulation "
            "(MinerU library not available) - Environment: %s, Mock Services: %s",
            settings.environment,
            settings.mock_services
        )

        file_size = pdf_path.stat().st_size

        # Simulate streaming processing with progress updates
        processing_steps = [
            "Loading PDF structure",
            "Extracting text content",
            "Detecting tables",
            "Processing formulas",
            "Extracting images",
            "Generating chunks",
            "Finalizing output"
        ]

        bytes_per_step = file_size // len(processing_steps)

        for i, step in enumerate(processing_steps):
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=bytes_per_step,
                    current_step=step
                )

            # Simulate processing time
            await asyncio.sleep(0.2)

        # Generate file hash for streaming mode
        operation_id = f"hash_{uuid.uuid4().hex[:8]}"
        file_hash = None

        async for chunk in stream_large_file(
            file_path=pdf_path,
            operation_id=operation_id,
            chunk_size=64 * 1024  # 64KB chunks for hashing
        ):
            if file_hash is None:
                import hashlib
                file_hash = hashlib.sha256()
            file_hash.update(chunk)

        final_hash = file_hash.hexdigest() if file_hash else "mock_hash"

        # Mock processing metadata
        metadata = ProcessingMetadata(
            pages=max(1, file_size // (1024 * 1024)),  # Estimate pages
            processing_time_ms=2000,  # Simulate processing time
            ocr_confidence=0.95,
            file_size_bytes=file_size,
            file_hash=final_hash,
            tables_found=1 if config.extract_tables else 0,
            formulas_found=2 if config.extract_formulas else 0,
            images_found=1 if config.extract_images else 0,
            chunks_created=0,
            text_extraction_quality=0.98,
            layout_preservation_quality=0.92,
            language_detected="en",
            mineru_version="mock-streaming-0.1.0",
            processing_options={
                "layout_mode": config.layout_mode,
                "ocr_language": config.ocr_language,
                "extract_tables": config.extract_tables,
                "extract_formulas": config.extract_formulas,
                "extract_images": config.extract_images,
                "streaming_enabled": True
            }
        )

        # Mock content based on file size
        filename = pdf_path.name
        content_size = max(1000, file_size // 100)  # Estimate extracted text size

        markdown_content = (
            f"# {filename}\\n\\n"
            f"This is mock content extracted from {filename} using streaming processing.\\n\\n"
            f"File size: {file_size} bytes\\n"
            f"Estimated content size: {content_size} characters\\n\\n"
            + "Sample extracted content...\\n" * (content_size // 30)
        )

        plain_text = (
            f"{filename}\\n\\n"
            f"This is mock content extracted from {filename} using streaming processing.\\n"
            f"File size: {file_size} bytes\\n"
            f"Estimated content size: {content_size} characters\\n\\n"
            + "Sample extracted content...\\n" * (content_size // 30)
        )

        return ProcessingResult(
            markdown_content=markdown_content,
            plain_text=plain_text,
            extracted_tables=[],
            extracted_formulas=[],
            extracted_images=[],
            chunk_data=[],
            processing_metadata=metadata
        )

    def get_streaming_capabilities(self) -> Dict[str, Any]:
        """Get information about streaming capabilities."""
        return {
            "streaming_supported": True,
            "memory_mapped_reading": True,
            "progress_tracking": True,
            "backpressure_handling": True,
            "max_file_size_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
            "recommended_streaming_threshold_mb": 50,
            "chunk_sizes": {
                "small_files": "64KB",
                "large_files": "1MB",
                "hash_calculation": "64KB"
            },
            "mineru_library_available": self.MinerUAPI is not None
        }