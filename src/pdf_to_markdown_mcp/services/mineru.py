"""
MinerU PDF processing service.

This module provides advanced PDF processing capabilities using the MinerU library,
including layout-aware text extraction, table detection, formula recognition, and OCR.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pdf_to_markdown_mcp.config import settings
from pdf_to_markdown_mcp.core.errors import (
    ProcessingError,
    ValidationError,
    processing_error,
    validation_error,
)
from pdf_to_markdown_mcp.core.streaming import (
    MemoryMappedFileReader,
    StreamingProgressTracker,
    stream_large_file,
)
from pdf_to_markdown_mcp.models.processing import (
    ChunkData,
    ProcessingMetadata,
    ProcessingResult,
)
from pdf_to_markdown_mcp.models.request import ProcessingOptions

logger = logging.getLogger(__name__)

# File size limit (500MB as per architecture)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Processing timeout (20 minutes for large PDFs with OCR)
# Note: OCR is CPU-only until paddlepaddle-gpu is installed
# Large PDFs (200+ pages) can take 10-15 minutes with CPU OCR
PROCESSING_TIMEOUT_SECONDS = 20 * 60

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

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize MinerU service.

        Args:
            config: Optional configuration dictionary for MinerU
        """
        self.config = config or {}
        self.logger = logger.getChild(self.__class__.__name__)

        # Import MinerU components (delayed import to handle missing dependencies)
        try:
            from mineru.cli.common import do_parse, read_fn
            from mineru.utils.enum_class import MakeMode

            self.do_parse = do_parse
            self.read_fn = read_fn
            self.MakeMode = MakeMode
            self._mineru_available = True

        except ImportError as e:
            self.logger.error("MinerU library not available: %s", e)
            self.do_parse = None
            self.read_fn = None
            self.MakeMode = None
            self._mineru_available = False

            if not self._mineru_available:
                raise processing_error(
                    "MinerU library not available. Please install MinerU."
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
            ProcessingError: If MinerU is not available
        """
        if not self._mineru_available:
            raise processing_error(
                "MinerU library not available. Please install MinerU."
            )

    async def process_pdf(
        self, pdf_path: Path, options: ProcessingOptions
    ) -> ProcessingResult:
        """Process PDF without streaming (compatibility method)."""
        return await self.process_pdf_streaming(pdf_path=pdf_path, options=options)

    async def process_pdf_streaming(
        self,
        pdf_path: Path,
        options: ProcessingOptions,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
        output_dir: Path | None = None,
        output_filename: str | None = None,
    ) -> ProcessingResult:
        """
        Process PDF file with MinerU library using streaming support.

        Args:
            pdf_path: Path to PDF file
            options: Processing options
            progress_callback: Optional callback for progress updates
            output_dir: Optional output directory for files
            output_filename: Optional specific output filename (with extension)

        Returns:
            ProcessingResult with extracted content and metadata

        Raises:
            ValidationError: If file validation fails
            ProcessingError: If PDF processing fails
        """
        start_time = time.time()
        operation_id = f"mineru_{uuid.uuid4().hex[:8]}"
        file_size = pdf_path.stat().st_size
        use_streaming = (
            file_size > STREAMING_THRESHOLD_BYTES
        )  # Use streaming for files > 25MB

        # Create progress tracker if callback provided
        progress_tracker = None
        if progress_callback:
            progress_tracker = StreamingProgressTracker(
                operation_id=operation_id,
                total_size=file_size,
                callback=progress_callback,
            )

        try:
            # Update progress: Starting validation
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0, current_step="Validating PDF file"
                )

            # Validate input file with streaming support
            if use_streaming:
                await self._validate_pdf_file_streaming(pdf_path, progress_tracker)
            else:
                await self.validate_pdf_file(pdf_path)

            self.logger.info(
                "Starting PDF processing: %s (streaming: %s)", pdf_path, use_streaming
            )

            # Update progress: Starting processing
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0, current_step="Configuring MinerU"
                )

            # Get MinerU configuration
            mineru_config = self._get_mineru_config(options)

            # Update progress: Processing with MinerU
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=0, current_step="Processing PDF with MinerU"
                )

            # Process PDF with timeout and streaming support
            if use_streaming:
                result = await asyncio.wait_for(
                    self._process_with_mineru_streaming(
                        pdf_path, mineru_config, output_dir, progress_tracker
                    ),
                    timeout=PROCESSING_TIMEOUT_SECONDS,
                )
            else:
                result = await asyncio.wait_for(
                    self._process_with_mineru(pdf_path, mineru_config),
                    timeout=PROCESSING_TIMEOUT_SECONDS,
                )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update metadata with actual processing time
            result.processing_metadata.processing_time_ms = processing_time_ms

            # Generate chunks if requested
            if options.chunk_for_embeddings:
                chunks = self._generate_chunks(
                    result.plain_text, options.chunk_size, options.chunk_overlap
                )
                result.chunk_data = chunks

            # Final progress update
            if progress_tracker:
                await progress_tracker.update_progress(
                    bytes_processed=file_size, current_step="PDF processing completed"
                )
                await progress_tracker.set_completion(success=True)

            self.logger.info(
                "PDF processing completed: %s (%.2fs, streaming: %s)",
                pdf_path,
                processing_time_ms / 1000,
                use_streaming,
            )

            return result

        except TimeoutError:
            if progress_tracker:
                await progress_tracker.set_completion(
                    success=False,
                    error=f"Processing timeout exceeded {PROCESSING_TIMEOUT_SECONDS}s",
                )
            self.logger.error("PDF processing timeout: %s", pdf_path)
            raise ProcessingError(
                f"Processing timeout exceeded {PROCESSING_TIMEOUT_SECONDS}s",
                file_path=str(pdf_path),
                error_code="TIMEOUT",
            )
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            if progress_tracker:
                await progress_tracker.set_completion(success=False, error=str(e))
            self.logger.exception("PDF processing failed: %s", pdf_path)
            raise processing_error(
                f"PDF processing failed: {e!s}"
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
                f"File not found: {pdf_path}", "pdf_path", str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}", "pdf_path", str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != ".pdf":
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                "file_type",
                pdf_path.suffix,
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                "file_size",
                file_size,
            )

        # Check if file is readable
        try:
            with open(pdf_path, "rb") as f:
                # Read first few bytes to verify it's a valid PDF
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    raise validation_error(
                        "File does not appear to be a valid PDF",
                        "pdf_header",
                        header.decode("ascii", errors="ignore"),
                    )
        except OSError as e:
            raise validation_error(
                f"Cannot read PDF file: {e!s}", "file_access", str(e)
            )

        return True

    def _get_mineru_config(self, options: ProcessingOptions) -> dict[str, Any]:
        """
        Generate MinerU configuration from processing options.

        Args:
            options: Processing options

        Returns:
            Dictionary with do_parse parameters
        """
        # Map language codes to MinerU language strings
        language_map = {
            "eng": "en",
            "chi_sim": "ch",
            "chi_tra": "chinese_cht",
            "fra": "en",  # MinerU doesn't have French, fallback to English
            "deu": "en",  # MinerU doesn't have German, fallback to English
            "spa": "en",  # MinerU doesn't have Spanish, fallback to English
            "jpn": "japan",
            "kor": "korean",
        }

        mineru_lang = language_map.get(options.ocr_language, "en")

        return {
            "backend": "pipeline",
            "parse_method": "auto",
            "p_formula_enable": options.extract_formulas,
            "p_table_enable": options.extract_tables,
            "p_lang": mineru_lang,
            "f_dump_md": True,
            "f_dump_middle_json": True,
            "f_dump_content_list": False,
            "f_dump_model_output": False,
            "f_dump_orig_pdf": False,
            "f_draw_layout_bbox": False,
            "f_draw_span_bbox": False,
            "f_make_md_mode": self.MakeMode.MM_MD if self.MakeMode else None,
        }

    async def _process_with_mineru(
        self, pdf_path: Path, config: dict[str, Any]
    ) -> ProcessingResult:
        """
        Internal method to process PDF with MinerU.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration dictionary

        Returns:
            ProcessingResult
        """
        if not self.do_parse:
            raise processing_error(
                "MinerU library not available"
            )

        # Create temporary output directory
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="mineru_"))

        try:
            # Read PDF bytes
            pdf_bytes = self.read_fn(pdf_path)
            pdf_file_name = pdf_path.stem

            # Run MinerU parsing in thread pool (it's CPU-bound)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.do_parse,
                str(output_dir),
                [pdf_file_name],
                [pdf_bytes],
                [config["p_lang"]],
                config["backend"],
                config["parse_method"],
                config["p_formula_enable"],
                config["p_table_enable"],
                None,  # server_url
                config["f_draw_layout_bbox"],
                config["f_draw_span_bbox"],
                config["f_dump_md"],
                config["f_dump_middle_json"],
                config["f_dump_model_output"],
                config["f_dump_orig_pdf"],
                config["f_dump_content_list"],
                config["f_make_md_mode"],
                0,  # start_page_id
                None,  # end_page_id
            )

            # Read generated markdown
            md_dir = output_dir / pdf_file_name / config["parse_method"]
            md_file = md_dir / f"{pdf_file_name}.md"

            if not md_file.exists():
                raise processing_error(f"MinerU did not generate markdown file: {md_file}")

            markdown_content = md_file.read_text(encoding="utf-8")

            # Extract plain text (strip markdown)
            plain_text = self._markdown_to_plain_text(markdown_content)

            # Try to read middle JSON for metadata
            middle_json_file = md_dir / f"{pdf_file_name}_middle.json"
            page_count = 1
            if middle_json_file.exists():
                import json
                middle_json = json.loads(middle_json_file.read_text(encoding="utf-8"))
                page_count = len(middle_json.get("pdf_info", []))

            # Create processing result
            file_hash = self._calculate_file_hash(pdf_path)
            file_size = pdf_path.stat().st_size

            metadata = ProcessingMetadata(
                file_hash=file_hash,
                file_size_bytes=file_size,
                pages=page_count,
                processing_time_ms=0,  # Will be updated by caller
                mineru_version="2.0+",
            )

            return ProcessingResult(
                markdown_content=markdown_content,
                plain_text=plain_text,
                processing_metadata=metadata,
            )

        finally:
            # Clean up temporary directory
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def _generate_chunks(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[ChunkData]:
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
                    token_count=self._estimate_token_count(chunk_text),
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

    def _markdown_to_plain_text(self, markdown: str) -> str:
        """
        Convert markdown to plain text by stripping formatting.

        Args:
            markdown: Markdown content

        Returns:
            Plain text content
        """
        import re

        text = markdown

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

        # Remove bold/italic
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

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
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def get_processing_stats(self) -> dict[str, Any]:
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
                "eng",
                "chi_sim",
                "chi_tra",
                "fra",
                "deu",
                "spa",
                "jpn",
                "kor",
            ],
            "features": [
                "layout_aware_extraction",
                "table_detection",
                "formula_recognition",
                "built_in_ocr",
                "automatic_chunking",
                "multi_language_support",
            ],
        }

    def __str__(self) -> str:
        """String representation of service."""
        return f"MinerUService(config={self.config})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MinerUService(config={self.config}, mineru_available={self.do_parse is not None})"

    async def _validate_pdf_file_streaming(
        self,
        pdf_path: Path,
        progress_tracker: StreamingProgressTracker | None = None,
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
                f"File not found: {pdf_path}", "pdf_path", str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}", "pdf_path", str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != ".pdf":
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                "file_type",
                pdf_path.suffix,
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                "file_size",
                file_size,
            )

        # Stream validation of PDF header for large files
        try:
            async with MemoryMappedFileReader(pdf_path, chunk_size=8192) as reader:
                # Read first chunk to verify PDF header
                header_chunk = await reader.read_chunk(8)

                if progress_tracker:
                    await progress_tracker.update_progress(
                        bytes_processed=len(header_chunk),
                        current_step="Validating PDF header",
                    )

                if not header_chunk.startswith(b"%PDF-"):
                    raise validation_error(
                        "File does not appear to be a valid PDF",
                        "pdf_header",
                        header_chunk.decode("ascii", errors="ignore"),
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
                            current_step="Validating PDF footer",
                        )

                    # Look for PDF end marker
                    if b"%%EOF" not in end_chunk:
                        self.logger.warning(
                            f"PDF {pdf_path} may be incomplete (no EOF marker found)"
                        )

        except OSError as e:
            raise validation_error(
                f"Cannot read PDF file: {e!s}", "file_access", str(e)
            )

        return True

    async def _process_with_mineru_streaming(
        self,
        pdf_path: Path,
        config: dict[str, Any],
        output_dir: Path | None = None,
        progress_tracker: StreamingProgressTracker | None = None,
    ) -> ProcessingResult:
        """
        Internal method to process PDF with MinerU using streaming.

        Args:
            pdf_path: Path to PDF file
            config: MinerU configuration dictionary
            output_dir: Optional output directory
            progress_tracker: Progress tracker for updates

        Returns:
            ProcessingResult
        """
        if progress_tracker:
            await progress_tracker.update_progress(
                bytes_processed=0, current_step="Initializing MinerU processing"
            )

        # Use the same processing logic as non-streaming
        # MinerU doesn't provide streaming progress callbacks
        result = await self._process_with_mineru(pdf_path, config)

        return result

    def get_streaming_capabilities(self) -> dict[str, Any]:
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
                "hash_calculation": "64KB",
            },
            "mineru_library_available": self.do_parse is not None,
        }
