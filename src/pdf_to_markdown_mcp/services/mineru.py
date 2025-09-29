"""
MinerU PDF processing service.

This module provides advanced PDF processing capabilities using the MinerU library,
including layout-aware text extraction, table detection, formula recognition, and OCR.
"""

import asyncio
import hashlib
import logging
import os
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

# Global model singleton for GPU resource sharing
_mineru_model_singleton = None
_model_lock = None

# Processing timeout (20 minutes for large PDFs with OCR)
# Note: OCR now uses GPU acceleration via paddlepaddle-gpu (Python 3.12)
# Large PDFs (200+ pages) should process much faster with GPU OCR
PROCESSING_TIMEOUT_SECONDS = 20 * 60

# Streaming threshold - use streaming for files > 25MB (reduced from 50MB for memory safety)
STREAMING_THRESHOLD_BYTES = 25 * 1024 * 1024

# Memory-safe chunk size for hash calculation
HASH_CHUNK_SIZE = 64 * 1024  # 64KB chunks


def get_shared_mineru_instance() -> "MinerUService":
    """Get or create shared MinerU instance for GPU resource management."""
    global _mineru_model_singleton, _model_lock

    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    if _mineru_model_singleton is None:
        with _model_lock:
            if _mineru_model_singleton is None:
                logger.info("Initializing shared MinerU instance for GPU optimization")

                # Record model loading event
                try:
                    from pdf_to_markdown_mcp.core.monitoring import metrics_collector
                    import torch

                    device = "gpu" if torch.cuda.is_available() else "cpu"
                    metrics_collector.record_model_loading("mineru", device, "started")

                    start_time = time.time()
                    _mineru_model_singleton = MinerUService()
                    load_time = time.time() - start_time

                    metrics_collector.record_model_loading("mineru", device, "completed")
                    logger.info(f"MinerU model loaded in {load_time:.2f}s on {device}")

                except Exception as e:
                    logger.warning(f"Failed to record model loading metrics: {e}")
                    _mineru_model_singleton = MinerUService()

    return _mineru_model_singleton


class MinerUService:
    """
    MinerU PDF processing service with advanced features.

    Provides layout-aware text extraction, table detection, formula recognition,
    built-in OCR, and automatic content chunking for embeddings.

    Uses singleton pattern for GPU resource management.
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
                f"File not found: {pdf_path}", field="pdf_path", value=str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}", field="pdf_path", value=str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != ".pdf":
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                field="file_type",
                value=pdf_path.suffix,
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                field="file_size",
                value=file_size,
            )

        # Check if file is readable
        try:
            with open(pdf_path, "rb") as f:
                # Read first few bytes to verify it's a valid PDF
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    raise validation_error(
                        "File does not appear to be a valid PDF",
                        field="pdf_header",
                        value=header.decode("ascii", errors="ignore"),
                    )
        except OSError as e:
            raise validation_error(
                f"Cannot read PDF file: {e!s}", field="file_access", value=str(e)
            )

        return True

    def _get_mineru_config(self, options: ProcessingOptions) -> dict[str, Any]:
        """
        Generate MinerU configuration from processing options with GPU optimization.

        Args:
            options: Processing options

        Returns:
            Dictionary with do_parse parameters optimized for GPU
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

        # Advanced GPU device configuration
        import os
        import torch

        # Configure GPU environment optimally
        self._configure_gpu_environment()

        device = self._get_optimal_device()
        self.logger.info(f"MinerU using device: {device}")

        config = {
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
            # GPU-optimized batch settings
            "batch_size": self._get_optimal_batch_size(),
            "device": device,
        }

        return config

    def _configure_gpu_environment(self) -> None:
        """
        Configure optimal GPU environment for MinerU processing.
        """
        import os
        import torch

        # Set optimal CUDA environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Configure PyTorch CUDA memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

        # Set MinerU device mode explicitly
        os.environ["MINERU_DEVICE_MODE"] = "cuda"

        # Configure PaddlePaddle for GPU
        os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
        os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.8"
        os.environ["FLAGS_allocator_strategy"] = "auto_growth"

        # Optimize for inference
        os.environ["FLAGS_use_mkldnn"] = "false"
        os.environ["FLAGS_use_pinned_memory"] = "true"

        self.logger.info("GPU environment configured for optimal MinerU performance")

    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for MinerU processing.

        Returns:
            Device string ("cuda", "cpu", etc.)
        """
        import torch
        import os

        # Check environment override first
        device_override = os.getenv("MINERU_DEVICE_MODE")
        if device_override:
            self.logger.info(f"Using device override: {device_override}")
            return device_override.lower()

        # Auto-detect optimal device
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)

            self.logger.info(
                f"CUDA available: {device_count} device(s), current: {current_device} "
                f"({device_name}, {memory_gb:.1f}GB)"
            )

            # Ensure sufficient GPU memory (>= 4GB recommended for MinerU)
            if memory_gb >= 4.0:
                return "cuda"
            else:
                self.logger.warning(f"GPU memory insufficient ({memory_gb:.1f}GB < 4GB), falling back to CPU")
                return "cpu"

        elif torch.backends.mps.is_available():
            self.logger.info("Using MPS (Apple Silicon GPU)")
            return "mps"

        else:
            self.logger.info("No GPU acceleration available, using CPU")
            return "cpu"

    def _get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Returns:
            Optimal batch size for processing
        """
        import torch
        import os

        if not torch.cuda.is_available():
            return 1  # CPU processing

        try:
            # Get GPU memory info
            device = torch.cuda.current_device()
            memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

            # Conservative batch size calculation based on GPU memory
            # MinerU models typically use 2-4GB base + ~500MB per batch item
            if memory_gb >= 20:  # RTX 3090/4090 class
                batch_size = min(32, int((memory_gb - 4) / 0.5))
            elif memory_gb >= 12:  # RTX 3060 Ti/4060 Ti class
                batch_size = min(16, int((memory_gb - 3) / 0.5))
            elif memory_gb >= 8:  # RTX 3060/4060 class
                batch_size = min(8, int((memory_gb - 2) / 0.5))
            else:
                batch_size = 4  # Minimum for GPU

            # Set environment variable for MinerU
            os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = str(batch_size)

            self.logger.info(f"Optimal batch size: {batch_size} (GPU memory: {memory_gb:.1f}GB)")
            return batch_size

        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 8  # Conservative default

    async def _process_with_mineru(
        self, pdf_path: Path, config: dict[str, Any]
    ) -> ProcessingResult:
        """
        Internal method to process PDF with MinerU using GPU acceleration.

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
            # Warm up GPU if using CUDA
            planned_device = config.get("device", "cpu")
            if planned_device == "cuda":
                try:
                    await self._warm_up_gpu()
                    # Verify GPU is still available after warm-up
                    if not self._is_cuda_available():
                        logger.warning("GPU became unavailable after warm-up, falling back to CPU")
                        planned_device = "cpu"
                        config["device"] = "cpu"
                        # Record GPU fallback
                        try:
                            from pdf_to_markdown_mcp.core.monitoring import metrics_collector
                            metrics_collector.record_gpu_fallback("gpu_unavailable_after_warmup")
                        except ImportError:
                            pass
                except Exception as e:
                    logger.warning(f"GPU warm-up failed, falling back to CPU: {e}")
                    planned_device = "cpu"
                    config["device"] = "cpu"
                    # Record GPU fallback
                    try:
                        from pdf_to_markdown_mcp.core.monitoring import metrics_collector
                        metrics_collector.record_gpu_fallback("warmup_failed")
                    except ImportError:
                        pass

            # Read PDF bytes
            pdf_bytes = self.read_fn(pdf_path)
            pdf_file_name = pdf_path.stem

            # Run MinerU parsing with GPU optimization
            # Note: Even though MinerU uses GPU internally, we still use thread pool
            # to prevent blocking the async event loop
            loop = asyncio.get_event_loop()

            # Use a dedicated thread pool for GPU-bound operations
            from concurrent.futures import ThreadPoolExecutor
            import threading

            # Create thread pool specifically for GPU operations
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mineru-gpu") as executor:
                await loop.run_in_executor(
                    executor,
                    self._run_mineru_processing,
                    output_dir,
                    pdf_file_name,
                    pdf_bytes,
                    config,
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
            processing_stats = {}

            if middle_json_file.exists():
                import json
                middle_json = json.loads(middle_json_file.read_text(encoding="utf-8"))
                page_count = len(middle_json.get("pdf_info", []))

                # Extract GPU processing statistics if available
                processing_stats = self._extract_processing_stats(middle_json)

            # Create processing result with GPU stats
            file_hash = self._calculate_file_hash(pdf_path)
            file_size = pdf_path.stat().st_size

            metadata = ProcessingMetadata(
                file_hash=file_hash,
                file_size_bytes=file_size,
                pages=page_count,
                processing_time_ms=0,  # Will be updated by caller
                mineru_version="2.0.6",
                processing_stats=processing_stats,
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

    async def _warm_up_gpu(self) -> None:
        """
        Warm up GPU by running a small inference to initialize CUDA context.
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Small tensor operation to initialize CUDA context
                device = torch.cuda.current_device()
                test_tensor = torch.randn(100, 100, device=device)
                _ = torch.mm(test_tensor, test_tensor)
                torch.cuda.synchronize()

                self.logger.info(f"GPU warmed up on device {device}")
        except Exception as e:
            self.logger.warning(f"GPU warm-up failed: {e}")

    def _run_mineru_processing(
        self,
        output_dir: Path,
        pdf_file_name: str,
        pdf_bytes: bytes,
        config: dict[str, Any],
    ) -> None:
        """
        Run MinerU processing in a dedicated thread with GPU optimization.
        """
        import os
        import gc
        import torch

        # Ensure GPU environment is set in this thread
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["MINERU_DEVICE_MODE"] = "cuda"

        try:
            # Clear any existing GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Run MinerU processing
            self.do_parse(
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

        finally:
            # Clean up GPU memory after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def _extract_processing_stats(self, middle_json: dict) -> dict[str, Any]:
        """
        Extract processing statistics from MinerU middle JSON.

        Args:
            middle_json: MinerU processing results

        Returns:
            Processing statistics dictionary
        """
        stats = {
            "gpu_used": self._get_optimal_device() == "cuda",
            "batch_size": int(os.getenv("MINERU_MIN_BATCH_INFERENCE_SIZE", "8")),
        }

        # Extract page-level statistics if available
        pdf_info = middle_json.get("pdf_info", [])
        if pdf_info:
            stats["total_pages"] = len(pdf_info)

            # Count different content types processed
            table_count = 0
            formula_count = 0
            image_count = 0

            for page in pdf_info:
                layout_dets = page.get("layout_dets", [])
                for det in layout_dets:
                    category = det.get("category", "")
                    if category == "table":
                        table_count += 1
                    elif category == "formula":
                        formula_count += 1
                    elif category in ["figure", "image"]:
                        image_count += 1

            stats.update({
                "tables_detected": table_count,
                "formulas_detected": formula_count,
                "images_detected": image_count,
            })

        # Add GPU memory usage if available
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB

                stats.update({
                    "gpu_memory_allocated_gb": round(memory_allocated, 2),
                    "gpu_memory_reserved_gb": round(memory_reserved, 2),
                })
        except Exception:
            pass  # GPU stats not critical

        return stats

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_gpu_memory_info(self) -> dict[str, float]:
        """Get GPU memory information."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                total_memory = props.total_memory / (1024**3)  # GB
                allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB

                return {
                    "total_gb": round(total_memory, 2),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "free_gb": round(total_memory - reserved, 2),
                }
        except Exception:
            pass

        return {"total_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}

    def _check_gpu_environment(self) -> dict[str, str]:
        """Check GPU environment configuration."""
        import os
        return {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "not_set"),
            "MINERU_DEVICE_MODE": os.getenv("MINERU_DEVICE_MODE", "not_set"),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", "not_set"),
            "FLAGS_fraction_of_gpu_memory_to_use": os.getenv("FLAGS_fraction_of_gpu_memory_to_use", "not_set"),
        }

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
                f"File not found: {pdf_path}", field="pdf_path", value=str(pdf_path)
            )

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise validation_error(
                f"Path is not a file: {pdf_path}", field="pdf_path", value=str(pdf_path)
            )

        # Check file extension
        if pdf_path.suffix.lower() != ".pdf":
            raise validation_error(
                f"Invalid file type. Expected PDF, got: {pdf_path.suffix}",
                field="file_type",
                value=pdf_path.suffix,
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise validation_error(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB",
                field="file_size",
                value=file_size,
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
                        field="pdf_header",
                        value=header_chunk.decode("ascii", errors="ignore"),
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
                f"Cannot read PDF file: {e!s}", field="file_access", value=str(e)
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
            "gpu_acceleration": {
                "cuda_available": self._is_cuda_available(),
                "optimal_device": self._get_optimal_device() if self._mineru_available else "unknown",
                "optimal_batch_size": self._get_optimal_batch_size() if self._mineru_available else 1,
                "gpu_memory_info": self._get_gpu_memory_info(),
                "environment_configured": self._check_gpu_environment(),
            },
        }
