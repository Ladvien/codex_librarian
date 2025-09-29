#!/usr/bin/env python3
"""
Standalone MinerU GPU Processing Service.

This service runs independently of Celery to avoid CUDA initialization issues.
It monitors a Redis queue for PDF processing jobs and returns results via Redis.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as redis
from pydantic import BaseModel

# Set up GPU environment before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MINERU_DEVICE_MODE"] = "cuda"
# Force OCR to use GPU
os.environ["USE_GPU"] = "True"
os.environ["PADDLE_USE_GPU"] = "True"
os.environ["FLAGS_use_gpu"] = "True"
# Set device for PaddleOCR
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# Ensure PyTorch uses CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
# Force PyTorch to use CUDA
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingRequest(BaseModel):
    """Request model for PDF processing."""
    request_id: str
    pdf_path: str
    options: Dict[str, Any] = {}
    callback_queue: str = "mineru_results"


class ProcessingResult(BaseModel):
    """Result model for PDF processing."""
    request_id: str
    success: bool
    markdown: Optional[str] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


class MinerUStandaloneService:
    """Standalone MinerU processing service with GPU support."""

    def __init__(self, redis_url: str = None):
        # Get redis URL from settings if not provided
        if redis_url is None:
            # Try to get from environment or use default
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_db = os.getenv("REDIS_DB", "0")
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.mineru_initialized = False
        self.running = True
        self.do_parse = None
        self.read_fn = None

        # Queue names
        self.request_queue = "mineru_requests"
        self.result_queue = "mineru_results"

    async def initialize(self):
        """Initialize Redis connection and MinerU."""
        logger.info("Initializing MinerU Standalone Service...")

        # Connect to Redis
        self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
        await self.redis_client.ping()
        logger.info("Connected to Redis")

        # Check GPU availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(
                f"GPU available: {device_count} device(s), "
                f"using {device_name} with {memory_gb:.1f}GB memory"
            )
        else:
            logger.warning("No GPU available, will use CPU (slower)")

        # Initialize MinerU
        try:
            # Import from the global MinerU package, not our local mineru.py
            import sys
            # Temporarily remove our services directory from path to avoid conflict
            services_path = str(Path(__file__).parent)
            if services_path in sys.path:
                sys.path.remove(services_path)

            from mineru.cli.common import do_parse, read_fn

            # Restore path
            if services_path not in sys.path:
                sys.path.append(services_path)
            self.do_parse = do_parse
            self.read_fn = read_fn
            self.mineru_initialized = True
            logger.info("MinerU initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MinerU: {e}")
            raise

    async def process_pdf(self, request: ProcessingRequest) -> ProcessingResult:
        """Process a single PDF file."""
        logger.info(f"Processing PDF: {request.pdf_path} (request: {request.request_id})")

        try:
            # Read PDF file
            pdf_path = Path(request.pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            # Create temporary output directory
            with tempfile.TemporaryDirectory(prefix="mineru_") as temp_dir:
                output_dir = Path(temp_dir) / "output"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Prepare MinerU configuration
                config = {
                    "backend": "pipeline",
                    "parse_method": "auto",
                    "p_formula_enable": request.options.get("extract_formulas", True),
                    "p_table_enable": request.options.get("extract_tables", True),
                    "p_lang": request.options.get("language", "en"),
                    "f_dump_md": True,
                    "f_dump_middle_json": True,
                    "f_dump_content_list": False,
                    "f_dump_model_output": False,
                    "f_dump_orig_pdf": False,
                }

                # Process with MinerU
                logger.info(f"Starting MinerU processing with config: {config}")

                # Run MinerU processing (blocking call)
                pdf_bytes = self.read_fn(str(pdf_path))

                # do_parse expects lists as arguments
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.do_parse,
                    str(output_dir),  # output directory
                    [pdf_path.name],  # list of PDF file names
                    [pdf_bytes],  # list of PDF bytes
                    [config.get("p_lang", "en")],  # list of languages
                    config.get("backend", "pipeline"),
                    config.get("parse_method", "auto"),
                    config.get("p_formula_enable", True),
                    config.get("p_table_enable", True),
                    None,  # server_url
                    False,  # f_draw_layout_bbox
                    False,  # f_draw_span_bbox
                    config.get("f_dump_md", True),
                    config.get("f_dump_middle_json", True),
                    config.get("f_dump_model_output", False),
                    config.get("f_dump_orig_pdf", False),
                    config.get("f_dump_content_list", False)
                )

                # Processing is complete, no return value from do_parse
                result = {"page_count": 0, "processing_time": 0}

                # Read the generated markdown
                # MinerU creates files in different possible locations
                parse_method = config.get("parse_method", "auto")

                # Try multiple possible paths where MinerU might save the file
                possible_paths = [
                    output_dir / pdf_path.name / parse_method / f"{pdf_path.stem}.md",
                    output_dir / pdf_path.stem / parse_method / f"{pdf_path.stem}.md",
                    output_dir / parse_method / f"{pdf_path.stem}.md",
                    # Also check for any .md files in the output directory
                ]

                md_path = None
                for path in possible_paths:
                    logger.info(f"Checking for markdown at: {path}")
                    if path.exists():
                        md_path = path
                        break

                # If still not found, search for any .md file in output_dir
                if not md_path:
                    logger.info(f"Searching for any .md files in {output_dir}")
                    md_files = list(output_dir.rglob("*.md"))
                    if md_files:
                        md_path = md_files[0]
                        logger.info(f"Found markdown file at: {md_path}")

                if md_path and md_path.exists():
                    markdown = md_path.read_text(encoding='utf-8')

                    # Save to actual output directory
                    output_base_dir = Path("/mnt/codex_fs/research/librarian_output")
                    output_base_dir.mkdir(parents=True, exist_ok=True)

                    # Create output filename based on PDF name
                    output_filename = pdf_path.stem + ".md"
                    output_path = output_base_dir / output_filename

                    # Save markdown file
                    output_path.write_text(markdown, encoding='utf-8')
                    logger.info(f"Saved markdown to: {output_path}")

                    return ProcessingResult(
                        request_id=request.request_id,
                        success=True,
                        markdown=markdown,
                        metadata={
                            "pages": result.get("page_count", 0),
                            "processing_time": result.get("processing_time", 0),
                            "output_path": str(output_path),
                        }
                    )
                else:
                    # Log directory structure for debugging
                    logger.error(f"MinerU output directory structure:")
                    for p in output_dir.rglob("*"):
                        logger.error(f"  {p}")
                    raise FileNotFoundError(f"MinerU did not generate markdown output. Checked paths: {possible_paths}")

        except Exception as e:
            logger.error(f"Error processing PDF: {e}\n{traceback.format_exc()}")
            return ProcessingResult(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )

    async def process_queue(self):
        """Main processing loop - monitors Redis queue for jobs."""
        logger.info(f"Starting queue processor, monitoring: {self.request_queue}")

        while self.running:
            try:
                # Block waiting for job (timeout after 1 second to check running status)
                result = await self.redis_client.blpop(self.request_queue, timeout=1)

                if result:
                    _, job_data = result

                    # Parse request
                    request_dict = json.loads(job_data)
                    request = ProcessingRequest(**request_dict)

                    logger.info(f"Received job: {request.request_id}")

                    # Process PDF
                    result = await self.process_pdf(request)

                    # Send result back via Redis
                    result_data = result.model_dump_json().encode('utf-8')
                    await self.redis_client.lpush(request.callback_queue, result_data)

                    logger.info(f"Completed job: {request.request_id}, success: {result.success}")

            except Exception as e:
                logger.error(f"Error in processing loop: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1)  # Brief pause on error

    async def shutdown(self):
        """Clean shutdown of service."""
        logger.info("Shutting down MinerU service...")
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Shutdown complete")

    async def run(self):
        """Main service entry point."""
        try:
            await self.initialize()
            await self.process_queue()
        finally:
            await self.shutdown()


def handle_signal(sig, frame):
    """Signal handler for graceful shutdown."""
    logger.info(f"Received signal {sig}, shutting down...")
    asyncio.create_task(service.shutdown())


async def main():
    """Main entry point."""
    global service

    # Get Redis URL from environment or use settings
    redis_url = os.getenv("REDIS_URL", None)

    # Create and run service (will use settings if redis_url is None)
    service = MinerUStandaloneService(redis_url)

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run service
    await service.run()


if __name__ == "__main__":
    # Set up GPU environment
    logger.info("Starting MinerU Standalone GPU Service")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"MINERU_DEVICE_MODE: {os.getenv('MINERU_DEVICE_MODE')}")

    # Run the service
    asyncio.run(main())