"""
MinerU Client for Celery Workers.

This client communicates with the standalone MinerU service via Redis queues,
allowing Celery workers to offload GPU-intensive PDF processing.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MinerUClient:
    """Client for communicating with standalone MinerU service."""

    def __init__(self, redis_url: str = None, timeout: int = 300):
        """
        Initialize MinerU client.

        Args:
            redis_url: Redis connection URL (uses config if not provided)
            timeout: Maximum time to wait for processing (seconds)
        """
        # Get redis URL from settings if not provided
        if redis_url is None:
            from ..config import settings
            redis_url = settings.redis.url
        self.redis_url = redis_url
        self.timeout = timeout
        self.redis_client = redis.from_url(redis_url, decode_responses=False)

        # Queue names (must match standalone service)
        self.request_queue = "mineru_requests"
        self.result_queue = "mineru_results"

    def process_pdf_sync(
        self,
        pdf_path: str,
        extract_formulas: bool = True,
        extract_tables: bool = True,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process a PDF file using the standalone MinerU service (synchronous).

        Args:
            pdf_path: Path to the PDF file
            extract_formulas: Whether to extract formulas
            extract_tables: Whether to extract tables
            language: OCR language

        Returns:
            Processing result dictionary
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Create request
        request = {
            "request_id": request_id,
            "pdf_path": str(pdf_path),
            "options": {
                "extract_formulas": extract_formulas,
                "extract_tables": extract_tables,
                "language": language,
            },
            "callback_queue": self.result_queue
        }

        try:
            # Send request to MinerU service
            logger.info(f"Sending PDF to MinerU service: {pdf_path} (request: {request_id})")
            request_data = json.dumps(request)
            self.redis_client.lpush(self.request_queue, request_data.encode('utf-8'))

            # Wait for result
            logger.info(f"Waiting for MinerU result (timeout: {self.timeout}s)")
            result = self.redis_client.blpop(self.result_queue, timeout=self.timeout)

            if result:
                _, result_data = result
                result_dict = json.loads(result_data)

                # Verify it's our result
                if result_dict.get("request_id") == request_id:
                    if result_dict.get("success"):
                        logger.info(f"MinerU processing successful for request {request_id}")
                        return {
                            "success": True,
                            "markdown": result_dict.get("markdown"),
                            "metadata": result_dict.get("metadata", {})
                        }
                    else:
                        error = result_dict.get("error", "Unknown error")
                        logger.error(f"MinerU processing failed: {error}")
                        return {
                            "success": False,
                            "error": error
                        }
                else:
                    logger.warning(f"Received result for different request: {result_dict.get('request_id')}")
                    # Put it back at the end of queue for the correct client (avoid infinite loop)
                    self.redis_client.rpush(self.result_queue, result_data)
                    return {
                        "success": False,
                        "error": "Request ID mismatch"
                    }
            else:
                logger.error(f"Timeout waiting for MinerU result (waited {self.timeout}s)")
                return {
                    "success": False,
                    "error": f"Processing timeout after {self.timeout} seconds"
                }

        except Exception as e:
            logger.error(f"Error communicating with MinerU service: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def process_pdf_async(
        self,
        pdf_path: str,
        extract_formulas: bool = True,
        extract_tables: bool = True,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process a PDF file using the standalone MinerU service (asynchronous).

        Args:
            pdf_path: Path to the PDF file
            extract_formulas: Whether to extract formulas
            extract_tables: Whether to extract tables
            language: OCR language

        Returns:
            Processing result dictionary
        """
        # For async version, we'd use redis.asyncio
        # For now, run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_pdf_sync,
            pdf_path,
            extract_formulas,
            extract_tables,
            language
        )

    def check_service_health(self) -> bool:
        """
        Check if the MinerU service is responsive.

        Returns:
            True if service is healthy
        """
        try:
            # Check Redis connection
            self.redis_client.ping()

            # Check if service is processing by looking at queue lengths
            request_queue_len = self.redis_client.llen(self.request_queue)
            result_queue_len = self.redis_client.llen(self.result_queue)

            logger.info(
                f"MinerU service queues - Requests: {request_queue_len}, "
                f"Results: {result_queue_len}"
            )
            return True

        except Exception as e:
            logger.error(f"MinerU service health check failed: {e}")
            return False


# Singleton client instance
_mineru_client: Optional[MinerUClient] = None


def get_mineru_client() -> MinerUClient:
    """Get or create singleton MinerU client."""
    global _mineru_client
    if _mineru_client is None:
        _mineru_client = MinerUClient()
    return _mineru_client