"""
Celery worker package for PDF to Markdown MCP Server.

This package contains the Celery application configuration and task definitions
for background processing of PDF documents.
"""

from .celery import app as celery_app
from .tasks import (
    process_pdf_document,
    generate_embeddings,
    cleanup_temp_files,
    health_check,
)

__all__ = [
    "celery_app",
    "process_pdf_document",
    "generate_embeddings",
    "cleanup_temp_files",
    "health_check",
]