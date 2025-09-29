"""
Celery worker package for PDF to Markdown MCP Server.

This package contains the Celery application configuration and task definitions
for background processing of PDF documents.
"""

from .celery import app as celery_app
from .tasks import (
    cleanup_temp_files,
    generate_embeddings,
    health_check,
    process_pdf_document,
)

__all__ = [
    "celery_app",
    "cleanup_temp_files",
    "generate_embeddings",
    "health_check",
    "process_pdf_document",
]
