"""
Test fixtures and utilities for PDF to Markdown MCP Server tests.

This module provides test data, factory functions, and utility classes
for creating consistent test scenarios across the test suite.
"""

from .factories import (
    ChunkFactory,
    DocumentFactory,
    EmbeddingFactory,
    ProcessingResultFactory,
)
from .test_data import (
    SAMPLE_CHUNKS,
    SAMPLE_FORMULAS,
    SAMPLE_IMAGES,
    SAMPLE_PDF_METADATA,
    SAMPLE_TABLES,
    create_sample_embeddings,
    create_sample_markdown,
    create_sample_pdf_content,
    create_sample_processing_result,
)
from .utils import (
    assert_database_state,
    assert_processing_result,
    cleanup_test_files,
    create_temp_pdf,
    create_test_directory,
)

__all__ = [
    # Test data
    "create_sample_pdf_content",
    "create_sample_markdown",
    "create_sample_processing_result",
    "create_sample_embeddings",
    "SAMPLE_PDF_METADATA",
    "SAMPLE_CHUNKS",
    "SAMPLE_TABLES",
    "SAMPLE_FORMULAS",
    "SAMPLE_IMAGES",
    # Factories
    "DocumentFactory",
    "ProcessingResultFactory",
    "EmbeddingFactory",
    "ChunkFactory",
    # Utilities
    "create_temp_pdf",
    "create_test_directory",
    "cleanup_test_files",
    "assert_processing_result",
    "assert_database_state",
]
