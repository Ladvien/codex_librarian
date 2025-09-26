"""
Test fixtures and utilities for PDF to Markdown MCP Server tests.

This module provides test data, factory functions, and utility classes
for creating consistent test scenarios across the test suite.
"""

from .test_data import (
    create_sample_pdf_content,
    create_sample_markdown,
    create_sample_processing_result,
    create_sample_embeddings,
    SAMPLE_PDF_METADATA,
    SAMPLE_CHUNKS,
    SAMPLE_TABLES,
    SAMPLE_FORMULAS,
    SAMPLE_IMAGES,
)

from .factories import (
    DocumentFactory,
    ProcessingResultFactory,
    EmbeddingFactory,
    ChunkFactory,
)

from .utils import (
    create_temp_pdf,
    create_test_directory,
    cleanup_test_files,
    assert_processing_result,
    assert_database_state,
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