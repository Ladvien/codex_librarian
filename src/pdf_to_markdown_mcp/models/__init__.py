"""
Pydantic models for PDF to Markdown MCP Server.

This package contains all data validation models used throughout the application.
"""

from pdf_to_markdown_mcp.models.document import (
    Document,
    DocumentContent,
    DocumentEmbedding,
    DocumentImage,
    ProcessingStatus,
)
from pdf_to_markdown_mcp.models.dto import (
    CreateDocumentDTO,
    DocumentDTO,
    DocumentListDTO,
    DocumentSearchResultDTO,
    ProcessingStatusType,
    UpdateDocumentDTO,
)
from pdf_to_markdown_mcp.models.processing import (
    ChunkData,
    FormulaData,
    ImageData,
    ProcessingMetadata,
    ProcessingResult,
    TableData,
)
from pdf_to_markdown_mcp.models.request import (
    BatchConvertRequest,
    ConfigurationRequest,
    ConvertSingleRequest,
    FindSimilarRequest,
    HybridSearchRequest,
    ProcessingOptions,
    SemanticSearchRequest,
)
from pdf_to_markdown_mcp.models.response import (
    BatchConvertResponse,
    ConfigurationResponse,
    ConvertSingleResponse,
    ErrorResponse,
    HealthResponse,
    SearchResponse,
    SearchResult,
    StatusResponse,
)

__all__ = [
    # Document models
    "Document",
    "DocumentContent",
    "DocumentEmbedding",
    "DocumentImage",
    "ProcessingStatus",
    # Request models
    "ConvertSingleRequest",
    "BatchConvertRequest",
    "SemanticSearchRequest",
    "HybridSearchRequest",
    "FindSimilarRequest",
    "ConfigurationRequest",
    "ProcessingOptions",
    # Response models
    "ConvertSingleResponse",
    "BatchConvertResponse",
    "SearchResponse",
    "SearchResult",
    "StatusResponse",
    "ConfigurationResponse",
    "ErrorResponse",
    "HealthResponse",
    # Processing models
    "TableData",
    "FormulaData",
    "ImageData",
    "ChunkData",
    "ProcessingMetadata",
    "ProcessingResult",
    # DTO models
    "ProcessingStatusType",
    "DocumentDTO",
    "CreateDocumentDTO",
    "UpdateDocumentDTO",
    "DocumentSearchResultDTO",
    "DocumentListDTO",
]
