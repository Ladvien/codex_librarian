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
from pdf_to_markdown_mcp.models.request import (
    ConvertSingleRequest,
    BatchConvertRequest,
    SemanticSearchRequest,
    HybridSearchRequest,
    FindSimilarRequest,
    ConfigurationRequest,
    ProcessingOptions,
)
from pdf_to_markdown_mcp.models.response import (
    ConvertSingleResponse,
    BatchConvertResponse,
    SearchResponse,
    SearchResult,
    StatusResponse,
    ConfigurationResponse,
    ErrorResponse,
    HealthResponse,
)
from pdf_to_markdown_mcp.models.processing import (
    TableData,
    FormulaData,
    ImageData,
    ChunkData,
    ProcessingMetadata,
    ProcessingResult,
)
from pdf_to_markdown_mcp.models.dto import (
    ProcessingStatusType,
    DocumentDTO,
    CreateDocumentDTO,
    UpdateDocumentDTO,
    DocumentSearchResultDTO,
    DocumentListDTO,
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
