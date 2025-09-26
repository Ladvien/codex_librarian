"""
Response models for all API endpoints.

These Pydantic models define the structure of API responses.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Types of errors that can occur."""

    VALIDATION = "validation_error"
    PROCESSING = "processing_error"
    EMBEDDING = "embedding_error"
    DATABASE = "database_error"
    SYSTEM = "system_error"
    NOT_FOUND = "not_found"
    PERMISSION = "permission_error"
    TIMEOUT = "timeout_error"


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorType = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        None, description="Additional error details"
    )
    correlation_id: str | None = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "error": "processing_error",
                "message": "Failed to extract text from PDF",
                "details": {
                    "file_path": "/path/to/file.pdf",
                    "error_code": "CORRUPT_PDF",
                },
                "correlation_id": "req_123456789",
                "timestamp": "2025-09-25T10:30:00Z",
            }
        }


class ConvertSingleResponse(BaseModel):
    """Response model for single PDF conversion."""

    success: bool = Field(..., description="Whether conversion was successful")
    document_id: int | None = Field(
        None, description="Database ID of processed document"
    )
    job_id: str | None = Field(
        None, description="Background job ID for status tracking"
    )

    message: str = Field(..., description="Status message")

    # File information
    source_path: Path = Field(..., description="Original file path")
    output_path: Path | None = Field(None, description="Output markdown file path")

    # Processing results
    processing_time_ms: int | None = Field(
        None, description="Processing time in milliseconds"
    )
    page_count: int | None = Field(None, description="Number of pages processed")
    chunk_count: int | None = Field(
        None, description="Number of text chunks created"
    )
    embedding_count: int | None = Field(
        None, description="Number of embeddings generated"
    )

    # Statistics
    file_size_bytes: int = Field(..., description="Original file size")
    has_images: bool | None = Field(
        None, description="Whether document contains images"
    )
    has_tables: bool | None = Field(
        None, description="Whether document contains tables"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "document_id": 42,
                "job_id": "pdf_proc_123456",
                "message": "PDF converted successfully",
                "source_path": "/path/to/document.pdf",
                "output_path": "/path/to/document.md",
                "processing_time_ms": 15000,
                "page_count": 25,
                "chunk_count": 150,
                "embedding_count": 150,
                "file_size_bytes": 2048000,
                "has_images": True,
                "has_tables": False,
            }
        }


class BatchConvertResponse(BaseModel):
    """Response model for batch PDF conversion."""

    success: bool = Field(
        ..., description="Whether batch operation was initiated successfully"
    )
    batch_id: str = Field(..., description="Batch operation ID")

    message: str = Field(..., description="Status message")

    # File counts
    files_found: int = Field(..., ge=0, description="Total PDF files found")
    files_queued: int = Field(..., ge=0, description="Files added to processing queue")
    files_skipped: int = Field(
        ..., ge=0, description="Files skipped (duplicates, errors)"
    )

    # Processing information
    estimated_time_minutes: int | None = Field(
        None, description="Estimated processing time"
    )
    queue_position: int | None = Field(
        None, description="Position in processing queue"
    )

    # File details
    queued_files: list[str] = Field(
        default_factory=list, description="List of files queued for processing"
    )
    skipped_files: list[dict[str, str]] = Field(
        default_factory=list, description="Files skipped with reasons"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "batch_id": "batch_987654321",
                "message": "Batch processing initiated",
                "files_found": 25,
                "files_queued": 23,
                "files_skipped": 2,
                "estimated_time_minutes": 45,
                "queue_position": 3,
                "queued_files": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
                "skipped_files": [
                    {"file": "corrupted.pdf", "reason": "File is corrupted"},
                    {"file": "duplicate.pdf", "reason": "Already processed"},
                ],
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""

    document_id: int = Field(..., description="Document database ID")
    chunk_id: int | None = Field(None, description="Chunk database ID")

    # Document information
    filename: str = Field(..., description="Original filename")
    source_path: str | None = Field(None, description="Original file path")

    # Content
    title: str | None = Field(None, description="Document title if available")
    content: str | None = Field(None, description="Matching content chunk")

    # Relevance
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Result rank (1-based)")

    # Context
    page_number: int | None = Field(
        None, description="Page number where content was found"
    )
    chunk_index: int | None = Field(None, description="Chunk index within page")

    # Metadata
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "document_id": 42,
                "chunk_id": 156,
                "filename": "machine_learning_guide.pdf",
                "source_path": "/docs/ml/machine_learning_guide.pdf",
                "title": "Introduction to Neural Networks",
                "content": "Neural networks are computational models inspired by biological neurons...",
                "similarity_score": 0.89,
                "rank": 1,
                "page_number": 15,
                "chunk_index": 3,
                "metadata": {"has_formulas": True, "word_count": 247},
            }
        }


class SearchResponse(BaseModel):
    """Response model for search operations."""

    success: bool = Field(..., description="Whether search was successful")
    query: str = Field(..., description="Original search query")

    # Results
    results: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(..., ge=0, description="Total number of results found")

    # Performance
    search_time_ms: int = Field(..., ge=0, description="Search execution time")

    # Search parameters
    top_k: int = Field(..., description="Maximum results requested")
    threshold: float | None = Field(None, description="Similarity threshold used")

    # Filters applied
    filters: dict[str, Any] | None = Field(
        None, description="Filters that were applied"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "machine learning algorithms",
                "results": [
                    {
                        "document_id": 42,
                        "filename": "ml_guide.pdf",
                        "content": "Various machine learning algorithms...",
                        "similarity_score": 0.92,
                        "rank": 1,
                    }
                ],
                "total_results": 15,
                "search_time_ms": 45,
                "top_k": 10,
                "threshold": 0.7,
            }
        }


class JobStatus(str, Enum):
    """Background job status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StatusResponse(BaseModel):
    """Response model for status queries."""

    # Job information
    job_id: str | None = Field(None, description="Specific job ID if requested")
    status: JobStatus | None = Field(None, description="Job status")

    # Progress information
    progress_percent: float | None = Field(
        None, ge=0.0, le=100.0, description="Completion percentage"
    )
    current_step: str | None = Field(None, description="Current processing step")

    # Timing
    started_at: datetime | None = Field(None, description="Job start time")
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )
    completed_at: datetime | None = Field(None, description="Job completion time")

    # Queue statistics
    queue_depth: int = Field(..., ge=0, description="Current queue depth")
    active_jobs: int = Field(..., ge=0, description="Number of active jobs")

    # System statistics
    total_documents: int = Field(..., ge=0, description="Total documents in database")
    processing_rate_per_hour: float | None = Field(
        None, description="Average processing rate"
    )

    # Error information
    error_message: str | None = Field(
        None, description="Error message if job failed"
    )
    retry_count: int | None = Field(None, description="Number of retry attempts")

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "job_id": "pdf_proc_123456",
                "status": "running",
                "progress_percent": 65.5,
                "current_step": "Generating embeddings",
                "started_at": "2025-09-25T10:00:00Z",
                "estimated_completion": "2025-09-25T10:15:00Z",
                "queue_depth": 5,
                "active_jobs": 3,
                "total_documents": 1247,
                "processing_rate_per_hour": 12.5,
            }
        }


class ConfigurationResponse(BaseModel):
    """Response model for configuration operations."""

    success: bool = Field(
        ..., description="Whether configuration update was successful"
    )
    message: str = Field(..., description="Status message")

    # Updated configuration
    watch_directories: list[str] | None = Field(
        None, description="Active watch directories"
    )
    embedding_provider: str | None = Field(
        None, description="Current embedding provider"
    )

    # Service status
    services_restarted: list[str] = Field(
        default_factory=list, description="Services that were restarted"
    )

    # Validation results
    validation_errors: list[str] = Field(
        default_factory=list, description="Configuration validation errors"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Configuration updated successfully",
                "watch_directories": ["/home/user/Documents", "/shared/pdfs"],
                "embedding_provider": "ollama",
                "services_restarted": ["file_watcher"],
                "validation_errors": [],
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""

    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")

    # Component health
    checks: dict[str, str] = Field(
        ..., description="Individual component health status"
    )

    # System information
    uptime_seconds: int | None = Field(None, description="Service uptime")
    memory_usage_mb: float | None = Field(None, description="Memory usage in MB")
    cpu_percent: float | None = Field(None, description="CPU usage percentage")

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "PDF to Markdown MCP Server",
                "version": "0.1.0",
                "checks": {
                    "database": "healthy",
                    "celery": "healthy",
                    "embeddings": "healthy",
                    "storage": "healthy",
                },
                "uptime_seconds": 3600,
                "memory_usage_mb": 256.5,
                "cpu_percent": 12.3,
            }
        }
