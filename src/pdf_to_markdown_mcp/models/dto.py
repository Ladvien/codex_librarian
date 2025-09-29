"""
Data Transfer Objects (DTOs) for API-Service layer communication.

These Pydantic models provide proper abstraction between API and Service layers,
preventing direct coupling to database models and maintaining clean architecture.

Following architecture principles from ARCHITECTURE.md:
- API → Services → Database (not API → Database)
- Type safety with Pydantic v2
- Immutable data structures where appropriate
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class ProcessingStatusType(str, Enum):
    """Processing status enumeration for documents."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentDTO(BaseModel):
    """
    Document Data Transfer Object for API-Service communication.

    Provides clean abstraction from database Document model,
    containing only the data needed by the API layer.
    """

    id: int | None = None
    filename: str = Field(..., min_length=1, max_length=255)
    file_path: str = Field(..., min_length=1)
    file_hash: str = Field(..., min_length=1)
    size_bytes: int = Field(..., gt=0)
    processing_status: ProcessingStatusType
    created_at: datetime | None = None
    updated_at: datetime | None = None
    processed_at: datetime | None = None

    # Processing metadata
    page_count: int | None = Field(None, ge=0)
    processing_time_seconds: float | None = Field(None, ge=0)
    error_message: str | None = None

    # Optional metadata
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    @validator("file_path")
    def validate_file_path(cls, v):
        """Validate file path format."""
        try:
            Path(v)
            return v
        except Exception:
            raise ValueError("Invalid file path format")

    @validator("file_hash")
    def validate_file_hash(cls, v):
        """Validate file hash format (SHA-256)."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("file_hash must be a valid SHA-256 hex string")
        return v.lower()

    class Config:
        """Pydantic configuration."""

        frozen = True  # Immutable DTO
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class CreateDocumentDTO(BaseModel):
    """
    DTO for document creation requests.

    Contains only the fields needed to create a new document record.
    """

    filename: str = Field(..., min_length=1, max_length=255)
    file_path: str = Field(..., min_length=1)
    file_hash: str = Field(..., min_length=1)
    size_bytes: int = Field(..., gt=0)

    @validator("file_path")
    def validate_file_path(cls, v):
        """Validate file path format."""
        try:
            path = Path(v)
            if not path.is_absolute():
                raise ValueError("File path must be absolute")
            return str(path)
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")

    @validator("file_hash")
    def validate_file_hash(cls, v):
        """Validate file hash format (SHA-256)."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("file_hash must be a valid SHA-256 hex string")
        return v.lower()


class UpdateDocumentDTO(BaseModel):
    """
    DTO for document update requests.

    Contains fields that can be updated after document creation.
    """

    processing_status: ProcessingStatusType | None = None
    page_count: int | None = Field(None, ge=0)
    processing_time_seconds: float | None = Field(None, ge=0)
    error_message: str | None = None
    processed_at: datetime | None = None
    metadata: dict[str, Any] | None = None


class DocumentSearchResultDTO(BaseModel):
    """
    DTO for document search results.

    Lightweight representation for search operations.
    """

    id: int
    filename: str
    file_path: str
    size_bytes: int
    processing_status: ProcessingStatusType
    created_at: datetime
    page_count: int | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class DocumentListDTO(BaseModel):
    """
    DTO for paginated document listings.

    Provides efficient document list representation.
    """

    documents: list[DocumentSearchResultDTO]
    total_count: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=100)
    total_pages: int = Field(..., ge=0)

    @validator("total_pages", always=True)
    def calculate_total_pages(cls, v, values):
        """Calculate total pages from total count and page size."""
        total_count = values.get("total_count", 0)
        page_size = values.get("page_size", 1)
        if page_size <= 0:
            return 0
        return (total_count + page_size - 1) // page_size


# Export all DTOs
__all__ = [
    "CreateDocumentDTO",
    "DocumentDTO",
    "DocumentListDTO",
    "DocumentSearchResultDTO",
    "ProcessingStatusType",
    "UpdateDocumentDTO",
]
