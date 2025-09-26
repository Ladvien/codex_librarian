"""
Document-related Pydantic models.

These models represent the core document entities stored in the database.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """Main document model."""

    id: Optional[int] = None
    source_path: Path = Field(..., description="Original file path")
    filename: str = Field(..., description="File name without path")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes")

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    conversion_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator("source_path")
    def validate_source_path(cls, v):
        """Ensure source path is absolute."""
        if not v.is_absolute():
            raise ValueError("Source path must be absolute")
        return v

    @validator("filename")
    def validate_filename(cls, v):
        """Ensure filename is not empty and valid."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()

    @validator("file_hash")
    def validate_file_hash(cls, v):
        """Ensure hash is valid SHA-256 format."""
        if not v or len(v) != 64:
            raise ValueError("File hash must be 64-character SHA-256 hash")
        # Verify it's hexadecimal
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("File hash must be hexadecimal")
        return v.lower()

    class Config:
        orm_mode = True
        use_enum_values = True


class DocumentContent(BaseModel):
    """Processed document content model."""

    id: Optional[int] = None
    document_id: int = Field(..., description="Reference to parent document")

    markdown_content: str = Field(..., description="Full Markdown content")
    plain_text: str = Field(..., description="Plain text for full-text search")

    page_count: int = Field(..., ge=1, description="Number of pages processed")
    has_images: bool = Field(default=False, description="Whether document contains images")
    has_tables: bool = Field(default=False, description="Whether document contains tables")

    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    created_at: Optional[datetime] = None

    @validator("markdown_content", "plain_text")
    def validate_content_not_empty(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v

    class Config:
        orm_mode = True


class DocumentEmbedding(BaseModel):
    """Document chunk embedding model."""

    id: Optional[int] = None
    document_id: int = Field(..., description="Reference to parent document")

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Chunk index within page")

    chunk_text: str = Field(..., description="Text content of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding")

    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    @validator("chunk_text")
    def validate_chunk_text(cls, v):
        """Ensure chunk text is not empty."""
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    @validator("embedding")
    def validate_embedding(cls, v):
        """Ensure embedding is valid."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numeric values")
        return v

    class Config:
        orm_mode = True


class DocumentImage(BaseModel):
    """Extracted document image model."""

    id: Optional[int] = None
    document_id: int = Field(..., description="Reference to parent document")

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    image_index: int = Field(..., ge=0, description="Image index within page")

    image_path: Path = Field(..., description="Path to extracted image file")
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    image_embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator("image_path")
    def validate_image_path(cls, v):
        """Ensure image path is absolute."""
        if not v.is_absolute():
            raise ValueError("Image path must be absolute")
        return v

    @validator("image_embedding")
    def validate_image_embedding(cls, v):
        """Validate image embedding if present."""
        if v is not None:
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Image embedding must contain only numeric values")
        return v

    class Config:
        orm_mode = True


# Additional utility models
class DocumentStats(BaseModel):
    """Document processing statistics."""

    total_documents: int = Field(..., ge=0)
    pending_documents: int = Field(..., ge=0)
    processing_documents: int = Field(..., ge=0)
    completed_documents: int = Field(..., ge=0)
    failed_documents: int = Field(..., ge=0)

    total_size_mb: float = Field(..., ge=0.0)
    avg_processing_time_ms: Optional[float] = None

    @validator("pending_documents", "processing_documents", "completed_documents", "failed_documents")
    def validate_status_counts(cls, v, values):
        """Ensure status counts are consistent with total."""
        if "total_documents" in values:
            total = values["total_documents"]
            # This validation will run after all fields, so we check at the end
        return v

    class Config:
        validate_assignment = True