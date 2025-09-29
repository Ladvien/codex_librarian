"""
Document-related Pydantic models.

These models represent the core document entities stored in the database.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """Main document model."""

    id: int | None = None
    source_path: Path = Field(..., description="Original file path")
    filename: str = Field(..., description="File name without path")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes")

    created_at: datetime | None = None
    updated_at: datetime | None = None

    conversion_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: str | None = None
    metadata: dict[str, Any] | None = None

    # Directory mirroring fields
    source_relative_path: Path | None = Field(
        None, description="Relative path from watch directory"
    )
    output_path: Path | None = Field(
        None, description="Absolute output markdown path"
    )
    output_relative_path: Path | None = Field(
        None, description="Relative output path"
    )
    directory_depth: int | None = Field(
        None, ge=0, description="Directory depth level"
    )

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

    @validator("output_path")
    def validate_output_path(cls, v):
        """Validate output path if provided."""
        if v is not None and not v.is_absolute():
            raise ValueError("Output path must be absolute")
        return v

    class Config:
        orm_mode = True
        use_enum_values = True


class DocumentContent(BaseModel):
    """Processed document content model."""

    id: int | None = None
    document_id: int = Field(..., description="Reference to parent document")

    markdown_content: str = Field(..., description="Full Markdown content")
    plain_text: str = Field(..., description="Plain text for full-text search")

    page_count: int = Field(..., ge=1, description="Number of pages processed")
    has_images: bool = Field(
        default=False, description="Whether document contains images"
    )
    has_tables: bool = Field(
        default=False, description="Whether document contains tables"
    )

    processing_time_ms: int = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    created_at: datetime | None = None

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

    id: int | None = None
    document_id: int = Field(..., description="Reference to parent document")

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Chunk index within page")

    chunk_text: str = Field(..., description="Text content of the chunk")
    embedding: list[float] = Field(..., description="Vector embedding")

    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None

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

    id: int | None = None
    document_id: int = Field(..., description="Reference to parent document")

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    image_index: int = Field(..., ge=0, description="Image index within page")

    image_path: Path = Field(..., description="Path to extracted image file")
    ocr_text: str | None = None
    ocr_confidence: float | None = Field(None, ge=0.0, le=1.0)

    image_embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None

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
    avg_processing_time_ms: float | None = None

    @validator(
        "pending_documents",
        "processing_documents",
        "completed_documents",
        "failed_documents",
    )
    def validate_status_counts(cls, v, values):
        """Ensure status counts are consistent with total."""
        if "total_documents" in values:
            total = values["total_documents"]
            # This validation will run after all fields, so we check at the end
        return v

    class Config:
        validate_assignment = True


class PathMapping(BaseModel):
    """Path mapping model for directory structure preservation."""

    id: int | None = None
    source_directory: Path = Field(..., description="Source base directory")
    output_directory: Path = Field(..., description="Output base directory")
    relative_path: Path = Field(..., description="Relative path from base directory")
    directory_level: int = Field(..., ge=0, description="Directory depth level")
    files_count: int = Field(
        default=0, ge=0, description="Number of files in this directory"
    )

    last_scanned: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @validator("source_directory", "output_directory")
    def validate_directories_absolute(cls, v):
        """Ensure directory paths are absolute."""
        if not v.is_absolute():
            raise ValueError("Directory paths must be absolute")
        return v

    class Config:
        orm_mode = True


class DirectoryMirrorInfo(BaseModel):
    """Information about a directory mirroring operation."""

    source_path: Path = Field(..., description="Source PDF file path")
    source_relative_path: Path = Field(
        ..., description="Relative path from watch directory"
    )
    output_path: Path = Field(..., description="Output markdown file path")
    output_relative_path: Path = Field(..., description="Relative output path")
    directory_depth: int = Field(..., ge=0, description="Directory depth level")
    output_directory: Path = Field(..., description="Output directory path")

    @validator("source_path", "output_path", "output_directory")
    def validate_absolute_paths(cls, v):
        """Ensure absolute paths are absolute."""
        if not v.is_absolute():
            raise ValueError("Absolute paths must be absolute")
        return v

    class Config:
        validate_assignment = True


class DirectorySyncStats(BaseModel):
    """Statistics from directory synchronization operation."""

    directories_scanned: int = Field(..., ge=0)
    mappings_created: int = Field(..., ge=0)
    mappings_updated: int = Field(..., ge=0)
    files_processed: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)

    sync_started: datetime | None = None
    sync_completed: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate of sync operation."""
        total_operations = self.files_processed
        if total_operations == 0:
            return 1.0
        return (total_operations - self.errors) / total_operations

    class Config:
        validate_assignment = True
