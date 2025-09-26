"""
Processing result models for MinerU PDF processing.

These models represent the output of PDF processing operations.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class TableData(BaseModel):
    """Extracted table data structure."""

    page: int = Field(..., ge=1, description="Page number where table was found")
    table_index: int = Field(..., ge=0, description="Table index on the page")
    headers: List[str] = Field(..., description="Column headers")
    rows: List[List[str]] = Field(..., description="Table row data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    bbox: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2]")


class FormulaData(BaseModel):
    """Extracted formula data structure."""

    page: int = Field(..., ge=1, description="Page number where formula was found")
    formula_index: int = Field(..., ge=0, description="Formula index on the page")
    latex: str = Field(..., description="LaTeX representation of formula")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    bbox: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    is_inline: bool = Field(default=False, description="Whether formula is inline or block")


class ImageData(BaseModel):
    """Extracted image data structure."""

    page: int = Field(..., ge=1, description="Page number where image was found")
    image_index: int = Field(..., ge=0, description="Image index on the page")
    path: str = Field(..., description="Path to extracted image file")
    ocr_text: Optional[str] = Field(None, description="OCR text from image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence")
    bbox: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    format: Optional[str] = Field(None, description="Image format (png, jpg, etc.)")


class ChunkData(BaseModel):
    """Text chunk data for embeddings."""

    chunk_index: int = Field(..., ge=0, description="Chunk index in document")
    text: str = Field(..., min_length=1, description="Chunk text content")
    start_char: int = Field(..., ge=0, description="Start character position in document")
    end_char: int = Field(..., gt=0, description="End character position in document")
    page: int = Field(..., ge=1, description="Primary page number for chunk")
    token_count: Optional[int] = Field(None, ge=0, description="Approximate token count")

    def __init__(self, **data):
        super().__init__(**data)
        # Validate that end_char > start_char
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")


class ProcessingMetadata(BaseModel):
    """Metadata from PDF processing operation."""

    pages: int = Field(..., ge=1, description="Total number of pages")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average OCR confidence")

    # File information
    file_size_bytes: Optional[int] = Field(None, ge=0, description="Original file size")
    file_hash: Optional[str] = Field(None, description="SHA-256 hash of file")

    # Processing statistics
    tables_found: Optional[int] = Field(None, ge=0, description="Number of tables extracted")
    formulas_found: Optional[int] = Field(None, ge=0, description="Number of formulas extracted")
    images_found: Optional[int] = Field(None, ge=0, description="Number of images extracted")
    chunks_created: Optional[int] = Field(None, ge=0, description="Number of text chunks created")

    # Quality metrics
    text_extraction_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text extraction quality score")
    layout_preservation_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Layout preservation quality")

    # Additional metadata
    language_detected: Optional[str] = Field(None, description="Primary language detected")
    mineru_version: Optional[str] = Field(None, description="MinerU library version used")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="Processing options used")


class ProcessingResult(BaseModel):
    """Complete result of PDF processing with MinerU."""

    markdown_content: str = Field(..., description="Full document in Markdown format")
    plain_text: str = Field(..., description="Plain text content for search")

    extracted_tables: List[TableData] = Field(default_factory=list, description="Extracted table data")
    extracted_formulas: List[FormulaData] = Field(default_factory=list, description="Extracted formula data")
    extracted_images: List[ImageData] = Field(default_factory=list, description="Extracted image data")
    chunk_data: List[ChunkData] = Field(default_factory=list, description="Text chunks for embeddings")

    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")

    def get_content_summary(self) -> Dict[str, Any]:
        """Get a summary of the processed content."""
        return {
            "total_pages": self.processing_metadata.pages,
            "text_length": len(self.plain_text),
            "markdown_length": len(self.markdown_content),
            "tables_count": len(self.extracted_tables),
            "formulas_count": len(self.extracted_formulas),
            "images_count": len(self.extracted_images),
            "chunks_count": len(self.chunk_data),
            "processing_time_ms": self.processing_metadata.processing_time_ms,
            "avg_ocr_confidence": self.processing_metadata.ocr_confidence,
        }

    def has_structured_content(self) -> bool:
        """Check if document contains structured content (tables, formulas, images)."""
        return (
            len(self.extracted_tables) > 0 or
            len(self.extracted_formulas) > 0 or
            len(self.extracted_images) > 0
        )

    def get_page_content(self, page_number: int) -> Dict[str, Any]:
        """Get content for a specific page."""
        page_tables = [t for t in self.extracted_tables if t.page == page_number]
        page_formulas = [f for f in self.extracted_formulas if f.page == page_number]
        page_images = [i for i in self.extracted_images if i.page == page_number]
        page_chunks = [c for c in self.chunk_data if c.page == page_number]

        return {
            "page": page_number,
            "tables": page_tables,
            "formulas": page_formulas,
            "images": page_images,
            "chunks": page_chunks,
        }