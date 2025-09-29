"""
Request models for all API endpoints.

These Pydantic models handle validation of incoming API requests.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class SupportedLanguage(str, Enum):
    """Supported OCR languages."""

    ENGLISH = "eng"
    CHINESE_SIMPLIFIED = "chi_sim"
    CHINESE_TRADITIONAL = "chi_tra"
    FRENCH = "fra"
    GERMAN = "deu"
    SPANISH = "spa"
    JAPANESE = "jpn"
    KOREAN = "kor"


class ProcessingOptions(BaseModel):
    """Common processing options for PDF conversion."""

    ocr_language: SupportedLanguage = Field(
        default=SupportedLanguage.ENGLISH, description="OCR language code"
    )
    preserve_layout: bool = Field(default=True, description="Preserve document layout")
    chunk_size: int = Field(
        default=1000, ge=100, le=5000, description="Text chunk size for embeddings"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks"
    )

    extract_images: bool = Field(default=True, description="Extract images from PDF")
    extract_tables: bool = Field(default=True, description="Extract tables from PDF")
    extract_formulas: bool = Field(
        default=True, description="Extract mathematical formulas"
    )
    chunk_for_embeddings: bool = Field(
        default=True, description="Generate text chunks for embeddings"
    )

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    @validator("ocr_language")
    def validate_ocr_language(cls, v):
        """Validate OCR language is supported."""
        if isinstance(v, str) and v not in [lang.value for lang in SupportedLanguage]:
            raise ValueError(
                f"Unsupported OCR language: {v}. Supported: {[lang.value for lang in SupportedLanguage]}"
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "ocr_language": "eng",
                "preserve_layout": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "extract_images": True,
                "extract_tables": True,
                "extract_formulas": True,
                "chunk_for_embeddings": True,
            }
        }


class ConvertSingleRequest(BaseModel):
    """Request model for single PDF conversion."""

    file_path: str | Path = Field(..., description="Path to PDF file")
    output_dir: str | Path | None = Field(
        None, description="Output directory for markdown files"
    )
    store_embeddings: bool = Field(
        default=True, description="Generate and store vector embeddings"
    )
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    @validator("file_path", pre=True)
    def validate_file_path(cls, v):
        """Ensure file path exists and is a PDF."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("File must be a PDF")

        # Check file size (max 500MB as per architecture)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 500MB"
            )

        # Basic PDF header validation
        try:
            with open(path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    raise ValueError("File does not appear to be a valid PDF")
        except OSError as e:
            raise ValueError(f"Cannot read file: {e!s}")

        return path

    @validator("output_dir", pre=True)
    def validate_output_dir(cls, v):
        """Ensure output directory is valid if provided."""
        if v is None:
            return None
        path = Path(v)
        if path.exists() and not path.is_dir():
            raise ValueError("Output path exists but is not a directory")
        return path

    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "output_dir": "/path/to/output",
                "store_embeddings": True,
                "options": {
                    "ocr_language": "eng",
                    "preserve_layout": True,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                },
            }
        }


class BatchConvertRequest(BaseModel):
    """Request model for batch PDF conversion."""

    directory: str | Path = Field(..., description="Directory to search for PDFs")
    pattern: str = Field(default="**/*.pdf", description="File pattern to match")
    recursive: bool = Field(
        default=True, description="Search subdirectories recursively"
    )
    output_base: str | Path | None = Field(
        None, description="Base output directory"
    )
    store_embeddings: bool = Field(
        default=True, description="Generate and store vector embeddings"
    )
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    max_files: int = Field(
        default=100, ge=1, le=1000, description="Maximum number of files to process"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Processing priority (1=highest, 10=lowest)"
    )

    @validator("directory", pre=True)
    def validate_directory(cls, v):
        """Ensure directory exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return path

    @validator("output_base", pre=True)
    def validate_output_base(cls, v):
        """Validate output base directory."""
        if v is None:
            return None
        path = Path(v)
        if path.exists() and not path.is_dir():
            raise ValueError("Output base path exists but is not a directory")
        return path

    class Config:
        schema_extra = {
            "example": {
                "directory": "/path/to/pdfs",
                "pattern": "**/*.pdf",
                "recursive": True,
                "output_base": "/path/to/output",
                "store_embeddings": True,
                "max_files": 50,
                "priority": 5,
            }
        }


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )

    filter: dict[str, Any] | None = Field(None, description="Filter criteria")
    include_content: bool = Field(
        default=True, description="Include chunk content in results"
    )

    @validator("query")
    def validate_query(cls, v):
        """Ensure query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "top_k": 10,
                "threshold": 0.7,
                "filter": {"document_id": 123},
                "include_content": True,
            }
        }


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search (semantic + keyword)."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    semantic_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for semantic search"
    )
    keyword_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for keyword search"
    )
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )

    filter: dict[str, Any] | None = Field(None, description="Filter criteria")
    include_content: bool = Field(
        default=True, description="Include chunk content in results"
    )

    @validator("query")
    def validate_query(cls, v):
        """Ensure query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("keyword_weight")
    def validate_weights_sum(cls, v, values):
        """Ensure semantic and keyword weights sum to 1.0."""
        if "semantic_weight" in values:
            total = values["semantic_weight"] + v
            if abs(total - 1.0) > 0.001:  # Allow for floating point precision
                raise ValueError("Semantic weight and keyword weight must sum to 1.0")
        return v

    class Config:
        schema_extra = {
            "example": {
                "query": "neural network training",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "top_k": 15,
                "include_content": True,
            }
        }


class FindSimilarRequest(BaseModel):
    """Request model for finding similar documents."""

    document_id: int = Field(..., description="Reference document ID")
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of similar documents to return"
    )
    min_similarity: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )

    include_self: bool = Field(
        default=False, description="Include the reference document in results"
    )

    class Config:
        schema_extra = {
            "example": {
                "document_id": 42,
                "top_k": 5,
                "min_similarity": 0.6,
                "include_self": False,
            }
        }


class ConfigurationRequest(BaseModel):
    """Request model for server configuration updates."""

    watch_directories: list[str | Path] | None = Field(
        None, description="Directories to monitor"
    )
    output_directory: str | Path | None = Field(
        None, description="Output directory for processed files"
    )
    embedding_config: dict[str, Any] | None = Field(
        None, description="Embedding service configuration"
    )
    ocr_settings: dict[str, Any] | None = Field(
        None, description="OCR processing settings"
    )
    processing_limits: dict[str, Any] | None = Field(
        None, description="Processing resource limits"
    )

    restart_watcher: bool = Field(
        default=False, description="Restart file watcher after changes"
    )

    @validator("watch_directories", pre=True)
    def validate_watch_directories(cls, v):
        """Validate watch directories exist."""
        if v is None:
            return None

        validated = []
        for dir_path in v:
            path = Path(dir_path)
            if not path.exists():
                raise ValueError(f"Watch directory does not exist: {dir_path}")
            if not path.is_dir():
                raise ValueError(f"Watch path is not a directory: {dir_path}")
            validated.append(path)

        return validated

    class Config:
        schema_extra = {
            "example": {
                "watch_directories": ["/home/user/Documents", "/shared/pdfs"],
                "embedding_config": {
                    "provider": "ollama",
                    "model": "nomic-embed-text",
                    "batch_size": 32,
                },
                "ocr_settings": {"language": "eng+fra", "dpi": 300},
                "restart_watcher": True,
            }
        }
