"""
SQLAlchemy models for PDF to Markdown MCP Server.

This module defines the complete database schema including:
- Document metadata and processing status
- Converted content storage
- Vector embeddings with PGVector
- Extracted images with CLIP embeddings
- Processing queue management

All models follow the schema specification in ARCHITECTURE.md.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for development without pgvector installed
    Vector = lambda dim: Text  # noqa: E731


Base = declarative_base()


class Document(Base):
    """
    Main documents table storing PDF file metadata and processing status.

    This table tracks all processed PDFs with their conversion status,
    file metadata, and processing information.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    source_path = Column(Text, unique=True, nullable=False, index=True)
    filename = Column(
        Text, nullable=False, index=True
    )  # Add index for filename searches
    file_hash = Column(
        Text, nullable=False, unique=True, index=True
    )  # Make hash unique
    file_size_bytes = Column(BigInteger)
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
        index=True,  # Add index for date range queries
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    conversion_status = Column(
        String(50),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        index=True,  # Add index for status filtering
    )
    error_message = Column(Text)
    meta_data = Column("metadata", JSON)  # Map to 'metadata' column in database

    # Directory mirroring fields
    source_relative_path = Column(Text, index=True)  # Relative to watch directory
    output_path = Column(Text, index=True)  # Absolute output markdown path
    output_relative_path = Column(Text, index=True)  # Relative to output directory
    directory_depth = Column(Integer)  # Directory depth level

    # Add check constraint for conversion status and directory depth
    __table_args__ = (
        CheckConstraint(
            "conversion_status IN ('pending', 'processing', 'completed', 'failed')",
            name="check_conversion_status",
        ),
        CheckConstraint(
            "directory_depth >= 0",
            name="check_directory_depth_positive",
        ),
    )

    # Relationships
    content = relationship(
        "DocumentContent",
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
    )
    embeddings = relationship(
        "DocumentEmbedding", back_populates="document", cascade="all, delete-orphan"
    )
    images = relationship(
        "DocumentImage", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.conversion_status}')>"


class DocumentContent(Base):
    """
    Converted content table storing the processed Markdown and plain text.

    This table contains the actual converted content from PDF processing,
    including processing metadata and content statistics.
    """

    __tablename__ = "document_content"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    markdown_content = Column(Text)
    plain_text = Column(Text)
    page_count = Column(Integer)
    has_images = Column(Boolean, default=False)
    has_tables = Column(Boolean, default=False)
    processing_time_ms = Column(Integer)
    created_at = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )

    # Embedding tracking fields
    embedding_status = Column(
        String(50), default="pending", server_default=text("'pending'")
    )
    embedding_generated_at = Column(DateTime)
    embedding_count = Column(Integer, default=0, server_default=text("0"))
    embedding_error = Column(Text)

    # Relationship
    document = relationship("Document", back_populates="content")

    def __repr__(self) -> str:
        return f"<DocumentContent(id={self.id}, document_id={self.document_id}, pages={self.page_count})>"


class DocumentEmbedding(Base):
    """
    Vector embeddings table for semantic search.

    This table stores text embeddings generated from document chunks,
    using PGVector for efficient similarity search operations.
    Embeddings are 1536-dimensional (OpenAI text-embedding-3-small).
    """

    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    chunk_text = Column(Text)
    embedding = Column(Vector(768))  # Text embeddings dimension (nomic-embed-text)
    meta_data = Column("metadata", JSON)  # Map to 'metadata' column
    created_at = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )

    # Add vector dimension validation constraint
    __table_args__ = (
        CheckConstraint(
            "vector_dims(embedding) = 768", name="check_text_embedding_dimensions"
        ),
        CheckConstraint("chunk_index >= 0", name="check_chunk_index_positive"),
    )

    # Relationship
    document = relationship("Document", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<DocumentEmbedding(id={self.id}, document_id={self.document_id}, chunk={self.chunk_index})>"


class DocumentImage(Base):
    """
    Extracted images table with CLIP embeddings for visual search.

    This table stores images extracted from PDFs along with their
    OCR text and CLIP embeddings for visual similarity search.
    """

    __tablename__ = "document_images"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer)
    image_index = Column(Integer)
    image_path = Column(Text)
    ocr_text = Column(Text)
    ocr_confidence = Column(Float)
    image_embedding = Column(Vector(512))  # CLIP embeddings dimension
    meta_data = Column("metadata", JSON)  # Map to 'metadata' column
    created_at = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )

    # Add vector dimension validation constraint
    __table_args__ = (
        CheckConstraint(
            "vector_dims(image_embedding) = 512",
            name="check_image_embedding_dimensions",
        ),
        CheckConstraint(
            "ocr_confidence >= 0 AND ocr_confidence <= 1",
            name="check_ocr_confidence_range",
        ),
        CheckConstraint("image_index >= 0", name="check_image_index_positive"),
    )

    # Relationship
    document = relationship("Document", back_populates="images")

    def __repr__(self) -> str:
        return f"<DocumentImage(id={self.id}, document_id={self.document_id}, page={self.page_number})>"


class ProcessingQueue(Base):
    """
    Processing queue table for task management.

    This table manages the queue of PDF files waiting to be processed,
    with priority, status tracking, and worker assignment.
    """

    __tablename__ = "processing_queue"

    id = Column(Integer, primary_key=True)
    file_path = Column(Text, nullable=False)
    priority = Column(Integer, default=5)
    status = Column(String(50), default="queued")
    attempts = Column(Integer, default=0)
    created_at = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    worker_id = Column(Text)
    error_message = Column(Text)

    # Add check constraint for status
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed', 'retrying')",
            name="check_queue_status",
        ),
    )

    def __repr__(self) -> str:
        return f"<ProcessingQueue(id={self.id}, file='{self.file_path}', status='{self.status}')>"


class PathMapping(Base):
    """
    Path mappings table for directory structure preservation.

    This table maintains mappings between source PDF directories and
    output Markdown directories, enabling automatic directory mirroring.
    """

    __tablename__ = "path_mappings"

    id = Column(Integer, primary_key=True)
    source_directory = Column(Text, nullable=False, index=True)
    output_directory = Column(Text, nullable=False, index=True)
    relative_path = Column(Text, nullable=False, index=True)
    directory_level = Column(Integer, index=True)
    files_count = Column(Integer, default=0)
    last_scanned = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )
    created_at = Column(
        DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Add constraints
    __table_args__ = (
        UniqueConstraint(
            "source_directory", "relative_path", name="uq_path_mappings_source_relative"
        ),
        CheckConstraint("directory_level >= 0", name="check_directory_level_positive"),
        CheckConstraint("files_count >= 0", name="check_files_count_non_negative"),
    )

    def __repr__(self) -> str:
        return f"<PathMapping(id={self.id}, source='{self.source_directory}', relative='{self.relative_path}')>"


class ServerConfiguration(Base):
    """
    Server configuration table for runtime settings persistence.

    This table stores server configuration that can be updated at runtime
    via the MCP configuration API, allowing dynamic reconfiguration without restarts.
    """

    __tablename__ = "server_configuration"

    id = Column(Integer, primary_key=True)
    config_key = Column(String, nullable=False, unique=True, index=True)
    config_value = Column(JSON, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_by = Column(String, nullable=True)
    description = Column(Text, nullable=True)

    # Add constraints
    __table_args__ = (
        UniqueConstraint("config_key", name="uq_server_configuration_config_key"),
    )

    def __repr__(self) -> str:
        return f"<ServerConfiguration(key='{self.config_key}', updated_at={self.updated_at})>"
