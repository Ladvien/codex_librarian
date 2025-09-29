"""
Unit tests for database models.

This module tests the SQLAlchemy models for proper validation,
relationships, and database operations following TDD principles.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from src.pdf_to_markdown_mcp.db.models import (
    Document,
    DocumentContent,
    DocumentEmbedding,
    DocumentImage,
    ProcessingQueue,
)
from tests.fixtures import (
    DocumentFactory,
    EmbeddingFactory,
    create_sample_embeddings,
)


class TestDocument:
    """Test Document model validation and relationships."""

    def test_create_document_with_valid_data(self, db_session):
        """Test creating a document with all required fields."""
        # Given
        document_data = DocumentFactory.create(
            file_path="/tmp/test.pdf",
            file_name="test.pdf",
            file_size=12345,
            status="pending",
        )

        # When
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # Then
        assert document.id is not None
        assert document.file_path == "/tmp/test.pdf"
        assert document.file_name == "test.pdf"
        assert document.file_size == 12345
        assert document.status == "pending"
        assert document.mime_type == "application/pdf"
        assert document.created_at is not None
        assert document.updated_at is not None

    def test_document_status_validation(self, db_session):
        """Test document status constraint validation."""
        # Given
        valid_statuses = ["pending", "processing", "completed", "failed"]

        for status in valid_statuses:
            # When
            document_data = DocumentFactory.create(status=status)
            document = Document(**document_data)
            db_session.add(document)

            # Then - Should not raise exception
            db_session.commit()
            assert document.status == status
            db_session.delete(document)
            db_session.commit()

    def test_document_status_invalid_value(self, db_session):
        """Test document with invalid status raises constraint error."""
        # Given
        document_data = DocumentFactory.create(status="invalid_status")
        document = Document(**document_data)
        db_session.add(document)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_file_path_uniqueness(self, db_session):
        """Test file_path uniqueness constraint."""
        # Given
        file_path = "/tmp/unique_test.pdf"
        document1_data = DocumentFactory.create(file_path=file_path)
        document2_data = DocumentFactory.create(file_path=file_path)

        document1 = Document(**document1_data)
        document2 = Document(**document2_data)

        # When
        db_session.add(document1)
        db_session.commit()

        db_session.add(document2)

        # Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_file_hash_uniqueness(self, db_session):
        """Test file_hash uniqueness constraint."""
        # Given
        file_hash = "unique_hash_123"
        document1_data = DocumentFactory.create(file_hash=file_hash)
        document2_data = DocumentFactory.create(file_hash=file_hash)

        document1 = Document(**document1_data)
        document2 = Document(**document2_data)

        # When
        db_session.add(document1)
        db_session.commit()

        db_session.add(document2)

        # Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_relationships(self, db_session):
        """Test document relationships with other models."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # When - Add related content
        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test",
            plain_text="Test",
            word_count=1,
            language="en",
        )
        db_session.add(content)
        db_session.commit()

        # Then
        assert len(document.content) == 1
        assert document.content[0].markdown_content == "# Test"
        assert document.content[0].document_id == document.id

    def test_document_created_at_auto_timestamp(self, db_session):
        """Test created_at is automatically set."""
        # Given
        document_data = DocumentFactory.create()
        del document_data["created_at"]  # Remove to test auto-generation

        # When
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # Then
        assert document.created_at is not None
        assert isinstance(document.created_at, datetime)
        # Should be very recent (within last minute)
        assert datetime.utcnow() - document.created_at < timedelta(minutes=1)

    def test_document_updated_at_auto_timestamp(self, db_session):
        """Test updated_at is automatically updated on modification."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        original_updated_at = document.updated_at

        # When - Update document
        document.status = "completed"
        db_session.commit()

        # Then
        assert document.updated_at > original_updated_at

    def test_document_string_representation(self, db_session):
        """Test document __repr__ method."""
        # Given
        document_data = DocumentFactory.create(file_name="test_doc.pdf")
        document = Document(**document_data)

        # When
        repr_str = repr(document)

        # Then
        assert "test_doc.pdf" in repr_str
        assert "Document" in repr_str


class TestDocumentContent:
    """Test DocumentContent model validation and relationships."""

    def test_create_document_content(self, db_session):
        """Test creating document content with valid data."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # When
        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test Document\n\nContent",
            plain_text="Test Document\n\nContent",
            word_count=3,
            language="en",
        )
        db_session.add(content)
        db_session.commit()

        # Then
        assert content.id is not None
        assert content.document_id == document.id
        assert content.markdown_content == "# Test Document\n\nContent"
        assert content.plain_text == "Test Document\n\nContent"
        assert content.word_count == 3
        assert content.language == "en"
        assert content.created_at is not None

    def test_document_content_foreign_key_constraint(self, db_session):
        """Test foreign key constraint on document_id."""
        # Given
        content = DocumentContent(
            document_id=999999,  # Non-existent document
            markdown_content="# Test",
            plain_text="Test",
            word_count=1,
            language="en",
        )
        db_session.add(content)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_content_relationship(self, db_session):
        """Test relationship between document and content."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test",
            plain_text="Test",
            word_count=1,
            language="en",
        )
        db_session.add(content)
        db_session.commit()

        # When
        retrieved_document = (
            db_session.query(Document).filter_by(id=document.id).first()
        )

        # Then
        assert len(retrieved_document.content) == 1
        assert retrieved_document.content[0].id == content.id

    def test_document_content_cascade_delete(self, db_session):
        """Test content is deleted when document is deleted."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test",
            plain_text="Test",
            word_count=1,
            language="en",
        )
        db_session.add(content)
        db_session.commit()

        content_id = content.id

        # When
        db_session.delete(document)
        db_session.commit()

        # Then
        assert (
            db_session.query(DocumentContent).filter_by(id=content_id).first() is None
        )


class TestDocumentEmbedding:
    """Test DocumentEmbedding model with vector operations."""

    @pytest.mark.database
    def test_create_document_embedding(self, db_session):
        """Test creating document embedding with vector data."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        embedding_vector = create_sample_embeddings(1, 1536)[0]

        # When
        embedding = DocumentEmbedding(
            document_id=document.id,
            chunk_index=0,
            chunk_text="Sample text chunk for embedding",
            embedding=embedding_vector,
            start_char=0,
            end_char=33,
            token_count=6,
        )
        db_session.add(embedding)
        db_session.commit()

        # Then
        assert embedding.id is not None
        assert embedding.document_id == document.id
        assert embedding.chunk_index == 0
        assert embedding.chunk_text == "Sample text chunk for embedding"
        assert len(embedding.embedding) == 1536
        assert embedding.start_char == 0
        assert embedding.end_char == 33
        assert embedding.token_count == 6

    def test_document_embedding_foreign_key(self, db_session):
        """Test foreign key constraint on document_id."""
        # Given
        embedding_vector = create_sample_embeddings(1, 1536)[0]
        embedding = DocumentEmbedding(
            document_id=999999,  # Non-existent document
            chunk_index=0,
            chunk_text="Test",
            embedding=embedding_vector,
            start_char=0,
            end_char=4,
            token_count=1,
        )
        db_session.add(embedding)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_embedding_relationship(self, db_session):
        """Test relationship between document and embeddings."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # Create multiple embeddings
        embeddings_data = EmbeddingFactory.create_batch(3, document_id=document.id)
        embeddings = []
        for emb_data in embeddings_data:
            embedding = DocumentEmbedding(**emb_data)
            embeddings.append(embedding)
            db_session.add(embedding)

        db_session.commit()

        # When
        retrieved_document = (
            db_session.query(Document).filter_by(id=document.id).first()
        )

        # Then
        assert len(retrieved_document.embeddings) == 3
        assert all(
            emb.document_id == document.id for emb in retrieved_document.embeddings
        )

    def test_document_embedding_unique_constraint(self, db_session):
        """Test unique constraint on (document_id, chunk_index)."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        embedding_vector = create_sample_embeddings(1, 1536)[0]

        # Create first embedding
        embedding1 = DocumentEmbedding(
            document_id=document.id,
            chunk_index=0,
            chunk_text="First chunk",
            embedding=embedding_vector,
            start_char=0,
            end_char=11,
            token_count=2,
        )
        db_session.add(embedding1)
        db_session.commit()

        # Create second embedding with same chunk_index
        embedding2 = DocumentEmbedding(
            document_id=document.id,
            chunk_index=0,  # Same chunk_index
            chunk_text="Second chunk",
            embedding=embedding_vector,
            start_char=12,
            end_char=24,
            token_count=2,
        )
        db_session.add(embedding2)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestDocumentImage:
    """Test DocumentImage model with CLIP embeddings."""

    def test_create_document_image(self, db_session):
        """Test creating document image with metadata."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        clip_embedding = create_sample_embeddings(1, 512)[0]  # CLIP uses 512D

        # When
        image = DocumentImage(
            document_id=document.id,
            image_index=0,
            file_path="/tmp/image.png",
            description="Sample image",
            ocr_text="Image caption text",
            embedding=clip_embedding,
            page_number=1,
            position_data={"x": 100, "y": 200, "width": 300, "height": 200},
        )
        db_session.add(image)
        db_session.commit()

        # Then
        assert image.id is not None
        assert image.document_id == document.id
        assert image.image_index == 0
        assert image.file_path == "/tmp/image.png"
        assert image.description == "Sample image"
        assert image.ocr_text == "Image caption text"
        assert len(image.embedding) == 512
        assert image.page_number == 1
        assert image.position_data["x"] == 100

    def test_document_image_unique_constraint(self, db_session):
        """Test unique constraint on (document_id, image_index)."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        clip_embedding = create_sample_embeddings(1, 512)[0]

        # Create first image
        image1 = DocumentImage(
            document_id=document.id,
            image_index=0,
            file_path="/tmp/image1.png",
            description="First image",
            ocr_text="First caption",
            embedding=clip_embedding,
            page_number=1,
            position_data={"x": 100, "y": 200, "width": 300, "height": 200},
        )
        db_session.add(image1)
        db_session.commit()

        # Create second image with same image_index
        image2 = DocumentImage(
            document_id=document.id,
            image_index=0,  # Same image_index
            file_path="/tmp/image2.png",
            description="Second image",
            ocr_text="Second caption",
            embedding=clip_embedding,
            page_number=1,
            position_data={"x": 400, "y": 200, "width": 300, "height": 200},
        )
        db_session.add(image2)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestProcessingQueue:
    """Test ProcessingQueue model for task management."""

    def test_create_processing_queue_entry(self, db_session):
        """Test creating queue entry with valid data."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        # When
        queue_entry = ProcessingQueue(
            task_id="test-task-123",
            document_id=document.id,
            task_type="pdf_processing",
            status="pending",
            priority=5,
            progress=0.0,
            retry_count=0,
        )
        db_session.add(queue_entry)
        db_session.commit()

        # Then
        assert queue_entry.id is not None
        assert queue_entry.task_id == "test-task-123"
        assert queue_entry.document_id == document.id
        assert queue_entry.task_type == "pdf_processing"
        assert queue_entry.status == "pending"
        assert queue_entry.priority == 5
        assert queue_entry.progress == 0.0
        assert queue_entry.retry_count == 0
        assert queue_entry.created_at is not None

    def test_processing_queue_status_validation(self, db_session):
        """Test queue status constraint validation."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        valid_statuses = ["pending", "running", "completed", "failed", "retrying"]

        for status in valid_statuses:
            # When
            queue_entry = ProcessingQueue(
                task_id=f"test-task-{status}",
                document_id=document.id,
                task_type="pdf_processing",
                status=status,
                priority=5,
                progress=0.0,
                retry_count=0,
            )
            db_session.add(queue_entry)

            # Then - Should not raise exception
            db_session.commit()
            assert queue_entry.status == status
            db_session.delete(queue_entry)
            db_session.commit()

    def test_processing_queue_task_type_validation(self, db_session):
        """Test queue task_type constraint validation."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        valid_task_types = [
            "pdf_processing",
            "embedding_generation",
            "maintenance",
            "cleanup",
        ]

        for task_type in valid_task_types:
            # When
            queue_entry = ProcessingQueue(
                task_id=f"test-task-{task_type}",
                document_id=document.id,
                task_type=task_type,
                status="pending",
                priority=5,
                progress=0.0,
                retry_count=0,
            )
            db_session.add(queue_entry)

            # Then - Should not raise exception
            db_session.commit()
            assert queue_entry.task_type == task_type
            db_session.delete(queue_entry)
            db_session.commit()

    def test_processing_queue_task_id_uniqueness(self, db_session):
        """Test task_id uniqueness constraint."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        task_id = "unique-task-123"

        # Create first queue entry
        entry1 = ProcessingQueue(
            task_id=task_id,
            document_id=document.id,
            task_type="pdf_processing",
            status="pending",
            priority=5,
            progress=0.0,
            retry_count=0,
        )
        db_session.add(entry1)
        db_session.commit()

        # Create second entry with same task_id
        entry2 = ProcessingQueue(
            task_id=task_id,  # Same task_id
            document_id=document.id,
            task_type="embedding_generation",
            status="pending",
            priority=3,
            progress=0.0,
            retry_count=0,
        )
        db_session.add(entry2)

        # When/Then
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_processing_queue_relationship(self, db_session):
        """Test relationship between queue entry and document."""
        # Given
        document_data = DocumentFactory.create()
        document = Document(**document_data)
        db_session.add(document)
        db_session.commit()

        queue_entry = ProcessingQueue(
            task_id="test-task-123",
            document_id=document.id,
            task_type="pdf_processing",
            status="pending",
            priority=5,
            progress=0.0,
            retry_count=0,
        )
        db_session.add(queue_entry)
        db_session.commit()

        # When
        retrieved_document = (
            db_session.query(Document).filter_by(id=document.id).first()
        )

        # Then
        assert len(retrieved_document.processing_tasks) == 1
        assert retrieved_document.processing_tasks[0].task_id == "test-task-123"
