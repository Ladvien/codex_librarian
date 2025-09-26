"""
Factory classes for creating test objects with consistent data.

This module provides factory classes that can generate test objects
with realistic data for testing various components of the system.
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from .test_data import (
    SAMPLE_CHUNKS,
    SAMPLE_FORMULAS,
    SAMPLE_IMAGES,
    SAMPLE_MARKDOWN_CONTENT,
    SAMPLE_PDF_CONTENT,
    SAMPLE_PLAIN_TEXT,
    SAMPLE_TABLES,
    create_sample_embeddings,
)


class BaseFactory:
    """Base factory class with common utility methods."""

    @staticmethod
    def _generate_uuid() -> str:
        """Generate a unique UUID string."""
        return str(uuid.uuid4())

    @staticmethod
    def _generate_hash(content: str | bytes) -> str:
        """Generate SHA256 hash for content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _random_timestamp(days_ago: int = 0) -> datetime:
        """Generate a timestamp optionally N days ago."""
        base_time = datetime.utcnow()
        if days_ago > 0:
            base_time -= timedelta(days=days_ago)
        return base_time


class DocumentFactory(BaseFactory):
    """Factory for creating Document model instances."""

    @classmethod
    def create(
        self,
        file_path: str | None = None,
        file_name: str | None = None,
        file_size: int | None = None,
        status: str = "pending",
        created_days_ago: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a document dictionary with realistic data."""
        doc_id = kwargs.get("id", 1)
        unique_id = self._generate_uuid()

        if not file_path:
            file_path = f"/tmp/test_document_{unique_id}.pdf"

        if not file_name:
            file_name = f"test_document_{doc_id}.pdf"

        if not file_size:
            file_size = len(SAMPLE_PDF_CONTENT)

        file_hash = self._generate_hash(f"{file_path}_{file_size}")
        timestamp = self._random_timestamp(created_days_ago)

        document = {
            "id": doc_id,
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "mime_type": "application/pdf",
            "status": status,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        document.update(kwargs)
        return document

    @classmethod
    def create_batch(
        self,
        count: int = 3,
        status_distribution: dict[str, int] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Create multiple documents with varied statuses."""
        if status_distribution is None:
            status_distribution = {
                "completed": count // 2,
                "processing": count // 4,
                "pending": count // 4,
                "failed": max(1, count - (count // 2) - (count // 4) - (count // 4)),
            }

        documents = []
        doc_id = kwargs.pop("start_id", 1)

        for status, status_count in status_distribution.items():
            for i in range(status_count):
                doc = self.create(
                    id=doc_id, status=status, created_days_ago=i, **kwargs
                )
                documents.append(doc)
                doc_id += 1

        return documents


class ProcessingResultFactory(BaseFactory):
    """Factory for creating ProcessingResult instances."""

    @classmethod
    def create(
        self,
        success: bool = True,
        include_tables: bool = True,
        include_formulas: bool = True,
        include_images: bool = True,
        chunk_count: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a processing result with configurable content."""
        if chunk_count is None:
            chunk_count = len(SAMPLE_CHUNKS)

        chunks = SAMPLE_CHUNKS[:chunk_count] if success else []
        tables = SAMPLE_TABLES.copy() if success and include_tables else []
        formulas = SAMPLE_FORMULAS.copy() if success and include_formulas else []
        images = SAMPLE_IMAGES.copy() if success and include_images else []

        word_count = len(SAMPLE_PLAIN_TEXT.split()) if success else 0
        processing_time = kwargs.get("processing_time", 2.5)
        confidence = kwargs.get("confidence", 0.95 if success else 0.0)

        result = {
            "success": success,
            "markdown_content": SAMPLE_MARKDOWN_CONTENT if success else "",
            "plain_text": SAMPLE_PLAIN_TEXT if success else "",
            "chunks": chunks,
            "tables": tables,
            "formulas": formulas,
            "images": images,
            "metadata": {
                "processing_time": processing_time,
                "page_count": 1 if success else 0,
                "word_count": word_count,
                "language": "en" if success else "unknown",
                "confidence": confidence,
                "file_size": len(SAMPLE_PDF_CONTENT),
                "file_hash": self._generate_hash(SAMPLE_PDF_CONTENT),
            },
        }

        if not success:
            result["error_message"] = kwargs.get(
                "error_message", "Processing failed: Invalid PDF format"
            )

        result.update(kwargs)
        return result

    @classmethod
    def create_failed(
        self,
        error_message: str = "Processing failed",
        partial_content: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a failed processing result."""
        result = self.create(success=False, error_message=error_message, **kwargs)

        if partial_content:
            result["markdown_content"] = (
                "# Partially Processed\n\nSome content was extracted..."
            )
            result["plain_text"] = (
                "Partially Processed\n\nSome content was extracted..."
            )
            result["metadata"]["confidence"] = 0.3

        return result


class EmbeddingFactory(BaseFactory):
    """Factory for creating embedding vectors and related data."""

    @classmethod
    def create_embedding(
        self, dimensions: int = 1536, base_value: float = 0.1, **kwargs
    ) -> list[float]:
        """Create a single embedding vector."""
        return [base_value + i * 0.0001 for i in range(dimensions)]

    @classmethod
    def create_batch(
        self, count: int = 4, dimensions: int = 1536, **kwargs
    ) -> list[list[float]]:
        """Create multiple embedding vectors."""
        return create_sample_embeddings(count, dimensions)

    @classmethod
    def create_document_embedding(
        self,
        document_id: int = 1,
        chunk_index: int = 0,
        chunk_text: str | None = None,
        embedding: list[float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a document embedding database record."""
        if chunk_text is None:
            chunk_text = SAMPLE_CHUNKS[chunk_index % len(SAMPLE_CHUNKS)]["text"]

        if embedding is None:
            embedding = self.create_embedding(base_value=0.1 * (chunk_index + 1))

        chunk_data = SAMPLE_CHUNKS[chunk_index % len(SAMPLE_CHUNKS)]

        record = {
            "id": kwargs.get("id", chunk_index + 1),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "start_char": chunk_data["start_char"],
            "end_char": chunk_data["end_char"],
            "token_count": chunk_data["token_count"],
            "created_at": self._random_timestamp(),
        }

        record.update(kwargs)
        return record


class ChunkFactory(BaseFactory):
    """Factory for creating text chunks and related data."""

    @classmethod
    def create(
        self,
        text: str | None = None,
        chunk_index: int = 0,
        start_char: int | None = None,
        end_char: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a text chunk with realistic properties."""
        if text is None:
            text = SAMPLE_CHUNKS[chunk_index % len(SAMPLE_CHUNKS)]["text"]

        if start_char is None:
            start_char = chunk_index * 100

        if end_char is None:
            end_char = start_char + len(text)

        token_count = len(text.split())

        chunk = {
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "token_count": token_count,
            "chunk_index": chunk_index,
        }

        chunk.update(kwargs)
        return chunk

    @classmethod
    def create_batch(
        self, count: int = 4, base_text: str | None = None, **kwargs
    ) -> list[dict[str, Any]]:
        """Create multiple chunks from text or sample data."""
        if base_text:
            # Split text into chunks
            words = base_text.split()
            words_per_chunk = len(words) // count
            chunks = []

            for i in range(count):
                start_idx = i * words_per_chunk
                end_idx = start_idx + words_per_chunk if i < count - 1 else len(words)
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                chunk = self.create(
                    text=chunk_text,
                    chunk_index=i,
                    start_char=base_text.find(chunk_text),
                    end_char=base_text.find(chunk_text) + len(chunk_text),
                    **kwargs,
                )
                chunks.append(chunk)

            return chunks
        else:
            # Use sample chunks
            return [
                self.create(chunk_index=i, **kwargs)
                for i in range(min(count, len(SAMPLE_CHUNKS)))
            ]


class TableFactory(BaseFactory):
    """Factory for creating table data."""

    @classmethod
    def create(
        self,
        headers: list[str] | None = None,
        rows: list[list[str]] | None = None,
        table_index: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """Create table data with headers and rows."""
        if headers is None:
            headers = [f"Column {i + 1}" for i in range(3)]

        if rows is None:
            rows = [
                [f"Row {i + 1}, Col {j + 1}" for j in range(len(headers))]
                for i in range(3)
            ]

        # Generate markdown representation
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
        data_rows = []
        for row in rows:
            data_rows.append("| " + " | ".join(row) + " |")

        markdown = "\n".join([header_row, separator] + data_rows)

        table = {
            "table_index": table_index,
            "headers": headers,
            "rows": rows,
            "markdown": markdown,
            "position": {
                "page": 1,
                "x": 100,
                "y": 400,
                "width": len(headers) * 100,
                "height": (len(rows) + 1) * 20,
            },
            "confidence": 0.92,
        }

        table.update(kwargs)
        return table


class FormulaFactory(BaseFactory):
    """Factory for creating mathematical formulas."""

    @classmethod
    def create(
        self,
        latex: str | None = None,
        text: str | None = None,
        formula_index: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """Create formula data with LaTeX and text representations."""
        formulas = [
            {"latex": "E = mc^2", "text": "E = mc^2"},
            {
                "latex": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
                "text": "x = (-b ± √(b²-4ac)) / 2a",
            },
            {
                "latex": "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}",
                "text": "∫_{-∞}^{∞} e^{-x²} dx = √π",
            },
            {
                "latex": "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}",
                "text": "Σ_{n=1}^{∞} 1/n² = π²/6",
            },
        ]

        formula_data = formulas[formula_index % len(formulas)]

        if latex is None:
            latex = formula_data["latex"]

        if text is None:
            text = formula_data["text"]

        formula = {
            "formula_index": formula_index,
            "latex": latex,
            "text": text,
            "position": {
                "page": 1,
                "x": 150 + formula_index * 10,
                "y": 300 - formula_index * 20,
                "width": len(text) * 8,
                "height": 20,
            },
            "confidence": 0.95,
        }

        formula.update(kwargs)
        return formula


class ImageFactory(BaseFactory):
    """Factory for creating image extraction data."""

    @classmethod
    def create(
        self,
        image_index: int = 0,
        description: str | None = None,
        ocr_text: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create image data with OCR text and metadata."""
        if description is None:
            description = f"Image {image_index + 1} extracted from PDF"

        if ocr_text is None:
            ocr_text = f"Sample image {image_index + 1} with caption text"

        image = {
            "image_index": image_index,
            "file_path": f"/tmp/extracted_image_{image_index}.png",
            "description": description,
            "ocr_text": ocr_text,
            "position": {
                "page": 1,
                "x": 100,
                "y": 200 - image_index * 150,
                "width": 300,
                "height": 200,
            },
            "format": "PNG",
            "size_bytes": 12345 + image_index * 1000,
            "confidence": 0.89,
        }

        image.update(kwargs)
        return image


class TaskFactory(BaseFactory):
    """Factory for creating Celery task and queue data."""

    @classmethod
    def create(
        self,
        task_id: str | None = None,
        document_id: int = 1,
        task_type: str = "pdf_processing",
        status: str = "pending",
        **kwargs,
    ) -> dict[str, Any]:
        """Create task queue entry."""
        if task_id is None:
            task_id = f"task-{self._generate_uuid()}"

        task = {
            "id": kwargs.get("id", 1),
            "task_id": task_id,
            "document_id": document_id,
            "task_type": task_type,
            "status": status,
            "priority": kwargs.get("priority", 5),
            "progress": kwargs.get("progress", 0.0),
            "error_message": kwargs.get("error_message"),
            "retry_count": kwargs.get("retry_count", 0),
            "created_at": self._random_timestamp(),
            "updated_at": self._random_timestamp(),
        }

        task.update(kwargs)
        return task

    @classmethod
    def create_batch(
        self,
        count: int = 5,
        status_distribution: dict[str, int] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Create multiple tasks with varied statuses."""
        if status_distribution is None:
            status_distribution = {
                "completed": count // 3,
                "running": count // 3,
                "pending": count // 3,
                "failed": max(1, count - 3 * (count // 3)),
            }

        tasks = []
        task_id = kwargs.pop("start_id", 1)

        for status, status_count in status_distribution.items():
            for i in range(status_count):
                task = self.create(
                    id=task_id,
                    document_id=task_id,  # Simple mapping
                    status=status,
                    **kwargs,
                )
                tasks.append(task)
                task_id += 1

        return tasks
