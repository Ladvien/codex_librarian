"""
Custom exceptions for PDF to Markdown MCP Server.

This module defines all custom exceptions used throughout the application
with proper error categorization for retry logic and error handling.
"""

from typing import Optional, Dict, Any


class PDFToMarkdownError(Exception):
    """Base exception for all PDF to Markdown MCP Server errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ValidationError(PDFToMarkdownError):
    """Raised when input validation fails."""

    pass


class ProcessingError(PDFToMarkdownError):
    """Raised when PDF processing fails."""

    def __init__(self, message: str, pdf_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.pdf_path = pdf_path
        if pdf_path:
            self.details["pdf_path"] = pdf_path


class EmbeddingError(PDFToMarkdownError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        if provider:
            self.details["provider"] = provider
        if model:
            self.details["model"] = model


class DatabaseError(PDFToMarkdownError):
    """Raised when database operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class ConfigurationError(PDFToMarkdownError):
    """Raised when configuration is invalid or missing."""

    pass


class ResourceError(PDFToMarkdownError):
    """Raised when system resources are exhausted."""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        if resource_type:
            self.details["resource_type"] = resource_type


class WorkerError(PDFToMarkdownError):
    """Raised when Celery worker operations fail."""

    def __init__(
        self,
        message: str,
        worker_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.worker_id = worker_id
        self.task_id = task_id
        if worker_id:
            self.details["worker_id"] = worker_id
        if task_id:
            self.details["task_id"] = task_id


class FileSystemError(PDFToMarkdownError):
    """Raised when file system operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation
        if file_path:
            self.details["file_path"] = file_path
        if operation:
            self.details["operation"] = operation


class OCRError(ProcessingError):
    """Raised when OCR processing fails."""

    def __init__(self, message: str, language: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.language = language
        if language:
            self.details["language"] = language


class ChunkingError(ProcessingError):
    """Raised when text chunking fails."""

    def __init__(
        self,
        message: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        if chunk_size:
            self.details["chunk_size"] = chunk_size
        if overlap:
            self.details["overlap"] = overlap


class SearchError(PDFToMarkdownError):
    """Raised when search operations fail."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        search_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.query = query
        self.search_type = search_type
        if query:
            self.details["query"] = query
        if search_type:
            self.details["search_type"] = search_type


class APIError(PDFToMarkdownError):
    """Raised when API operations fail."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.endpoint = endpoint
        if status_code:
            self.details["status_code"] = status_code
        if endpoint:
            self.details["endpoint"] = endpoint


class QueueError(WorkerError):
    """Raised when task queue operations fail."""

    def __init__(self, message: str, queue_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.queue_name = queue_name
        if queue_name:
            self.details["queue_name"] = queue_name


class RetryableError(PDFToMarkdownError):
    """Base class for errors that should be retried."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.max_retries = max_retries
        if retry_after:
            self.details["retry_after"] = retry_after
        if max_retries:
            self.details["max_retries"] = max_retries


class NonRetryableError(PDFToMarkdownError):
    """Base class for errors that should not be retried."""

    pass


# Convenience functions for creating specific error types


def validation_error(
    message: str, field: Optional[str] = None, value: Optional[Any] = None
) -> ValidationError:
    """Create a validation error with field information."""
    details = {}
    if field:
        details["field"] = field
    if value is not None:
        details["value"] = str(value)
    return ValidationError(message, details=details)


def processing_error(
    message: str, pdf_path: str, stage: Optional[str] = None
) -> ProcessingError:
    """Create a processing error with context information."""
    details = {}
    if stage:
        details["processing_stage"] = stage
    return ProcessingError(message, pdf_path=pdf_path, details=details)


def embedding_error(
    message: str, provider: str, model: str, text_length: Optional[int] = None
) -> EmbeddingError:
    """Create an embedding error with provider information."""
    details = {}
    if text_length:
        details["text_length"] = text_length
    return EmbeddingError(message, provider=provider, model=model, details=details)


def database_error(
    message: str, operation: str, table: Optional[str] = None
) -> DatabaseError:
    """Create a database error with operation information."""
    details = {}
    if table:
        details["table"] = table
    return DatabaseError(message, operation=operation, details=details)


def resource_error(
    message: str,
    resource_type: str,
    current_usage: Optional[str] = None,
    limit: Optional[str] = None,
) -> ResourceError:
    """Create a resource error with usage information."""
    details = {}
    if current_usage:
        details["current_usage"] = current_usage
    if limit:
        details["limit"] = limit
    return ResourceError(message, resource_type=resource_type, details=details)
