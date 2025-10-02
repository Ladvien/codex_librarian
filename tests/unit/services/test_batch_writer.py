"""
Test suite for batch database writer service.

Comprehensive test coverage for BatchWriter class including:
- Initialization and configuration
- Start/stop lifecycle management
- Queue operations (document_content, document_update)
- Worker loop batch flushing (size and time triggers)
- Database write operations
- Error handling and retry logic
- Metrics collection
- Thread safety
- Singleton pattern
"""

import time
import threading
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from collections import deque

import pytest

from pdf_to_markdown_mcp.services.batch_writer import (
    BatchWriter,
    get_batch_writer,
    stop_batch_writer,
)


class TestBatchWriterInitialization:
    """Test BatchWriter initialization and configuration."""

    def test_batch_writer_default_initialization(self):
        """Test BatchWriter with default configuration."""
        writer = BatchWriter()

        assert writer.batch_size == 5
        assert writer.max_delay_seconds == 10.0
        assert writer.enable_metrics is True
        assert isinstance(writer.queue, deque)
        assert len(writer.queue) == 0
        assert isinstance(writer.lock, type(threading.Lock()))
        assert writer.running is False
        assert writer.worker_thread is None

    def test_batch_writer_custom_initialization(self):
        """Test BatchWriter with custom configuration."""
        writer = BatchWriter(
            batch_size=10,
            max_delay_seconds=5.0,
            enable_metrics=False
        )

        assert writer.batch_size == 10
        assert writer.max_delay_seconds == 5.0
        assert writer.enable_metrics is False

    def test_batch_writer_metrics_initialization(self):
        """Test BatchWriter metrics are initialized correctly."""
        writer = BatchWriter()

        assert writer.total_writes == 0
        assert writer.total_records == 0
        assert writer.total_batches == 0
        assert writer.total_wait_time == 0.0
        assert writer.last_flush_time > 0

    def test_batch_writer_error_tracking_initialization(self):
        """Test BatchWriter error tracking is initialized."""
        writer = BatchWriter()

        assert writer.last_error is None
        assert writer.error_count == 0


class TestBatchWriterLifecycle:
    """Test BatchWriter start/stop lifecycle."""

    def test_start_batch_writer(self):
        """Test starting BatchWriter creates worker thread."""
        writer = BatchWriter()

        try:
            writer.start()

            assert writer.running is True
            assert writer.worker_thread is not None
            assert writer.worker_thread.is_alive()
            assert writer.worker_thread.daemon is True
            assert writer.worker_thread.name == "BatchWriterThread"
        finally:
            writer.stop()

    def test_start_batch_writer_already_running(self):
        """Test starting already running BatchWriter logs warning."""
        writer = BatchWriter()

        try:
            writer.start()

            # Try starting again
            with patch('pdf_to_markdown_mcp.services.batch_writer.logger') as mock_logger:
                writer.start()
                mock_logger.warning.assert_called_once_with("BatchWriter already running")
        finally:
            writer.stop()

    def test_stop_batch_writer(self):
        """Test stopping BatchWriter halts worker thread."""
        writer = BatchWriter()
        writer.start()

        # Wait for thread to start
        time.sleep(0.1)

        writer.stop(flush=False)

        assert writer.running is False
        # Thread should complete within timeout
        if writer.worker_thread:
            writer.worker_thread.join(timeout=1.0)
            assert not writer.worker_thread.is_alive()

    def test_stop_batch_writer_with_flush(self):
        """Test stopping BatchWriter with flush option."""
        writer = BatchWriter()
        writer.start()

        # Add items to queue
        writer.queue_document_update(
            document_id=1,
            status="completed"
        )

        with patch.object(writer, '_flush_batch') as mock_flush:
            writer.stop(flush=True)
            mock_flush.assert_called_once_with(force=True)

    def test_stop_batch_writer_not_running(self):
        """Test stopping non-running BatchWriter is safe."""
        writer = BatchWriter()

        # Should not raise any errors
        writer.stop()

        assert writer.running is False


class TestQueueDocumentContent:
    """Test queueing document content records."""

    def test_queue_document_content_basic(self):
        """Test queueing basic document content."""
        writer = BatchWriter()

        result = writer.queue_document_content(
            document_id=123,
            markdown_content="# Test Document",
            plain_text="Test Document",
            page_count=5
        )

        assert result is True
        assert len(writer.queue) == 1

        request = writer.queue[0]
        assert request["type"] == "document_content"
        assert request["data"]["document_id"] == 123
        assert request["data"]["markdown_content"] == "# Test Document"
        assert request["data"]["plain_text"] == "Test Document"
        assert request["data"]["page_count"] == 5
        assert request["data"]["has_images"] is False
        assert request["data"]["has_tables"] is False
        assert request["data"]["processing_time_ms"] == 0

    def test_queue_document_content_with_all_fields(self):
        """Test queueing document content with all optional fields."""
        writer = BatchWriter()

        result = writer.queue_document_content(
            document_id=456,
            markdown_content="# Advanced Document\n\n![image](img.png)",
            plain_text="Advanced Document",
            page_count=10,
            has_images=True,
            has_tables=True,
            processing_time_ms=1500,
            correlation_id="test-correlation-123"
        )

        assert result is True
        assert len(writer.queue) == 1

        request = writer.queue[0]
        assert request["data"]["has_images"] is True
        assert request["data"]["has_tables"] is True
        assert request["data"]["processing_time_ms"] == 1500
        assert request["correlation_id"] == "test-correlation-123"
        assert "queued_at" in request

    def test_queue_document_content_multiple(self):
        """Test queueing multiple document content records."""
        writer = BatchWriter()

        for i in range(5):
            writer.queue_document_content(
                document_id=i,
                markdown_content=f"# Document {i}",
                plain_text=f"Document {i}",
                page_count=i + 1
            )

        assert len(writer.queue) == 5

        # Verify order is preserved
        for i, request in enumerate(writer.queue):
            assert request["data"]["document_id"] == i

    def test_queue_document_content_thread_safety(self):
        """Test queueing document content is thread-safe."""
        writer = BatchWriter()
        results = []

        def queue_items(start_id, count):
            for i in range(start_id, start_id + count):
                result = writer.queue_document_content(
                    document_id=i,
                    markdown_content=f"# Doc {i}",
                    plain_text=f"Doc {i}",
                    page_count=1
                )
                results.append(result)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=queue_items, args=(i * 10, 10))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results)
        assert len(writer.queue) == 50


class TestQueueDocumentUpdate:
    """Test queueing document status updates."""

    def test_queue_document_update_basic(self):
        """Test queueing basic document update."""
        writer = BatchWriter()

        result = writer.queue_document_update(
            document_id=789,
            status="completed"
        )

        assert result is True
        assert len(writer.queue) == 1

        request = writer.queue[0]
        assert request["type"] == "document_update"
        assert request["data"]["document_id"] == 789
        assert request["data"]["status"] == "completed"
        assert request["data"]["metadata"] == {}
        assert isinstance(request["data"]["updated_at"], datetime)

    def test_queue_document_update_with_metadata(self):
        """Test queueing document update with metadata."""
        writer = BatchWriter()

        metadata = {
            "error_count": 2,
            "retry_attempts": 1,
            "last_error": "Timeout"
        }

        result = writer.queue_document_update(
            document_id=101,
            status="failed",
            metadata=metadata,
            correlation_id="update-correlation-456"
        )

        assert result is True
        request = writer.queue[0]
        assert request["data"]["metadata"] == metadata
        assert request["correlation_id"] == "update-correlation-456"

    def test_queue_document_update_multiple(self):
        """Test queueing multiple document updates."""
        writer = BatchWriter()

        statuses = ["processing", "completed", "failed", "completed", "processing"]

        for i, status in enumerate(statuses):
            writer.queue_document_update(document_id=i, status=status)

        assert len(writer.queue) == 5

    def test_queue_mixed_operations(self):
        """Test queueing mixed document_content and document_update operations."""
        writer = BatchWriter()

        # Queue content
        writer.queue_document_content(
            document_id=1,
            markdown_content="# Doc 1",
            plain_text="Doc 1",
            page_count=1
        )

        # Queue update
        writer.queue_document_update(
            document_id=2,
            status="completed"
        )

        # Queue more content
        writer.queue_document_content(
            document_id=3,
            markdown_content="# Doc 3",
            plain_text="Doc 3",
            page_count=1
        )

        assert len(writer.queue) == 3
        assert writer.queue[0]["type"] == "document_content"
        assert writer.queue[1]["type"] == "document_update"
        assert writer.queue[2]["type"] == "document_content"


class TestWorkerLoopBatchFlushing:
    """Test worker loop batch flushing logic."""

    def test_worker_loop_flush_on_batch_size(self):
        """Test worker loop flushes when batch size is reached."""
        writer = BatchWriter(batch_size=3, max_delay_seconds=100.0)

        with patch.object(writer, '_flush_batch') as mock_flush:
            writer.start()

            # Add items to reach batch size
            for i in range(3):
                writer.queue_document_update(document_id=i, status="completed")

            # Wait for worker to process
            time.sleep(0.3)

            # Should have flushed at least once
            assert mock_flush.call_count >= 1

            writer.stop(flush=False)

    def test_worker_loop_flush_on_max_delay(self):
        """Test worker loop flushes when max delay is exceeded."""
        writer = BatchWriter(batch_size=100, max_delay_seconds=0.2)

        with patch.object(writer, '_flush_batch') as mock_flush:
            writer.start()

            # Add one item (won't reach batch size)
            writer.queue_document_update(document_id=1, status="completed")

            # Wait for max delay to be exceeded
            time.sleep(0.4)

            # Should have flushed due to time
            assert mock_flush.call_count >= 1

            writer.stop(flush=False)

    def test_worker_loop_no_flush_empty_queue(self):
        """Test worker loop doesn't flush when queue is empty."""
        writer = BatchWriter(batch_size=3, max_delay_seconds=0.2)

        with patch.object(writer, '_flush_batch') as mock_flush:
            writer.start()

            # Wait without adding items
            time.sleep(0.4)

            # Should not flush empty queue
            mock_flush.assert_not_called()

            writer.stop(flush=False)

    def test_worker_loop_handles_exceptions(self):
        """Test worker loop continues after exceptions."""
        writer = BatchWriter(batch_size=2)

        call_count = [0]

        def failing_flush():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Test exception")

        with patch.object(writer, '_flush_batch', side_effect=failing_flush):
            writer.start()

            # Add items
            writer.queue_document_update(document_id=1, status="completed")
            writer.queue_document_update(document_id=2, status="completed")

            # Wait for processing
            time.sleep(0.3)

            # Should have tried multiple times despite error
            assert call_count[0] >= 1
            assert writer.error_count >= 1
            assert writer.last_error is not None

            writer.stop(flush=False)


class TestFlushBatch:
    """Test batch flushing to database."""

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_basic(self, mock_get_session):
        """Test basic batch flush operation."""
        writer = BatchWriter()
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Add items to queue
        writer.queue_document_update(document_id=1, status="completed")
        writer.queue_document_update(document_id=2, status="completed")

        writer._flush_batch()

        # Queue should be empty after flush
        assert len(writer.queue) == 0

        # Database session should be used
        mock_session.commit.assert_called_once()

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_empty_queue(self, mock_get_session):
        """Test flushing empty queue does nothing."""
        writer = BatchWriter()

        writer._flush_batch()

        # Should not create session for empty queue
        mock_get_session.assert_not_called()

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_respects_batch_size(self, mock_get_session):
        """Test flush respects batch_size limit."""
        writer = BatchWriter(batch_size=3)
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Add more items than batch size
        for i in range(10):
            writer.queue_document_update(document_id=i, status="completed")

        writer._flush_batch()

        # Should have processed only batch_size items
        assert len(writer.queue) == 7  # 10 - 3 = 7 remaining

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_force_flushes_all(self, mock_get_session):
        """Test force flush processes all items."""
        writer = BatchWriter(batch_size=3)
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Add more items than batch size
        for i in range(10):
            writer.queue_document_update(document_id=i, status="completed")

        writer._flush_batch(force=True)

        # Should have processed all items
        assert len(writer.queue) == 0

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_updates_metrics(self, mock_get_session):
        """Test flush updates metrics correctly."""
        writer = BatchWriter()
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Add items
        for i in range(3):
            writer.queue_document_update(document_id=i, status="completed")

        initial_batches = writer.total_batches
        initial_records = writer.total_records

        writer._flush_batch()

        assert writer.total_batches == initial_batches + 1
        assert writer.total_records == initial_records + 3

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_flush_batch_database_error_retry(self, mock_get_session):
        """Test flush requeues items on database error."""
        writer = BatchWriter()
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("Database error")
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Add items
        writer.queue_document_update(document_id=1, status="completed")
        writer.queue_document_update(document_id=2, status="completed")

        writer._flush_batch()

        # Items should be back in queue
        assert len(writer.queue) == 2

        # Error tracking
        assert writer.error_count == 1
        assert writer.last_error is not None


class TestWriteDocumentContent:
    """Test writing document content to database."""

    @patch('pdf_to_markdown_mcp.db.models.DocumentContent')
    def test_write_document_content_basic(self, mock_content_class):
        """Test writing basic document content."""
        writer = BatchWriter()
        mock_session = MagicMock()

        request = {
            "type": "document_content",
            "data": {
                "document_id": 123,
                "markdown_content": "# Test",
                "plain_text": "Test",
                "page_count": 1,
                "has_images": False,
                "has_tables": False,
                "processing_time_ms": 100
            },
            "correlation_id": "test-123"
        }

        writer._write_document_content(mock_session, request)

        # Should create DocumentContent instance
        mock_content_class.assert_called_once_with(
            document_id=123,
            markdown_content="# Test",
            plain_text="Test",
            page_count=1,
            has_images=False,
            has_tables=False,
            processing_time_ms=100
        )

        # Should add to session (flush removed as performance optimization)
        mock_session.add.assert_called_once()

    @patch('pdf_to_markdown_mcp.db.models.DocumentContent')
    def test_write_document_content_with_images_and_tables(self, mock_content_class):
        """Test writing document content with images and tables."""
        writer = BatchWriter()
        mock_session = MagicMock()

        request = {
            "type": "document_content",
            "data": {
                "document_id": 456,
                "markdown_content": "# Advanced\n\n![img](test.png)\n\n| A | B |\n|---|---|",
                "plain_text": "Advanced",
                "page_count": 5,
                "has_images": True,
                "has_tables": True,
                "processing_time_ms": 2500
            },
            "correlation_id": None
        }

        writer._write_document_content(mock_session, request)

        call_kwargs = mock_content_class.call_args[1]
        assert call_kwargs["has_images"] is True
        assert call_kwargs["has_tables"] is True


class TestWriteDocumentUpdate:
    """Test writing document status updates to database."""

    @patch('pdf_to_markdown_mcp.db.models.Document')
    def test_write_document_update_basic(self, mock_document_class):
        """Test writing basic document update."""
        writer = BatchWriter()
        mock_session = MagicMock()

        # Mock document query
        mock_document = MagicMock()
        mock_document.meta_data = {}
        mock_session.query.return_value.filter.return_value.first.return_value = mock_document

        updated_at = datetime.utcnow()
        request = {
            "type": "document_update",
            "data": {
                "document_id": 789,
                "status": "completed",
                "metadata": {},
                "updated_at": updated_at
            },
            "correlation_id": "update-789"
        }

        writer._write_document_update(mock_session, request)

        # Should update document status
        assert mock_document.conversion_status == "completed"
        assert mock_document.updated_at == updated_at

    @patch('pdf_to_markdown_mcp.db.models.Document')
    def test_write_document_update_with_metadata(self, mock_document_class):
        """Test writing document update with metadata."""
        writer = BatchWriter()
        mock_session = MagicMock()

        # Mock document with existing metadata
        mock_document = MagicMock()
        mock_document.meta_data = {"existing_key": "existing_value"}
        mock_session.query.return_value.filter.return_value.first.return_value = mock_document

        request = {
            "type": "document_update",
            "data": {
                "document_id": 101,
                "status": "failed",
                "metadata": {"error": "Processing failed", "attempt": 3},
                "updated_at": datetime.utcnow()
            },
            "correlation_id": None
        }

        writer._write_document_update(mock_session, request)

        # Should merge metadata
        assert "existing_key" in mock_document.meta_data
        assert "error" in mock_document.meta_data
        assert "attempt" in mock_document.meta_data

    @patch('pdf_to_markdown_mcp.db.models.Document')
    def test_write_document_update_document_not_found(self, mock_document_class):
        """Test writing update for non-existent document."""
        writer = BatchWriter()
        mock_session = MagicMock()

        # Mock document not found
        mock_session.query.return_value.filter.return_value.first.return_value = None

        request = {
            "type": "document_update",
            "data": {
                "document_id": 999,
                "status": "completed",
                "metadata": {},
                "updated_at": datetime.utcnow()
            },
            "correlation_id": None
        }

        # Should not raise exception
        with patch('pdf_to_markdown_mcp.services.batch_writer.logger') as mock_logger:
            writer._write_document_update(mock_session, request)

            # Should log warning
            mock_logger.warning.assert_called_once()


class TestGetMetrics:
    """Test metrics collection."""

    def test_get_metrics_initial_state(self):
        """Test metrics in initial state."""
        writer = BatchWriter(batch_size=10, max_delay_seconds=20.0)

        metrics = writer.get_metrics()

        assert metrics["running"] is False
        assert metrics["queue_size"] == 0
        assert metrics["total_batches"] == 0
        assert metrics["total_records"] == 0
        assert metrics["avg_wait_time_seconds"] == 0
        assert metrics["error_count"] == 0
        assert metrics["last_error"] is None
        assert metrics["batch_size"] == 10
        assert metrics["max_delay_seconds"] == 20.0

    def test_get_metrics_with_queue(self):
        """Test metrics with items in queue."""
        writer = BatchWriter()

        for i in range(5):
            writer.queue_document_update(document_id=i, status="completed")

        metrics = writer.get_metrics()

        assert metrics["queue_size"] == 5

    def test_get_metrics_after_processing(self):
        """Test metrics after processing batches."""
        writer = BatchWriter()

        # Manually update metrics (simulating processing)
        writer.total_batches = 3
        writer.total_records = 15
        writer.total_wait_time = 4.5
        writer.error_count = 1
        writer.last_error = "Test error"

        metrics = writer.get_metrics()

        assert metrics["total_batches"] == 3
        assert metrics["total_records"] == 15
        assert metrics["avg_wait_time_seconds"] == 0.3  # 4.5 / 15
        assert metrics["error_count"] == 1
        assert metrics["last_error"] == "Test error"

    def test_get_metrics_running_state(self):
        """Test metrics reflect running state."""
        writer = BatchWriter()

        try:
            writer.start()
            time.sleep(0.1)

            metrics = writer.get_metrics()
            assert metrics["running"] is True
        finally:
            writer.stop()


class TestSingletonPattern:
    """Test singleton pattern for get_batch_writer."""

    def teardown_method(self):
        """Clean up singleton after each test."""
        stop_batch_writer(flush=False)

    def test_get_batch_writer_creates_singleton(self):
        """Test get_batch_writer creates singleton instance."""
        writer1 = get_batch_writer(auto_start=False)
        writer2 = get_batch_writer(auto_start=False)

        # Should be the same instance
        assert writer1 is writer2

    def test_get_batch_writer_auto_start(self):
        """Test get_batch_writer auto-starts by default."""
        writer = get_batch_writer(auto_start=True)

        try:
            assert writer.running is True
            assert writer.worker_thread is not None
        finally:
            writer.stop()

    def test_get_batch_writer_custom_config(self):
        """Test get_batch_writer with custom configuration."""
        writer = get_batch_writer(
            batch_size=20,
            max_delay_seconds=15.0,
            auto_start=False
        )

        assert writer.batch_size == 20
        assert writer.max_delay_seconds == 15.0

    def test_get_batch_writer_ignores_subsequent_config(self):
        """Test get_batch_writer ignores config on subsequent calls."""
        writer1 = get_batch_writer(batch_size=10, auto_start=False)

        # Second call with different config should return same instance
        writer2 = get_batch_writer(batch_size=50, auto_start=False)

        assert writer1 is writer2
        assert writer2.batch_size == 10  # Original config preserved

    def test_stop_batch_writer_singleton(self):
        """Test stop_batch_writer stops and clears singleton."""
        writer = get_batch_writer(auto_start=True)

        # Add item to queue
        writer.queue_document_update(document_id=1, status="completed")

        with patch.object(writer, '_flush_batch') as mock_flush:
            stop_batch_writer(flush=True)

            # Should flush
            mock_flush.assert_called_once_with(force=True)

        # Singleton should be cleared
        # Next call should create new instance
        new_writer = get_batch_writer(auto_start=False)

        # Should be different instance (singleton was cleared)
        assert new_writer is not writer


class TestThreadSafety:
    """Test thread safety of concurrent operations."""

    def test_concurrent_queueing(self):
        """Test concurrent queueing from multiple threads."""
        writer = BatchWriter()
        num_threads = 10
        items_per_thread = 20

        def queue_worker(thread_id):
            for i in range(items_per_thread):
                writer.queue_document_content(
                    document_id=thread_id * 1000 + i,
                    markdown_content=f"# Content {i}",
                    plain_text=f"Content {i}",
                    page_count=1
                )

        threads = []
        for tid in range(num_threads):
            thread = threading.Thread(target=queue_worker, args=(tid,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All items should be queued
        assert len(writer.queue) == num_threads * items_per_thread

    def test_concurrent_queue_and_flush(self):
        """Test concurrent queueing while flushing."""
        writer = BatchWriter(batch_size=5)

        with patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session'):
            writer.start()

            # Queue items from multiple threads while worker is flushing
            def queue_worker():
                for i in range(10):
                    writer.queue_document_update(document_id=i, status="completed")
                    time.sleep(0.01)

            threads = [threading.Thread(target=queue_worker) for _ in range(3)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            writer.stop()

            # Should have processed all items successfully
            # (exact count may vary due to timing)
            assert writer.error_count == 0

    def test_metrics_thread_safety(self):
        """Test metrics access is thread-safe."""
        writer = BatchWriter()
        results = []

        def get_metrics_worker():
            for _ in range(100):
                metrics = writer.get_metrics()
                results.append(metrics)

        threads = [threading.Thread(target=get_metrics_worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All metrics calls should succeed
        assert len(results) == 500


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_markdown_content(self):
        """Test queueing document with empty markdown content."""
        writer = BatchWriter()

        result = writer.queue_document_content(
            document_id=1,
            markdown_content="",
            plain_text="",
            page_count=0
        )

        assert result is True
        assert len(writer.queue) == 1

    def test_large_content(self):
        """Test queueing document with large content."""
        writer = BatchWriter()

        large_content = "# Test\n\n" + ("Large content block. " * 10000)

        result = writer.queue_document_content(
            document_id=1,
            markdown_content=large_content,
            plain_text=large_content,
            page_count=100
        )

        assert result is True
        assert len(writer.queue) == 1

    def test_special_characters_in_content(self):
        """Test queueing content with special characters."""
        writer = BatchWriter()

        special_content = "# Test\n\n特殊字符 \U0001F600 \n\t\r\n"

        result = writer.queue_document_content(
            document_id=1,
            markdown_content=special_content,
            plain_text=special_content,
            page_count=1
        )

        assert result is True
        request = writer.queue[0]
        assert request["data"]["markdown_content"] == special_content

    @patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session')
    def test_unknown_request_type(self, mock_get_session):
        """Test handling unknown request type in batch."""
        writer = BatchWriter()
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Manually add unknown request type
        with writer.lock:
            writer.queue.append({
                "type": "unknown_type",
                "data": {},
                "queued_at": time.time()
            })

        # Should not raise exception
        with patch('pdf_to_markdown_mcp.services.batch_writer.logger') as mock_logger:
            writer._flush_batch()

            # Should log warning
            mock_logger.warning.assert_called_once()

    def test_queue_overflow_handling(self):
        """Test handling very large queue."""
        writer = BatchWriter(batch_size=10)

        # Queue many items
        for i in range(1000):
            writer.queue_document_update(document_id=i, status="completed")

        assert len(writer.queue) == 1000

        # Multiple flushes should process all items
        with patch('pdf_to_markdown_mcp.services.batch_writer.get_db_session'):
            for _ in range(100):  # More than enough to clear queue
                if not writer.queue:
                    break
                writer._flush_batch()

        assert len(writer.queue) == 0

    def test_negative_timing_values(self):
        """Test handling of edge case timing values."""
        writer = BatchWriter(batch_size=5, max_delay_seconds=0.1)

        # This should not cause issues
        writer.last_flush_time = time.time() + 1000  # Future time

        metrics = writer.get_metrics()
        assert metrics is not None
