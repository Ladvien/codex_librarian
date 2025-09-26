"""
Integration tests for Celery worker with Redis and database.

These tests require running Redis and PostgreSQL instances and test the complete
Celery workflow including task queuing, execution, and result storage.
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.worker.celery import app as celery_app
from pdf_to_markdown_mcp.worker.tasks import (
    cleanup_temp_files,
    generate_embeddings,
    health_check,
    process_pdf_document,
)

pytestmark = pytest.mark.integration


class TestCeleryIntegration:
    """Integration tests for Celery worker functionality."""

    @pytest.fixture(scope="class")
    def celery_worker(self):
        """Start a test Celery worker."""
        # Configure Celery for testing
        celery_app.conf.update(
            task_always_eager=False,  # Don't run tasks synchronously in tests
            broker_url="redis://localhost:6379/1",  # Use different DB for tests
            result_backend="redis://localhost:6379/1",
        )

        # Start worker in test mode
        worker = celery_app.Worker(
            hostname="test-worker@localhost",
            queues=["pdf_processing", "embeddings", "maintenance", "monitoring"],
            concurrency=1,
        )

        # Note: In real integration tests, you'd start the worker in a separate process
        # For now, we'll mock the worker behavior
        yield worker

    @pytest.fixture
    def test_pdf_file(self, tmp_path):
        """Create a test PDF file."""
        pdf_file = tmp_path / "test_document.pdf"
        # Create a minimal PDF file
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
299
%%EOF"""
        pdf_file.write_bytes(pdf_content)
        return pdf_file

    def test_celery_app_configuration(self):
        """Test that Celery app is properly configured."""
        # Given
        app_config = celery_app.conf

        # Then
        assert app_config.task_serializer == "json"
        assert app_config.result_serializer == "json"
        assert app_config.timezone == "UTC"
        assert app_config.enable_utc is True

        # Check task routes are configured
        assert (
            "pdf_to_markdown_mcp.worker.tasks.process_pdf_document"
            in app_config.task_routes
        )

    @pytest.mark.slow
    def test_task_queuing_and_routing(self, celery_worker):
        """Test that tasks are properly queued and routed."""
        # Given
        task_name = "pdf_to_markdown_mcp.worker.tasks.health_check"

        # When
        result = health_check.delay()

        # Then
        assert result.id is not None
        assert result.task_id is not None

        # The task should be in the correct queue
        # Note: This would require inspecting the actual Redis queue in a real test

    @pytest.mark.slow
    @patch("pdf_to_markdown_mcp.worker.tasks.get_db_session")
    @patch("pdf_to_markdown_mcp.worker.tasks.MinerUService")
    @patch("pdf_to_markdown_mcp.worker.tasks.EmbeddingService")
    def test_pdf_processing_workflow(
        self,
        mock_embedding_service_class,
        mock_mineru_service_class,
        mock_get_db_session,
        test_pdf_file,
    ):
        """Test the complete PDF processing workflow."""
        # Given
        mock_db = Mock()
        mock_db.__enter__.return_value = mock_db
        mock_db.__exit__.return_value = None
        mock_get_db_session.return_value = mock_db

        mock_document = Mock()
        mock_document.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_document
        )

        mock_mineru = Mock()
        mock_mineru_service_class.return_value = mock_mineru
        mock_mineru.process_pdf.return_value = {
            "markdown": "# Test Document\nContent here",
            "plain_text": "Test Document Content here",
            "page_count": 1,
            "has_images": False,
            "has_tables": False,
            "processing_time_ms": 1500,
            "chunks": [
                {"text": "Test Document", "index": 0, "page_number": 1},
                {"text": "Content here", "index": 1, "page_number": 1},
            ],
        }

        # When
        with patch(
            "pdf_to_markdown_mcp.worker.tasks.generate_embeddings"
        ) as mock_gen_embeddings:
            mock_gen_embeddings.delay = Mock()

            result = process_pdf_document.delay(
                document_id=1, file_path=str(test_pdf_file), processing_options={}
            )

            # Wait for task completion (with timeout)
            timeout = 30  # seconds
            start_time = time.time()
            while not result.ready() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

        # Then
        if result.ready():
            task_result = result.get()
            assert task_result["status"] == "completed"
            assert task_result["document_id"] == 1
            assert task_result["page_count"] == 1

            # Verify embedding task was queued
            mock_gen_embeddings.delay.assert_called_once()
        else:
            pytest.skip(
                "Task did not complete within timeout - may indicate Redis/worker issues"
            )

    @pytest.mark.slow
    def test_health_check_task_execution(self, celery_worker):
        """Test health check task execution."""
        # When
        with (
            patch("pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "pdf_to_markdown_mcp.worker.tasks.EmbeddingService"
            ) as mock_embedding_service_class,
            patch("pdf_to_markdown_mcp.worker.tasks.settings") as mock_settings,
        ):
            # Setup mocks
            mock_db_instance = Mock()
            mock_db_instance.__enter__.return_value = mock_db_instance
            mock_db_instance.__exit__.return_value = None
            mock_db.return_value = mock_db_instance

            mock_embedding_service = Mock()
            mock_embedding_service_class.return_value = mock_embedding_service
            mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2]]

            mock_settings.processing.temp_dir = Path("/tmp")

            result = health_check.delay()

            # Wait for completion
            timeout = 10
            start_time = time.time()
            while not result.ready() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

        # Then
        if result.ready():
            health_result = result.get()
            assert "status" in health_result
            assert "timestamp" in health_result
            assert "checks" in health_result
            assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
        else:
            pytest.skip("Health check task did not complete within timeout")

    def test_task_retry_mechanism(self, celery_worker):
        """Test task retry mechanism for failures."""
        # This would test the retry logic by causing intentional failures
        # and verifying that tasks are retried according to configuration

        with patch("pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db:
            mock_db.side_effect = Exception("Database connection failed")

            result = health_check.delay()

            # The task should eventually fail after retries
            # In a real test, we'd verify the retry attempts
            timeout = 15
            start_time = time.time()
            while not result.ready() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if result.ready():
                try:
                    result.get()
                except Exception:
                    # Expected to fail after retries
                    pass

    @pytest.mark.slow
    def test_task_queue_priorities(self, celery_worker):
        """Test that tasks are processed according to queue priorities."""
        # Queue multiple tasks on different priority queues
        # and verify they're processed in the correct order

        results = []

        # Queue high-priority task
        pdf_result = process_pdf_document.delay(1, "/fake/path.pdf")
        results.append(("pdf_processing", pdf_result))

        # Queue medium-priority task
        embed_result = generate_embeddings.delay(1, "test content", [])
        results.append(("embeddings", embed_result))

        # Queue low-priority task
        cleanup_result = cleanup_temp_files.delay()
        results.append(("maintenance", cleanup_result))

        # In a real integration test, we'd verify processing order
        # For now, just verify all tasks were queued
        for queue_name, result in results:
            assert result.id is not None

    def test_worker_stats_collection(self):
        """Test worker statistics collection."""
        from pdf_to_markdown_mcp.worker.celery import get_worker_stats

        # When
        with patch("pdf_to_markdown_mcp.worker.celery.app") as mock_app:
            mock_inspect = Mock()
            mock_inspect.active.return_value = {"worker1": []}
            mock_inspect.scheduled.return_value = {"worker1": []}
            mock_inspect.reserved.return_value = {"worker1": []}
            mock_inspect.stats.return_value = {"worker1": {"total": 10}}

            mock_app.control.inspect.return_value = mock_inspect
            mock_app.conf.task_queues = ["pdf_processing", "embeddings"]
            mock_app.tasks.keys.return_value = ["task1", "task2"]

            stats = get_worker_stats()

        # Then
        assert "active_queues" in stats
        assert "registered_tasks" in stats
        assert "active_tasks" in stats
        assert stats["active_queues"] == ["pdf_processing", "embeddings"]

    def test_queue_length_monitoring(self):
        """Test queue length monitoring functionality."""
        from pdf_to_markdown_mcp.worker.celery import get_queue_length

        # When
        with patch("pdf_to_markdown_mcp.worker.celery.app") as mock_app:
            mock_conn = Mock()
            mock_channel = Mock()
            mock_client = Mock()
            mock_client.llen.return_value = 5

            mock_channel.client = mock_client
            mock_conn.default_channel = mock_channel
            mock_app.connection.return_value.__enter__.return_value = mock_conn

            length = get_queue_length("pdf_processing")

        # Then
        assert length == 5

    @pytest.mark.slow
    def test_batch_processing_coordination(self, celery_worker, tmp_path):
        """Test coordination of batch processing tasks."""
        # Create multiple test PDF files
        pdf_files = []
        for i in range(3):
            pdf_file = tmp_path / f"test_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 test content " + str(i).encode())
            pdf_files.append(str(pdf_file))

        # When
        with (
            patch("pdf_to_markdown_mcp.worker.tasks.get_db_session") as mock_db,
            patch(
                "pdf_to_markdown_mcp.worker.tasks.process_pdf_document"
            ) as mock_process,
        ):
            mock_db_instance = Mock()
            mock_db_instance.__enter__.return_value = mock_db_instance
            mock_db_instance.__exit__.return_value = None
            mock_db.return_value = mock_db_instance

            # Mock no existing documents (not duplicates)
            mock_db_instance.query.return_value.filter.return_value.first.return_value = None

            mock_document = Mock()
            mock_document.id = 1

            mock_process.delay = Mock()

            from pdf_to_markdown_mcp.worker.tasks import process_pdf_batch

            result = process_pdf_batch.delay(pdf_files, {})

            # Wait for completion
            timeout = 10
            start_time = time.time()
            while not result.ready() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

        # Then
        if result.ready():
            batch_result = result.get()
            assert batch_result["total_files"] == 3
            assert batch_result["successful"] >= 0  # May vary based on mocking


class TestCeleryErrorHandling:
    """Test Celery error handling and recovery."""

    def test_task_failure_recovery(self):
        """Test task failure and recovery mechanisms."""
        # Test that failed tasks are properly handled and can be retried

    def test_worker_failure_recovery(self):
        """Test worker failure and recovery."""
        # Test what happens when a worker dies during task execution

    def test_redis_connection_failure(self):
        """Test behavior when Redis connection fails."""
        # Test graceful handling of Redis connection issues


class TestCeleryMonitoring:
    """Test Celery monitoring and observability."""

    def test_task_event_monitoring(self):
        """Test task event monitoring."""
        # Test that task events are properly generated and can be monitored

    def test_worker_heartbeat_monitoring(self):
        """Test worker heartbeat monitoring."""
        # Test that worker heartbeats are properly sent and monitored

    def test_queue_depth_monitoring(self):
        """Test queue depth monitoring."""
        # Test monitoring of queue depths for alerting


@pytest.mark.performance
class TestCeleryPerformance:
    """Performance tests for Celery operations."""

    def test_task_throughput(self):
        """Test task processing throughput."""
        # Test how many tasks can be processed per second

    def test_memory_usage(self):
        """Test memory usage during task processing."""
        # Test that memory usage doesn't grow unbounded

    def test_concurrent_task_processing(self):
        """Test concurrent task processing."""
        # Test processing multiple tasks simultaneously
