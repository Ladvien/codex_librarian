"""
Unit tests for Celery configuration and application setup.

Tests the Celery app configuration, queue setup, and task routing.
"""

from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.worker.celery import (
    CallbackTask,
    _setup_queues,
    _setup_task_routes,
    create_celery_app,
    get_queue_length,
    get_worker_stats,
)


class TestCeleryAppConfiguration:
    """Test Celery application configuration."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.celery.broker_url = "redis://localhost:6379/0"
        settings.celery.result_backend = "redis://localhost:6379/0"
        settings.celery.worker_concurrency = 4
        settings.celery.task_soft_time_limit = 300
        settings.celery.task_time_limit = 600
        return settings

    @patch("pdf_to_markdown_mcp.worker.celery.settings")
    def test_create_celery_app_basic_config(self, mock_settings_module):
        """Test that Celery app is created with correct basic configuration."""
        # Given
        mock_settings_module.celery.broker_url = "redis://localhost:6379/0"
        mock_settings_module.celery.result_backend = "redis://localhost:6379/0"
        mock_settings_module.celery.worker_concurrency = 4
        mock_settings_module.celery.task_soft_time_limit = 300
        mock_settings_module.celery.task_time_limit = 600

        # When
        app = create_celery_app()

        # Then
        assert app.conf.broker_url == "redis://localhost:6379/0"
        assert app.conf.result_backend == "redis://localhost:6379/0"
        assert app.conf.task_serializer == "json"
        assert app.conf.accept_content == ["json"]
        assert app.conf.result_serializer == "json"
        assert app.conf.timezone == "UTC"
        assert app.conf.enable_utc is True

    def test_setup_task_routes(self):
        """Test that task routes are configured correctly."""
        # When
        routes = _setup_task_routes()

        # Then
        assert "pdf_to_markdown_mcp.worker.tasks.process_pdf_document" in routes
        assert "pdf_to_markdown_mcp.worker.tasks.generate_embeddings" in routes
        assert "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files" in routes
        assert "pdf_to_markdown_mcp.worker.tasks.health_check" in routes

        # Check specific routing
        pdf_route = routes["pdf_to_markdown_mcp.worker.tasks.process_pdf_document"]
        assert pdf_route["queue"] == "pdf_processing"
        assert pdf_route["routing_key"] == "pdf_processing"

        embed_route = routes["pdf_to_markdown_mcp.worker.tasks.generate_embeddings"]
        assert embed_route["queue"] == "embeddings"

    def test_setup_queues(self):
        """Test that queues are configured with correct priorities."""
        # When
        queues = _setup_queues()

        # Then
        assert len(queues) == 4
        queue_names = [q.name for q in queues]
        assert "pdf_processing" in queue_names
        assert "embeddings" in queue_names
        assert "maintenance" in queue_names
        assert "monitoring" in queue_names

        # Find PDF processing queue and check priority
        pdf_queue = next(q for q in queues if q.name == "pdf_processing")
        assert pdf_queue.queue_arguments["x-max-priority"] == 10

    @patch("pdf_to_markdown_mcp.worker.celery.settings")
    def test_celery_app_includes_beat_schedule(self, mock_settings_module):
        """Test that beat scheduler is configured."""
        # Given
        mock_settings_module.celery.broker_url = "redis://localhost:6379/0"
        mock_settings_module.celery.result_backend = "redis://localhost:6379/0"
        mock_settings_module.celery.worker_concurrency = 4
        mock_settings_module.celery.task_soft_time_limit = 300
        mock_settings_module.celery.task_time_limit = 600

        # When
        app = create_celery_app()

        # Then
        assert "cleanup-temp-files" in app.conf.beat_schedule
        assert "health-check" in app.conf.beat_schedule
        cleanup_task = app.conf.beat_schedule["cleanup-temp-files"]
        assert (
            cleanup_task["task"]
            == "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files"
        )
        assert cleanup_task["schedule"] == 3600.0


class TestCallbackTask:
    """Test custom CallbackTask class."""

    @pytest.fixture
    def callback_task(self):
        """Create a CallbackTask instance for testing."""
        task = CallbackTask()
        task.name = "test_task"
        task.request = Mock()
        task.request.id = "test_task_id"
        return task

    def test_update_state_progress_metadata(self, callback_task):
        """Test that progress state updates include required metadata."""
        # Given
        callback_task.update_state = Mock()

        # When
        callback_task.update_state(state="PROGRESS", meta={"current": 5, "total": 10})

        # Then
        callback_task.update_state.assert_called_once()
        call_args = callback_task.update_state.call_args
        assert call_args[1]["state"] == "PROGRESS"
        meta = call_args[1]["meta"]
        assert meta["current"] == 5
        assert meta["total"] == 10
        assert "message" in meta
        assert "timestamp" in meta

    def test_categorize_error_transient(self, callback_task):
        """Test that transient errors are categorized correctly."""
        # Given
        connection_error = ConnectionError("Redis connection failed")

        # When
        category = callback_task._categorize_error(connection_error)

        # Then
        assert category == "transient"

    def test_categorize_error_validation(self, callback_task):
        """Test that validation errors are categorized correctly."""
        # Given
        from pdf_to_markdown_mcp.core.exceptions import ValidationError

        validation_error = ValidationError("Invalid input")

        # When
        category = callback_task._categorize_error(validation_error)

        # Then
        assert category == "validation"

    def test_categorize_error_unknown(self, callback_task):
        """Test that unknown errors are categorized as unknown."""
        # Given
        unknown_error = RuntimeError("Something unexpected")

        # When
        category = callback_task._categorize_error(unknown_error)

        # Then
        assert category == "unknown"

    def test_on_failure_logs_error(self, callback_task):
        """Test that task failures are logged with context."""
        # Given
        callback_task.update_state = Mock()
        exc = Exception("Test error")
        task_id = "test_task_id"
        args = ["arg1", "arg2"]
        kwargs = {"key": "value"}

        with patch("pdf_to_markdown_mcp.worker.celery.logger") as mock_logger:
            # When
            callback_task.on_failure(exc, task_id, args, kwargs, None)

            # Then
            mock_logger.error.assert_called_once()
            callback_task.update_state.assert_called_once_with(
                task_id=task_id,
                state="FAILURE",
                meta={
                    "error": str(exc),
                    "error_category": "unknown",
                    "task_args": args,
                    "task_kwargs": kwargs,
                },
            )


class TestWorkerStats:
    """Test worker statistics and monitoring functions."""

    @patch("pdf_to_markdown_mcp.worker.celery.app")
    def test_get_worker_stats_success(self, mock_app):
        """Test successful worker stats retrieval."""
        # Given
        mock_inspect = Mock()
        mock_inspect.active.return_value = {"worker1": []}
        mock_inspect.scheduled.return_value = {"worker1": []}
        mock_inspect.reserved.return_value = {"worker1": []}
        mock_inspect.stats.return_value = {"worker1": {"total": 10}}

        mock_app.control.inspect.return_value = mock_inspect
        mock_app.conf.task_queues = ["queue1", "queue2"]
        mock_app.tasks.keys.return_value = ["task1", "task2"]

        # When
        stats = get_worker_stats()

        # Then
        assert "active_queues" in stats
        assert "registered_tasks" in stats
        assert "active_tasks" in stats
        assert "scheduled_tasks" in stats
        assert "reserved_tasks" in stats
        assert "worker_stats" in stats
        assert stats["active_queues"] == ["queue1", "queue2"]
        assert stats["registered_tasks"] == ["task1", "task2"]

    @patch("pdf_to_markdown_mcp.worker.celery.app")
    def test_get_worker_stats_error(self, mock_app):
        """Test worker stats retrieval error handling."""
        # Given
        mock_app.control.inspect.side_effect = Exception("Inspection failed")

        # When
        stats = get_worker_stats()

        # Then
        assert "error" in stats
        assert stats["error"] == "Inspection failed"

    @patch("pdf_to_markdown_mcp.worker.celery.app")
    def test_get_queue_length_success(self, mock_app):
        """Test successful queue length retrieval."""
        # Given
        mock_conn = Mock()
        mock_channel = Mock()
        mock_client = Mock()
        mock_client.llen.return_value = 5

        mock_channel.client = mock_client
        mock_conn.default_channel = mock_channel
        mock_app.connection.return_value.__enter__.return_value = mock_conn

        # When
        length = get_queue_length("test_queue")

        # Then
        assert length == 5
        mock_client.llen.assert_called_once_with("celery.test_queue")

    @patch("pdf_to_markdown_mcp.worker.celery.app")
    def test_get_queue_length_error(self, mock_app):
        """Test queue length retrieval error handling."""
        # Given
        mock_app.connection.side_effect = Exception("Connection failed")

        # When
        length = get_queue_length("test_queue")

        # Then
        assert length == -1


class TestCelerySignals:
    """Test Celery signal handlers."""

    def test_signal_handlers_exist(self):
        """Test that required signal handlers are registered."""
        # This test verifies that the signal handlers are defined
        # In a real application, you might want to test their behavior
        from pdf_to_markdown_mcp.worker import celery

        # The handlers should be defined as functions
        assert hasattr(celery, "worker_ready_handler")
        assert hasattr(celery, "worker_shutting_down_handler")
        assert hasattr(celery, "task_prerun_handler")
        assert hasattr(celery, "task_postrun_handler")

    @patch("pdf_to_markdown_mcp.worker.celery.logger")
    def test_worker_ready_handler(self, mock_logger):
        """Test worker ready signal handler."""
        from pdf_to_markdown_mcp.worker.celery import worker_ready_handler

        # When
        worker_ready_handler(sender="test_worker")

        # Then
        mock_logger.info.assert_called_once_with("Celery worker ready: test_worker")

    @patch("pdf_to_markdown_mcp.worker.celery.logger")
    def test_task_prerun_handler_with_timing(self, mock_logger):
        """Test task prerun handler sets start time."""
        from pdf_to_markdown_mcp.worker.celery import task_prerun_handler

        # Given
        mock_task = Mock()
        mock_task.name = "test_task"

        # When
        with patch("time.time", return_value=1234567890):
            task_prerun_handler(
                sender=None, task_id="test_id", task=mock_task, args=(), kwargs={}
            )

        # Then
        mock_logger.info.assert_called_once_with("Starting task test_task [test_id]")
        assert hasattr(mock_task, "start_time")
        assert mock_task.start_time == 1234567890
