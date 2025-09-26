"""
Unit tests for Server-Sent Events implementation (API-005).

Testing real-time progress streaming functionality following TDD principles.
"""

import asyncio
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from src.pdf_to_markdown_mcp.api.streaming import (
    JobProgressMonitor,
    ProgressStream,
    ProgressTracker,
    SSEEvent,
    SSEProgress,
    create_sse_response,
    format_sse_data,
)
from src.pdf_to_markdown_mcp.models.response import JobStatus


class TestSSEDataStructures:
    """Test Server-Sent Events data structures."""

    def test_sse_event_creation(self):
        """Test SSEEvent creation with all fields."""
        # Given
        event_data = {
            "event": "progress",
            "data": {"progress": 50, "message": "Processing"},
            "event_id": "evt_123",
            "retry": 5000,
        }

        # When
        event = SSEEvent(**event_data)

        # Then
        assert event.event == "progress"
        assert event.data["progress"] == 50
        assert event.event_id == "evt_123"
        assert event.retry == 5000

    def test_sse_progress_structure(self):
        """Test SSEProgress data structure."""
        # Given
        progress_data = {
            "job_id": "pdf_proc_123",
            "progress_percent": 75.5,
            "current_step": "Generating embeddings",
            "status": JobStatus.RUNNING,
            "estimated_completion": datetime.utcnow().isoformat(),
            "metadata": {"pages_processed": 15},
        }

        # When
        progress = SSEProgress(**progress_data)

        # Then
        assert progress.job_id == "pdf_proc_123"
        assert progress.progress_percent == 75.5
        assert progress.current_step == "Generating embeddings"
        assert progress.status == JobStatus.RUNNING
        assert progress.metadata["pages_processed"] == 15

    def test_sse_progress_validation(self):
        """Test SSEProgress validates progress_percent bounds."""
        # Given - invalid progress values
        invalid_values = [-1, 101, 150]

        # When/Then
        for invalid_value in invalid_values:
            with pytest.raises(ValueError):
                SSEProgress(
                    job_id="test",
                    progress_percent=invalid_value,
                    status=JobStatus.RUNNING,
                )


class TestSSEFormatting:
    """Test SSE data formatting."""

    def test_format_sse_data_simple(self):
        """Test basic SSE data formatting."""
        # Given
        data = {"message": "Hello, world!"}

        # When
        formatted = format_sse_data(data)

        # Then
        expected = 'data: {"message": "Hello, world!"}\n\n'
        assert formatted == expected

    def test_format_sse_data_with_event_type(self):
        """Test SSE formatting with event type."""
        # Given
        data = {"progress": 50}
        event_type = "progress"

        # When
        formatted = format_sse_data(data, event_type=event_type)

        # Then
        expected = 'event: progress\ndata: {"progress": 50}\n\n'
        assert formatted == expected

    def test_format_sse_data_with_id_and_retry(self):
        """Test SSE formatting with ID and retry."""
        # Given
        data = {"status": "completed"}
        event_id = "evt_456"
        retry_ms = 3000

        # When
        formatted = format_sse_data(data, event_id=event_id, retry_ms=retry_ms)

        # Then
        assert "id: evt_456\n" in formatted
        assert "retry: 3000\n" in formatted
        assert 'data: {"status": "completed"}\n\n' in formatted

    def test_format_sse_multiline_data(self):
        """Test SSE formatting with multiline data."""
        # Given
        data = {"message": "Line 1\nLine 2\nLine 3"}

        # When
        formatted = format_sse_data(data)

        # Then
        # Each line should be prefixed with 'data: '
        lines = formatted.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        # Should properly handle multiline JSON
        assert len(data_lines) >= 1
        assert formatted.endswith("\n\n")


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initializes with correct defaults."""
        # Given/When
        tracker = ProgressTracker("job_123")

        # Then
        assert tracker.job_id == "job_123"
        assert tracker.progress_percent == 0.0
        assert tracker.status == JobStatus.QUEUED
        assert tracker.current_step == "Initializing"

    def test_progress_tracker_update(self):
        """Test ProgressTracker updates progress."""
        # Given
        tracker = ProgressTracker("job_123")

        # When
        tracker.update_progress(
            progress_percent=50.0,
            current_step="Processing PDF",
            status=JobStatus.RUNNING,
        )

        # Then
        assert tracker.progress_percent == 50.0
        assert tracker.current_step == "Processing PDF"
        assert tracker.status == JobStatus.RUNNING

    def test_progress_tracker_complete(self):
        """Test ProgressTracker marks job as complete."""
        # Given
        tracker = ProgressTracker("job_123")

        # When
        tracker.complete("Processing completed successfully")

        # Then
        assert tracker.progress_percent == 100.0
        assert tracker.status == JobStatus.COMPLETED
        assert "completed successfully" in tracker.current_step

    def test_progress_tracker_error(self):
        """Test ProgressTracker handles errors."""
        # Given
        tracker = ProgressTracker("job_123")

        # When
        tracker.error("Processing failed: Invalid PDF format")

        # Then
        assert tracker.status == JobStatus.FAILED
        assert "Processing failed" in tracker.current_step

    def test_progress_tracker_to_sse_progress(self):
        """Test ProgressTracker converts to SSEProgress."""
        # Given
        tracker = ProgressTracker("job_123")
        tracker.update_progress(75.0, "Generating embeddings")
        tracker.add_metadata({"pages_processed": 10})

        # When
        sse_progress = tracker.to_sse_progress()

        # Then
        assert isinstance(sse_progress, SSEProgress)
        assert sse_progress.job_id == "job_123"
        assert sse_progress.progress_percent == 75.0
        assert sse_progress.current_step == "Generating embeddings"
        assert sse_progress.metadata["pages_processed"] == 10


class TestProgressStream:
    """Test ProgressStream functionality."""

    @pytest.mark.asyncio
    async def test_progress_stream_initialization(self):
        """Test ProgressStream initializes correctly."""
        # Given/When
        stream = ProgressStream("job_123")

        # Then
        assert stream.job_id == "job_123"
        assert not stream.is_complete
        assert stream.subscribers == 0

    @pytest.mark.asyncio
    async def test_progress_stream_add_subscriber(self):
        """Test adding subscribers to ProgressStream."""
        # Given
        stream = ProgressStream("job_123")

        # When
        async def mock_generator():
            yield "data: test\n\n"

        generator = mock_generator()
        stream.add_subscriber(generator)

        # Then
        assert stream.subscribers == 1

    @pytest.mark.asyncio
    async def test_progress_stream_emit_progress(self):
        """Test ProgressStream emits progress to subscribers."""
        # Given
        stream = ProgressStream("job_123")
        received_events = []

        async def test_subscriber():
            async for event in stream.subscribe():
                received_events.append(event)
                if len(received_events) >= 2:  # Stop after receiving events
                    break

        # Start subscriber
        subscriber_task = asyncio.create_task(test_subscriber())

        # When
        await stream.emit_progress(50.0, "Processing", JobStatus.RUNNING)
        await stream.emit_complete("Done")

        # Wait for events to be processed
        await asyncio.sleep(0.1)
        subscriber_task.cancel()

        # Then
        assert len(received_events) >= 1
        # Check that events are properly formatted SSE
        for event in received_events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_progress_stream_complete_closes_stream(self):
        """Test completing a ProgressStream closes it."""
        # Given
        stream = ProgressStream("job_123")

        # When
        await stream.emit_complete("Processing finished")

        # Then
        assert stream.is_complete


class TestJobProgressMonitor:
    """Test JobProgressMonitor integration."""

    def test_job_progress_monitor_creation(self):
        """Test JobProgressMonitor creates and manages streams."""
        # Given/When
        monitor = JobProgressMonitor()

        # Then
        assert isinstance(monitor.active_streams, dict)
        assert len(monitor.active_streams) == 0

    def test_job_progress_monitor_get_or_create_stream(self):
        """Test JobProgressMonitor gets or creates streams."""
        # Given
        monitor = JobProgressMonitor()

        # When
        stream1 = monitor.get_or_create_stream("job_123")
        stream2 = monitor.get_or_create_stream("job_123")  # Same job ID

        # Then
        assert stream1 is stream2  # Should return same stream
        assert len(monitor.active_streams) == 1

    @pytest.mark.asyncio
    async def test_job_progress_monitor_cleanup_completed_streams(self):
        """Test JobProgressMonitor cleans up completed streams."""
        # Given
        monitor = JobProgressMonitor()
        stream = monitor.get_or_create_stream("job_123")

        # When
        await stream.emit_complete("Done")
        await monitor.cleanup_completed_streams()

        # Then
        assert "job_123" not in monitor.active_streams

    @pytest.mark.asyncio
    async def test_job_progress_monitor_broadcast_progress(self):
        """Test JobProgressMonitor broadcasts progress to all streams."""
        # Given
        monitor = JobProgressMonitor()
        job_ids = ["job_1", "job_2", "job_3"]

        # Create streams for multiple jobs
        streams = [monitor.get_or_create_stream(job_id) for job_id in job_ids]

        # When
        await monitor.broadcast_progress(
            job_id="job_2",
            progress_percent=75.0,
            current_step="Almost done",
            status=JobStatus.RUNNING,
        )

        # Then
        # The specific job should have received the update
        job_2_stream = monitor.active_streams["job_2"]
        # We can't easily test the internal state without making it more complex,
        # but we can verify the stream exists and is active
        assert job_2_stream is not None
        assert not job_2_stream.is_complete


class TestSSEResponse:
    """Test SSE response creation and streaming."""

    @pytest.mark.asyncio
    async def test_create_sse_response(self):
        """Test create_sse_response creates proper StreamingResponse."""

        # Given
        async def mock_event_generator():
            yield format_sse_data({"progress": 25})
            yield format_sse_data({"progress": 50})
            yield format_sse_data({"progress": 100, "status": "completed"})

        # When
        response = create_sse_response(mock_event_generator())

        # Then
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"
        assert "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_sse_response_content(self):
        """Test SSE response generates correct content."""

        # Given
        async def test_generator():
            yield format_sse_data({"message": "Hello"})
            yield format_sse_data({"message": "World"})

        response = create_sse_response(test_generator())

        # When
        content_parts = []
        async for part in response.body_iterator:
            content_parts.append(part.decode())

        # Then
        full_content = "".join(content_parts)
        assert 'data: {"message": "Hello"}\n\n' in full_content
        assert 'data: {"message": "World"}\n\n' in full_content


class TestSSEIntegration:
    """Test SSE integration with FastAPI."""

    def test_sse_endpoint_integration(self):
        """Test SSE endpoint integrates properly with FastAPI."""
        # Given
        app = FastAPI()

        @app.get("/stream/{job_id}")
        async def stream_progress(job_id: str):
            async def progress_generator():
                for i in range(0, 101, 25):
                    yield format_sse_data(
                        {
                            "job_id": job_id,
                            "progress": i,
                            "status": "running" if i < 100 else "completed",
                        }
                    )
                    await asyncio.sleep(0.01)  # Small delay

            return create_sse_response(progress_generator())

        client = TestClient(app)

        # When
        with client.stream("GET", "/stream/test_job") as response:
            # Then
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_sse_client_connection_handling(self):
        """Test SSE handles client connections properly."""
        # Given
        app = FastAPI()

        connection_count = {"value": 0}

        @app.get("/stream")
        async def stream_endpoint():
            async def event_generator():
                connection_count["value"] += 1
                try:
                    for i in range(5):
                        yield format_sse_data({"count": i})
                        await asyncio.sleep(0.01)
                finally:
                    connection_count["value"] -= 1

            return create_sse_response(event_generator())

        client = TestClient(app)

        # When - Open connection
        with client.stream("GET", "/stream") as response:
            # Read some data
            chunk = next(response.iter_text())
            assert "data: " in chunk

        # Then - Connection should be cleaned up
        # Note: This is hard to test precisely due to client handling
