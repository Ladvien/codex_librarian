"""
Server-Sent Events (SSE) implementation for real-time progress streaming.

This module provides comprehensive SSE functionality for streaming
processing progress, job status updates, and other real-time events.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pdf_to_markdown_mcp.models.response import JobStatus

logger = logging.getLogger(__name__)


class SSEEventType(str, Enum):
    """Types of Server-Sent Events."""

    PROGRESS = "progress"
    STATUS = "status"
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    METADATA = "metadata"


@dataclass
class SSEEvent:
    """Server-Sent Event data structure."""

    event: str
    data: dict[str, Any]
    event_id: str | None = None
    retry: int | None = None  # Retry timeout in milliseconds

    def to_sse_format(self) -> str:
        """Format event as SSE data."""
        lines = []

        if self.event_id:
            lines.append(f"id: {self.event_id}")

        if self.retry:
            lines.append(f"retry: {self.retry}")

        lines.append(f"event: {self.event}")

        # Handle multiline data
        data_json = json.dumps(self.data)
        for line in data_json.split("\n"):
            lines.append(f"data: {line}")

        lines.append("")  # Empty line to end event
        return "\n".join(lines) + "\n"


class SSEProgress(BaseModel):
    """Progress update data for SSE streaming."""

    job_id: str = Field(..., description="Job identifier")
    progress_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Progress percentage"
    )
    current_step: str = Field(..., description="Current processing step")
    status: JobStatus = Field(..., description="Job status")

    # Timing information
    started_at: datetime | None = Field(None, description="Job start time")
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional progress metadata"
    )

    class Config:
        use_enum_values = True


class ProgressTracker:
    """Tracks progress for individual jobs and generates SSE events."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.progress_percent = 0.0
        self.current_step = "Initializing"
        self.status = JobStatus.QUEUED
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error_message: str | None = None
        self.metadata: dict[str, Any] = {}
        self.events: list[SSEEvent] = []

    def update_progress(
        self,
        progress_percent: float,
        current_step: str,
        status: JobStatus = JobStatus.RUNNING,
        metadata: dict[str, Any] | None = None,
    ):
        """Update job progress and generate SSE event."""
        self.progress_percent = min(100.0, max(0.0, progress_percent))
        self.current_step = current_step
        self.status = status

        if not self.started_at and status == JobStatus.RUNNING:
            self.started_at = datetime.utcnow()

        if metadata:
            self.metadata.update(metadata)

        # Calculate ETA if we have progress
        estimated_completion = None
        if self.started_at and progress_percent > 0:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            total_estimated = (elapsed / progress_percent) * 100
            remaining = total_estimated - elapsed
            estimated_completion = datetime.utcnow() + timedelta(seconds=remaining)

        # Create SSE event
        event = SSEEvent(
            event=SSEEventType.PROGRESS,
            data={
                "job_id": self.job_id,
                "progress_percent": self.progress_percent,
                "current_step": self.current_step,
                "status": status.value,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "estimated_completion": (
                    estimated_completion.isoformat() if estimated_completion else None
                ),
                "metadata": self.metadata.copy(),
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        self.events.append(event)
        return event

    def complete(self, message: str = "Processing completed successfully"):
        """Mark job as complete."""
        self.completed_at = datetime.utcnow()
        self.progress_percent = 100.0
        self.current_step = message
        self.status = JobStatus.COMPLETED

        event = SSEEvent(
            event=SSEEventType.COMPLETE,
            data={
                "job_id": self.job_id,
                "status": JobStatus.COMPLETED.value,
                "message": message,
                "completed_at": self.completed_at.isoformat(),
                "total_time_seconds": (
                    (self.completed_at - self.started_at).total_seconds()
                    if self.started_at
                    else None
                ),
                "metadata": self.metadata.copy(),
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        self.events.append(event)
        return event

    def error(self, error_message: str):
        """Mark job as failed with error."""
        self.error_message = error_message
        self.status = JobStatus.FAILED
        self.current_step = f"Error: {error_message}"

        event = SSEEvent(
            event=SSEEventType.ERROR,
            data={
                "job_id": self.job_id,
                "status": JobStatus.FAILED.value,
                "error_message": error_message,
                "metadata": self.metadata.copy(),
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        self.events.append(event)
        return event

    def add_metadata(self, metadata: dict[str, Any]):
        """Add metadata to the tracker."""
        self.metadata.update(metadata)

        # Generate metadata event
        event = SSEEvent(
            event=SSEEventType.METADATA,
            data={
                "job_id": self.job_id,
                "metadata": self.metadata.copy(),
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        self.events.append(event)
        return event

    def to_sse_progress(self) -> SSEProgress:
        """Convert to SSEProgress model."""
        return SSEProgress(
            job_id=self.job_id,
            progress_percent=self.progress_percent,
            current_step=self.current_step,
            status=self.status,
            started_at=self.started_at,
            metadata=self.metadata.copy(),
        )


class ProgressStream:
    """Manages SSE streaming for a single job."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.subscribers: int = 0
        self.is_complete = False
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Subscribe to progress events for this job."""
        self.subscribers += 1

        try:
            # Send initial heartbeat
            heartbeat_event = SSEEvent(
                event=SSEEventType.HEARTBEAT,
                data={
                    "job_id": self.job_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                event_id=str(uuid.uuid4()),
            )
            yield heartbeat_event.to_sse_format()

            # Stream events from queue
            while not self._closed:
                try:
                    # Wait for event with timeout for heartbeat
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=30.0)
                    yield event.to_sse_format()

                    # Check if this was a completion event
                    if event.event in [SSEEventType.COMPLETE, SSEEventType.ERROR]:
                        break

                except TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat = SSEEvent(
                        event=SSEEventType.HEARTBEAT,
                        data={
                            "job_id": self.job_id,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        retry=30000,  # 30 seconds
                    )
                    yield heartbeat.to_sse_format()

        except Exception as e:
            logger.exception(f"Error in progress stream for job {self.job_id}")
            error_event = SSEEvent(
                event=SSEEventType.ERROR,
                data={
                    "job_id": self.job_id,
                    "error": "Stream error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            yield error_event.to_sse_format()

        finally:
            self.subscribers -= 1
            logger.info(f"Subscriber disconnected from job {self.job_id} stream")

    def add_subscriber(self, generator: AsyncGenerator):
        """Add a subscriber to the stream."""
        self.subscribers += 1

    async def emit_progress(
        self,
        progress_percent: float,
        current_step: str,
        status: JobStatus = JobStatus.RUNNING,
        metadata: dict[str, Any] | None = None,
    ):
        """Emit a progress event to all subscribers."""
        if self._closed:
            return

        event = SSEEvent(
            event=SSEEventType.PROGRESS,
            data={
                "job_id": self.job_id,
                "progress_percent": progress_percent,
                "current_step": current_step,
                "status": status.value,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        await self.event_queue.put(event)

    async def emit_complete(self, message: str = "Processing completed"):
        """Emit completion event and close stream."""
        if self._closed:
            return

        event = SSEEvent(
            event=SSEEventType.COMPLETE,
            data={
                "job_id": self.job_id,
                "status": JobStatus.COMPLETED.value,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        await self.event_queue.put(event)
        self.is_complete = True
        self._closed = True

    async def emit_error(self, error_message: str):
        """Emit error event and close stream."""
        if self._closed:
            return

        event = SSEEvent(
            event=SSEEventType.ERROR,
            data={
                "job_id": self.job_id,
                "status": JobStatus.FAILED.value,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            },
            event_id=str(uuid.uuid4()),
        )

        await self.event_queue.put(event)
        self.is_complete = True
        self._closed = True

    def close(self):
        """Close the stream."""
        self._closed = True


class JobProgressMonitor:
    """Global monitor for managing multiple job progress streams."""

    def __init__(self):
        self.active_streams: dict[str, ProgressStream] = {}
        self._cleanup_task: asyncio.Task | None = None

    def get_or_create_stream(self, job_id: str) -> ProgressStream:
        """Get existing stream or create a new one."""
        if job_id not in self.active_streams:
            self.active_streams[job_id] = ProgressStream(job_id)
            logger.info(f"Created progress stream for job {job_id}")

        return self.active_streams[job_id]

    async def broadcast_progress(
        self,
        job_id: str,
        progress_percent: float,
        current_step: str,
        status: JobStatus = JobStatus.RUNNING,
        metadata: dict[str, Any] | None = None,
    ):
        """Broadcast progress update to job stream."""
        stream = self.get_or_create_stream(job_id)
        await stream.emit_progress(progress_percent, current_step, status, metadata)

    async def complete_job(self, job_id: str, message: str = "Processing completed"):
        """Mark job as complete and close its stream."""
        if job_id in self.active_streams:
            await self.active_streams[job_id].emit_complete(message)

    async def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed and close its stream."""
        if job_id in self.active_streams:
            await self.active_streams[job_id].emit_error(error_message)

    async def cleanup_completed_streams(self):
        """Remove completed streams to free memory."""
        to_remove = [
            job_id
            for job_id, stream in self.active_streams.items()
            if stream.is_complete and stream.subscribers == 0
        ]

        for job_id in to_remove:
            stream = self.active_streams.pop(job_id)
            stream.close()
            logger.info(f"Cleaned up completed stream for job {job_id}")

    def get_active_job_count(self) -> int:
        """Get number of active job streams."""
        return len(self.active_streams)

    def get_subscriber_count(self) -> int:
        """Get total number of active subscribers across all streams."""
        return sum(stream.subscribers for stream in self.active_streams.values())


# Global progress monitor instance
progress_monitor = JobProgressMonitor()


def format_sse_data(
    data: dict[str, Any],
    event_type: str | None = None,
    event_id: str | None = None,
    retry_ms: int | None = None,
) -> str:
    """Format data as Server-Sent Event string."""
    lines = []

    if event_id:
        lines.append(f"id: {event_id}")

    if retry_ms:
        lines.append(f"retry: {retry_ms}")

    if event_type:
        lines.append(f"event: {event_type}")

    # Handle multiline JSON data
    data_json = json.dumps(data)
    for line in data_json.split("\n"):
        lines.append(f"data: {line}")

    lines.append("")  # Empty line to end event
    return "\n".join(lines) + "\n"


def create_sse_response(
    event_generator: AsyncGenerator[str, None],
) -> StreamingResponse:
    """Create a StreamingResponse for Server-Sent Events."""
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


async def create_job_progress_stream(job_id: str) -> AsyncGenerator[str, None]:
    """Create a progress stream for a specific job."""
    stream = progress_monitor.get_or_create_stream(job_id)

    async for event_data in stream.subscribe():
        yield event_data
