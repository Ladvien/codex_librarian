"""
Streaming and Async I/O utilities for large file processing.

This module provides comprehensive streaming support for handling large PDF files
(up to 500MB) without loading them entirely into memory, with progress tracking,
backpressure handling, and memory monitoring.
"""

import asyncio
import gc
import logging
import mmap
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
)

import psutil

from pdf_to_markdown_mcp.core.exceptions import (
    ProcessingError,
    ResourceError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")
ChunkType = bytes
ProgressCallback = Callable[[int, int, str | None], None]

# Constants for streaming configuration
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB chunks
LARGE_FILE_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for files > 10MB
MAX_MEMORY_USAGE_PERCENT = 75.0  # Max memory usage before backpressure
MEMORY_CHECK_INTERVAL = 100  # Check memory every N chunks
BACKPRESSURE_SLEEP_MS = 50  # Sleep time during backpressure
MAX_CONCURRENT_STREAMS = 5  # Maximum concurrent streaming operations


class StreamingProtocol(Protocol):
    """Protocol for streaming operations."""

    async def read_chunk(self, size: int) -> bytes:
        """Read a chunk of data."""
        ...

    async def write_chunk(self, data: bytes) -> None:
        """Write a chunk of data."""
        ...

    def get_progress(self) -> tuple[int, int]:
        """Get current progress (processed, total)."""
        ...


@dataclass
class StreamingMetrics:
    """Metrics for streaming operations."""

    start_time: float = field(default_factory=time.time)
    bytes_processed: int = 0
    total_bytes: int = 0
    chunks_processed: int = 0
    avg_chunk_size: float = 0.0
    processing_rate_mbps: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    backpressure_events: int = 0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)

    def update(self, bytes_delta: int = 0, chunk_delta: int = 0) -> None:
        """Update metrics with processing progress."""
        self.bytes_processed += bytes_delta
        self.chunks_processed += chunk_delta

        if self.chunks_processed > 0:
            self.avg_chunk_size = self.bytes_processed / self.chunks_processed

        # Calculate processing rate
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.processing_rate_mbps = (self.bytes_processed / (1024 * 1024)) / elapsed

        # Update memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.memory_usage_mb = current_memory
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)

        self.last_update = time.time()

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.bytes_processed / self.total_bytes) * 100.0)

    @property
    def estimated_completion_time(self) -> float | None:
        """Estimate completion time in seconds."""
        if self.processing_rate_mbps <= 0 or self.total_bytes <= 0:
            return None

        remaining_mb = (self.total_bytes - self.bytes_processed) / (1024 * 1024)
        return remaining_mb / self.processing_rate_mbps

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "progress_percent": self.progress_percent,
            "bytes_processed": self.bytes_processed,
            "total_bytes": self.total_bytes,
            "chunks_processed": self.chunks_processed,
            "avg_chunk_size": self.avg_chunk_size,
            "processing_rate_mbps": self.processing_rate_mbps,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "backpressure_events": self.backpressure_events,
            "error_count": self.error_count,
            "elapsed_time": time.time() - self.start_time,
            "estimated_completion": self.estimated_completion_time,
        }


class MemoryMonitor:
    """Monitor memory usage and implement backpressure."""

    def __init__(self, max_memory_percent: float = MAX_MEMORY_USAGE_PERCENT):
        self.max_memory_percent = max_memory_percent
        self._check_count = 0

    def should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        self._check_count += 1

        # Only check memory periodically for performance
        if self._check_count % MEMORY_CHECK_INTERVAL != 0:
            return False

        memory_percent = psutil.virtual_memory().percent
        return memory_percent > self.max_memory_percent

    async def wait_for_memory(self, max_wait_seconds: float = 30.0) -> None:
        """Wait for memory usage to decrease."""
        start_time = time.time()

        while self.should_apply_backpressure():
            if time.time() - start_time > max_wait_seconds:
                raise ResourceError("Memory pressure timeout exceeded")

            # Force garbage collection
            gc.collect()

            # Short async sleep to yield control
            await asyncio.sleep(BACKPRESSURE_SLEEP_MS / 1000.0)

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return {
            "system_memory_percent": memory.percent,
            "system_available_gb": memory.available / (1024**3),
            "process_memory_mb": process.memory_info().rss / (1024**2),
            "max_memory_percent": self.max_memory_percent,
        }


class StreamingBuffer(Generic[T]):
    """Async buffer for streaming operations with backpressure."""

    def __init__(self, max_size: int = 10):
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._closed = False
        self._error: Exception | None = None

    async def put(self, item: T) -> None:
        """Put item into buffer with backpressure."""
        if self._closed:
            raise ValueError("Buffer is closed")

        if self._error:
            raise self._error

        await self._queue.put(item)

    async def get(self) -> T:
        """Get item from buffer."""
        if self._closed and self._queue.empty():
            raise StopAsyncIteration

        if self._error and self._queue.empty():
            raise self._error

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except TimeoutError:
            if self._closed and self._queue.empty():
                raise StopAsyncIteration
            raise

    async def close(self) -> None:
        """Close the buffer."""
        self._closed = True

    def set_error(self, error: Exception) -> None:
        """Set error state."""
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            return await self.get()
        except (StopAsyncIteration, asyncio.CancelledError):
            raise StopAsyncIteration


class MemoryMappedFileReader:
    """Memory-mapped file reader for efficient large file processing."""

    def __init__(
        self, file_path: Path, chunk_size: int | None = None, read_ahead: bool = True
    ):
        self.file_path = file_path
        self.file_size = file_path.stat().st_size

        # Choose optimal chunk size based on file size
        if chunk_size is None:
            self.chunk_size = (
                LARGE_FILE_CHUNK_SIZE
                if self.file_size > 10 * 1024 * 1024
                else DEFAULT_CHUNK_SIZE
            )
        else:
            self.chunk_size = chunk_size

        self.read_ahead = read_ahead
        self._file = None
        self._mmap = None
        self._current_position = 0
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def open(self) -> None:
        """Open the file with memory mapping."""
        if self._file is not None:
            return

        try:
            self._file = open(self.file_path, "rb")

            # Use memory mapping for large files
            if self.file_size > DEFAULT_CHUNK_SIZE:
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

                # Advise the kernel about our access pattern
                if hasattr(mmap, "MADV_SEQUENTIAL") and self.read_ahead:
                    self._mmap.madvise(mmap.MADV_SEQUENTIAL)

            logger.debug(
                f"Opened memory-mapped file: {self.file_path} "
                f"({self.file_size} bytes, chunk_size={self.chunk_size})"
            )

        except Exception as e:
            await self.close()
            raise ProcessingError(f"Failed to open file for streaming: {e}")

    async def close(self) -> None:
        """Close the file and cleanup resources."""
        if self._closed:
            return

        try:
            if self._mmap:
                self._mmap.close()
                self._mmap = None

            if self._file:
                self._file.close()
                self._file = None

            self._closed = True
            logger.debug(f"Closed memory-mapped file: {self.file_path}")

        except Exception as e:
            logger.warning(f"Error closing memory-mapped file: {e}")

    async def read_chunk(self, size: int | None = None) -> bytes:
        """Read a chunk of data."""
        if self._closed:
            raise ValueError("File is closed")

        if self._current_position >= self.file_size:
            return b""

        read_size = size or self.chunk_size
        remaining = self.file_size - self._current_position
        read_size = min(read_size, remaining)

        try:
            if self._mmap:
                # Read from memory map
                data = self._mmap[
                    self._current_position : self._current_position + read_size
                ]
            else:
                # Read from file directly for small files
                self._file.seek(self._current_position)
                data = self._file.read(read_size)

            self._current_position += len(data)
            return data

        except Exception as e:
            raise ProcessingError(f"Error reading chunk: {e}")

    async def read_range(self, start: int, end: int) -> bytes:
        """Read a specific range of bytes."""
        if self._closed:
            raise ValueError("File is closed")

        if start < 0 or end > self.file_size or start >= end:
            raise ValidationError(f"Invalid range: {start}-{end}")

        try:
            if self._mmap:
                return self._mmap[start:end]
            else:
                self._file.seek(start)
                return self._file.read(end - start)
        except Exception as e:
            raise ProcessingError(f"Error reading range {start}-{end}: {e}")

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        """Stream file in chunks."""
        while True:
            chunk = await self.read_chunk()
            if not chunk:
                break
            yield chunk

    @property
    def position(self) -> int:
        """Current read position."""
        return self._current_position

    @property
    def progress_percent(self) -> float:
        """Read progress percentage."""
        if self.file_size <= 0:
            return 100.0
        return min(100.0, (self._current_position / self.file_size) * 100.0)


class StreamingProgressTracker:
    """Track and broadcast streaming progress with Server-Sent Events support."""

    def __init__(
        self,
        operation_id: str,
        total_size: int,
        callback: ProgressCallback | None = None,
    ):
        self.operation_id = operation_id
        self.total_size = total_size
        self.callback = callback
        self.metrics = StreamingMetrics(total_bytes=total_size)
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to progress updates."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Unsubscribe from progress updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    async def update_progress(
        self, bytes_processed: int, current_step: str | None = None
    ) -> None:
        """Update processing progress."""
        # Update metrics
        self.metrics.update(bytes_delta=bytes_processed)

        # Create progress event
        progress_data = {
            "operation_id": self.operation_id,
            "current_step": current_step,
            "timestamp": time.time(),
            **self.metrics.to_dict(),
        }

        # Call callback if provided
        if self.callback:
            try:
                self.callback(
                    self.metrics.bytes_processed, self.metrics.total_bytes, current_step
                )
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        # Notify subscribers
        with self._lock:
            for subscriber in self._subscribers[
                :
            ]:  # Copy to avoid modification during iteration
                try:
                    subscriber(progress_data)
                except Exception as e:
                    logger.warning(f"Progress subscriber error: {e}")

    async def set_completion(
        self, success: bool = True, error: str | None = None
    ) -> None:
        """Mark operation as completed."""
        completion_data = {
            "operation_id": self.operation_id,
            "completed": True,
            "success": success,
            "error": error,
            "timestamp": time.time(),
            **self.metrics.to_dict(),
        }

        # Notify subscribers
        with self._lock:
            for subscriber in self._subscribers[:]:
                try:
                    subscriber(completion_data)
                except Exception as e:
                    logger.warning(f"Completion subscriber error: {e}")


class ConcurrencyLimiter:
    """Limit concurrent streaming operations."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_STREAMS):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_operations: dict[str, StreamingProgressTracker] = {}
        self._lock = threading.Lock()

    @asynccontextmanager
    async def acquire(self, operation_id: str, tracker: StreamingProgressTracker):
        """Acquire concurrency slot for streaming operation."""
        async with self._semaphore:
            with self._lock:
                self._active_operations[operation_id] = tracker

            try:
                yield
            finally:
                with self._lock:
                    self._active_operations.pop(operation_id, None)

    def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get status of all active operations."""
        with self._lock:
            return {
                op_id: tracker.metrics.to_dict()
                for op_id, tracker in self._active_operations.items()
            }

    def get_capacity_info(self) -> dict[str, int]:
        """Get concurrency capacity information."""
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "active_count": len(self._active_operations),
                "available_slots": self.max_concurrent - len(self._active_operations),
            }


# Global concurrency limiter instance
_global_limiter = ConcurrencyLimiter()


async def stream_large_file(
    file_path: Path,
    operation_id: str,
    progress_callback: ProgressCallback | None = None,
    chunk_size: int | None = None,
    memory_limit_percent: float = MAX_MEMORY_USAGE_PERCENT,
) -> AsyncIterator[bytes]:
    """
    Stream a large file with memory management and progress tracking.

    Args:
        file_path: Path to the file to stream
        operation_id: Unique identifier for this operation
        progress_callback: Optional progress callback function
        chunk_size: Size of chunks to read (auto-determined if None)
        memory_limit_percent: Maximum memory usage before applying backpressure

    Yields:
        Chunks of file data as bytes

    Raises:
        ProcessingError: If streaming fails
        ResourceError: If memory limits are exceeded
        ValidationError: If file is invalid
    """
    # Validate file
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    file_size = file_path.stat().st_size

    # Create progress tracker
    progress_tracker = StreamingProgressTracker(
        operation_id=operation_id, total_size=file_size, callback=progress_callback
    )

    # Create memory monitor
    memory_monitor = MemoryMonitor(max_memory_percent=memory_limit_percent)

    # Stream the file with concurrency limits
    async with _global_limiter.acquire(operation_id, progress_tracker):
        async with MemoryMappedFileReader(file_path, chunk_size) as reader:
            try:
                chunk_count = 0
                async for chunk in reader.stream_chunks():
                    # Check memory pressure and apply backpressure if needed
                    if memory_monitor.should_apply_backpressure():
                        progress_tracker.metrics.backpressure_events += 1
                        logger.debug(f"Applying backpressure for {operation_id}")
                        await memory_monitor.wait_for_memory()

                    # Update progress
                    await progress_tracker.update_progress(
                        bytes_processed=len(chunk),
                        current_step=f"Reading chunk {chunk_count + 1}",
                    )

                    chunk_count += 1
                    yield chunk

                # Mark completion
                await progress_tracker.set_completion(success=True)

                logger.info(
                    f"Completed streaming {file_path} "
                    f"({file_size} bytes in {chunk_count} chunks)"
                )

            except Exception as e:
                progress_tracker.metrics.error_count += 1
                await progress_tracker.set_completion(success=False, error=str(e))
                raise ProcessingError(f"Streaming failed for {operation_id}: {e}")


async def stream_processing_with_backpressure(
    input_stream: AsyncIterator[T],
    processor: Callable[[T], Any],
    max_concurrent: int = 5,
    buffer_size: int = 10,
) -> AsyncIterator[Any]:
    """
    Process streaming data with backpressure and concurrency control.

    Args:
        input_stream: Input data stream
        processor: Function to process each item
        max_concurrent: Maximum concurrent processing operations
        buffer_size: Size of output buffer

    Yields:
        Processed items
    """
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create output buffer
    output_buffer = StreamingBuffer[Any](max_size=buffer_size)

    # Create executor for CPU-bound processing
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_item(item: T) -> None:
        """Process a single item."""
        async with semaphore:
            try:
                # Run processor in thread pool for CPU-bound work
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, processor, item)
                await output_buffer.put(result)
            except Exception as e:
                output_buffer.set_error(e)

    # Start background task to process input stream
    async def producer():
        try:
            tasks = []
            async for item in input_stream:
                task = asyncio.create_task(process_item(item))
                tasks.append(task)

            # Wait for all processing to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            output_buffer.set_error(e)
        finally:
            await output_buffer.close()
            executor.shutdown(wait=True)

    # Start producer task
    producer_task = asyncio.create_task(producer())

    try:
        # Yield processed items
        async for result in output_buffer:
            yield result
    finally:
        # Cleanup
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


def get_streaming_stats() -> dict[str, Any]:
    """Get global streaming statistics."""
    memory_stats = psutil.virtual_memory()

    return {
        "concurrency": _global_limiter.get_capacity_info(),
        "active_operations": _global_limiter.get_active_operations(),
        "memory": {
            "system_memory_percent": memory_stats.percent,
            "system_available_gb": memory_stats.available / (1024**3),
            "process_memory_mb": psutil.Process().memory_info().rss / (1024**2),
        },
        "configuration": {
            "default_chunk_size": DEFAULT_CHUNK_SIZE,
            "large_file_chunk_size": LARGE_FILE_CHUNK_SIZE,
            "max_memory_usage_percent": MAX_MEMORY_USAGE_PERCENT,
            "max_concurrent_streams": MAX_CONCURRENT_STREAMS,
        },
    }


async def cleanup_streaming_resources() -> None:
    """Cleanup any remaining streaming resources."""
    # Force garbage collection
    gc.collect()

    logger.info("Streaming resources cleanup completed")
