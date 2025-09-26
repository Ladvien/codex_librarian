"""File system monitoring with Watchdog for automatic PDF processing."""

import fnmatch
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# Try to import magic, fall back to basic validation if not available
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    magic = None
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    """Configuration for the directory watcher."""

    watch_directories: list[str] = field(default_factory=list)
    recursive: bool = True
    patterns: list[str] = field(default_factory=lambda: ["*.pdf", "*.PDF"])
    ignore_patterns: list[str] = field(
        default_factory=lambda: ["**/.*", "**/tmp/*", "**/temp/*"]
    )
    stability_timeout: float = 5.0
    max_file_size_mb: int = 500
    enable_deduplication: bool = True


class FileValidator:
    """Validates PDF files and extracts metadata."""

    def __init__(self):
        """Initialize file validator with MIME type detection."""
        if MAGIC_AVAILABLE:
            self.mime = magic.Magic(mime=True)
        else:
            self.mime = None

    def validate_pdf(self, file_path: Path) -> dict[str, Any]:
        """Validate PDF file and extract metadata.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            "valid": False,
            "mime_type": None,
            "size_bytes": 0,
            "hash": None,
            "error": None,
        }

        try:
            # MIME type validation
            if self.mime is not None:
                mime_type = self.mime.from_file(str(file_path))
                result["mime_type"] = mime_type

                if mime_type != "application/pdf":
                    result["error"] = f"Invalid MIME type: {mime_type}"
                    return result
            else:
                # Fallback validation - check file extension and basic PDF header
                if file_path.suffix.lower() not in [".pdf"]:
                    result["error"] = f"Invalid file extension: {file_path.suffix}"
                    return result

                # Check for PDF magic bytes
                with open(file_path, "rb") as f:
                    header = f.read(4)
                    if header != b"%PDF":
                        result["error"] = "File does not start with PDF magic bytes"
                        return result

                result["mime_type"] = "application/pdf"  # Assumed based on validation

            # File size and hash calculation
            result["size_bytes"] = file_path.stat().st_size
            result["hash"] = self.calculate_file_hash(file_path)
            result["valid"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for deduplication.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash string
        """
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()


class SmartFileDetector:
    """Detects when files have finished being written and are stable."""

    def __init__(self, stability_timeout: float = 5.0):
        """Initialize smart file detector.

        Args:
            stability_timeout: Time in seconds to wait for file stability
        """
        self.stable_files: dict[Path, tuple[int, float]] = {}
        self.stability_timeout = stability_timeout

    def is_file_stable(self, file_path: Path) -> bool:
        """Check if file has stopped changing.

        Args:
            file_path: Path to check for stability

        Returns:
            True if file is stable, False otherwise
        """
        try:
            current_time = time.time()
            current_size = file_path.stat().st_size

            if file_path in self.stable_files:
                last_size, last_time = self.stable_files[file_path]
                if current_size == last_size:
                    return (current_time - last_time) > self.stability_timeout

            self.stable_files[file_path] = (current_size, current_time)
            return False

        except OSError:
            # File might have been deleted or become inaccessible
            if file_path in self.stable_files:
                del self.stable_files[file_path]
            return False


class PDFFileHandler(FileSystemEventHandler):
    """Handles file system events for PDF files."""

    def __init__(self, task_queue, config: WatcherConfig, directory_mirror=None):
        """Initialize PDF file handler.

        Args:
            task_queue: Task queue for processing PDFs
            config: Watcher configuration
            directory_mirror: Optional DirectoryMirror for structure preservation
        """
        super().__init__()
        self.task_queue = task_queue
        self.config = config
        self.directory_mirror = directory_mirror
        self.processed_files: set[str] = set()
        self.detector = SmartFileDetector(config.stability_timeout)
        self.validator = FileValidator()

    def is_pdf_file(self, file_path: str) -> bool:
        """Check if file matches PDF patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file matches PDF patterns
        """
        filename = Path(file_path).name
        return any(
            fnmatch.fnmatch(filename, pattern) for pattern in self.config.patterns
        )

    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on ignore patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file should be ignored
        """
        return any(
            fnmatch.fnmatch(file_path, pattern)
            for pattern in self.config.ignore_patterns
        )

    def process_new_file(self, file_path: str) -> None:
        """Process a newly detected PDF file.

        Args:
            file_path: Path to the PDF file to process
        """
        try:
            path_obj = Path(file_path)

            # Check if file is stable (finished being written)
            if not self.detector.is_file_stable(path_obj):
                logger.debug(
                    f"File not yet stable, will check again later: {file_path}"
                )
                return

            # Validate the PDF file
            validation_result = self.validator.validate_pdf(path_obj)

            if not validation_result["valid"]:
                logger.warning(
                    f"Invalid PDF file detected: {file_path} - {validation_result['error']}"
                )
                return

            # Check file size limits
            size_mb = validation_result["size_bytes"] / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.warning(
                    f"PDF file too large ({size_mb:.1f}MB > {self.config.max_file_size_mb}MB): {file_path}"
                )
                return

            # Check for duplicates if deduplication is enabled
            if self.config.enable_deduplication:
                file_hash = validation_result["hash"]
                if file_hash in self.processed_files:
                    logger.info(
                        f"Duplicate PDF file detected (hash: {file_hash[:8]}...): {file_path}"
                    )
                    return
                self.processed_files.add(file_hash)

            # Calculate mirror paths if directory mirror is available
            mirror_info = None
            if self.directory_mirror:
                try:
                    mirror_info = self.directory_mirror.process_file_for_mirroring(
                        path_obj
                    )
                    if mirror_info:
                        logger.info(
                            f"Directory mirroring prepared: {file_path} -> {mirror_info['output_path']}"
                        )
                    else:
                        logger.warning(
                            f"Failed to prepare directory mirroring for {file_path}"
                        )
                except Exception as e:
                    logger.error(f"Error in directory mirroring for {file_path}: {e}")
                    # Continue without mirroring rather than failing completely

            # Queue the file for processing
            logger.info(f"Queuing PDF for processing: {file_path}")
            self.task_queue.queue_pdf_processing(
                file_path, validation_result, mirror_info=mirror_info
            )

        except Exception as e:
            logger.error(f"Error processing new PDF file {file_path}: {e}")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if not self.is_pdf_file(event.src_path):
            return

        if self._should_ignore_file(event.src_path):
            logger.debug(f"Ignoring file due to ignore patterns: {event.src_path}")
            return

        logger.debug(f"PDF file created: {event.src_path}")
        self.process_new_file(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if not self.is_pdf_file(event.src_path):
            return

        if self._should_ignore_file(event.src_path):
            return

        logger.debug(f"PDF file modified: {event.src_path}")
        self.process_new_file(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if hasattr(event, "dest_path") and self.is_pdf_file(event.dest_path):
            if not self._should_ignore_file(event.dest_path):
                logger.debug(f"PDF file moved to: {event.dest_path}")
                self.process_new_file(event.dest_path)


class DirectoryWatcher:
    """Main directory watcher class that coordinates file monitoring."""

    def __init__(self, task_queue, config: WatcherConfig, directory_mirror=None):
        """Initialize directory watcher.

        Args:
            task_queue: Task queue for processing files
            config: Watcher configuration
            directory_mirror: Optional DirectoryMirror for structure preservation
        """
        self.task_queue = task_queue
        self.config = config
        self.directory_mirror = directory_mirror
        self.observer: Observer | None = None
        self.handler = PDFFileHandler(task_queue, config, directory_mirror)

    def start(self) -> None:
        """Start watching configured directories."""
        if self.observer is not None:
            logger.warning("Watcher is already running")
            return

        self.observer = Observer()

        for directory in self.config.watch_directories:
            try:
                watch_path = Path(directory)
                if not watch_path.exists():
                    logger.warning(f"Watch directory does not exist: {directory}")
                    continue

                if not watch_path.is_dir():
                    logger.warning(f"Watch path is not a directory: {directory}")
                    continue

                self.observer.schedule(
                    self.handler, str(watch_path), recursive=self.config.recursive
                )
                logger.info(
                    f"Started watching directory: {directory} (recursive={self.config.recursive})"
                )

            except Exception as e:
                logger.error(f"Failed to start watching directory {directory}: {e}")

        self.observer.start()
        logger.info("Directory watcher started successfully")

    def stop(self) -> None:
        """Stop watching directories."""
        if self.observer is None:
            logger.warning("Watcher is not running")
            return

        self.observer.stop()
        self.observer.join()
        self.observer = None
        logger.info("Directory watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is currently running.

        Returns:
            True if watcher is active
        """
        return self.observer is not None and self.observer.is_alive()

    def add_watch_directory(self, directory: str) -> None:
        """Add a new directory to watch dynamically.

        Args:
            directory: Path to directory to watch
        """
        if directory not in self.config.watch_directories:
            self.config.watch_directories.append(directory)

            if self.observer is not None:
                try:
                    watch_path = Path(directory)
                    if watch_path.exists() and watch_path.is_dir():
                        self.observer.schedule(
                            self.handler,
                            str(watch_path),
                            recursive=self.config.recursive,
                        )
                        logger.info(f"Added new watch directory: {directory}")
                    else:
                        logger.warning(
                            f"Cannot add non-existent directory: {directory}"
                        )

                except Exception as e:
                    logger.error(f"Failed to add watch directory {directory}: {e}")

    def remove_watch_directory(self, directory: str) -> None:
        """Remove a directory from watching.

        Args:
            directory: Path to directory to stop watching
        """
        if directory in self.config.watch_directories:
            self.config.watch_directories.remove(directory)

            # Note: Watchdog doesn't provide a direct way to unschedule specific paths
            # This would require restarting the observer with new configuration
            logger.info(f"Removed watch directory from config: {directory}")
            logger.warning("Observer restart required to apply directory removal")

    def update_config(self, new_config: WatcherConfig) -> None:
        """Update watcher configuration dynamically.

        Args:
            new_config: New configuration to apply
        """
        old_config = self.config
        self.config = new_config
        self.handler.config = new_config

        # Update detector timeout if changed
        if old_config.stability_timeout != new_config.stability_timeout:
            self.handler.detector.stability_timeout = new_config.stability_timeout

        logger.info("Watcher configuration updated")

    def get_status(self) -> dict[str, Any]:
        """Get current watcher status and statistics.

        Returns:
            Dictionary with watcher status information
        """
        return {
            "is_running": self.is_running(),
            "watch_directories": self.config.watch_directories,
            "recursive": self.config.recursive,
            "patterns": self.config.patterns,
            "ignore_patterns": self.config.ignore_patterns,
            "stability_timeout": self.config.stability_timeout,
            "max_file_size_mb": self.config.max_file_size_mb,
            "enable_deduplication": self.config.enable_deduplication,
            "processed_files_count": len(self.handler.processed_files),
            "stable_files_tracked": len(self.handler.detector.stable_files),
        }
