"""Service factory for creating configured watcher instances."""

import logging

from ..db.session import get_db_session
from .mirror import create_directory_mirror
from .task_queue import TaskQueue, create_task_queue
from .watcher import DirectoryWatcher, WatcherConfig

logger = logging.getLogger(__name__)


def create_watcher_service(
    config: WatcherConfig,
    task_queue: TaskQueue | None = None,
    enable_directory_mirroring: bool = True,
    output_base_dir: str | None = None,
) -> DirectoryWatcher:
    """Create a properly configured DirectoryWatcher with TaskQueue.

    Args:
        config: Watcher configuration
        task_queue: Optional custom task queue, creates default if None
        enable_directory_mirroring: Whether to enable directory structure preservation
        output_base_dir: Base directory for markdown output (required if mirroring enabled)

    Returns:
        Configured DirectoryWatcher instance
    """
    if task_queue is None:
        # Create default task queue with database session factory
        task_queue = create_task_queue(db_session_factory=get_db_session)
        logger.info("Created default TaskQueue for watcher service")

    # Create directory mirror if enabled and we have watch directories
    directory_mirror = None
    if enable_directory_mirroring and config.watch_directories and output_base_dir:
        try:
            # Use first watch directory as base for mirroring
            # TODO: Support multiple base directories in future
            watch_base = config.watch_directories[0]
            directory_mirror = create_directory_mirror(
                watch_base_dir=watch_base,
                output_base_dir=output_base_dir,
                db_session_factory=get_db_session,
            )
            logger.info(f"Created DirectoryMirror: {watch_base} -> {output_base_dir}")
        except Exception as e:
            logger.error(f"Failed to create DirectoryMirror: {e}")
            logger.warning("Proceeding without directory mirroring")

    watcher = DirectoryWatcher(task_queue, config, directory_mirror)
    logger.info(
        f"Created DirectoryWatcher for {len(config.watch_directories)} directories "
        f"(mirroring={'enabled' if directory_mirror else 'disabled'})"
    )

    return watcher


def create_default_watcher_config(watch_directories: list[str]) -> WatcherConfig:
    """Create default watcher configuration for common use cases.

    Args:
        watch_directories: List of directories to watch

    Returns:
        WatcherConfig with sensible defaults
    """
    return WatcherConfig(
        watch_directories=watch_directories,
        recursive=True,
        patterns=["*.pdf", "*.PDF"],
        ignore_patterns=[
            "**/.*",  # Hidden files/directories
            "**/tmp/*",  # Temporary directories
            "**/temp/*",  # Temporary directories
            "**/cache/*",  # Cache directories
            "**/__pycache__/*",  # Python cache
            "**/node_modules/*",  # Node.js modules
            "**/.git/*",  # Git directories
        ],
        stability_timeout=5.0,  # 5 seconds for file stability
        max_file_size_mb=500,  # 500MB maximum
        enable_deduplication=True,
    )


class WatcherManager:
    """Manager for multiple watcher instances with lifecycle management."""

    def __init__(self):
        """Initialize watcher manager."""
        self.watchers = {}
        self._task_queue = None

    def get_task_queue(self) -> TaskQueue:
        """Get or create shared task queue instance."""
        if self._task_queue is None:
            self._task_queue = create_task_queue()
            logger.info("Created shared TaskQueue for WatcherManager")
        return self._task_queue

    def add_watcher(
        self,
        name: str,
        config: WatcherConfig,
        output_base_dir: str | None = None,
        enable_mirroring: bool = True,
    ) -> DirectoryWatcher:
        """Add a named watcher instance.

        Args:
            name: Unique name for the watcher
            config: Watcher configuration
            output_base_dir: Base directory for markdown output
            enable_mirroring: Whether to enable directory mirroring

        Returns:
            Created DirectoryWatcher instance

        Raises:
            ValueError: If watcher name already exists
        """
        if name in self.watchers:
            raise ValueError(f"Watcher '{name}' already exists")

        watcher = create_watcher_service(
            config,
            self.get_task_queue(),
            enable_directory_mirroring=enable_mirroring,
            output_base_dir=output_base_dir,
        )
        self.watchers[name] = {
            "instance": watcher,
            "config": config,
            "started": False,
            "output_base_dir": output_base_dir,
            "mirroring_enabled": enable_mirroring,
        }

        logger.info(
            f"Added watcher '{name}' with {len(config.watch_directories)} directories "
            f"(mirroring={'enabled' if enable_mirroring and output_base_dir else 'disabled'})"
        )
        return watcher

    def start_watcher(self, name: str) -> None:
        """Start a specific watcher.

        Args:
            name: Name of watcher to start

        Raises:
            KeyError: If watcher doesn't exist
            RuntimeError: If watcher is already running
        """
        if name not in self.watchers:
            raise KeyError(f"Watcher '{name}' not found")

        watcher_info = self.watchers[name]
        if watcher_info["started"]:
            raise RuntimeError(f"Watcher '{name}' is already running")

        watcher_info["instance"].start()
        watcher_info["started"] = True
        logger.info(f"Started watcher '{name}'")

    def stop_watcher(self, name: str) -> None:
        """Stop a specific watcher.

        Args:
            name: Name of watcher to stop

        Raises:
            KeyError: If watcher doesn't exist
        """
        if name not in self.watchers:
            raise KeyError(f"Watcher '{name}' not found")

        watcher_info = self.watchers[name]
        if watcher_info["started"]:
            watcher_info["instance"].stop()
            watcher_info["started"] = False
            logger.info(f"Stopped watcher '{name}'")

    def start_all(self) -> None:
        """Start all registered watchers."""
        for name, watcher_info in self.watchers.items():
            if not watcher_info["started"]:
                try:
                    watcher_info["instance"].start()
                    watcher_info["started"] = True
                    logger.info(f"Started watcher '{name}'")
                except Exception as e:
                    logger.error(f"Failed to start watcher '{name}': {e}")

    def stop_all(self) -> None:
        """Stop all running watchers."""
        for name, watcher_info in self.watchers.items():
            if watcher_info["started"]:
                try:
                    watcher_info["instance"].stop()
                    watcher_info["started"] = False
                    logger.info(f"Stopped watcher '{name}'")
                except Exception as e:
                    logger.error(f"Failed to stop watcher '{name}': {e}")

    def remove_watcher(self, name: str) -> None:
        """Remove a watcher instance.

        Args:
            name: Name of watcher to remove

        Raises:
            KeyError: If watcher doesn't exist
        """
        if name not in self.watchers:
            raise KeyError(f"Watcher '{name}' not found")

        # Stop if running
        self.stop_watcher(name)

        # Remove from registry
        del self.watchers[name]
        logger.info(f"Removed watcher '{name}'")

    def get_status(self) -> dict:
        """Get status of all managed watchers.

        Returns:
            Dictionary with watcher status information
        """
        status = {
            "total_watchers": len(self.watchers),
            "running_watchers": 0,
            "watchers": {},
        }

        for name, watcher_info in self.watchers.items():
            watcher_status = watcher_info["instance"].get_status()
            watcher_status["started"] = watcher_info["started"]
            status["watchers"][name] = watcher_status

            if watcher_info["started"]:
                status["running_watchers"] += 1

        return status

    def update_watcher_config(self, name: str, new_config: WatcherConfig) -> None:
        """Update configuration for a specific watcher.

        Args:
            name: Name of watcher to update
            new_config: New watcher configuration

        Raises:
            KeyError: If watcher doesn't exist
        """
        if name not in self.watchers:
            raise KeyError(f"Watcher '{name}' not found")

        watcher_info = self.watchers[name]
        watcher_info["instance"].update_config(new_config)
        watcher_info["config"] = new_config

        logger.info(f"Updated configuration for watcher '{name}'")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop all watchers."""
        self.stop_all()


# Global watcher manager instance
_watcher_manager = None


def get_watcher_manager() -> WatcherManager:
    """Get global watcher manager instance (singleton pattern).

    Returns:
        WatcherManager instance
    """
    global _watcher_manager
    if _watcher_manager is None:
        _watcher_manager = WatcherManager()
        logger.info("Created global WatcherManager instance")
    return _watcher_manager
