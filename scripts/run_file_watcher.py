#!/usr/bin/env python3
"""
Standalone file watcher service for PDF to Markdown MCP Server.

This script runs the file watcher as a separate process to avoid issues
with uvicorn's multiprocessing model where Observer threads don't survive fork().
"""

import logging
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Add src directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.config import settings
from pdf_to_markdown_mcp.core.watcher import WatcherConfig
from pdf_to_markdown_mcp.core.watcher_service import create_watcher_service, WatcherManager
from pdf_to_markdown_mcp.db.session import SessionLocal
from pdf_to_markdown_mcp.services.config_service import ConfigurationService
from pdf_to_markdown_mcp.worker.indexer import index_watch_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances for signal handling and configuration monitoring
watcher_service = None
config_monitor = None


class ConfigurationMonitor:
    """
    Simplified configuration monitor for dynamic reload.

    Monitors database configuration changes and reloads watcher service
    when changes are detected.
    """

    def __init__(self, watcher_manager: WatcherManager, poll_interval: float = 30.0):
        self.watcher_manager = watcher_manager
        self.poll_interval = poll_interval
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._current_config_hash: Optional[str] = None
        self._reload_requested = False

    def _get_config_hash(self, config_dict: dict) -> str:
        """Generate hash for configuration to detect changes."""
        import json
        import hashlib
        normalized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _load_current_configuration(self) -> dict:
        """Load current configuration from database."""
        try:
            with SessionLocal() as db:
                return ConfigurationService.load_from_database(db)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def _apply_configuration_changes(self, new_config: dict) -> None:
        """Apply configuration changes to watcher manager."""
        try:
            logger.info("Applying configuration changes...")

            # Apply to global settings
            ConfigurationService.apply_database_config_to_settings(new_config)

            # Stop current watchers
            logger.info("Stopping watchers for reconfiguration...")
            self.watcher_manager.stop_all()

            # Wait for clean shutdown
            time.sleep(1.0)

            # Create new watcher configuration
            watcher_config = WatcherConfig(
                watch_directories=settings.watcher.watch_directories,
                patterns=settings.watcher.file_patterns,
                recursive=settings.watcher.recursive,
                stability_timeout=5.0,
                max_file_size_mb=settings.processing.max_file_size_mb,
                enable_deduplication=True,
            )

            # Remove all existing watchers and create new one
            for name in list(self.watcher_manager.watchers.keys()):
                self.watcher_manager.remove_watcher(name)

            # Add new watcher with updated configuration
            self.watcher_manager.add_watcher(
                name="main",
                config=watcher_config,
                output_base_dir=str(settings.watcher.output_directory),
                enable_mirroring=True,
            )

            # Start the new watcher
            self.watcher_manager.start_watcher("main")

            logger.info("Configuration changes applied successfully")

        except Exception as e:
            logger.error(f"Error applying configuration changes: {e}", exc_info=True)

    def _poll_once(self) -> None:
        """Perform one configuration polling cycle."""
        try:
            # Check if reload was requested via signal
            if self._reload_requested:
                logger.info("Processing configuration reload request...")
                self._reload_requested = False
                self.force_reload()
                return

            # Regular polling check
            current_config = self._load_current_configuration()
            if not current_config:
                return

            current_hash = self._get_config_hash(current_config)

            # Initialize hash on first run
            if self._current_config_hash is None:
                self._current_config_hash = current_hash
                logger.info("Initialized configuration monitoring")
                return

            # Check for changes
            if current_hash != self._current_config_hash:
                logger.info(f"Configuration change detected: {self._current_config_hash[:8]} -> {current_hash[:8]}")
                self._current_config_hash = current_hash
                self._apply_configuration_changes(current_config)

        except Exception as e:
            logger.error(f"Error during configuration polling: {e}", exc_info=True)

    def force_reload(self) -> None:
        """Force immediate configuration reload."""
        logger.info("Force reloading configuration...")
        current_config = self._load_current_configuration()
        if current_config:
            self._current_config_hash = self._get_config_hash(current_config)
            self._apply_configuration_changes(current_config)
        else:
            logger.warning("No configuration found during force reload")

    def _polling_loop(self) -> None:
        """Main polling loop that runs in background thread."""
        logger.info("Configuration monitoring started")

        while self._running:
            try:
                self._poll_once()

                # Sleep for poll interval, but check running status frequently
                sleep_time = 0
                while sleep_time < self.poll_interval and self._running:
                    time.sleep(0.1)
                    sleep_time += 0.1

            except Exception as e:
                logger.error(f"Error in configuration polling loop: {e}", exc_info=True)
                time.sleep(1.0)  # Continue even on error

        logger.info("Configuration monitoring stopped")

    def start(self) -> None:
        """Start configuration monitoring."""
        if self._running:
            logger.warning("Configuration monitor is already running")
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._polling_loop,
            name="ConfigurationMonitor",
            daemon=True
        )
        self._poll_thread.start()
        logger.info("Configuration monitor started")

    def stop(self) -> None:
        """Stop configuration monitoring."""
        if not self._running:
            return

        logger.info("Stopping configuration monitor...")
        self._running = False

        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5.0)
            if self._poll_thread.is_alive():
                logger.warning("Configuration monitor thread did not stop gracefully")

        logger.info("Configuration monitor stopped")

    def trigger_reload(self) -> None:
        """Trigger configuration reload via signal."""
        self._reload_requested = True


def signal_handler(signum, frame):
    """Handle shutdown and reload signals gracefully."""
    global watcher_service, config_monitor

    if signum == signal.SIGUSR1:
        logger.info("Received SIGUSR1 - triggering configuration reload")
        if config_monitor:
            config_monitor.trigger_reload()
        else:
            logger.warning("Configuration monitor not available for reload")
        return

    # Handle shutdown signals
    logger.info(f"Received signal {signum}, shutting down file watcher...")

    if config_monitor:
        try:
            config_monitor.stop()
        except Exception as e:
            logger.error(f"Error stopping configuration monitor: {e}")

    if watcher_service:
        try:
            if hasattr(watcher_service, 'stop_all'):
                watcher_service.stop_all()
            else:
                watcher_service.stop()
            logger.info("File watcher stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping watcher: {e}")

    sys.exit(0)


def main():
    """Run the file watcher service with dynamic configuration monitoring."""
    global watcher_service, config_monitor

    logger.info("Starting PDF File Watcher Service with Dynamic Configuration")

    # Load initial configuration from database
    logger.info("Loading initial configuration from database...")
    try:
        with SessionLocal() as db:
            config_dict = ConfigurationService.load_from_database(db)
            ConfigurationService.apply_database_config_to_settings(config_dict)

        logger.info("Initial configuration loaded from database")
    except Exception as e:
        logger.warning(f"Failed to load configuration from database: {e}")
        logger.info("Using default configuration from environment/settings")

    logger.info(f"Watch directories: {settings.watcher.watch_directories}")
    logger.info(f"Output directory: {settings.watcher.output_directory}")
    logger.info(f"File patterns: {settings.watcher.file_patterns}")

    # Trigger initial directory indexing if enabled
    if settings.indexer.initial_index_on_startup:
        logger.info("Initial directory indexing enabled - triggering indexer task...")
        try:
            task_result = index_watch_directories.delay()
            logger.info(f"Indexer task queued with ID: {task_result.id}")
        except Exception as e:
            logger.error(f"Failed to queue indexer task: {e}")
            logger.warning("Continuing without initial index - resync will catch up later")
    else:
        logger.info("Initial directory indexing disabled (INDEXER_INITIAL_INDEX_ON_STARTUP=false)")

    # Register signal handlers (including SIGUSR1 for config reload)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    logger.info("Signal handlers registered (SIGINT, SIGTERM, SIGUSR1)")

    try:
        # Create watcher manager
        watcher_manager = WatcherManager()
        watcher_service = watcher_manager

        # Create initial watcher configuration
        watcher_config = WatcherConfig(
            watch_directories=settings.watcher.watch_directories,
            patterns=settings.watcher.file_patterns,
            recursive=settings.watcher.recursive,
            stability_timeout=5.0,
            max_file_size_mb=settings.processing.max_file_size_mb,
            enable_deduplication=True,
        )

        # Add main watcher to manager
        watcher_manager.add_watcher(
            name="main",
            config=watcher_config,
            output_base_dir=str(settings.watcher.output_directory),
            enable_mirroring=True,
        )

        # Start configuration monitor
        config_monitor = ConfigurationMonitor(
            watcher_manager=watcher_manager,
            poll_interval=30.0  # Check every 30 seconds
        )
        config_monitor.start()

        logger.info("Starting file watcher...")
        watcher_manager.start_watcher("main")
        logger.info("File watcher started successfully")

        # Keep the process running
        logger.info("File watcher is running with dynamic configuration monitoring.")
        logger.info("Send SIGUSR1 signal to trigger immediate configuration reload.")
        logger.info("Press Ctrl+C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in file watcher service: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean shutdown
        if config_monitor:
            try:
                config_monitor.stop()
                logger.info("Configuration monitor stopped")
            except Exception as e:
                logger.error(f"Error stopping configuration monitor: {e}")

        if watcher_service:
            try:
                if hasattr(watcher_service, 'stop_all'):
                    watcher_service.stop_all()
                else:
                    watcher_service.stop()
                logger.info("File watcher stopped")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()