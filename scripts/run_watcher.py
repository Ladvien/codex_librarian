#!/usr/bin/env python3
"""
Script to run the PDF file watcher service.

This script demonstrates how to set up and run the file system monitoring
for automatic PDF processing. It can be used standalone or as a service.
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import List

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.core import (
    create_watcher_service,
    create_default_watcher_config,
    WatcherManager,
)
from pdf_to_markdown_mcp.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('watcher.log') if os.path.exists('logs') else logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class WatcherService:
    """Service wrapper for running PDF file watchers."""

    def __init__(self):
        """Initialize watcher service."""
        self.manager = WatcherManager()
        self.running = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def add_watcher_from_config(self, name: str, directories: List[str]):
        """Add a watcher from configuration.

        Args:
            name: Unique name for the watcher
            directories: List of directories to watch
        """
        # Validate directories
        valid_directories = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                valid_directories.append(str(dir_path.absolute()))
                logger.info(f"Added watch directory: {dir_path.absolute()}")
            else:
                logger.warning(f"Skipping invalid directory: {directory}")

        if not valid_directories:
            logger.error(f"No valid directories found for watcher '{name}'")
            return False

        # Create configuration
        config = create_default_watcher_config(valid_directories)
        logger.info(f"Created watcher config: recursive={config.recursive}, "
                   f"patterns={config.patterns}, "
                   f"stability_timeout={config.stability_timeout}s")

        # Add watcher
        try:
            self.manager.add_watcher(name, config)
            logger.info(f"Added watcher '{name}' for {len(valid_directories)} directories")
            return True
        except Exception as e:
            logger.error(f"Failed to add watcher '{name}': {e}")
            return False

    def start_all_watchers(self):
        """Start all configured watchers."""
        try:
            self.manager.start_all()
            self.running = True
            logger.info("All watchers started successfully")

            # Print status
            status = self.manager.get_status()
            logger.info(f"Running {status['running_watchers']}/{status['total_watchers']} watchers")

        except Exception as e:
            logger.error(f"Failed to start watchers: {e}")
            return False

        return True

    def run_monitoring_loop(self):
        """Run monitoring loop with periodic status updates."""
        logger.info("Starting monitoring loop (Ctrl+C to stop)")

        try:
            while self.running:
                time.sleep(30)  # Status update every 30 seconds

                # Get status
                status = self.manager.get_status()
                logger.info(f"Status: {status['running_watchers']}/{status['total_watchers']} watchers running")

                # Log individual watcher status
                for watcher_name, watcher_status in status['watchers'].items():
                    if watcher_status['is_running']:
                        logger.debug(f"Watcher '{watcher_name}': "
                                   f"{watcher_status['processed_files_count']} files processed, "
                                   f"{watcher_status['stable_files_tracked']} files tracked")

        except KeyboardInterrupt:
            logger.info("Monitoring loop interrupted")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    def shutdown(self):
        """Shutdown all watchers gracefully."""
        logger.info("Shutting down watcher service...")
        self.running = False

        try:
            self.manager.stop_all()
            logger.info("All watchers stopped successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point for watcher service."""
    logger.info("Starting PDF File Watcher Service")

    # Create service
    service = WatcherService()
    service.setup_signal_handlers()

    # Get watch directories from environment or command line
    watch_dirs = []

    # Check for command line arguments
    if len(sys.argv) > 1:
        watch_dirs = sys.argv[1:]
        logger.info(f"Using watch directories from command line: {watch_dirs}")
    else:
        # Check for environment variable
        env_dirs = os.getenv('WATCHER_DIRECTORIES', '').strip()
        if env_dirs:
            watch_dirs = [d.strip() for d in env_dirs.split(',') if d.strip()]
            logger.info(f"Using watch directories from environment: {watch_dirs}")
        else:
            # Default directories
            default_dirs = [
                './pdfs',
                './input',
                './documents'
            ]
            watch_dirs = [d for d in default_dirs if Path(d).exists()]

            if not watch_dirs:
                logger.error("No watch directories found. Please specify directories as arguments "
                           "or set WATCHER_DIRECTORIES environment variable")
                logger.info("Example: python run_watcher.py /path/to/pdfs /another/path")
                logger.info("Example: WATCHER_DIRECTORIES='/path/to/pdfs,/another/path' python run_watcher.py")
                sys.exit(1)

            logger.info(f"Using default directories that exist: {watch_dirs}")

    # Add main watcher
    if not service.add_watcher_from_config('main', watch_dirs):
        logger.error("Failed to configure main watcher")
        sys.exit(1)

    # Start watchers
    if not service.start_all_watchers():
        logger.error("Failed to start watchers")
        sys.exit(1)

    # Run monitoring loop
    try:
        service.run_monitoring_loop()
    finally:
        service.shutdown()

    logger.info("PDF File Watcher Service stopped")


if __name__ == '__main__':
    main()