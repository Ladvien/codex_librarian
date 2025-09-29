"""
Configuration Monitor for dynamic configuration reload.

This module provides the ConfigurationMonitor class which handles:
- Database configuration polling with configurable intervals
- Signal handling (SIGUSR1) for immediate configuration reload
- Configuration change detection and version tracking
- Redis pub/sub for configuration change notifications
- Integration with watcher manager for seamless reconfiguration
"""

import asyncio
import hashlib
import json
import logging
import signal
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

import redis
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..services.config_service import ConfigurationService

logger = logging.getLogger(__name__)


class ConfigurationVersion(BaseModel):
    """
    Represents a specific version of the configuration with hash-based comparison.

    Used to detect configuration changes by comparing hash values.
    """

    config_data: dict[str, Any]
    version_hash: str
    timestamp: datetime

    def __init__(self, config_data: dict[str, Any], **kwargs):
        """
        Initialize configuration version with automatic hash generation.

        Args:
            config_data: Configuration dictionary
        """
        # Generate hash from normalized config data
        normalized_data = json.dumps(config_data, sort_keys=True, default=str)
        version_hash = hashlib.sha256(normalized_data.encode()).hexdigest()

        super().__init__(
            config_data=config_data,
            version_hash=version_hash,
            timestamp=datetime.utcnow(),
            **kwargs
        )

    def __eq__(self, other):
        """Compare configurations by hash."""
        if not isinstance(other, ConfigurationVersion):
            return False
        return self.version_hash == other.version_hash

    def __ne__(self, other):
        """Not equal comparison."""
        return not self.__eq__(other)


class ConfigurationChange(BaseModel):
    """
    Represents a detected configuration change.

    Contains old and new configuration versions plus change metadata.
    """

    old_version: Optional[ConfigurationVersion]
    new_version: ConfigurationVersion
    change_type: str
    timestamp: datetime = None

    def __init__(self, **kwargs):
        """Initialize with current timestamp if not provided."""
        if kwargs.get("timestamp") is None:
            kwargs["timestamp"] = datetime.utcnow()
        super().__init__(**kwargs)


class ConfigurationMonitor:
    """
    Monitors configuration changes and applies them to the file watcher service.

    Features:
    - Periodic database polling for configuration changes
    - Signal handling for immediate reload requests (SIGUSR1)
    - Configuration versioning and change detection
    - Redis pub/sub notifications for configuration changes
    - Integration with watcher manager for seamless reconfiguration
    """

    def __init__(
        self,
        db_session_factory: Callable[[], Session],
        watcher_manager: Any,  # Type hint avoided to prevent circular imports
        redis_client: Optional[redis.Redis] = None,
        poll_interval: float = 30.0,
    ):
        """
        Initialize configuration monitor.

        Args:
            db_session_factory: Factory function to create database sessions
            watcher_manager: Watcher manager instance to control watchers
            redis_client: Optional Redis client for pub/sub notifications
            poll_interval: Configuration polling interval in seconds
        """
        self.db_session_factory = db_session_factory
        self.watcher_manager = watcher_manager
        self.redis_client = redis_client
        self.poll_interval = poll_interval

        # Internal state
        self._current_version: Optional[ConfigurationVersion] = None
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._reload_requested = False

        # Register signal handlers
        signal.signal(signal.SIGUSR1, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(
            f"ConfigurationMonitor initialized (poll_interval={poll_interval}s, "
            f"redis={'enabled' if redis_client else 'disabled'})"
        )

    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle signals for configuration reload and shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if signum == signal.SIGUSR1:
            logger.info("Received SIGUSR1 - triggering configuration reload")
            self._reload_requested = True
        elif signum in (signal.SIGTERM, signal.SIGINT):
            logger.info(f"Received signal {signum} - stopping configuration monitor")
            self._running = False

    async def load_current_configuration(self) -> dict[str, Any]:
        """
        Load current configuration from database.

        Returns:
            Current configuration dictionary
        """
        with self.db_session_factory() as db:
            return ConfigurationService.load_from_database(db)

    async def detect_configuration_changes(self) -> Optional[ConfigurationChange]:
        """
        Check for configuration changes since last check.

        Returns:
            ConfigurationChange if changes detected, None otherwise
        """
        try:
            current_config = await self.load_current_configuration()
            new_version = ConfigurationVersion(current_config)

            # First time - initialize current version
            if self._current_version is None:
                self._current_version = new_version
                logger.info("Initialized configuration version tracking")
                return None

            # Check for changes
            if self._current_version == new_version:
                return None

            # Configuration changed
            change = ConfigurationChange(
                old_version=self._current_version,
                new_version=new_version,
                change_type="configuration_updated",
            )

            self._current_version = new_version

            logger.info(
                f"Configuration change detected: "
                f"{self._current_version.version_hash[:8]} -> {new_version.version_hash[:8]}"
            )

            return change

        except Exception as e:
            logger.error(f"Error detecting configuration changes: {e}", exc_info=True)
            return None

    async def apply_configuration_changes(self, change: ConfigurationChange) -> None:
        """
        Apply configuration changes to the system.

        Args:
            change: Configuration change to apply
        """
        try:
            logger.info(f"Applying configuration changes: {change.change_type}")

            # Apply configuration to global settings
            ConfigurationService.apply_database_config_to_settings(
                change.new_version.config_data
            )

            # Stop all watchers
            logger.info("Stopping all watchers for reconfiguration...")
            self.watcher_manager.stop_all()

            # Wait a moment for clean shutdown
            await asyncio.sleep(1.0)

            # Start watchers with new configuration
            logger.info("Starting watchers with new configuration...")
            self.watcher_manager.start_all()

            # Publish change notification if Redis available
            if self.redis_client:
                await self.publish_configuration_change(change)

            logger.info("Configuration changes applied successfully")

        except Exception as e:
            logger.error(f"Error applying configuration changes: {e}", exc_info=True)
            raise

    async def publish_configuration_change(self, change: ConfigurationChange) -> None:
        """
        Publish configuration change notification to Redis.

        Args:
            change: Configuration change to publish
        """
        try:
            message = {
                "change_type": change.change_type,
                "timestamp": change.timestamp.isoformat(),
                "old_version_hash": change.old_version.version_hash if change.old_version else None,
                "new_version_hash": change.new_version.version_hash,
                "config_keys": list(change.new_version.config_data.keys()),
            }

            self.redis_client.publish("config_changes", json.dumps(message))
            logger.debug("Published configuration change notification to Redis")

        except Exception as e:
            logger.error(f"Error publishing configuration change: {e}")

    async def force_reload(self) -> None:
        """
        Force immediate configuration reload.

        Useful for testing or manual configuration refresh.
        """
        logger.info("Force reloading configuration...")
        change = await self.detect_configuration_changes()
        if change:
            await self.apply_configuration_changes(change)
        else:
            logger.info("No configuration changes detected during force reload")

    async def _poll_once(self) -> None:
        """
        Perform one configuration polling cycle.

        Checks for configuration changes and applies them if found.
        """
        try:
            # Check if reload was requested via signal
            if self._reload_requested:
                logger.info("Processing reload request...")
                self._reload_requested = False
                await self.force_reload()
                return

            # Regular polling check
            change = await self.detect_configuration_changes()
            if change:
                await self.apply_configuration_changes(change)

        except Exception as e:
            logger.error(f"Error during configuration polling: {e}", exc_info=True)

    def _polling_loop(self) -> None:
        """
        Main polling loop that runs in background thread.

        Continuously polls for configuration changes at specified intervals.
        """
        logger.info("Configuration monitoring started")

        while self._running:
            try:
                # Run async polling function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._poll_once())
                loop.close()

                # Sleep for poll interval, but check running status frequently
                sleep_time = 0
                while sleep_time < self.poll_interval and self._running:
                    time.sleep(0.1)
                    sleep_time += 0.1

            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)
                # Continue running even if error occurs
                time.sleep(1.0)

        logger.info("Configuration monitoring stopped")

    def start(self) -> None:
        """
        Start configuration monitoring.

        Begins background polling thread and signal handling.
        """
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
        """
        Stop configuration monitoring.

        Gracefully shuts down background polling thread.
        """
        if not self._running:
            return

        logger.info("Stopping configuration monitor...")
        self._running = False

        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5.0)
            if self._poll_thread.is_alive():
                logger.warning("Configuration monitor thread did not stop gracefully")

        logger.info("Configuration monitor stopped")

    def get_status(self) -> dict[str, Any]:
        """
        Get current status of configuration monitor.

        Returns:
            Status dictionary with monitoring information
        """
        return {
            "running": self._running,
            "poll_interval": self.poll_interval,
            "current_version_hash": self._current_version.version_hash if self._current_version else None,
            "current_version_timestamp": self._current_version.timestamp.isoformat() if self._current_version else None,
            "redis_enabled": self.redis_client is not None,
            "reload_requested": self._reload_requested,
        }

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop()