"""
Configuration Service for runtime configuration management.

This service handles loading, saving, and managing server configuration
with database persistence, allowing dynamic reconfiguration without restarts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.config import settings
from pdf_to_markdown_mcp.db.models import ServerConfiguration

logger = logging.getLogger(__name__)


class ConfigurationService:
    """
    Manage runtime configuration with database persistence.

    This service provides methods to load configuration from database,
    save changes, and fallback to .env defaults when needed.
    """

    @staticmethod
    def load_from_database(db: Session) -> dict[str, Any]:
        """
        Load configuration from database.

        Args:
            db: Database session

        Returns:
            Dictionary of configuration key-value pairs
        """
        try:
            configs = db.query(ServerConfiguration).all()

            if not configs:
                logger.info("No configuration found in database, using .env defaults")
                return ConfigurationService._get_env_defaults()

            config_dict = {}
            for config in configs:
                config_dict[config.config_key] = config.config_value

            logger.info(f"Loaded {len(config_dict)} configuration entries from database")
            return config_dict

        except Exception as e:
            logger.error(f"Error loading configuration from database: {e}")
            logger.info("Falling back to .env defaults")
            return ConfigurationService._get_env_defaults()

    @staticmethod
    def save_to_database(
        config_key: str, config_value: Any, db: Session, updated_by: str = "api"
    ) -> None:
        """
        Save configuration to database.

        Args:
            config_key: Configuration key
            config_value: Configuration value (will be stored as JSONB)
            db: Database session
            updated_by: Who/what made the update
        """
        try:
            # Check if config exists
            existing = (
                db.query(ServerConfiguration)
                .filter_by(config_key=config_key)
                .first()
            )

            if existing:
                # Update existing
                existing.config_value = config_value
                existing.updated_at = datetime.utcnow()
                existing.updated_by = updated_by
                logger.info(f"Updated configuration: {config_key}")
            else:
                # Create new
                new_config = ServerConfiguration(
                    config_key=config_key,
                    config_value=config_value,
                    updated_by=updated_by,
                )
                db.add(new_config)
                logger.info(f"Created configuration: {config_key}")

            db.commit()

        except Exception as e:
            logger.error(f"Error saving configuration to database: {e}")
            db.rollback()
            raise

    @staticmethod
    def apply_database_config_to_settings(config_dict: dict[str, Any]) -> None:
        """
        Apply database configuration to global settings.

        Args:
            config_dict: Configuration dictionary from database
        """
        try:
            # Update watch_directories if present
            if "watch_directories" in config_dict:
                watch_dirs = config_dict["watch_directories"]
                if watch_dirs:  # Only update if not empty
                    settings.watcher.watch_directories = watch_dirs
                    logger.info(f"Applied watch_directories from database: {watch_dirs}")

            # Update output_directory if present
            if "output_directory" in config_dict:
                output_dir = config_dict["output_directory"]
                if output_dir:  # Only update if not empty
                    settings.watcher.output_directory = Path(output_dir)
                    logger.info(f"Applied output_directory from database: {output_dir}")

            # Update file_patterns if present
            if "file_patterns" in config_dict:
                patterns = config_dict["file_patterns"]
                if patterns:
                    settings.watcher.file_patterns = patterns
                    logger.info(f"Applied file_patterns from database: {patterns}")

        except Exception as e:
            logger.error(f"Error applying database config to settings: {e}")
            raise

    @staticmethod
    def seed_database_from_env(db: Session) -> None:
        """
        Seed database with configuration from .env if not already present.

        Args:
            db: Database session
        """
        try:
            # Check if watch_directories already exists
            existing = (
                db.query(ServerConfiguration)
                .filter_by(config_key="watch_directories")
                .first()
            )

            if not existing:
                # Seed from .env
                logger.info("Seeding database configuration from .env")

                ConfigurationService.save_to_database(
                    "watch_directories",
                    settings.watcher.watch_directories,
                    db,
                    updated_by="env_seed",
                )

                ConfigurationService.save_to_database(
                    "output_directory",
                    str(settings.watcher.output_directory),
                    db,
                    updated_by="env_seed",
                )

                ConfigurationService.save_to_database(
                    "file_patterns",
                    settings.watcher.file_patterns,
                    db,
                    updated_by="env_seed",
                )

                logger.info("Database configuration seeded from .env")
            else:
                logger.info("Database configuration already exists, skipping seed")

        except Exception as e:
            logger.error(f"Error seeding database from .env: {e}")
            raise

    @staticmethod
    def _get_env_defaults() -> dict[str, Any]:
        """
        Get default configuration from .env settings.

        Returns:
            Dictionary of configuration key-value pairs
        """
        return {
            "watch_directories": settings.watcher.watch_directories,
            "output_directory": str(settings.watcher.output_directory),
            "file_patterns": settings.watcher.file_patterns,
        }

    @staticmethod
    def get_current_config() -> dict[str, Any]:
        """
        Get current active configuration from settings.

        Returns:
            Dictionary of current configuration
        """
        return {
            "watch_directories": settings.watcher.watch_directories,
            "output_directory": str(settings.watcher.output_directory),
            "file_patterns": settings.watcher.file_patterns,
            "recursive": settings.watcher.recursive,
            "debounce_seconds": settings.watcher.debounce_seconds,
            "embedding": {
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
                "dimensions": settings.embedding.dimensions,
                "batch_size": settings.embedding.batch_size,
            },
            "processing": {
                "max_file_size_mb": settings.processing.max_file_size_mb,
                "processing_timeout_seconds": settings.processing.processing_timeout_seconds,
                "ocr_language": settings.processing.ocr_language,
                "chunk_size": settings.processing.chunk_size,
                "chunk_overlap": settings.processing.chunk_overlap,
            },
        }

    @staticmethod
    def publish_configuration_change(
        old_config: dict[str, Any], new_config: dict[str, Any], updated_by: str = "system"
    ) -> None:
        """
        Publish configuration change notification to Redis if available.

        Args:
            old_config: Previous configuration
            new_config: New configuration
            updated_by: Who/what made the update
        """
        try:
            import redis
            import json
            import hashlib
            from datetime import datetime

            # Try to connect to Redis with default settings
            try:
                redis_client = redis.Redis(
                    host=getattr(settings.redis, "host", "localhost"),
                    port=getattr(settings.redis, "port", 6379),
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                # Test connection
                redis_client.ping()
            except (redis.ConnectionError, AttributeError):
                logger.debug("Redis not available for configuration change notifications")
                return

            # Create configuration change message
            old_hash = hashlib.sha256(
                json.dumps(old_config, sort_keys=True, default=str).encode()
            ).hexdigest()
            new_hash = hashlib.sha256(
                json.dumps(new_config, sort_keys=True, default=str).encode()
            ).hexdigest()

            message = {
                "change_type": "configuration_updated",
                "timestamp": datetime.utcnow().isoformat(),
                "updated_by": updated_by,
                "old_config_hash": old_hash,
                "new_config_hash": new_hash,
                "changed_keys": [
                    key for key in new_config.keys()
                    if key not in old_config or old_config[key] != new_config[key]
                ],
            }

            # Publish to Redis
            redis_client.publish("config_changes", json.dumps(message))
            logger.debug("Published configuration change notification to Redis")

        except Exception as e:
            logger.debug(f"Failed to publish configuration change to Redis: {e}")

    @staticmethod
    def get_configuration_version_info(db: Session) -> dict[str, Any]:
        """
        Get configuration version information including last update timestamp.

        Args:
            db: Database session

        Returns:
            Dictionary with version information
        """
        try:
            configs = db.query(ServerConfiguration).all()
            if not configs:
                return {
                    "version_count": 0,
                    "last_updated": None,
                    "last_updated_by": None,
                    "configuration_keys": [],
                }

            # Find most recent update
            most_recent = max(configs, key=lambda c: c.updated_at)

            return {
                "version_count": len(configs),
                "last_updated": most_recent.updated_at.isoformat(),
                "last_updated_by": most_recent.updated_by,
                "configuration_keys": [c.config_key for c in configs],
            }

        except Exception as e:
            logger.error(f"Error getting configuration version info: {e}")
            return {
                "version_count": 0,
                "last_updated": None,
                "last_updated_by": None,
                "configuration_keys": [],
                "error": str(e),
            }