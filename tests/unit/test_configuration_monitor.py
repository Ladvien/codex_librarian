"""
Tests for ConfigurationMonitor that handles dynamic configuration reload.

This module tests the ConfigurationMonitor class which implements:
- Database configuration polling
- Signal handling for configuration reload
- Configuration change detection and notifications
- Redis pub/sub for configuration change events
"""

import asyncio
import signal
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.core.configuration_monitor import (
    ConfigurationChange,
    ConfigurationMonitor,
    ConfigurationVersion,
)
from pdf_to_markdown_mcp.db.models import ServerConfiguration
from pdf_to_markdown_mcp.services.config_service import ConfigurationService


class TestConfigurationMonitor:
    """Test ConfigurationMonitor following TDD"""

    @pytest.fixture
    def mock_db_session(self):
        """Setup mock database session"""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_watcher_manager(self):
        """Setup mock watcher manager"""
        manager = Mock()
        manager.stop_all = Mock()
        manager.start_all = Mock()
        manager.update_watcher_config = Mock()
        return manager

    @pytest.fixture
    def mock_redis(self):
        """Setup mock Redis client"""
        redis_mock = Mock()
        redis_mock.pubsub = Mock()
        redis_mock.publish = Mock()
        return redis_mock

    @pytest.fixture
    def config_monitor(self, mock_db_session, mock_watcher_manager, mock_redis):
        """Setup ConfigurationMonitor with mocked dependencies"""
        return ConfigurationMonitor(
            db_session_factory=lambda: mock_db_session,
            watcher_manager=mock_watcher_manager,
            redis_client=mock_redis,
            poll_interval=1.0,  # Short interval for testing
        )

    def test_configuration_version_creation(self):
        """Test that ConfigurationVersion correctly tracks configuration state"""
        # Given
        config_data = {
            "watch_directories": ["/test/dir1", "/test/dir2"],
            "output_directory": "/test/output",
            "file_patterns": ["*.pdf"],
        }

        # When
        version = ConfigurationVersion(config_data)

        # Then
        assert version.config_data == config_data
        assert version.version_hash is not None
        assert len(version.version_hash) == 64  # SHA-256 hash length
        assert version.timestamp is not None

    def test_configuration_version_equality(self):
        """Test that ConfigurationVersion detects identical configurations"""
        # Given
        config_data1 = {"watch_directories": ["/test/dir"]}
        config_data2 = {"watch_directories": ["/test/dir"]}
        config_data3 = {"watch_directories": ["/different/dir"]}

        # When
        version1 = ConfigurationVersion(config_data1)
        version2 = ConfigurationVersion(config_data2)
        version3 = ConfigurationVersion(config_data3)

        # Then
        assert version1 == version2  # Same config should have same hash
        assert version1 != version3  # Different config should have different hash

    def test_configuration_change_creation(self):
        """Test that ConfigurationChange correctly captures configuration changes"""
        # Given
        old_config = {"watch_directories": ["/old/dir"]}
        new_config = {"watch_directories": ["/new/dir"]}
        change_type = "watch_directories_updated"

        # When
        change = ConfigurationChange(
            old_version=ConfigurationVersion(old_config),
            new_version=ConfigurationVersion(new_config),
            change_type=change_type,
        )

        # Then
        assert change.old_version.config_data == old_config
        assert change.new_version.config_data == new_config
        assert change.change_type == change_type
        assert change.timestamp is not None

    def test_monitor_initialization(self, config_monitor):
        """Test that ConfigurationMonitor initializes correctly"""
        # Then
        assert config_monitor.poll_interval == 1.0
        assert config_monitor._running is False
        assert config_monitor._poll_thread is None
        assert config_monitor._current_version is None

    @pytest.mark.asyncio
    async def test_load_current_configuration(self, config_monitor, mock_db_session):
        """Test that monitor loads current configuration from database"""
        # Given
        expected_config = {
            "watch_directories": ["/test/dir"],
            "output_directory": "/test/output",
            "file_patterns": ["*.pdf"],
        }

        with patch.object(
            ConfigurationService, "load_from_database", return_value=expected_config
        ):
            # When
            current_config = await config_monitor.load_current_configuration()

            # Then
            assert current_config == expected_config
            ConfigurationService.load_from_database.assert_called_once_with(
                mock_db_session
            )

    @pytest.mark.asyncio
    async def test_detect_configuration_changes_no_change(self, config_monitor):
        """Test that monitor correctly detects no configuration changes"""
        # Given
        config_data = {"watch_directories": ["/test/dir"]}
        config_monitor._current_version = ConfigurationVersion(config_data)

        with patch.object(
            config_monitor, "load_current_configuration", return_value=config_data
        ):
            # When
            change = await config_monitor.detect_configuration_changes()

            # Then
            assert change is None

    @pytest.mark.asyncio
    async def test_detect_configuration_changes_with_change(self, config_monitor):
        """Test that monitor correctly detects configuration changes"""
        # Given
        old_config = {"watch_directories": ["/old/dir"]}
        new_config = {"watch_directories": ["/new/dir"]}
        config_monitor._current_version = ConfigurationVersion(old_config)

        with patch.object(
            config_monitor, "load_current_configuration", return_value=new_config
        ):
            # When
            change = await config_monitor.detect_configuration_changes()

            # Then
            assert change is not None
            assert change.old_version.config_data == old_config
            assert change.new_version.config_data == new_config
            assert change.change_type == "configuration_updated"

    @pytest.mark.asyncio
    async def test_apply_configuration_changes(
        self, config_monitor, mock_watcher_manager
    ):
        """Test that monitor applies configuration changes to watcher manager"""
        # Given
        old_config = {"watch_directories": ["/old/dir"]}
        new_config = {"watch_directories": ["/new/dir"], "output_directory": "/out"}
        change = ConfigurationChange(
            old_version=ConfigurationVersion(old_config),
            new_version=ConfigurationVersion(new_config),
            change_type="configuration_updated",
        )

        with patch.object(
            ConfigurationService, "apply_database_config_to_settings"
        ) as mock_apply:
            # When
            await config_monitor.apply_configuration_changes(change)

            # Then
            mock_apply.assert_called_once_with(new_config)
            mock_watcher_manager.stop_all.assert_called_once()
            mock_watcher_manager.start_all.assert_called_once()

    def test_signal_handler_sigusr1(self, config_monitor):
        """Test that SIGUSR1 signal triggers configuration reload"""
        # Given
        config_monitor._reload_requested = False

        # When
        config_monitor._signal_handler(signal.SIGUSR1, None)

        # Then
        assert config_monitor._reload_requested is True

    def test_signal_handler_sigterm(self, config_monitor):
        """Test that SIGTERM signal triggers shutdown"""
        # Given
        config_monitor._running = True

        # When
        config_monitor._signal_handler(signal.SIGTERM, None)

        # Then
        assert config_monitor._running is False

    def test_start_monitoring(self, config_monitor):
        """Test that monitoring starts correctly"""
        # When
        config_monitor.start()

        # Then
        assert config_monitor._running is True
        assert config_monitor._poll_thread is not None
        assert config_monitor._poll_thread.is_alive()

        # Cleanup
        config_monitor.stop()
        config_monitor._poll_thread.join(timeout=2.0)

    def test_stop_monitoring(self, config_monitor):
        """Test that monitoring stops correctly"""
        # Given
        config_monitor.start()
        assert config_monitor._running is True

        # When
        config_monitor.stop()

        # Then
        assert config_monitor._running is False

        # Wait for thread to finish
        if config_monitor._poll_thread:
            config_monitor._poll_thread.join(timeout=2.0)

    @pytest.mark.asyncio
    async def test_publish_configuration_change(self, config_monitor, mock_redis):
        """Test that configuration changes are published to Redis"""
        # Given
        change = ConfigurationChange(
            old_version=ConfigurationVersion({"watch_directories": ["/old"]}),
            new_version=ConfigurationVersion({"watch_directories": ["/new"]}),
            change_type="configuration_updated",
        )

        # When
        await config_monitor.publish_configuration_change(change)

        # Then
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "config_changes"  # Channel name
        # Message should be JSON serializable
        import json
        message = json.loads(call_args[0][1])
        assert message["change_type"] == "configuration_updated"
        assert "old_version_hash" in message
        assert "new_version_hash" in message

    def test_context_manager(self, config_monitor):
        """Test that ConfigurationMonitor works as context manager"""
        # Given/When/Then
        with config_monitor as monitor:
            assert monitor._running is True
            assert monitor._poll_thread is not None

        # After context exit, should be stopped
        assert monitor._running is False
        if monitor._poll_thread:
            monitor._poll_thread.join(timeout=2.0)

    @pytest.mark.asyncio
    async def test_force_reload(self, config_monitor):
        """Test that force_reload immediately checks for configuration changes"""
        # Given
        new_config = {"watch_directories": ["/new/dir"]}

        with patch.object(
            config_monitor, "detect_configuration_changes"
        ) as mock_detect, patch.object(
            config_monitor, "apply_configuration_changes"
        ) as mock_apply:

            mock_change = Mock()
            mock_detect.return_value = mock_change

            # When
            await config_monitor.force_reload()

            # Then
            mock_detect.assert_called_once()
            mock_apply.assert_called_once_with(mock_change)

    @pytest.mark.asyncio
    async def test_polling_loop_with_reload_request(self, config_monitor):
        """Test that polling loop handles reload requests"""
        # Given
        config_monitor._reload_requested = True

        with patch.object(
            config_monitor, "detect_configuration_changes", return_value=None
        ) as mock_detect:
            # When
            await config_monitor._poll_once()

            # Then
            mock_detect.assert_called_once()
            assert config_monitor._reload_requested is False

    @pytest.mark.asyncio
    async def test_error_handling_in_polling(self, config_monitor):
        """Test that polling handles errors gracefully"""
        # Given
        with patch.object(
            config_monitor, "detect_configuration_changes",
            side_effect=Exception("Database error")
        ), patch("pdf_to_markdown_mcp.core.configuration_monitor.logger") as mock_logger:

            # When
            await config_monitor._poll_once()

            # Then
            mock_logger.error.assert_called_once()
            # Monitor should continue running despite error
            assert config_monitor._running is True