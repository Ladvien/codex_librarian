"""
Unit tests for configuration API endpoints.

Tests the configure MCP tool following TDD principles.
"""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.models.request import ConfigurationRequest


class TestConfigureEndpoint:
    """Test configure endpoint following TDD principles."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings."""
        with patch("pdf_to_markdown_mcp.api.config.settings") as mock_settings:
            # Set up default mock values
            mock_settings.watcher.watch_directories = ["/default/path"]
            mock_settings.embedding.provider = "ollama"
            mock_settings.embedding.model = "nomic-embed-text"
            mock_settings.embedding.dimensions = 1536
            mock_settings.embedding.batch_size = 32
            mock_settings.embedding.ollama_url = "http://localhost:11434"
            mock_settings.processing.max_file_size_mb = 500
            mock_settings.processing.processing_timeout_seconds = 300
            mock_settings.processing.ocr_language = "eng"
            mock_settings.processing.preserve_layout = True
            mock_settings.processing.chunk_size = 1000
            mock_settings.processing.chunk_overlap = 200
            mock_settings.celery.broker_url = "redis://localhost:6379/0"
            mock_settings.celery.worker_concurrency = 4
            mock_settings.celery.task_soft_time_limit = 300
            mock_settings.celery.task_time_limit = 600
            mock_settings.database.host = "localhost"
            mock_settings.database.port = 5432
            mock_settings.database.name = "pdf_mcp"
            mock_settings.database.pool_size = 10
            mock_settings.database.max_overflow = 20
            yield mock_settings

    @pytest.fixture
    def valid_watch_directories(self, tmp_path):
        """Create valid directories for testing."""
        dirs = []
        for i in range(2):
            dir_path = tmp_path / f"watch_dir_{i}"
            dir_path.mkdir()
            dirs.append(str(dir_path))
        return dirs

    @pytest.mark.asyncio
    async def test_configure_updates_watch_directories_successfully(
        self, mock_db_session, mock_settings, valid_watch_directories
    ):
        """Test successful update of watch directories."""
        # Given
        request_data = {
            "watch_directories": valid_watch_directories,
            "restart_watcher": True,
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is True
        assert "Configuration updated successfully" in response.message
        assert "Services restarted: file_watcher" in response.message
        assert response.watch_directories == valid_watch_directories
        assert "file_watcher" in response.services_restarted
        assert len(response.validation_errors) == 0

        # Verify settings were updated
        assert mock_settings.watcher.watch_directories == valid_watch_directories

    @pytest.mark.asyncio
    async def test_configure_validates_nonexistent_directories(
        self, mock_db_session, mock_settings
    ):
        """Test validation error for nonexistent directories."""
        # Given
        nonexistent_dirs = ["/nonexistent/path1", "/nonexistent/path2"]

        # When/Then - Should raise validation error during request creation
        with pytest.raises(ValueError, match="Watch directory does not exist"):
            ConfigurationRequest(watch_directories=nonexistent_dirs)

    @pytest.mark.asyncio
    async def test_configure_updates_embedding_config_successfully(
        self, mock_db_session, mock_settings
    ):
        """Test successful update of embedding configuration."""
        # Given
        request_data = {
            "embedding_config": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "batch_size": 64,
                "dimensions": 1536,
            }
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is True
        assert response.embedding_provider == "openai"
        assert len(response.validation_errors) == 0

        # Verify settings were updated
        assert mock_settings.embedding.provider == "openai"
        assert mock_settings.embedding.model == "text-embedding-ada-002"
        assert mock_settings.embedding.batch_size == 64
        assert mock_settings.embedding.dimensions == 1536

    @pytest.mark.asyncio
    async def test_configure_rejects_invalid_embedding_provider(
        self, mock_db_session, mock_settings
    ):
        """Test validation error for invalid embedding provider."""
        # Given
        request_data = {"embedding_config": {"provider": "invalid_provider"}}
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is False
        assert len(response.validation_errors) > 0
        assert any(
            "Invalid embedding provider" in error
            for error in response.validation_errors
        )

    @pytest.mark.asyncio
    async def test_configure_updates_ocr_settings_successfully(
        self, mock_db_session, mock_settings
    ):
        """Test successful update of OCR settings."""
        # Given
        request_data = {
            "ocr_settings": {"language": "fra", "dpi": 300, "preserve_layout": False}
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is True
        assert len(response.validation_errors) == 0

        # Verify settings were updated
        assert mock_settings.processing.ocr_language == "fra"
        assert mock_settings.processing.preserve_layout is False

    @pytest.mark.asyncio
    async def test_configure_validates_ocr_dpi_limits(
        self, mock_db_session, mock_settings
    ):
        """Test validation of OCR DPI limits."""
        # Given
        request_data = {
            "ocr_settings": {
                "dpi": 50  # Below minimum of 72
            }
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is False
        assert len(response.validation_errors) > 0
        assert any(
            "DPI must be at least 72" in error for error in response.validation_errors
        )

    @pytest.mark.asyncio
    async def test_configure_updates_processing_limits_successfully(
        self, mock_db_session, mock_settings
    ):
        """Test successful update of processing limits."""
        # Given
        request_data = {
            "processing_limits": {
                "max_file_size_mb": 750,
                "processing_timeout_seconds": 600,
                "chunk_size": 1200,
            }
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is True
        assert len(response.validation_errors) == 0

        # Verify settings were updated
        assert mock_settings.processing.max_file_size_mb == 750
        assert mock_settings.processing.processing_timeout_seconds == 600
        assert mock_settings.processing.chunk_size == 1200

    @pytest.mark.asyncio
    async def test_configure_validates_processing_limits(
        self, mock_db_session, mock_settings
    ):
        """Test validation of processing limits."""
        # Given
        request_data = {
            "processing_limits": {
                "max_file_size_mb": -10,  # Invalid negative value
                "processing_timeout_seconds": 0,  # Invalid zero value
                "chunk_size": 50,  # Below minimum of 100
            }
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is False
        assert len(response.validation_errors) >= 3
        assert any(
            "Max file size must be positive" in error
            for error in response.validation_errors
        )
        assert any(
            "Processing timeout must be positive" in error
            for error in response.validation_errors
        )
        assert any(
            "Chunk size must be at least 100" in error
            for error in response.validation_errors
        )

    @pytest.mark.asyncio
    async def test_configure_handles_multiple_updates_with_partial_errors(
        self, mock_db_session, mock_settings, valid_watch_directories
    ):
        """Test configuration update with some valid and some invalid settings."""
        # Given
        request_data = {
            "watch_directories": valid_watch_directories,  # Valid
            "embedding_config": {
                "provider": "invalid_provider",  # Invalid
                "batch_size": 32,  # Valid
            },
            "processing_limits": {
                "max_file_size_mb": 1000,  # Valid
                "chunk_size": 50,  # Invalid (below minimum)
            },
        }
        request = ConfigurationRequest(**request_data)

        # When
        from pdf_to_markdown_mcp.api.config import update_configuration

        response = await update_configuration(request, mock_db_session)

        # Then
        assert response.success is False  # Overall failure due to validation errors
        assert len(response.validation_errors) >= 2
        assert any(
            "Invalid embedding provider" in error
            for error in response.validation_errors
        )
        assert any(
            "Chunk size must be at least 100" in error
            for error in response.validation_errors
        )

        # Valid settings should still be updated
        assert mock_settings.watcher.watch_directories == valid_watch_directories
        assert mock_settings.embedding.batch_size == 32
        assert mock_settings.processing.max_file_size_mb == 1000


class TestGetCurrentConfigurationEndpoint:
    """Test get_current_configuration endpoint."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with complete configuration."""
        with patch("pdf_to_markdown_mcp.api.config.settings") as mock_settings:
            # Set up comprehensive mock values
            mock_settings.watcher.watch_directories = [
                "/home/user/docs",
                "/shared/pdfs",
            ]
            mock_settings.embedding.provider = "ollama"
            mock_settings.embedding.model = "nomic-embed-text"
            mock_settings.embedding.dimensions = 1536
            mock_settings.embedding.batch_size = 32
            mock_settings.embedding.ollama_url = "http://localhost:11434"
            mock_settings.processing.max_file_size_mb = 500
            mock_settings.processing.processing_timeout_seconds = 300
            mock_settings.processing.ocr_language = "eng"
            mock_settings.processing.preserve_layout = True
            mock_settings.processing.chunk_size = 1000
            mock_settings.processing.chunk_overlap = 200
            mock_settings.celery.broker_url = "redis://localhost:6379/0"
            mock_settings.celery.worker_concurrency = 4
            mock_settings.celery.task_soft_time_limit = 300
            mock_settings.celery.task_time_limit = 600
            mock_settings.database.host = "localhost"
            mock_settings.database.port = 5432
            mock_settings.database.name = "pdf_mcp"
            mock_settings.database.pool_size = 10
            mock_settings.database.max_overflow = 20
            yield mock_settings

    @pytest.mark.asyncio
    async def test_get_current_configuration_returns_complete_config(
        self, mock_settings
    ):
        """Test that get_current_configuration returns all configuration sections."""
        # When
        from pdf_to_markdown_mcp.api.config import get_current_configuration

        result = await get_current_configuration()

        # Then
        assert "watch_directories" in result
        assert "embedding" in result
        assert "processing" in result
        assert "celery" in result
        assert "database" in result

        # Verify watch directories
        assert result["watch_directories"] == ["/home/user/docs", "/shared/pdfs"]

        # Verify embedding configuration
        embedding_config = result["embedding"]
        assert embedding_config["provider"] == "ollama"
        assert embedding_config["model"] == "nomic-embed-text"
        assert embedding_config["dimensions"] == 1536
        assert embedding_config["batch_size"] == 32
        assert embedding_config["ollama_url"] == "http://localhost:11434"

        # Verify processing configuration
        processing_config = result["processing"]
        assert processing_config["max_file_size_mb"] == 500
        assert processing_config["processing_timeout_seconds"] == 300
        assert processing_config["ocr_language"] == "eng"
        assert processing_config["preserve_layout"] is True
        assert processing_config["chunk_size"] == 1000
        assert processing_config["chunk_overlap"] == 200

        # Verify Celery configuration
        celery_config = result["celery"]
        assert celery_config["broker_url"] == "redis://localhost:6379/0"
        assert celery_config["worker_concurrency"] == 4

        # Verify database configuration
        database_config = result["database"]
        assert database_config["host"] == "localhost"
        assert database_config["port"] == 5432
        assert database_config["name"] == "pdf_mcp"


class TestValidateConfigurationEndpoint:
    """Test validate_configuration endpoint."""

    @pytest.mark.asyncio
    async def test_validate_configuration_accepts_valid_config(self, tmp_path):
        """Test validation of completely valid configuration."""
        # Given
        valid_dir = tmp_path / "valid_dir"
        valid_dir.mkdir()

        config_data = {
            "watch_directories": [str(valid_dir)],
            "embedding_config": {"provider": "ollama", "batch_size": 32},
            "processing_limits": {"max_file_size_mb": 500},
        }

        # When
        from pdf_to_markdown_mcp.api.config import validate_configuration

        result = await validate_configuration(config_data)

        # Then
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_validate_configuration_detects_errors(self):
        """Test validation detects configuration errors."""
        # Given
        config_data = {
            "watch_directories": ["/nonexistent/directory"],
            "embedding_config": {"provider": "invalid_provider", "batch_size": -5},
            "processing_limits": {"max_file_size_mb": -100},
        }

        # When
        from pdf_to_markdown_mcp.api.config import validate_configuration

        result = await validate_configuration(config_data)

        # Then
        assert result["valid"] is False
        assert len(result["errors"]) >= 4  # Multiple validation errors
        assert any("Directory does not exist" in error for error in result["errors"])
        assert any(
            "Provider must be 'ollama' or 'openai'" in error
            for error in result["errors"]
        )
        assert any(
            "Batch size must be a positive integer" in error
            for error in result["errors"]
        )
        assert any(
            "Max file size must be positive" in error for error in result["errors"]
        )

    @pytest.mark.asyncio
    async def test_validate_configuration_generates_warnings(self):
        """Test validation generates appropriate warnings."""
        # Given
        config_data = {
            "processing_limits": {
                "max_file_size_mb": 1500  # Large but valid, should generate warning
            }
        }

        # When
        from pdf_to_markdown_mcp.api.config import validate_configuration

        result = await validate_configuration(config_data)

        # Then
        assert result["valid"] is True  # Still valid despite warning
        assert len(result["warnings"]) >= 1
        assert any(
            "Large max file size may impact performance" in warning
            for warning in result["warnings"]
        )


class TestResetConfigurationEndpoint:
    """Test reset_configuration endpoint."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for reset testing."""
        with patch("pdf_to_markdown_mcp.api.config.settings") as mock_settings:
            mock_settings.watcher.watch_directories = ["/default/watch"]
            mock_settings.embedding.provider = "ollama"
            yield mock_settings

    @pytest.mark.asyncio
    async def test_reset_configuration_resets_to_defaults(self, mock_settings):
        """Test that reset_configuration resets settings to defaults."""
        # When
        from pdf_to_markdown_mcp.api.config import reset_configuration

        response = await reset_configuration()

        # Then
        assert response.success is True
        assert response.message == "Configuration reset to default values"
        assert "file_watcher" in response.services_restarted
        assert "embedding_service" in response.services_restarted
        assert len(response.validation_errors) == 0
        assert response.watch_directories == ["/default/watch"]
        assert response.embedding_provider == "ollama"
