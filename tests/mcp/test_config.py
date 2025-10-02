"""
Tests for MCP server configuration module.

Tests configuration loading, validation, and defaults.
"""

import os
from unittest.mock import patch

import pytest

from src.pdf_to_markdown_mcp.mcp.config import MCPConfig


class TestMCPConfigFromEnv:
    """Test configuration loading from environment variables."""

    def test_from_env_with_minimal_config(self) -> None:
        """Test loading configuration with only required variables."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
        }

        with patch.dict(os.environ, env, clear=True):
            config = MCPConfig.from_env()

            assert config.database_url == "postgresql://user:pass@localhost:5432/test_db"
            assert config.ollama_url == "http://localhost:11434"
            assert config.ollama_model == "nomic-embed-text"
            assert config.db_pool_min_size == 2
            assert config.db_pool_max_size == 10
            assert config.db_pool_timeout == 30
            assert config.log_level == "INFO"
            assert config.search_default_limit == 10
            assert config.search_max_limit == 50
            assert config.search_default_similarity == 0.7

    def test_from_env_with_all_config(self) -> None:
        """Test loading configuration with all variables specified."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "OLLAMA_URL": "http://custom:11434",
            "OLLAMA_MODEL": "custom-model",
            "DB_POOL_MIN_SIZE": "5",
            "DB_POOL_MAX_SIZE": "20",
            "DB_POOL_TIMEOUT": "60",
            "MCP_LOG_LEVEL": "DEBUG",
            "SEARCH_DEFAULT_LIMIT": "15",
            "SEARCH_MAX_LIMIT": "100",
            "SEARCH_DEFAULT_SIMILARITY": "0.8",
        }

        with patch.dict(os.environ, env, clear=True):
            config = MCPConfig.from_env()

            assert config.database_url == "postgresql://user:pass@localhost:5432/test_db"
            assert config.ollama_url == "http://custom:11434"
            assert config.ollama_model == "custom-model"
            assert config.db_pool_min_size == 5
            assert config.db_pool_max_size == 20
            assert config.db_pool_timeout == 60
            assert config.log_level == "DEBUG"
            assert config.search_default_limit == 15
            assert config.search_max_limit == 100
            assert config.search_default_similarity == 0.8

    def test_from_env_missing_database_url(self) -> None:
        """Test configuration fails when DATABASE_URL is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                MCPConfig.from_env()

    def test_from_env_invalid_integer(self) -> None:
        """Test configuration fails with invalid integer values."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "DB_POOL_MIN_SIZE": "not_a_number",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Invalid configuration value"):
                MCPConfig.from_env()

    def test_from_env_invalid_float(self) -> None:
        """Test configuration fails with invalid float values."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "SEARCH_DEFAULT_SIMILARITY": "not_a_float",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Invalid configuration value"):
                MCPConfig.from_env()

    def test_from_env_pool_min_size_too_small(self) -> None:
        """Test validation fails when DB_POOL_MIN_SIZE is less than 1."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "DB_POOL_MIN_SIZE": "0",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="DB_POOL_MIN_SIZE must be at least 1"):
                MCPConfig.from_env()

    def test_from_env_pool_max_less_than_min(self) -> None:
        """Test validation fails when DB_POOL_MAX_SIZE < DB_POOL_MIN_SIZE."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "DB_POOL_MIN_SIZE": "10",
            "DB_POOL_MAX_SIZE": "5",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="DB_POOL_MAX_SIZE.*must be >= DB_POOL_MIN_SIZE"):
                MCPConfig.from_env()

    def test_from_env_search_default_limit_too_large(self) -> None:
        """Test validation fails when SEARCH_DEFAULT_LIMIT > SEARCH_MAX_LIMIT."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "SEARCH_DEFAULT_LIMIT": "100",
            "SEARCH_MAX_LIMIT": "50",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SEARCH_DEFAULT_LIMIT.*must be between 1 and"):
                MCPConfig.from_env()

    def test_from_env_search_default_limit_zero(self) -> None:
        """Test validation fails when SEARCH_DEFAULT_LIMIT is zero."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "SEARCH_DEFAULT_LIMIT": "0",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SEARCH_DEFAULT_LIMIT.*must be between 1 and"):
                MCPConfig.from_env()

    def test_from_env_similarity_threshold_too_high(self) -> None:
        """Test validation fails when similarity threshold > 1.0."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "SEARCH_DEFAULT_SIMILARITY": "1.5",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SEARCH_DEFAULT_SIMILARITY.*must be between 0.0 and 1.0"):
                MCPConfig.from_env()

    def test_from_env_similarity_threshold_negative(self) -> None:
        """Test validation fails when similarity threshold is negative."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "SEARCH_DEFAULT_SIMILARITY": "-0.1",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SEARCH_DEFAULT_SIMILARITY.*must be between 0.0 and 1.0"):
                MCPConfig.from_env()


class TestMCPConfigValidate:
    """Test configuration validation after loading."""

    def test_validate_valid_config(self) -> None:
        """Test validation passes with valid configuration."""
        config = MCPConfig(
            database_url="postgresql://user:pass@localhost:5432/test_db",
            ollama_url="http://localhost:11434",
        )

        # Should not raise
        config.validate()

    def test_validate_database_url_invalid_prefix(self) -> None:
        """Test validation fails when DATABASE_URL doesn't start with postgresql://."""
        config = MCPConfig(
            database_url="mysql://user:pass@localhost:3306/test_db",
            ollama_url="http://localhost:11434",
        )

        with pytest.raises(ValueError, match="DATABASE_URL must start with 'postgresql://'"):
            config.validate()

    def test_validate_ollama_url_invalid_http(self) -> None:
        """Test validation fails with invalid Ollama URL."""
        config = MCPConfig(
            database_url="postgresql://user:pass@localhost:5432/test_db",
            ollama_url="ftp://localhost:11434",
        )

        with pytest.raises(ValueError, match="OLLAMA_URL must be a valid HTTP"):
            config.validate()

    def test_validate_ollama_url_https(self) -> None:
        """Test validation passes with HTTPS Ollama URL."""
        config = MCPConfig(
            database_url="postgresql://user:pass@localhost:5432/test_db",
            ollama_url="https://localhost:11434",
        )

        # Should not raise
        config.validate()

    def test_validate_log_level_invalid(self) -> None:
        """Test validation fails with invalid log level."""
        config = MCPConfig(
            database_url="postgresql://user:pass@localhost:5432/test_db",
            ollama_url="http://localhost:11434",
            log_level="INVALID",
        )

        with pytest.raises(ValueError, match="MCP_LOG_LEVEL must be one of"):
            config.validate()

    def test_validate_log_level_lowercase(self) -> None:
        """Test log level is normalized to uppercase."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
            "MCP_LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, env, clear=True):
            config = MCPConfig.from_env()
            assert config.log_level == "DEBUG"

    def test_validate_all_log_levels(self) -> None:
        """Test all valid log levels are accepted."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        for level in valid_levels:
            config = MCPConfig(
                database_url="postgresql://user:pass@localhost:5432/test_db",
                ollama_url="http://localhost:11434",
                log_level=level,
            )
            # Should not raise
            config.validate()


class TestMCPConfigConstants:
    """Test configuration class constants."""

    def test_embedding_dimensions_constant(self) -> None:
        """Test EMBEDDING_DIMENSIONS constant is correct."""
        assert MCPConfig.EMBEDDING_DIMENSIONS == 768
