"""
MCP server configuration module.

Reads ONLY from environment variables passed by MCP client.
No .env file dependency - all configuration comes from MCP client settings.
"""

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MCPConfig:
    """
    Configuration for the MCP server.

    All settings loaded from environment variables passed by MCP client.
    No .env file required.

    Required Environment Variables:
        DATABASE_URL: PostgreSQL connection string (required)
        OLLAMA_URL: Ollama API endpoint (default: http://localhost:11434)

    Optional Environment Variables:
        OLLAMA_MODEL: Embedding model name (default: nomic-embed-text)
        DB_POOL_MIN_SIZE: Minimum connection pool size (default: 2)
        DB_POOL_MAX_SIZE: Maximum connection pool size (default: 10)
        DB_POOL_TIMEOUT: Connection acquisition timeout in seconds (default: 30)
        MCP_LOG_LEVEL: Logging level (default: INFO)
        SEARCH_DEFAULT_LIMIT: Default number of search results (default: 10)
        SEARCH_MAX_LIMIT: Maximum number of search results (default: 50)
        SEARCH_DEFAULT_SIMILARITY: Default similarity threshold (default: 0.7)
    """

    # Required settings
    database_url: str
    ollama_url: str

    # Optional settings with defaults
    ollama_model: str = "nomic-embed-text"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_pool_timeout: int = 30
    log_level: str = "INFO"
    search_default_limit: int = 10
    search_max_limit: int = 50
    search_default_similarity: float = 0.7

    # Class constants
    EMBEDDING_DIMENSIONS: ClassVar[int] = 768  # nomic-embed-text dimensions

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """
        Load configuration from environment variables.

        Raises:
            ValueError: If required environment variables are missing
            ValueError: If configuration values are invalid

        Returns:
            MCPConfig instance with validated settings
        """
        # Required settings
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set it in your MCP client configuration."
            )

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Optional settings with validation
        try:
            db_pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
            db_pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
            db_pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
            search_default_limit = int(os.getenv("SEARCH_DEFAULT_LIMIT", "10"))
            search_max_limit = int(os.getenv("SEARCH_MAX_LIMIT", "50"))
            search_default_similarity = float(
                os.getenv("SEARCH_DEFAULT_SIMILARITY", "0.7")
            )
        except ValueError as e:
            raise ValueError(f"Invalid configuration value: {e}") from e

        # Validate pool sizes
        if db_pool_min_size < 1:
            raise ValueError("DB_POOL_MIN_SIZE must be at least 1")
        if db_pool_max_size < db_pool_min_size:
            raise ValueError(
                f"DB_POOL_MAX_SIZE ({db_pool_max_size}) must be >= "
                f"DB_POOL_MIN_SIZE ({db_pool_min_size})"
            )

        # Validate search limits
        if search_default_limit < 1 or search_default_limit > search_max_limit:
            raise ValueError(
                f"SEARCH_DEFAULT_LIMIT ({search_default_limit}) must be "
                f"between 1 and SEARCH_MAX_LIMIT ({search_max_limit})"
            )

        # Validate similarity threshold
        if not 0.0 <= search_default_similarity <= 1.0:
            raise ValueError(
                f"SEARCH_DEFAULT_SIMILARITY ({search_default_similarity}) "
                "must be between 0.0 and 1.0"
            )

        return cls(
            database_url=database_url,
            ollama_url=ollama_url,
            ollama_model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
            db_pool_min_size=db_pool_min_size,
            db_pool_max_size=db_pool_max_size,
            db_pool_timeout=db_pool_timeout,
            log_level=os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
            search_default_limit=search_default_limit,
            search_max_limit=search_max_limit,
            search_default_similarity=search_default_similarity,
        )

    def validate(self) -> None:
        """
        Validate configuration after loading.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate database URL format
        if not self.database_url.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must start with 'postgresql://'")

        # Validate Ollama URL format
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL must be a valid HTTP(S) URL")

        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"MCP_LOG_LEVEL must be one of: {', '.join(valid_log_levels)}"
            )