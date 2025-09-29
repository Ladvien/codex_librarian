"""
Configuration management for PDF to Markdown MCP Server.

Uses pydantic-settings for environment variable handling and type validation.
"""

import json
import logging
from pathlib import Path

from pydantic import ConfigDict, Field, validator
from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="pdf_to_markdown_mcp", env="DB_NAME")
    user: str = Field(default="pdf_user", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")

    # Connection pool settings
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")

    # PGVector configuration
    pgvector_enabled: bool = Field(default=True, env="PGVECTOR_ENABLED")

    @property
    def url(self) -> str:
        """Generate database connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    class Config:
        env_prefix = "DB_"


class EmbeddingSettings(BaseSettings):
    """Embedding generation configuration."""

    provider: str = Field(
        default="ollama", env="EMBEDDING_PROVIDER"
    )  # "ollama" or "openai"
    model: str = Field(default="nomic-embed-text", env="EMBEDDING_MODEL")
    dimensions: int = Field(default=768, env="EMBEDDING_DIMENSIONS")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # Ollama settings
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")

    # OpenAI settings
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")

    class Config:
        env_prefix = "EMBEDDING_"


class ProcessingSettings(BaseSettings):
    """PDF processing configuration."""

    max_file_size_mb: int = Field(default=500, env="MAX_FILE_SIZE_MB")
    processing_timeout_seconds: int = Field(default=300, env="PROCESSING_TIMEOUT")
    temp_dir: Path = Field(default=Path("/tmp/pdf_processing"), env="TEMP_DIR")

    # MinerU settings
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    preserve_layout: bool = Field(default=True, env="PRESERVE_LAYOUT")
    extract_images: bool = Field(default=True, env="EXTRACT_IMAGES")
    extract_tables: bool = Field(default=True, env="EXTRACT_TABLES")
    mineru_device: str = Field(default="cuda", env="MINERU_DEVICE_MODE")

    # Chunking settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    @validator("temp_dir")
    def create_temp_dir(cls, v):
        """Ensure temp directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_prefix = "PROCESSING_"


class CelerySettings(BaseSettings):
    """Enhanced Celery task queue configuration with Redis optimization."""

    # Redis Broker Configuration
    broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    result_backend: str = Field(
        default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND"
    )

    # Redis Connection Pool Settings
    broker_connection_retry_on_startup: bool = Field(
        default=True, env="CELERY_BROKER_CONNECTION_RETRY"
    )
    broker_pool_limit: int = Field(default=10, env="CELERY_BROKER_POOL_LIMIT")
    broker_connection_timeout: int = Field(
        default=30, env="CELERY_BROKER_CONNECTION_TIMEOUT"
    )
    broker_connection_retry: bool = Field(
        default=True, env="CELERY_BROKER_CONNECTION_RETRY_ENABLED"
    )
    broker_connection_max_retries: int = Field(
        default=3, env="CELERY_BROKER_CONNECTION_MAX_RETRIES"
    )

    # Result Backend Configuration
    result_expires: int = Field(default=3600, env="CELERY_RESULT_EXPIRES")  # 1 hour
    result_persistent: bool = Field(default=True, env="CELERY_RESULT_PERSISTENT")
    result_compression: str = Field(default="gzip", env="CELERY_RESULT_COMPRESSION")
    result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")

    # Task Routing with Enhanced Queue Configuration
    task_routes: dict = Field(
        default={
            "pdf_to_markdown_mcp.worker.tasks.process_pdf_document": {
                "queue": "pdf_processing"
            },
            "pdf_to_markdown_mcp.worker.tasks.generate_embeddings": {
                "queue": "embeddings"
            },
            "pdf_to_markdown_mcp.worker.tasks.process_document_images": {
                "queue": "embeddings"
            },
            "pdf_to_markdown_mcp.worker.tasks.process_pdf_batch": {
                "queue": "pdf_processing"
            },
            "pdf_to_markdown_mcp.worker.tasks.cleanup_temp_files": {
                "queue": "maintenance"
            },
            "pdf_to_markdown_mcp.worker.tasks.health_check": {"queue": "monitoring"},
        }
    )

    # Enhanced Worker Settings
    worker_concurrency: int = Field(default=4, env="CELERY_WORKER_CONCURRENCY")
    worker_max_tasks_per_child: int = Field(
        default=1000, env="CELERY_WORKER_MAX_TASKS_PER_CHILD"
    )
    worker_disable_rate_limits: bool = Field(
        default=False, env="CELERY_WORKER_DISABLE_RATE_LIMITS"
    )

    # Task Execution Limits
    task_soft_time_limit: int = Field(default=300, env="CELERY_TASK_SOFT_TIME_LIMIT")
    task_time_limit: int = Field(default=600, env="CELERY_TASK_TIME_LIMIT")
    task_acks_late: bool = Field(default=True, env="CELERY_TASK_ACKS_LATE")
    task_reject_on_worker_lost: bool = Field(
        default=True, env="CELERY_TASK_REJECT_ON_WORKER_LOST"
    )

    # Enhanced Retry Configuration
    task_default_max_retries: int = Field(
        default=3, env="CELERY_TASK_DEFAULT_MAX_RETRIES"
    )
    task_default_retry_delay: int = Field(
        default=60, env="CELERY_TASK_DEFAULT_RETRY_DELAY"
    )
    task_retry_backoff: bool = Field(default=True, env="CELERY_TASK_RETRY_BACKOFF")
    task_retry_jitter: bool = Field(default=False, env="CELERY_TASK_RETRY_JITTER")

    # Monitoring and Events
    worker_send_task_events: bool = Field(
        default=True, env="CELERY_WORKER_SEND_TASK_EVENTS"
    )
    task_send_sent_event: bool = Field(default=True, env="CELERY_TASK_SEND_SENT_EVENT")

    # Queue Management
    task_default_queue: str = Field(
        default="pdf_processing", env="CELERY_TASK_DEFAULT_QUEUE"
    )
    task_queue_max_priority: int = Field(
        default=10, env="CELERY_TASK_QUEUE_MAX_PRIORITY"
    )
    task_message_ttl: int = Field(default=3600, env="CELERY_TASK_MESSAGE_TTL")  # 1 hour

    # Beat Scheduler (for periodic tasks)
    beat_schedule_filename: str = Field(
        default="/tmp/celerybeat-schedule", env="CELERY_BEAT_SCHEDULE_FILENAME"
    )
    beat_max_loop_interval: int = Field(
        default=300, env="CELERY_BEAT_MAX_LOOP_INTERVAL"
    )

    # Security Settings
    task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    accept_content: list[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")

    # Logging Configuration
    worker_log_format: str = Field(
        default="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        env="CELERY_WORKER_LOG_FORMAT",
    )
    worker_task_log_format: str = Field(
        default="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
        env="CELERY_WORKER_TASK_LOG_FORMAT",
    )

    # Redis Optimization Settings
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=30, env="REDIS_SOCKET_TIMEOUT")
    redis_socket_connect_timeout: int = Field(
        default=30, env="REDIS_SOCKET_CONNECT_TIMEOUT"
    )
    redis_retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_health_check_interval: int = Field(
        default=30, env="REDIS_HEALTH_CHECK_INTERVAL"
    )

    class Config:
        env_prefix = "CELERY_"


class WatcherSettings(BaseSettings):
    """File watcher configuration."""

    watch_directories: list[str] = Field(
        default_factory=lambda: ["/home/user/Documents"], env="WATCH_DIRECTORIES"
    )
    output_directory: str = Field(
        default="/home/user/Documents/output",
        env="OUTPUT_DIRECTORY"
    )
    file_patterns: list[str] = Field(default=["*.pdf"], env="FILE_PATTERNS")
    recursive: bool = Field(default=True, env="WATCH_RECURSIVE")

    # Processing settings
    debounce_seconds: float = Field(default=2.0, env="DEBOUNCE_SECONDS")
    queue_batch_size: int = Field(default=10, env="QUEUE_BATCH_SIZE")

    @validator("watch_directories", pre=True)
    def parse_watch_directories(cls, v):
        """Parse JSON list from environment variable."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v if v else ["/home/user/Documents"]

    @validator("file_patterns", pre=True)
    def parse_file_patterns(cls, v):
        """Parse JSON list from environment variable."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    class Config:
        env_prefix = "WATCHER_"


class IndexerSettings(BaseSettings):
    """Directory indexer configuration."""

    initial_index_on_startup: bool = Field(
        default=True, env="INITIAL_INDEX_ON_STARTUP"
    )
    index_batch_size: int = Field(default=100, ge=10, le=1000, env="INDEX_BATCH_SIZE")
    resync_interval_minutes: int = Field(
        default=30, ge=5, le=1440, env="RESYNC_INTERVAL_MINUTES"
    )
    skip_existing_completed: bool = Field(
        default=True, env="SKIP_EXISTING_COMPLETED"
    )
    handle_deleted_files: bool = Field(default=True, env="HANDLE_DELETED_FILES")
    queue_pending_interval_minutes: int = Field(
        default=3, ge=1, le=60, env="QUEUE_PENDING_INTERVAL_MINUTES"
    )
    queue_pending_batch_size: int = Field(
        default=50, ge=10, le=200, env="QUEUE_PENDING_BATCH_SIZE"
    )

    class Config:
        env_prefix = "INDEXER_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )

    # File logging
    log_file: Path | None = Field(default=None, env="LOG_FILE")
    max_file_size_mb: int = Field(default=100, env="LOG_MAX_FILE_SIZE_MB")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    @validator("level")
    def validate_log_level(cls, v):
        """Validate log level is acceptable."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_prefix = "LOG_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: str | None = Field(default=None, env="REDIS_PASSWORD")

    @property
    def url(self) -> str:
        """Generate Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    # API authentication
    api_key: str | None = Field(default=None, env="API_KEY")
    require_auth: bool = Field(default=False, env="REQUIRE_AUTH")

    # Security headers
    enable_security_headers: bool = Field(default=True, env="ENABLE_SECURITY_HEADERS")
    enable_https_redirect: bool = Field(default=False, env="ENABLE_HTTPS_REDIRECT")

    # File upload security
    max_upload_size_mb: int = Field(default=500, env="MAX_UPLOAD_SIZE_MB")
    allowed_file_types: list[str] = Field(
        default=["application/pdf"], env="ALLOWED_FILE_TYPES"
    )
    scan_uploaded_files: bool = Field(default=True, env="SCAN_UPLOADED_FILES")

    # Input validation
    strict_validation: bool = Field(default=True, env="STRICT_VALIDATION")
    sanitize_filenames: bool = Field(default=True, env="SANITIZE_FILENAMES")

    @validator("allowed_file_types", pre=True)
    def parse_allowed_file_types(cls, v):
        """Parse allowed file types from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    class Config:
        env_prefix = ""  # No prefix for security settings


class MonitoringSettings(BaseSettings):
    """Monitoring and health check configuration."""

    # Health checks
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")

    # Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    prometheus_enabled: bool = Field(default=False, env="PROMETHEUS_ENABLED")

    # Performance monitoring
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    profiling_sample_rate: float = Field(default=0.1, env="PROFILING_SAMPLE_RATE")

    class Config:
        env_prefix = ""  # No prefix for monitoring settings


class Settings(BaseSettings):
    """Main application settings."""

    # Application info
    app_name: str = Field(default="PDF to Markdown MCP Server", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # API settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")

    # CORS settings - Environment-specific security
    cors_origins: list[str] = Field(default_factory=list, env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=False, env="CORS_CREDENTIALS")
    cors_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], env="CORS_METHODS"
    )
    cors_headers: list[str] = Field(
        default=["Content-Type", "Authorization", "X-Correlation-ID"],
        env="CORS_HEADERS",
    )

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")

    # Advanced configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay_seconds: int = Field(default=5, env="RETRY_DELAY_SECONDS")
    exponential_backoff: bool = Field(default=True, env="EXPONENTIAL_BACKOFF")

    # Memory management
    max_memory_usage_mb: int = Field(default=2048, env="MAX_MEMORY_USAGE_MB")
    memory_cleanup_interval: int = Field(default=300, env="MEMORY_CLEANUP_INTERVAL")

    # Performance tuning
    async_pool_size: int = Field(default=100, env="ASYNC_POOL_SIZE")
    connection_pool_size: int = Field(default=20, env="CONNECTION_POOL_SIZE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    watcher: WatcherSettings = Field(default_factory=WatcherSettings)
    indexer: IndexerSettings = Field(default_factory=IndexerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @validator("cors_origins", pre=True)
    def parse_and_secure_cors_origins(cls, v, values):
        """Parse CORS settings and apply security-based defaults."""
        # First, parse from JSON if it's a string
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                v = [v]

        # Get environment for security logic
        environment = values.get("environment", "development")

        # If no explicit CORS origins provided or default wildcard, apply security rules
        if not v or v == ["*"]:
            if environment == "production":
                # Production: No origins allowed by default (must be explicitly configured)
                return []
            elif environment == "staging":
                # Staging: Allow specific staging domains
                return ["https://staging.myapp.com", "https://api-staging.myapp.com"]
            else:  # development, testing
                # Development: Allow localhost for development
                return [
                    "http://localhost:3000",
                    "http://localhost:3001",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:3001",
                    "http://127.0.0.1:8080",
                ]

        # If origins explicitly provided, validate them
        if environment == "production":
            for origin in v:
                if origin == "*":
                    raise ValueError(
                        "Wildcard CORS origins (*) not allowed in production"
                    )
                if not origin.startswith(
                    ("https://", "http://localhost:", "http://127.0.0.1:")
                ):
                    raise ValueError(
                        f"Invalid origin for production: {origin}. Must use HTTPS or localhost for testing."
                    )

        return v

    @validator("cors_methods", "cors_headers", pre=True)
    def parse_cors_settings(cls, v):
        """Parse CORS methods and headers from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production", "testing"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # This allows extra environment variables to be ignored
    )


# Global settings instance
settings = Settings()


def configure_logging():
    """Configure application logging based on settings."""
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format=settings.logging.format,
    )

    if settings.logging.log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            settings.logging.log_file,
            maxBytes=settings.logging.max_file_size_mb * 1024 * 1024,
            backupCount=settings.logging.backup_count,
        )
        file_handler.setFormatter(logging.Formatter(settings.logging.format))

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def validate_configuration():
    """Validate configuration settings and dependencies."""
    errors = []
    warnings = []

    # Check required directories exist
    for directory in [settings.processing.temp_dir]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directory {directory}: {e}")

    # Validate database password is set in production
    if settings.environment == "production" and not settings.database.password:
        errors.append("Database password must be set in production environment")

    # Validate API key is set if auth is required
    if settings.security.require_auth and not settings.security.api_key:
        errors.append("API key must be set when authentication is required")

    # Check embedding configuration
    if (
        settings.embedding.provider == "openai"
        and not settings.embedding.openai_api_key
    ):
        warnings.append("OpenAI API key not set but OpenAI embedding provider selected")

    # Validate rate limiting settings
    if settings.rate_limit_per_hour < settings.rate_limit_per_minute * 60:
        warnings.append("Hourly rate limit may be inconsistent with per-minute limit")

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    if warnings:
        import warnings as warn

        for warning in warnings:
            warn.warn(f"Configuration warning: {warning}", UserWarning)

    return True
