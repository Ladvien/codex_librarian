# Configuration Guide

The PDF to Markdown MCP Server is configured through environment variables. All settings have sensible defaults but should be customized for production use.

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration values
3. Restart services after making changes

## Configuration Sections

### Application Settings

```bash
# Application identification
APP_NAME="PDF to Markdown MCP Server"
APP_VERSION="0.1.0"

# Server configuration
DEBUG=false              # Enable debug mode (development only)
HOST=0.0.0.0            # Bind address
PORT=8000               # Server port
RELOAD=false            # Auto-reload on changes (development only)
```

### Database Configuration

PostgreSQL with PGVector extension is required for vector similarity search.

```bash
# PostgreSQL connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_to_markdown_mcp
DB_USER=pdf_user
DB_PASSWORD=your_secure_password_here

# Connection pool settings
DB_POOL_SIZE=15         # Connection pool size
DB_MAX_OVERFLOW=30      # Max additional connections
DB_POOL_TIMEOUT=30      # Connection timeout (seconds)

# Database URL (automatically constructed, but can be overridden)
DATABASE_URL=postgresql://pdf_user:password@localhost:5432/pdf_to_markdown_mcp
```

### Redis Configuration

Redis is used for Celery task queue and caching.

```bash
# Redis connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=          # Leave empty if no password

# Redis URL (automatically constructed)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Embedding Providers

Choose between Ollama (local) and OpenAI (API) for generating embeddings.

```bash
# Embedding provider selection
EMBEDDING_PROVIDER=ollama  # Options: ollama, openai, both

# Ollama configuration (for local embeddings)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mxbai-embed-large
OLLAMA_TIMEOUT=30

# OpenAI configuration (for API-based embeddings)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=text-embedding-ada-002
OPENAI_TIMEOUT=30

# Embedding settings
EMBEDDING_DIMENSION=1536  # Must match model dimensions
```

### File Processing

Configure PDF processing and file monitoring.

```bash
# File monitoring
WATCH_DIRECTORIES=/mnt/codex_fs/research/
OUTPUT_DIRECTORY=/mnt/codex_fs/research/librarian_output/
WATCH_ENABLED=true

# PDF processing settings
EXTRACT_TABLES=true       # Extract tables from PDFs
EXTRACT_IMAGES=true       # Extract images from PDFs
EXTRACT_FORMULAS=true     # Extract mathematical formulas
OCR_ENABLED=true          # Enable OCR for scanned documents

# Text chunking for embeddings
CHUNK_SIZE=1000          # Characters per chunk
CHUNK_OVERLAP=200        # Overlap between chunks
MAX_CHUNK_SIZE=2000      # Maximum chunk size
```

### MinerU Configuration

MinerU handles PDF processing with advanced features.

```bash
# MinerU settings
MINERU_TEMP_DIR=/tmp/mineru
MINERU_OUTPUT_FORMAT=markdown
MINERU_DPI=200           # Image extraction DPI
MINERU_TIMEOUT=300       # Processing timeout (seconds)

# Language settings for OCR
MINERU_LANGUAGE=eng      # OCR language code
MINERU_LANGUAGES=eng,chi_sim,chi_tra  # Multiple languages
```

### Logging Configuration

Control logging levels and output formats.

```bash
# Logging settings
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json          # json or text
LOG_FILE=logs/app.log    # Log file path (optional)

# Structured logging
STRUCTLOG_PRETTY=false   # Pretty print logs (development only)
STRUCTLOG_COLORS=false   # Colored output (development only)
```

### Security Settings

Configure security and access controls.

```bash
# API security
API_KEY_REQUIRED=false   # Require API key for endpoints
API_KEYS=key1,key2,key3  # Comma-separated API keys

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100  # Requests per minute
RATE_LIMIT_WINDOW=60     # Window in seconds

# CORS settings
CORS_ENABLED=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
CORS_METHODS=GET,POST,PUT,DELETE
CORS_HEADERS=*
```

### Development Settings

Settings for development and testing environments.

```bash
# Development mode
DEV_MODE=false           # Enable development features
DEV_RELOAD=false         # Auto-reload on file changes
DEV_DEBUG_SQL=false      # Log SQL queries

# Testing settings
TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
TEST_REDIS_URL=redis://localhost:6379/1
```

## Production Recommendations

### Security
- Use strong passwords for database and Redis
- Enable API key authentication in production
- Configure CORS origins specifically
- Use HTTPS in production (configure reverse proxy)

### Performance
- Tune connection pool sizes based on load
- Use Redis persistence for production
- Monitor embedding provider response times
- Configure appropriate timeouts

### Monitoring
- Enable structured logging with JSON format
- Set up log aggregation (ELK stack, etc.)
- Monitor database connection pool usage
- Track Celery task performance

### Scalability
- Use multiple Celery workers for high throughput
- Consider Redis Cluster for high availability
- Scale PostgreSQL with read replicas if needed
- Use load balancer for multiple FastAPI instances

## Environment-Specific Configurations

### Development
```bash
DEBUG=true
LOG_LEVEL=DEBUG
DEV_MODE=true
RELOAD=true
STRUCTLOG_PRETTY=true
STRUCTLOG_COLORS=true
```

### Production
```bash
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json
RATE_LIMIT_ENABLED=true
API_KEY_REQUIRED=true
DB_POOL_SIZE=20
```

### Testing
```bash
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
REDIS_URL=redis://localhost:6379/1
LOG_LEVEL=WARNING
EMBEDDING_PROVIDER=ollama
```

## Configuration Validation

The application validates all configuration at startup. Invalid configurations will cause startup to fail with descriptive error messages.

To validate your configuration without starting the server:

```bash
python -c "from pdf_to_markdown_mcp.config import settings; print('Configuration valid!')"
```

## Configuration Reference

For a complete list of all available configuration options, see the [settings model](../apidocs/pdf_to_markdown_mcp.config.html) in the API documentation.
