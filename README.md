# PDF to Markdown MCP Server

A comprehensive MCP (Model Context Protocol) server that converts PDFs to searchable Markdown with vector embeddings stored in PostgreSQL with PGVector extension.

## Features

- 🔄 PDF to Markdown conversion using MinerU with OCR support
- 🔍 Vector similarity search with PGVector extension
- 🧠 Dual embedding providers (Ollama local + OpenAI API)
- 📊 Background processing with Celery and Redis
- 📁 Automatic file monitoring with Watchdog
- 🗄️ PostgreSQL database with comprehensive schema
- 🚀 FastAPI web server with real-time progress streaming
- 🛡️ Comprehensive error handling and logging
- 🧪 Extensive test suite with TDD approach
- 📊 Task monitoring and progress tracking

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with PGVector extension
- Redis server
- uv package manager (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ladvien/pdf-to-markdown-mcp.git
cd pdf-to-markdown-mcp
```

2. **Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create and activate virtual environment:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies:**
```bash
uv pip install -e ".[dev]"
```

5. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your database and service configurations
```

6. **Set up the database:**
```bash
# Create PostgreSQL database and user
sudo -u postgres createdb pdf_to_markdown_mcp
sudo -u postgres psql -c "CREATE USER pdf_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE pdf_to_markdown_mcp TO pdf_user;"

# Enable PGVector extension
sudo -u postgres psql -d pdf_to_markdown_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run database migrations
alembic upgrade head
```

7. **Start services:**
```bash
# Start Redis (if not running)
redis-server

# Start the FastAPI server
uvicorn pdf_to_markdown_mcp.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Celery worker
celery -A pdf_to_markdown_mcp.worker.celery_app worker --loglevel=debug

# Optional: Start file monitoring
python scripts/run_watcher.py
```

## Usage

### API Endpoints

The server provides several MCP-compatible endpoints:

#### Convert PDF to Markdown
```bash
curl -X POST "http://localhost:8000/api/v1/convert" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "output_path": "/path/to/output.md",
    "extract_tables": true,
    "extract_images": true
  }'
```

#### Search Documents
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 10,
    "similarity_threshold": 0.7
  }'
```

#### Monitor Processing Tasks
```bash
curl "http://localhost:8000/api/v1/tasks/status/{task_id}"
```

### Python Client Usage

```python
from pdf_to_markdown_mcp.services.pdf_processor import PDFProcessor
from pdf_to_markdown_mcp.services.embedding_service import EmbeddingService

# Initialize services
pdf_processor = PDFProcessor()
embedding_service = EmbeddingService()

# Process a PDF
result = await pdf_processor.convert_pdf(
    input_path="/path/to/document.pdf",
    output_path="/path/to/output.md"
)

# Generate embeddings
embeddings = await embedding_service.generate_embeddings(
    text="Your text content here"
)
```

### File Monitoring

Enable automatic PDF processing by running the file watcher:

```bash
python scripts/run_watcher.py --watch-dir /path/to/pdf/directory
```

This will automatically process new PDFs added to the specified directory.

## Architecture

This implementation follows a production-ready microservices architecture:

- **FastAPI** - Modern async web framework with automatic OpenAPI documentation
- **PostgreSQL + PGVector** - Database with vector similarity search capabilities
- **Celery + Redis** - Distributed task queue for background processing
- **MinerU** - Advanced PDF processing with OCR, table extraction, and formula recognition
- **Ollama/OpenAI** - Flexible embedding generation with fallback providers
- **Watchdog** - File system monitoring for automatic processing
- **SQLAlchemy** - Database ORM with async support
- **Pydantic** - Data validation and settings management

### Directory Structure

```
pdf-to-markdown-mcp/
├── src/pdf_to_markdown_mcp/
│   ├── api/                 # FastAPI routes and endpoints
│   ├── core/                # Core business logic
│   ├── db/                  # Database models and operations
│   ├── models/              # Pydantic data models
│   ├── services/            # External service integrations
│   ├── worker/              # Celery task definitions
│   ├── config.py            # Application configuration
│   └── main.py              # FastAPI application entry point
├── tests/                   # Comprehensive test suite
├── scripts/                 # Utility scripts
├── alembic/                 # Database migrations
└── docs/                    # Documentation
```

## Configuration

The application is configured via environment variables. Key settings include:

### Database Settings
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`

### Embedding Providers
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `EMBEDDING_PROVIDER` (ollama|openai|both)

### Processing Settings
- `WATCH_DIRECTORIES` - Comma-separated list of directories to monitor
- `OUTPUT_DIRECTORY` - Where to save processed Markdown files
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Text chunking parameters

See [.env.example](.env.example) for complete configuration options.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pdf_to_markdown_mcp --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Security audit
bandit -r src/

# Run all quality checks
pre-commit run --all-files
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Scripts

The `scripts/` directory contains utility scripts for development and deployment:

### Setup and Installation
- `setup.sh` - Complete setup script for new installations
- `init_database.sh` - Database initialization and configuration

### Service Management
- `start_worker_services.sh` - Start Redis and Celery worker services
- `stop_worker_services.sh` - Stop all worker services

### File Processing
- `run_watcher.py` - File monitoring and automatic processing
- `demo_watcher.py` - Demonstration of file watching capabilities
- `simple_watcher_demo.py` - Simple file watcher example

### Monitoring
- `celery_monitor.py` - Celery task monitoring and statistics
- `run_celery_worker.py` - Celery worker startup script

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow TDD principles - write tests first!
4. Ensure all tests pass and code quality checks succeed
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Principles

- **Test-Driven Development (TDD)** - Always write tests before implementation
- **Type Safety** - Use type hints and Pydantic models everywhere
- **Code Quality** - All code must pass black, ruff, mypy, and bandit checks
- **Documentation** - Document all public APIs and complex logic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MinerU](https://github.com/opendatalab/MinerU) for excellent PDF processing capabilities
- [PGVector](https://github.com/pgvector/pgvector) for vector similarity search in PostgreSQL
- [FastAPI](https://fastapi.tiangolo.com/) for the modern web framework
- [Celery](https://docs.celeryproject.org/) for distributed task processing

For detailed architecture information and development guidelines, see [CLAUDE.md](CLAUDE.md).
