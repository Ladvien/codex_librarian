# 📚 PDF to Markdown MCP Server

> Transform your PDFs into searchable, AI-ready markdown with semantic vector search capabilities.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready **Model Context Protocol (MCP)** server that converts PDF documents into searchable Markdown with vector embeddings. Perfect for building RAG systems, semantic search applications, or automated document processing pipelines.

---

## ✨ What Makes This Special?

- **🔍 Intelligent PDF Processing** - MinerU-powered extraction with OCR, table detection, and formula recognition
- **🧠 Dual Embedding Providers** - Use local Ollama models or OpenAI's API with automatic fallback
- **⚡ Background Processing** - Celery + Redis task queue handles heavy processing asynchronously
- **📊 Vector Search** - PostgreSQL with PGVector extension for blazing-fast semantic similarity search
- **👁️ Automatic File Monitoring** - Watchdog integration automatically processes new PDFs as they arrive
- **🚀 Production Ready** - Comprehensive error handling, logging, monitoring, and security features
- **🧪 TDD Approach** - Extensive test suite with 80%+ coverage following Test-Driven Development
- **📖 Rich Examples** - Complete working examples showing every feature in action

---

## 🎯 Quick Start

Get up and running in under 5 minutes!

### Prerequisites

Make sure you have these installed:

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **PostgreSQL 15+** with PGVector extension - [Installation Guide](https://github.com/pgvector/pgvector#installation)
- **Redis** - [Quick Start](https://redis.io/docs/getting-started/)
- **uv** (recommended) - [Installation](https://github.com/astral-sh/uv)

#### Optional: GPU Acceleration

For **5-10x faster PDF processing** with NVIDIA GPUs:

- **NVIDIA GPU** with 8GB+ VRAM (tested on RTX 3090)
- **CUDA 12.4+** - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ladvien/pdf-to-markdown-mcp.git
cd pdf-to-markdown-mcp

# 2. Install uv package manager (fast!)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 4. (Optional) Enable GPU acceleration for 5-10x faster processing
# For NVIDIA GPUs with CUDA 12.4+:
uv pip uninstall torch torchvision
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU detection:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Set up your environment
cp .env.example .env
# Edit .env with your database credentials and settings
# For GPU: uncomment MINERU_DEVICE_MODE=cuda in .env
```

### 🗄️ Database Setup

```bash
# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE pdf_to_markdown_mcp;
CREATE USER pdf_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE pdf_to_markdown_mcp TO pdf_user;
\c pdf_to_markdown_mcp
CREATE EXTENSION IF NOT EXISTS vector;
EOF

# Run database migrations
alembic upgrade head
```

### 🎬 Start Services

```bash
# Terminal 1: Start Redis (if not running as service)
redis-server

# Terminal 2: Start the FastAPI server
uvicorn pdf_to_markdown_mcp.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start Celery worker for background processing
celery -A pdf_to_markdown_mcp.worker.celery_app worker --loglevel=info
```

**🎉 You're ready!** Visit [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API documentation.

---

## 💡 Usage Examples

### Example 1: Kitchen Sink - Complete Pipeline 🌟

**The ultimate example** demonstrating the entire system working together. This example includes:
- Directory watching for new PDFs
- Automatic PDF processing with MinerU
- Embedding generation with Ollama/OpenAI
- Vector storage in PostgreSQL with PGVector
- Interactive semantic search

```bash
# Watch a directory and automatically process all PDFs
python examples/watch_and_mirror.py

# Process existing files once (batch mode)
python examples/watch_and_mirror.py --batch

# Interactive semantic search mode
python examples/watch_and_mirror.py --search

# Custom directories
python examples/watch_and_mirror.py \
  --watch-dir ./my-pdfs \
  --output-dir ./markdown-output
```

### 🔄 Running as a Systemd Service (Daemonized)

For production use, run the PDF watcher as a **systemd service** that automatically starts on boot and restarts on failure:

#### 1. Install the Service

```bash
# Copy the service file to systemd directory
sudo cp systemd/codex-watcher.service /etc/systemd/system/

# Create log file with correct permissions
sudo touch /var/log/codex-watcher.log
sudo chown ladvien:ladvien /var/log/codex-watcher.log

# Reload systemd to recognize the new service
sudo systemctl daemon-reload
```

#### 2. Configure the Service

Edit `/etc/systemd/system/codex-watcher.service` if needed to customize:
- User/Group
- Watch directory path
- Output directory path
- Environment file location

Example service file:
```ini
[Unit]
Description=Codex Librarian PDF Watcher Service
After=network.target postgresql.service

[Service]
Type=simple
User=ladvien
Group=ladvien
WorkingDirectory=/mnt/datadrive_m2/codex_librarian
Environment="PATH=/mnt/datadrive_m2/codex_librarian/.venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/mnt/datadrive_m2/codex_librarian/.env
ExecStart=/mnt/datadrive_m2/codex_librarian/.venv/bin/python /mnt/datadrive_m2/codex_librarian/examples/watch_and_mirror.py --watch-dir /mnt/codex_fs/research/codex_articles --output-dir /mnt/codex_fs/research/codex_articles_markdown
Restart=always
RestartSec=10
StandardOutput=append:/var/log/codex-watcher.log
StandardError=append:/var/log/codex-watcher.log

[Install]
WantedBy=multi-user.target
```

#### 3. Manage the Service

```bash
# Enable the service to start on boot
sudo systemctl enable codex-watcher.service

# Start the service
sudo systemctl start codex-watcher.service

# Check service status
systemctl status codex-watcher.service

# View recent logs
tail -f /var/log/codex-watcher.log

# Stop the service
sudo systemctl stop codex-watcher.service

# Restart the service
sudo systemctl restart codex-watcher.service

# Disable automatic startup
sudo systemctl disable codex-watcher.service
```

#### 4. Monitor the Service

```bash
# Check service status and memory usage
systemctl status codex-watcher.service --no-pager

# View real-time logs
tail -f /var/log/codex-watcher.log

# Count processed files
find /mnt/codex_fs/research/codex_articles_markdown -name "*.md" | wc -l

# Check database statistics
psql -U ladvien -d codex_librarian -c "
  SELECT
    COUNT(*) as total_docs,
    COUNT(CASE WHEN conversion_status='completed' THEN 1 END) as completed,
    COUNT(CASE WHEN conversion_status='failed' THEN 1 END) as failed
  FROM documents;
"

# View embedding statistics
psql -U ladvien -d codex_librarian -c "
  SELECT
    COUNT(*) as total_chunks,
    COUNT(DISTINCT document_id) as unique_docs
  FROM document_embeddings;
"
```

#### 5. Service Features

✅ **Auto-start on boot** - Service starts automatically when system boots
✅ **Auto-restart on failure** - Service restarts automatically if it crashes
✅ **Persistent logging** - All output logged to `/var/log/codex-watcher.log`
✅ **Graceful shutdown** - Handles SIGTERM signals properly
✅ **Resource isolation** - Runs under specified user/group permissions
✅ **Environment management** - Loads configuration from `.env` file

#### 6. Troubleshooting the Service

```bash
# Check for service errors
journalctl -u codex-watcher.service -n 50

# View service configuration
systemctl cat codex-watcher.service

# Check if service is enabled
systemctl is-enabled codex-watcher.service

# Check if service is active
systemctl is-active codex-watcher.service

# Reload service after config changes
sudo systemctl daemon-reload
sudo systemctl restart codex-watcher.service
```

**What it does:**
1. 📁 Scans watch directory for PDF files
2. 🔄 Processes each PDF through MinerU (OCR, tables, formulas)
3. 💾 Saves markdown to output directory, mirroring folder structure
4. 🗄️ Stores document metadata in PostgreSQL
5. 🧠 Generates embeddings and stores vectors in PGVector
6. 🔍 Enables semantic search across all processed documents

**Sample Output:**
```
======================================================================
🚀 PDF to Markdown - Kitchen Sink Example
======================================================================
⏰ Started: 2025-09-26 14:30:00
======================================================================

✅ Settings loaded
✅ Database initialized
✅ MinerU service initialized
✅ Database service initialized
✅ Embedding service initialized

🔍 Scanning for existing PDFs...
   Found 3 PDF files

📄 research/paper.pdf
   ✅ Markdown saved: output/research/paper.md
   ✅ Database record created (ID: 1)
   ✅ Embeddings generated and stored
   📊 Pages: 12, Tables: 3, Formulas: 8
   ✅ Created 24 chunks with embeddings

👀 Starting file watcher...
   Press Ctrl+C to stop
```

### Example 2: API Usage

```bash
# Convert a PDF using the REST API
curl -X POST "http://localhost:8000/api/v1/convert" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "options": {
      "extract_tables": true,
      "extract_formulas": true,
      "extract_images": false,
      "preserve_layout": true,
      "ocr_language": "eng"
    }
  }'

# Search for similar documents
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "similarity_threshold": 0.7
  }'
```

### Example 3: Python Integration

```python
import asyncio
from pathlib import Path
from pdf_to_markdown_mcp.services.mineru import MinerUService
from pdf_to_markdown_mcp.services.embeddings import EmbeddingService
from pdf_to_markdown_mcp.models.request import ProcessingOptions

async def process_pdf():
    # Initialize services
    mineru = MinerUService()
    embeddings = EmbeddingService()

    # Process PDF with custom options
    result = await mineru.process_pdf(
        pdf_path=Path("document.pdf"),
        options=ProcessingOptions(
            extract_tables=True,
            extract_formulas=True,
            preserve_layout=True,
            ocr_language="eng"
        )
    )

    # Save markdown
    Path("output.md").write_text(result.markdown_content)

    # Generate embeddings for semantic search
    vectors = await embeddings.generate_embeddings([result.plain_text])

    print(f"✅ Processed {result.processing_metadata.pages} pages")
    print(f"📊 Found {result.processing_metadata.tables_found} tables")
    print(f"🧮 Found {result.processing_metadata.formulas_found} formulas")

asyncio.run(process_pdf())
```

### Example 4: Batch Processing Script

```bash
# Create a simple batch processing script
cat > process_all.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from pathlib import Path
from pdf_to_markdown_mcp.services.mineru import MinerUService

async def batch_process(pdf_dir: Path, output_dir: Path):
    mineru = MinerUService()
    pdfs = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdfs)} PDFs to process")

    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        result = await mineru.process_pdf(pdf_path=pdf)

        # Save with same name but .md extension
        output_path = output_dir / f"{pdf.stem}.md"
        output_path.write_text(result.markdown_content)
        print(f"  ✅ Saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(batch_process(
        pdf_dir=Path("./pdfs"),
        output_dir=Path("./markdown")
    ))
EOF

# Run it
python process_all.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│            (REST API + WebSocket Streaming)                 │
└────────────┬────────────────────────────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
┌────▼─────┐    ┌────▼──────┐
│ Watchdog │    │   Celery  │
│  Monitor │    │  Workers  │
└────┬─────┘    └────┬──────┘
     │               │
     │         ┌─────▼──────┐
     │         │   MinerU   │ (PDF Processing)
     │         │   Engine   │ • OCR
     │         └─────┬──────┘ • Tables
     │               │        • Formulas
     │         ┌─────▼────────┐
     └────────►│  Embeddings  │
               │   Service    │ (Ollama/OpenAI)
               └─────┬────────┘
                     │
              ┌──────▼──────────┐
              │   PostgreSQL    │
              │   + PGVector    │ (Vector Database)
              └─────────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | Async API endpoints with auto-docs |
| **PDF Processing** | MinerU | OCR, table extraction, formula recognition |
| **Task Queue** | Celery + Redis | Background processing & job management |
| **Database** | PostgreSQL 15+ | Document storage & metadata |
| **Vector Search** | PGVector | Semantic similarity search |
| **Embeddings** | Ollama / OpenAI | Vector generation for semantic search |
| **File Monitoring** | Watchdog | Automatic processing of new files |
| **ORM** | SQLAlchemy 2.0 | Database operations with async support |
| **Validation** | Pydantic v2 | Request/response validation & settings |

---

## 📂 Project Structure

```
pdf-to-markdown-mcp/
├── src/pdf_to_markdown_mcp/
│   ├── api/                    # FastAPI routes
│   │   ├── convert.py         # PDF conversion endpoints
│   │   ├── search.py          # Semantic search endpoints
│   │   ├── status.py          # Task status & monitoring
│   │   └── health.py          # Health checks
│   ├── core/                   # Business logic
│   │   ├── processor.py       # Main processing pipeline
│   │   ├── chunker.py         # Text chunking for embeddings
│   │   ├── watcher.py         # File system monitoring
│   │   └── search_engine.py  # Semantic search logic
│   ├── services/               # External integrations
│   │   ├── mineru.py          # MinerU PDF processing
│   │   ├── embeddings.py      # Embedding generation
│   │   └── database.py        # Database operations
│   ├── db/                     # Database layer
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── session.py         # Database sessions
│   │   └── queries.py         # Query functions
│   ├── worker/                 # Background tasks
│   │   ├── celery.py          # Celery configuration
│   │   └── tasks.py           # Task definitions
│   ├── models/                 # Data models
│   │   ├── document.py        # Document models
│   │   ├── request.py         # Request schemas
│   │   └── response.py        # Response schemas
│   ├── config.py              # Configuration management
│   └── main.py                # Application entry point
├── examples/                   # Working examples
│   ├── watch_and_mirror.py   # Complete pipeline demo
│   └── README.md             # Examples documentation
├── scripts/                    # Utility scripts
│   ├── setup.sh              # Complete setup automation
│   ├── init_database.sh      # Database initialization
│   └── start_worker_services.sh  # Service management
├── tests/                      # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── security/             # Security tests
├── alembic/                    # Database migrations
├── docs/                       # Documentation
├── .env.example               # Environment template
├── pyproject.toml             # Project configuration
├── CLAUDE.md                  # AI development guide
└── README.md                  # You are here!
```

---

## ⚙️ Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

### Essential Settings

```bash
# Database (PostgreSQL with PGVector)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_to_markdown_mcp
DB_USER=pdf_user
DB_PASSWORD=your_secure_password

# Redis (Celery broker)
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0

# Embedding Provider (choose one or both)
EMBEDDING_PROVIDER=ollama              # "ollama" or "openai"
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
OPENAI_API_KEY=sk-...                 # If using OpenAI

# File Processing
INPUT_DIRECTORY=/mnt/codex_fs/research/
OUTPUT_DIRECTORY=/mnt/codex_fs/research/librarian_output/
MAX_FILE_SIZE_MB=500
PROCESSING_TIMEOUT_SECONDS=300

# MinerU Options
MINERU_OCR_LANGUAGE=eng
MINERU_EXTRACT_TABLES=true
MINERU_EXTRACT_FORMULAS=true
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Monitoring
WATCH_DIRECTORIES='["/path/to/watch"]'
FILE_PATTERNS='["*.pdf", "*.PDF"]'
AUTO_PROCESS_NEW_FILES=true
```

See [`.env.example`](.env.example) for complete configuration options with detailed comments.

---

## 🛠️ Development

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=pdf_to_markdown_mcp --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/security/          # Security tests only

# Run with parallel execution
pytest -n auto

# Watch mode for TDD
pytest-watch
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/pdf_to_markdown_mcp/

# Security audit with bandit
bandit -r src/pdf_to_markdown_mcp/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Database Migrations

```bash
# Create new migration after model changes
alembic revision --autogenerate -m "Add new table for xyz"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# View current revision
alembic current
```

---

## 📊 API Reference

### Core Endpoints

#### POST `/api/v1/convert`
Convert a PDF to Markdown

**Request:**
```json
{
  "file_path": "/path/to/document.pdf",
  "output_path": "/path/to/output.md",
  "options": {
    "extract_tables": true,
    "extract_formulas": true,
    "extract_images": false,
    "preserve_layout": true,
    "ocr_language": "eng"
  }
}
```

**Response:**
```json
{
  "task_id": "abc-123-def-456",
  "status": "processing",
  "estimated_time": 30,
  "message": "PDF processing started"
}
```

#### POST `/api/v1/search`
Semantic search across processed documents

**Request:**
```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "similarity_threshold": 0.7,
  "filter": {
    "document_ids": [1, 2, 3],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": 1,
      "chunk_id": 42,
      "similarity": 0.89,
      "content": "Machine learning algorithms...",
      "metadata": {
        "filename": "ml-paper.pdf",
        "page": 5
      }
    }
  ],
  "total": 1,
  "query_time_ms": 45
}
```

#### GET `/api/v1/tasks/status/{task_id}`
Check processing status

**Response:**
```json
{
  "task_id": "abc-123-def-456",
  "status": "completed",
  "progress": 100,
  "result": {
    "document_id": 1,
    "pages_processed": 12,
    "chunks_created": 24,
    "output_path": "/output/document.md"
  }
}
```

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "celery": "running",
    "mineru": "available"
  },
  "uptime_seconds": 3600
}
```

---

## 🔒 Security Features

- **Input Validation** - All inputs validated with Pydantic schemas
- **SQL Injection Prevention** - SQLAlchemy ORM with parameterized queries
- **Path Traversal Protection** - Strict path validation and sanitization
- **File Type Validation** - MIME type checking for uploaded files
- **Rate Limiting** - API endpoint rate limiting
- **Security Headers** - CORS, CSP, and security headers configured
- **Audit Logging** - All operations logged with correlation IDs
- **Secrets Management** - Environment-based configuration (never in code)

Run security audit:
```bash
bandit -r src/pdf_to_markdown_mcp/
```

---

## 📈 Performance Optimization

### GPU Acceleration ⚡

**Enable NVIDIA GPU acceleration for 5-10x faster processing:**

```bash
# Install CUDA-enabled PyTorch (requires CUDA 12.4+)
uv pip uninstall torch torchvision
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create ~/mineru.json for persistent GPU mode
cat > ~/mineru.json <<'EOF'
{
  "device-mode": "cuda",
  "models-dir": "~/.cache/mineru/models"
}
EOF

# Or set environment variable
export MINERU_DEVICE_MODE=cuda
```

**Performance Gains (RTX 3090, 24GB VRAM):**
- Layout analysis: **24 pages/sec** (vs 5 pages/sec CPU)
- Table detection: **11 pages/sec** (vs 2 pages/sec CPU)
- Formula recognition: **13.7 formulas/sec** (vs 2 formulas/sec CPU)
- Overall speedup: **5-10x** on complex PDFs

**GPU Requirements:**
- Minimum: 8GB VRAM
- Recommended: 16GB+ VRAM for batch processing
- Tested: RTX 3090 (24GB), RTX 4090, A100

### Tips for Large-Scale Processing

1. **Batch Processing** - Process multiple PDFs in parallel using Celery workers
   ```bash
   # Start multiple workers
   celery -A pdf_to_markdown_mcp.worker.celery_app worker \
     --concurrency=8 --loglevel=info
   ```

2. **Database Connection Pooling** - Configure pool size in `.env`
   ```bash
   DB_POOL_SIZE=20
   DB_MAX_OVERFLOW=40
   ```

3. **Embedding Batch Size** - Optimize for your hardware
   ```bash
   EMBEDDING_BATCH_SIZE=32  # Increase for GPU systems
   ```

4. **Async Processing** - Use async endpoints for better throughput
   ```python
   # FastAPI handles async natively
   @router.post("/convert")
   async def convert_pdf(request: ConvertRequest):
       return await process_pdf_async(request)
   ```

5. **Vector Index Optimization** - Create appropriate PGVector indexes
   ```sql
   -- Add IVFFlat index for faster similarity search (after data insertion)
   CREATE INDEX ON document_chunks
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

---

## 🐛 Troubleshooting

### Common Issues

**Problem:** `ImportError: No module named 'mineru'`
```bash
# Solution: Install MinerU dependencies
uv pip install mineru[core]==2.0.6
```

**Problem:** `psycopg2.OperationalError: could not connect to server`
```bash
# Solution: Ensure PostgreSQL is running and credentials are correct
sudo systemctl status postgresql
# Check .env file for correct DB_HOST, DB_PORT, DB_USER, DB_PASSWORD
```

**Problem:** `ConnectionRefusedError: [Errno 111] Connection refused` (Redis)
```bash
# Solution: Start Redis server
redis-server
# Or as a service:
sudo systemctl start redis
```

**Problem:** PGVector extension not found
```bash
# Solution: Install and enable PGVector extension
sudo -u postgres psql -d pdf_to_markdown_mcp \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Problem:** Celery tasks not processing
```bash
# Solution: Check Celery worker logs
celery -A pdf_to_markdown_mcp.worker.celery_app inspect active
# Restart workers if needed
pkill -f celery
celery -A pdf_to_markdown_mcp.worker.celery_app worker --loglevel=debug
```

### Debug Mode

Enable detailed logging:
```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
SQL_ECHO=true
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project wouldn't be possible without these amazing open-source libraries:

- **[MinerU](https://github.com/opendatalab/MinerU)** - Excellent PDF processing with OCR and layout preservation
- **[PGVector](https://github.com/pgvector/pgvector)** - Vector similarity search in PostgreSQL
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[Celery](https://docs.celeryq.dev/)** - Distributed task queue for Python
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - The Python SQL toolkit and ORM
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation using Python type hints

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/Ladvien/pdf-to-markdown-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ladvien/pdf-to-markdown-mcp/discussions)
- **Email**: cthomasbrittain@yahoo.com

---

<div align="center">

**Built with ❤️ by [C. Thomas Brittain](https://github.com/Ladvien)**

If this project helps you, consider giving it a ⭐ on GitHub!

</div>