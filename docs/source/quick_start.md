# Quick Start Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with PGVector extension
- Redis server
- uv package manager (recommended)

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Ladvien/pdf-to-markdown-mcp.git
cd pdf-to-markdown-mcp

# Run the setup script (recommended)
./scripts/setup.sh
```

The setup script will:
- Install uv package manager if not present
- Create virtual environment
- Install dependencies
- Create `.env` file from template
- Check system prerequisites
- Initialize database (if PostgreSQL is available)
- Run initial tests

### 2. Manual Installation

If you prefer manual setup:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

### 3. Database Setup

```bash
# Initialize database (PostgreSQL must be running)
./scripts/init_database.sh

# Or manually:
sudo -u postgres createdb pdf_to_markdown_mcp
sudo -u postgres psql -c "CREATE USER pdf_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE pdf_to_markdown_mcp TO pdf_user;"
sudo -u postgres psql -d pdf_to_markdown_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run database migrations
alembic upgrade head
```

## Starting Services

### Option 1: Using the Management Script

```bash
# Start all worker services (Redis + Celery)
./scripts/start_worker_services.sh

# In another terminal, start the FastAPI server
uvicorn pdf_to_markdown_mcp.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Manual Service Startup

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
source .venv/bin/activate
celery -A pdf_to_markdown_mcp.worker.celery_app worker --loglevel=debug

# Terminal 3: Start FastAPI server
source .venv/bin/activate
uvicorn pdf_to_markdown_mcp.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 4: Optional - Start file watcher
source .venv/bin/activate
python scripts/run_watcher.py --watch-dir /path/to/pdf/directory
```

## First Steps

1. **Access the API documentation**: http://localhost:8000/docs
2. **Test the API**: Use the interactive documentation to test endpoints
3. **Process your first PDF**: Use the `/api/v1/convert` endpoint
4. **Set up file monitoring**: Configure watch directories in `.env`

## Next Steps

- Read the [Configuration Guide](config.md) to customize your setup
- Explore [Advanced Usage](advanced_usage.md) for production deployment
- Check the [API Documentation](../apidocs/index.html) for detailed reference
