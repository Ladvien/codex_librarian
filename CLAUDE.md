# PDF to Markdown MCP Server

## Project Overview

GPU-accelerated PDF processing system that converts PDFs to markdown with vector embeddings. Uses MinerU standalone service for CUDA-accelerated processing, Celery for task orchestration, and PostgreSQL with PGVector for embeddings storage.

**Key Capabilities:**
- Watches directories for new PDFs (configured via API)
- GPU-accelerated conversion (5-10x faster than CPU)
- Vector embeddings for semantic search
- **MCP semantic search tool** - `search_library` for LLMs
- MCP-compatible REST API
- Persistent configuration survives restarts

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI + Celery + FastMCP
- **Database**: PostgreSQL 17+ with PGVector
- **Queue**: Redis (Docker)
- **GPU**: CUDA 12.4+ (MinerU)
- **Embeddings**: Ollama (nomic-embed-text)
- **Process Management**: systemd services
- **MCP Server**: FastMCP with semantic search tool

## Architecture

```
┌─────────────────────────────────┐
│ MCP Client / API Consumer       │
│ Configure via HTTP API          │
└────────────┬────────────────────┘
             │ HTTP/REST :8000
             ▼
┌─────────────────────────────────┐
│ FastAPI Server                  │
│ - 4 uvicorn workers             │
│ - /api/v1/configure             │
│ - /api/v1/configuration         │
└────────────┬────────────────────┘
             │
             ├──▶ PostgreSQL (192.168.1.104)
             │   └─ server_configuration table
             │   └─ documents, document_content
             │   └─ document_embeddings (PGVector)
             │
             ├──▶ Redis (Docker :6379)
             │   └─ mineru_requests queue
             │   └─ mineru_results queue
             │   └─ celery queue
             │   └─ embeddings queue
             │
             ├──▶ MinerU Standalone Service
             │   └─ GPU-accelerated processing
             │   └─ Redis queue consumer
             │
             └──▶ Celery Workers
                 └─ pdf-celery-worker.service
                 └─ pdf-celery-beat.service
```

## Project Structure

```
/mnt/datadrive_m2/codex_librarian/
├── src/pdf_to_markdown_mcp/
│   ├── api/           # FastAPI endpoints
│   ├── worker/        # Celery tasks
│   ├── services/      # Business logic
│   │   ├── mineru_standalone.py  # GPU service
│   │   ├── config_service.py     # Config persistence
│   │   └── document_service.py   # Document CRUD
│   ├── db/            # Database models, migrations
│   └── config.py      # Settings, environment
├── systemd/           # Service definitions
├── scripts/           # Monitoring & diagnostics
├── alembic/           # Database migrations
├── .env               # Environment configuration
└── CLAUDE.md          # This file
```

**Key Paths:**
- Input PDFs: `/mnt/codex_fs/research/codex_articles/`
- Output Markdown: `/mnt/codex_fs/research/librarian_output/`
- MinerU Log: `/tmp/mineru.log`
- Celery Log: `/var/log/celery-worker.log`

## Essential Commands

### Service Management
```bash
# Status check
./scripts/check_services.sh

# View logs
./scripts/view_logs.sh all          # All services
./scripts/view_logs.sh api          # API only
./scripts/view_logs.sh worker       # Celery worker
./scripts/view_logs.sh follow       # Real-time

# Restart services
./scripts/restart_services.sh

# Stop services
./scripts/stop_services.sh
```

### Manual Service Control
```bash
# Start services
sudo systemctl start pdf-celery-worker
sudo systemctl start pdf-celery-beat
sudo systemctl start pdf-api-server

# Check status
systemctl status pdf-api-server
systemctl status pdf-celery-worker

# View systemd logs
sudo journalctl -u pdf-api-server -f
```

### Database Access
```bash
# Connect to database
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian

# Check document status
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "
SELECT conversion_status, COUNT(*)
FROM documents
GROUP BY conversion_status;"
```

### GPU Monitoring
```bash
# Real-time GPU status
watch -n 1 nvidia-smi

# Check MinerU GPU usage
nvidia-smi | grep python

# Monitor MinerU processing
tail -f /tmp/mineru.log | grep -E "Layout|MFD|OCR|Completed"
```

### Redis Queue Inspection
```bash
# Check queue lengths
redis-cli llen mineru_requests
redis-cli llen mineru_results
redis-cli llen celery
redis-cli llen embeddings

# Clear queues (emergency only)
redis-cli FLUSHDB
```

### Health Checks
```bash
# Quick health check
./scripts/quick_health_check.sh

# Full diagnostic
python scripts/system_diagnostic.py

# API health
curl http://localhost:8000/health | jq .
curl http://192.168.1.110:8000/health | jq .
```

## Configuration API

### Get Current Configuration
```bash
curl http://192.168.1.110:8000/api/v1/configuration | jq .
```

### Update Watch Directories
```bash
# Single directory
curl -X POST http://192.168.1.110:8000/api/v1/configure \
  -H "Content-Type: application/json" \
  -d '{
    "watch_directories": ["/mnt/codex_fs/research/"],
    "restart_watcher": true
  }'

# Multiple directories
curl -X POST http://192.168.1.110:8000/api/v1/configure \
  -H "Content-Type: application/json" \
  -d '{
    "watch_directories": [
      "/mnt/codex_fs/research/",
      "/mnt/codex_fs/papers/",
      "/mnt/codex_fs/books/"
    ],
    "restart_watcher": true
  }'
```

### Reset to Defaults
```bash
curl -X POST http://192.168.1.110:8000/api/v1/configuration/reset
```

## Code Conventions

### Python Style
- Use type hints for all function signatures
- Async functions for I/O operations
- Pydantic models for validation
- FastAPI dependency injection for services

### Error Handling
- Log errors with context (document ID, filename)
- Update database status on failures
- Retry logic for transient failures
- Never swallow exceptions silently

### Database Transactions
- Use context managers: `with get_db_session() as db:`
- Commit explicitly after successful operations
- Rollback on errors
- Always close sessions

### Task Queue Best Practices
- Idempotent tasks (safe to retry)
- Store task_id in database for tracking
- Set reasonable timeouts
- Log task start/completion

## Deployment Workflow

### Initial Deployment

**1. Install systemd services:**
```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pdf-api-server pdf-celery-worker pdf-celery-beat
```

**2. Create log files:**
```bash
sudo touch /var/log/pdf-api-server.log
sudo touch /var/log/celery-worker.log
sudo touch /var/log/celery-beat.log
sudo chown ladvien:ladvien /var/log/pdf-*.log /var/log/celery-*.log
sudo chmod 644 /var/log/pdf-*.log /var/log/celery-*.log
```

**3. Configure firewall:**
```bash
# UFW
sudo ufw allow from 192.168.1.0/24 to any port 8000
sudo ufw reload

# Or firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

**4. Start services:**
```bash
sudo systemctl start pdf-celery-worker
sleep 3
sudo systemctl start pdf-celery-beat
sleep 2
sudo systemctl start pdf-api-server
sleep 3
./scripts/check_services.sh
```

**5. Verify deployment:**
```bash
curl http://localhost:8000/health | jq .
curl http://192.168.1.110:8000/health | jq .
```

### Configuration Updates

**Priority order (highest to lowest):**
1. Runtime API changes → `server_configuration` table
2. Database config → Persists across restarts
3. `.env` file → Initial/default configuration

**Update config via API:**
```bash
curl -X POST http://192.168.1.110:8000/api/v1/configure \
  -H "Content-Type: application/json" \
  -d '{"watch_directories": ["/new/path/"], "restart_watcher": true}'
```

## Verification Checklist

### Critical Success Criteria
- ✅ GPU actively processing PDFs (>1GB GPU memory during processing)
- ✅ Markdown files created in OUTPUT_DIR
- ✅ Vector embeddings stored in PostgreSQL
- ✅ No errors in service logs

### Step 1: Verify GPU Usage
```bash
nvidia-smi                           # Check GPU available
nvidia-smi | grep python             # Check MinerU using GPU
ps aux | grep mineru_standalone      # Verify process running
```

**Expected:** Python process using 1-8GB GPU memory during processing

### Step 2: Verify Services
```bash
systemctl status pdf-api-server      # Should be active (running)
systemctl status pdf-celery-worker   # Should be active (running)
systemctl status pdf-celery-beat     # Should be active (running)
redis-cli ping                       # Should return PONG
curl http://localhost:11434/api/tags | jq '.models[].name' | grep nomic
```

### Step 3: Verify Database
```bash
export PGPASSWORD='YOUR_DB_PASSWORD'

# Check document statistics
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "
SELECT conversion_status, COUNT(*), MAX(updated_at) as last_update
FROM documents
GROUP BY conversion_status;"

# Check embeddings
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "
SELECT COUNT(*) as total_embeddings,
       COUNT(DISTINCT document_id) as docs_with_embeddings,
       MAX(created_at) as most_recent
FROM document_embeddings;"
```

### Step 4: Verify File System
```bash
# Check output directory
ls -la /mnt/codex_fs/research/librarian_output/

# Check recent files (last hour)
find /mnt/codex_fs/research/librarian_output -name "*.md" -mmin -60 -ls

# Verify content
head -100 $(ls -t /mnt/codex_fs/research/librarian_output/*.md | head -1)
```

### Step 5: Monitor Processing
```bash
# Check MinerU activity
tail -100 /tmp/mineru.log | grep "Completed job"

# Real-time monitoring
tail -f /tmp/mineru.log | grep -E "Layout|MFD|OCR|Processing"

# Check queue status
redis-cli llen mineru_requests
redis-cli llen mineru_results
```

## Troubleshooting

### GPU Not Being Used

**Symptoms:**
- Processing slow
- No GPU memory in `nvidia-smi`
- CPU processing in logs

**Fix:**
```bash
# Kill and restart with GPU
pkill -f mineru_standalone.py
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 REDIS_PORT=6379 \
  nohup python src/pdf_to_markdown_mcp/services/mineru_standalone.py \
  > /tmp/mineru.log 2>&1 &

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Markdown Files Not Created

**Symptoms:**
- Documents marked "completed" but no files
- Empty OUTPUT_DIR

**Fix:**
```bash
# Check permissions
touch /mnt/codex_fs/research/librarian_output/test.txt && \
  rm /mnt/codex_fs/research/librarian_output/test.txt

# Check environment
grep OUTPUT_DIRECTORY .env

# Restart worker
sudo systemctl restart pdf-celery-worker
```

### Embeddings Not Generated

**Symptoms:**
- Empty document_embeddings table
- Documents stuck in "pending" status

**Fix:**
```bash
# Restart Ollama
sudo systemctl restart ollama
ollama pull nomic-embed-text

# Test embedding generation via system diagnostic
python scripts/system_diagnostic.py

# Or check embeddings health via API
curl http://localhost:8000/health | jq '.components.embeddings'
```

### Request ID Mismatch

**Symptoms:**
- "Request ID mismatch" errors
- Multiple retries failing

**Fix:**
```bash
# Clear queues and restart
redis-cli FLUSHDB
pkill -f mineru_standalone.py
sudo systemctl restart pdf-celery-worker

# Start fresh MinerU
source .venv/bin/activate
REDIS_PORT=6379 nohup python src/pdf_to_markdown_mcp/services/mineru_standalone.py \
  > /tmp/mineru.log 2>&1 &
```

### Full System Restart

**Use this for major issues:**
```bash
# 1. Stop all services
sudo systemctl stop pdf-celery-worker pdf-celery-beat
pkill -f mineru_standalone.py

# 2. Clear caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
> /tmp/mineru.log

# 3. Clear Redis
redis-cli FLUSHDB

# 4. Start services in order
source .venv/bin/activate

# Start MinerU
CUDA_VISIBLE_DEVICES=0 REDIS_PORT=6379 \
  nohup python src/pdf_to_markdown_mcp/services/mineru_standalone.py \
  > /tmp/mineru.log 2>&1 &
sleep 5

# Start Celery
sudo systemctl start pdf-celery-worker pdf-celery-beat

# Verify
./scripts/check_services.sh
```

### Reset Stuck Documents

**For documents stuck in "processing" state:**
```bash
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian << EOF
-- Reset documents stuck in processing
UPDATE documents
SET conversion_status = 'pending'
WHERE conversion_status = 'processing'
AND updated_at < NOW() - INTERVAL '1 hour';

-- Reset stuck embeddings
UPDATE document_content
SET embedding_status = 'pending'
WHERE embedding_status = 'processing'
AND embedding_generated_at < NOW() - INTERVAL '1 hour';
EOF
```

## Health Metrics

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| GPU Memory | 1-8 GB | < 500 MB | 0 MB |
| Pending Docs | < 100 | 100-500 | > 500 |
| Processing Speed | > 10 pages/min | 5-10 | < 5 |
| Embeddings/Doc | 20-200 | < 10 | 0 |
| Queue Lengths | < 50 | 50-200 | > 200 |
| Recent Files (1hr) | > 5 | 1-5 | 0 |
| Error Rate | < 5% | 5-20% | > 20% |

## Custom Shortcuts

### QCHECK - Quick Health Check
```bash
# Run comprehensive health check
./scripts/quick_health_check.sh
```

### QGPU - GPU Status
```bash
# Check GPU usage and MinerU process
nvidia-smi && nvidia-smi | grep python
```

### QLOGS - View Recent Errors
```bash
# Show recent errors from all services
tail -100 /var/log/celery-worker.log | grep -i error
tail -100 /tmp/mineru.log | grep -i error
```

### QSTATS - Database Statistics
```bash
# Show document and embedding statistics
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "
SELECT 
    conversion_status,
    COUNT(*) as count,
    MAX(updated_at) as last_update
FROM documents
GROUP BY conversion_status;"
```

### QRESET - Emergency Reset
```bash
# Full system restart (use carefully)
./scripts/restart_services.sh
```

## Testing

### Test Suite Overview

The project has **three distinct test categories**:

1. **Unit Tests** (fast, mocked) - SQLite, CPU-only, no external services
2. **Integration Tests** (real services) - PostgreSQL, GPU, Redis, Ollama
3. **End-to-End Tests** (full pipeline) - Complete workflow validation

### Quick Test Commands

```bash
# Fast unit tests (10-30 seconds) - no prerequisites
./scripts/test_unit.sh

# Integration tests (2-5 minutes) - requires GPU + services
./scripts/test_integration.sh

# Validate environment before integration tests
./scripts/validate_test_env.sh

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m e2e               # End-to-end tests only
pytest -m gpu               # GPU-dependent tests only

# Run with coverage
./scripts/test_unit.sh --cov
```

### Integration Test Prerequisites

Integration and e2e tests require:
- ✅ NVIDIA GPU with CUDA 12.4+
- ✅ PostgreSQL 17+ with PGVector (192.168.1.104)
- ✅ Redis server (localhost:6379)
- ✅ Ollama with nomic-embed-text model
- ✅ MinerU standalone service running

**Validate prerequisites:**
```bash
./scripts/validate_test_env.sh
```

Example output:
```
✅ PostgreSQL connection
✅ PGVector extension
✅ nvidia-smi available
  GPU: NVIDIA GeForce RTX 3090
  Memory: 24576 MiB
✅ CUDA available in Python
  CUDA Version: 12.4
✅ Redis server
✅ Ollama service
✅ Ollama model (nomic-embed-text)
⚠️  MinerU process running (optional)
⚠️  Celery worker (optional)

All required prerequisites met!
```

### Key Differences: Unit vs Integration Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Database | SQLite (mocked) | PostgreSQL + PGVector |
| GPU | CPU-only | Real CUDA GPU |
| Services | All mocked | Real Redis, Ollama, MinerU |
| Duration | 10-30 seconds | 2-5 minutes |
| Purpose | Fast feedback | Real validation |
| CI/CD | Every commit | Pre-deployment |

### Test Organization

```
tests/
├── unit/              # Fast mocked tests
├── integration/       # Real service tests
├── e2e/              # Full pipeline tests
├── fixtures/
│   ├── real_database.py    # Real PostgreSQL
│   ├── real_gpu.py          # GPU validation
│   └── real_services.py     # Real Redis/Ollama
└── README.md         # Comprehensive test documentation
```

**For complete testing documentation**, see **`tests/README.md`**

### Common Test Issues

**GPU tests failing:**
```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify environment
export CUDA_VISIBLE_DEVICES=0
export MINERU_DEVICE_MODE=cuda
```

**Integration tests skipped:**
```bash
# Check what's missing
./scripts/validate_test_env.sh

# Start missing services
sudo systemctl start redis
sudo systemctl start ollama
ollama pull nomic-embed-text
```

**See `tests/README.md` for detailed troubleshooting.**

## Do Not Section

### Never
- ❌ Edit files in `/mnt/codex_fs/` directly (read-only source data)
- ❌ Modify systemd service files without testing
- ❌ Clear Redis queues during active processing
- ❌ Change database schema without Alembic migration
- ❌ Commit `.env` file to version control
- ❌ Run MinerU multiple times (only one instance)

### Always
- ✅ Use Alembic for database migrations
- ✅ Test configuration changes in development first
- ✅ Check GPU availability before processing
- ✅ Monitor logs after deployment
- ✅ Verify backups before major changes
- ✅ Document configuration changes
- ✅ Update this CLAUDE.md when adding features

## Environment Details

- **Host**: 192.168.1.110
- **API Port**: 8000
- **Database**: 192.168.1.104:5432
- **Redis**: localhost:6379 (Docker)
- **Ollama**: localhost:11434
- **User**: ladvien
- **Python**: 3.11+ (.venv)
- **GPU**: NVIDIA RTX 3090 (CUDA 12.4+)

## Key Features

- ✅ MCP-style configuration via API
- ✅ Database-persisted configuration
- ✅ Single source of truth (WATCH_DIRECTORIES + OUTPUT_DIRECTORY)
- ✅ Network accessible API
- ✅ GPU-accelerated processing (5-10x faster)
- ✅ Production-ready systemd services
- ✅ Multiple watch directories support
- ✅ Auto-restart on failures
- ✅ Comprehensive logging

## Getting Help

**If experiencing issues:**

1. Run diagnostics:
   ```bash
   python scripts/system_diagnostic.py > diagnostic_report.txt
   ```

2. Collect logs:
   ```bash
   ./scripts/view_logs.sh all > all_logs.txt
   tail -100 /tmp/mineru.log > mineru_log.txt
   nvidia-smi > gpu_state.txt
   ```

3. Check database:
   ```bash
   psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "\conninfo"
   ```

4. Review this document for troubleshooting steps

## MCP Semantic Search Server

### Overview

The MCP server provides a `search_library` tool that enables LLMs (like Claude) to semantically search the document library using hybrid search (vector similarity + BM25 keyword matching).

**Key Features:**
- 🔍 Hybrid search (vector + keyword) via Reciprocal Rank Fusion
- 📄 Returns markdown file paths for direct document access
- ⚡ Sub-second response times
- 🔧 Zero .env dependency - all configuration via MCP client
- 📅 Date range filtering support
- 🎯 Configurable similarity thresholds

### Quick Setup

**1. Add to Claude Desktop config:**

```json
{
  "mcpServers": {
    "codex-librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "pdf_to_markdown_mcp.mcp.server"],
      "cwd": "/mnt/datadrive_m2/codex_librarian",
      "env": {
        "DATABASE_URL": "postgresql://codex_librarian:PASSWORD@192.168.1.104:5432/codex_librarian",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

**2. Restart Claude Desktop**

**3. Test the tool:**
```
Search my library for "neural networks"
```

### Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | ✅ | - | PostgreSQL connection string |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | No | `nomic-embed-text` | Embedding model |
| `DB_POOL_MIN_SIZE` | No | `2` | Min connections |
| `DB_POOL_MAX_SIZE` | No | `10` | Max connections |
| `MCP_LOG_LEVEL` | No | `INFO` | Logging level |
| `SEARCH_DEFAULT_LIMIT` | No | `10` | Default results |
| `SEARCH_MAX_LIMIT` | No | `50` | Max results |
| `SEARCH_DEFAULT_SIMILARITY` | No | `0.7` | Similarity threshold |

### Tool Usage

```python
search_library(
    query: str,              # Natural language query (required)
    limit: int = 10,         # Max results (1-50)
    min_similarity: float = 0.7,  # Similarity threshold (0.0-1.0)
    tags: list[str] = None,  # Tag filters (future)
    date_from: str = None,   # ISO 8601 date filter
    date_to: str = None      # ISO 8601 date filter
)
```

**Response includes:**
- `document_id` - Database ID
- `filename` - Original PDF filename
- `markdown_path` - Path to converted markdown file
- `similarity_score` - Relevance score (0.0-1.0)
- `excerpt` - Text snippet from document
- `page_number` - Source page number
- `created_at` - Processing timestamp

### Troubleshooting

**MCP server won't start:**
```bash
# Check configuration
cat ~/.config/claude/claude_desktop_config.json

# Test database connection
psql "postgresql://user:pass@host:5432/db" -c "SELECT 1"

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

**No results found:**
```sql
-- Check document count
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "
SELECT COUNT(*) FROM documents WHERE conversion_status = 'completed';
SELECT COUNT(*) FROM document_embeddings;
"
```

**Slow searches:**
```sql
-- Ensure HNSW index exists
SELECT indexname FROM pg_indexes WHERE tablename = 'document_embeddings';

-- Create if missing
CREATE INDEX idx_embeddings_hnsw ON document_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### Documentation

See **`docs/MCP_SETUP.md`** for:
- Detailed setup instructions
- Configuration examples
- Performance tuning
- Security considerations
- Advanced usage

### Architecture

The MCP server runs as a separate process from the FastAPI server:

```
┌──────────────────────┐
│ Claude Desktop       │
│ (MCP Client)         │
└──────────┬───────────┘
           │ stdio
           ▼
┌──────────────────────┐
│ FastMCP Server       │
│ - search_library     │
└──────────┬───────────┘
           │
           ├──▶ PostgreSQL + PGVector (vector similarity)
           └──▶ Ollama (query embeddings)
```

## Notes for AI Assistants

When helping with this project:
- Always check GPU status before processing tasks
- Verify services are running before suggesting code changes
- Use provided scripts for diagnostics and monitoring
- Follow the deployment workflow for new features
- Test database changes with Alembic migrations first
- Monitor logs after making changes
- Use the verification checklist after deployments
- **For MCP issues**: Check `docs/MCP_SETUP.md` first