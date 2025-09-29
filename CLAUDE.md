# PDF to Markdown MCP Server

## Project Overview

GPU-accelerated PDF processing system that converts PDFs to markdown with vector embeddings. Uses MinerU standalone service for CUDA-accelerated processing, Celery for task orchestration, and PostgreSQL with PGVector for embeddings storage.

**Key Capabilities:**
- Watches directories for new PDFs (configured via API)
- GPU-accelerated conversion (5-10x faster than CPU)
- Vector embeddings for semantic search
- MCP-compatible REST API
- Persistent configuration survives restarts

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI + Celery
- **Database**: PostgreSQL 17+ with PGVector
- **Queue**: Redis (Docker)
- **GPU**: CUDA 12.4+ (MinerU)
- **Embeddings**: Ollama (nomic-embed-text)
- **Process Management**: systemd services

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

## Notes for AI Assistants

When helping with this project:
- Always check GPU status before processing tasks
- Verify services are running before suggesting code changes
- Use provided scripts for diagnostics and monitoring
- Follow the deployment workflow for new features
- Test database changes with Alembic migrations first
- Monitor logs after making changes
- Use the verification checklist after deployments