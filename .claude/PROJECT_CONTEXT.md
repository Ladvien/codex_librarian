# PDF to Markdown MCP Project Context

## Project Overview
PDF to Markdown conversion system with GPU acceleration, using MinerU for processing and PostgreSQL with PGVector for embedding storage.

## Critical System Information

### Service Endpoints
- **FastAPI**: http://localhost:8000
- **PostgreSQL**: 192.168.1.104:5432
- **Redis**: localhost:6379
- **Ollama**: http://localhost:11434

### Key Directories
- **Project Root**: `/mnt/datadrive_m2/codex_librarian/`
- **Input PDFs**: `/mnt/codex_fs/research/codex_articles/`
- **Output Markdown**: `/mnt/codex_fs/research/librarian_output/`
- **Virtual Environment**: `/mnt/datadrive_m2/codex_librarian/.venv`

### Critical Files
- **MinerU Log**: `/tmp/mineru.log`
- **Celery Log**: `/var/log/celery-worker.log`
- **Environment**: `.env` (contains DATABASE_URL, API keys, etc.)

## Common Commands

### Service Management
```bash
# Check system health
./scripts/quick_health_check.sh

# Run full diagnostics
python scripts/system_diagnostic.py

# Restart services (requires sudo)
sudo systemctl restart pdf-celery-worker pdf-celery-beat

# Start MinerU with GPU
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 REDIS_PORT=6379 nohup python src/pdf_to_markdown_mcp/services/mineru_standalone.py > /tmp/mineru.log 2>&1 &
```

### Database Operations
```bash
# Connect to database
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian

# Apply migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Description"
```

### Queue Management
```bash
# Check Redis queues
redis-cli llen mineru_requests
redis-cli llen celery

# Clear all queues (use carefully!)
redis-cli FLUSHDB
```

## Known Issues & Solutions

### Issue: Multiple MinerU Processes
**Symptom**: Request ID mismatches, high GPU memory usage
**Solution**:
```bash
pkill -f mineru_standalone.py
# Then restart single instance
```

### Issue: Embeddings Not Generating
**Symptom**: document_embeddings table not growing
**Solution**: Check async/sync in tasks.py line 817-820

### Issue: Port Conflicts
**Symptom**: Connection refused errors
**Solution**: Ensure all services use configured ports from .env, not hardcoded

## Testing Procedures

### Quick Health Check
1. GPU usage: `nvidia-smi | grep python`
2. Service status: `systemctl status pdf-celery-worker`
3. Recent files: `ls -lt /mnt/codex_fs/research/librarian_output/*.md | head -5`
4. Database stats: Check documents table for completed count

### Full Verification
Run: `python scripts/system_diagnostic.py`
Review: PRODUCTION_CHECKLIST.md for comprehensive steps

## Development Workflow

1. **Always activate venv first**: `source .venv/bin/activate`
2. **Run tests before changes**: `pytest tests/`
3. **Format code**: `black src/ tests/`
4. **Check types**: `mypy src/`
5. **Security scan**: `bandit -r src/`

## Security Notes
- Database password is in .env (YOUR_DB_PASSWORD)
- Never commit .env file
- Use environment variables for all secrets
- Path validation temporarily disabled for testing

## Contact & Resources
- Project Docs: See README.md, TROUBLESHOOTING.md, PRODUCTION_CHECKLIST.md
- Architecture: See CLAUDE.md for development guidelines
- Deployment: See DEPLOYMENT.md for production setup