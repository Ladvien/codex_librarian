# Quick Reference - PDF to Markdown MCP

## 🚀 Most Used Commands

```bash
# 1. Check system health
./scripts/quick_health_check.sh

# 2. View GPU usage
nvidia-smi | grep python

# 3. Check recent markdown files
ls -lt /mnt/codex_fs/research/librarian_output/*.md | head -5

# 4. Monitor MinerU processing
tail -f /tmp/mineru.log | grep "Processing\|Completed"

# 5. Database document stats
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "SELECT conversion_status, COUNT(*) FROM documents GROUP BY conversion_status;"
```

## 🔧 Fix Common Issues

### Restart MinerU with GPU
```bash
pkill -f mineru_standalone.py
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 REDIS_PORT=6379 nohup python src/pdf_to_markdown_mcp/services/mineru_standalone.py > /tmp/mineru.log 2>&1 &
```

### Clear Stuck Queues
```bash
redis-cli FLUSHDB
sudo systemctl restart pdf-celery-worker
```

### Reset Failed Documents
```bash
export PGPASSWORD='YOUR_DB_PASSWORD'
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "UPDATE documents SET conversion_status = 'pending' WHERE conversion_status = 'failed';"
```

## 📁 Key File Locations

| What | Where |
|------|-------|
| Project Root | `/mnt/datadrive_m2/codex_librarian/` |
| Input PDFs | `/mnt/codex_fs/research/codex_articles/` |
| Output Markdown | `/mnt/codex_fs/research/librarian_output/` |
| MinerU Log | `/tmp/mineru.log` |
| Celery Log | `/var/log/celery-worker.log` |
| Environment | `.env` |

## 🎯 Critical Success Indicators

✅ **System is healthy when:**
- GPU memory > 1GB during processing
- Markdown files appearing in output dir
- Embeddings count growing in database
- No "Request ID mismatch" errors
- Queue lengths < 50