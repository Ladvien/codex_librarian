# MCP Server Setup Guide

## Overview

The codex-librarian MCP server provides semantic search capabilities for your document library through the `search_library` tool. It uses hybrid search (vector similarity + BM25 keyword matching) with Reciprocal Rank Fusion for state-of-the-art retrieval performance.

**Key Features:**
- ✅ Zero `.env` file dependency - all configuration via MCP client
- ✅ Hybrid search (vector + keyword) for best results
- ✅ Returns markdown file paths for direct document access
- ✅ Sub-second search response times
- ✅ Filtering by date range
- ✅ Configurable result limits and similarity thresholds

## Quick Start

### Prerequisites

1. **PostgreSQL with PGVector** - Running and accessible
2. **Ollama** - Running locally with `nomic-embed-text` model
3. **Python 3.11+** - With uv package manager
4. **MCP Client** - Claude Desktop or compatible client

### Installation

1. **Install dependencies:**
   ```bash
   cd /path/to/codex_librarian
   uv sync
   ```

2. **Verify Ollama is running:**
   ```bash
   ollama pull nomic-embed-text
   curl http://localhost:11434/api/tags
   ```

3. **Test database connection:**
   ```bash
   psql -h <db_host> -U <db_user> -d <db_name> -c "SELECT COUNT(*) FROM documents"
   ```

### Configuration

Add to your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/claude/claude_desktop_config.json`

#### Minimal Configuration

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

#### Full Configuration (with all options)

```json
{
  "mcpServers": {
    "codex-librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "pdf_to_markdown_mcp.mcp.server"],
      "cwd": "/mnt/datadrive_m2/codex_librarian",
      "env": {
        "DATABASE_URL": "postgresql://codex_librarian:PASSWORD@192.168.1.104:5432/codex_librarian",
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "nomic-embed-text",
        "DB_POOL_MIN_SIZE": "2",
        "DB_POOL_MAX_SIZE": "10",
        "DB_POOL_TIMEOUT": "30",
        "MCP_LOG_LEVEL": "INFO",
        "SEARCH_DEFAULT_LIMIT": "10",
        "SEARCH_MAX_LIMIT": "50",
        "SEARCH_DEFAULT_SIMILARITY": "0.7"
      }
    }
  }
}
```

### Configuration Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `DATABASE_URL` | ✅ Yes | - | PostgreSQL connection string with credentials |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | No | `nomic-embed-text` | Embedding model name |
| `DB_POOL_MIN_SIZE` | No | `2` | Minimum database connections |
| `DB_POOL_MAX_SIZE` | No | `10` | Maximum database connections |
| `DB_POOL_TIMEOUT` | No | `30` | Connection timeout in seconds |
| `MCP_LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `SEARCH_DEFAULT_LIMIT` | No | `10` | Default number of results |
| `SEARCH_MAX_LIMIT` | No | `50` | Maximum number of results |
| `SEARCH_DEFAULT_SIMILARITY` | No | `0.7` | Default similarity threshold (0.0-1.0) |

## Usage

### Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

### Search Examples

**Basic search:**
```
Can you search my library for papers about "transformer architectures"?
```

**With custom limit:**
```
Search my library for "machine learning optimization" and show me the top 20 results.
```

**With date filter:**
```
Find documents about "neural networks" processed after 2025-01-01.
```

**With similarity threshold:**
```
Search for "attention mechanisms" with at least 80% similarity.
```

### Tool Parameters

The `search_library` tool accepts these parameters:

```python
search_library(
    query: str,              # Required: Natural language query
    limit: int = 10,         # Optional: Max results (1-50)
    min_similarity: float = 0.7,  # Optional: Similarity threshold (0.0-1.0)
    tags: list[str] = None,  # Optional: Tag filters (not yet implemented)
    date_from: str = None,   # Optional: ISO 8601 date (e.g., "2025-01-01")
    date_to: str = None      # Optional: ISO 8601 date
)
```

### Response Format

```json
{
  "results": [
    {
      "document_id": 123,
      "filename": "transformer_paper.pdf",
      "source_path": "/mnt/codex_fs/research/transformer_paper.pdf",
      "markdown_path": "/mnt/codex_fs/research/librarian_output/transformer_paper.md",
      "similarity_score": 0.92,
      "excerpt": "Attention mechanisms allow the model to focus...",
      "page_number": 5,
      "chunk_index": 12,
      "created_at": "2025-09-29T10:30:00Z"
    }
  ],
  "query": "transformer architectures",
  "total_results": 3,
  "search_duration_ms": 45
}
```

## Troubleshooting

### MCP Server Won't Start

**Check configuration:**
```bash
# Verify DATABASE_URL is set correctly
echo $DATABASE_URL

# Test database connection
psql "$DATABASE_URL" -c "SELECT 1"
```

**Check logs:**
```bash
# Claude Desktop logs location:
# macOS: ~/Library/Logs/Claude/
# Windows: %APPDATA%\Claude\logs\
# Linux: ~/.config/claude/logs/

# Look for MCP server errors
tail -f ~/.config/claude/logs/mcp.log
```

### "Database connection failed"

**Causes:**
- Database not running
- Incorrect credentials in `DATABASE_URL`
- Network/firewall blocking connection
- PostgreSQL not accepting connections

**Solution:**
```bash
# Test connection
psql -h <host> -U <user> -d <database> -c "\conninfo"

# Check PostgreSQL is running
systemctl status postgresql  # or: pg_isready

# Verify PGVector extension
psql "$DATABASE_URL" -c "SELECT * FROM pg_extension WHERE extname = 'vector'"
```

### "Failed to generate embedding"

**Causes:**
- Ollama not running
- Incorrect `OLLAMA_URL`
- Model not downloaded

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve &

# Pull embedding model
ollama pull nomic-embed-text

# Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test query"
}'
```

### Slow Search Performance

**Causes:**
- Missing database indexes
- Large result sets
- Slow embedding generation

**Solutions:**
```sql
-- Check if HNSW index exists
SELECT indexname FROM pg_indexes
WHERE tablename = 'document_embeddings';

-- Create HNSW index if missing (see CLAUDE.md)
CREATE INDEX idx_embeddings_hnsw ON document_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Analyze tables
ANALYZE documents;
ANALYZE document_content;
ANALYZE document_embeddings;
```

### No Results Found

**Causes:**
- No documents processed yet
- Similarity threshold too high
- Query embedding mismatch

**Solutions:**
```sql
-- Check document count
SELECT COUNT(*) FROM documents WHERE conversion_status = 'completed';

-- Check embedding count
SELECT COUNT(*) FROM document_embeddings;

-- Test with lower similarity threshold
# In Claude: "Search with min_similarity=0.5"
```

## Advanced Configuration

### Using Remote Ollama

If Ollama is running on another machine:

```json
{
  "env": {
    "OLLAMA_URL": "http://192.168.1.100:11434",
    ...
  }
}
```

### Using Different Embedding Model

```json
{
  "env": {
    "OLLAMA_MODEL": "all-minilm",
    ...
  }
}
```

**Note:** Changing the embedding model requires re-embedding all documents.

### Adjusting Connection Pool

For high-concurrency scenarios:

```json
{
  "env": {
    "DB_POOL_MIN_SIZE": "5",
    "DB_POOL_MAX_SIZE": "20",
    ...
  }
}
```

### Debug Logging

```json
{
  "env": {
    "MCP_LOG_LEVEL": "DEBUG",
    ...
  }
}
```

## Performance Tuning

### Database Indexes

Ensure these indexes exist:

```sql
-- HNSW index for vector similarity (most important)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
ON document_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_content_fts
ON document_content USING gin(to_tsvector('english', chunk_text));

-- Date range filtering
CREATE INDEX IF NOT EXISTS idx_documents_created_at
ON documents(created_at);

-- Status filtering
CREATE INDEX IF NOT EXISTS idx_documents_status
ON documents(conversion_status);
```

### Expected Performance

| Library Size | Search Time (P95) |
|-------------|-------------------|
| < 10K docs | < 100ms |
| 10K - 100K docs | < 300ms |
| 100K - 1M docs | < 500ms |
| > 1M docs | < 1s |

### Optimization Tips

1. **Keep Ollama local** - Network latency adds 50-200ms
2. **Use HNSW indexes** - 10-100x faster than sequential scan
3. **Limit result count** - More results = more processing time
4. **Adjust similarity threshold** - Higher threshold = fewer results to process
5. **Regular VACUUM** - `VACUUM ANALYZE document_embeddings;`

## Security Considerations

### Database Credentials

**Never commit credentials!** Always set them via MCP client config.

### Network Access

If database is remote:
- Use SSL/TLS connections: `postgresql://user:pass@host:5432/db?sslmode=require`
- Consider VPN or SSH tunneling for additional security
- Restrict PostgreSQL to specific IPs via `pg_hba.conf`

### Read-Only Access

For production, consider using a read-only database user:

```sql
CREATE USER mcp_reader WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE codex_librarian TO mcp_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO mcp_reader;
```

## Support

For issues or questions:

1. **Check CLAUDE.md** - Comprehensive system documentation
2. **Review logs** - Claude Desktop MCP logs
3. **Run diagnostics** - `python scripts/system_diagnostic.py`
4. **GitHub Issues** - https://github.com/Ladvien/codex_librarian/issues

## Changelog

### v0.1.0 (2025-09-29)

- Initial release
- `search_library` tool with hybrid search
- Environment-based configuration
- PostgreSQL + PGVector integration
- Ollama embeddings support
- Reciprocal Rank Fusion ranking
- Date range filtering