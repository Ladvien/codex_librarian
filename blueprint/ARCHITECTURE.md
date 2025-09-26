# PDF to Markdown MCP Server - Architecture

## System Overview
A Python-based Model Context Protocol (MCP) server that converts PDFs to searchable Markdown with vector embeddings stored in PostgreSQL with PGVector. The system uses MinerU for advanced PDF processing and provides semantic search capabilities.

## Core Components

### Technology Stack
- **Language**: Python 3.11+
- **Package Manager**: uv
- **Web Framework**: FastAPI with Pydantic v2
- **Database**: PostgreSQL 17+ with PGVector extension
- **PDF Processing**: MinerU library
- **Embeddings**: Ollama (local) or OpenAI API
- **Testing**: pytest with TDD approach
- **Code Quality**: ruff, black, mypy, bandit
- **Development OS**: Manjaro Linux

### System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  File System    │────▶│  Watchdog    │────▶│  Task Queue    │
│  (Recursive)    │     │  (Python)    │     │  (Celery)      │
└─────────────────┘     └──────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  MCP Server     │◀────│  Converter   │◀────│  PDF Processor │
│  (FastAPI)      │     │  Orchestrator│     │  (MinerU)      │
└─────────────────┘     └──────────────┘     └────────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  PostgreSQL     │     │  Embedding   │     │  MinerU OCR    │
│  + PGVector     │◀────│  Generator   │     │  (Built-in)    │
└─────────────────┘     │  (Ollama/    │     └────────────────┘
                        │   OpenAI)    │
                        └──────────────┘
```

## Database Schema

```sql
-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversion_status TEXT CHECK (conversion_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    metadata JSONB
);

-- Converted content table
CREATE TABLE document_content (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    markdown_content TEXT,
    plain_text TEXT,
    page_count INTEGER,
    has_images BOOLEAN DEFAULT FALSE,
    has_tables BOOLEAN DEFAULT FALSE,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings table
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding vector(1536),  -- Adjust dimensions based on model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted images table
CREATE TABLE document_images (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    image_index INTEGER,
    image_path TEXT,
    ocr_text TEXT,
    ocr_confidence FLOAT,
    image_embedding vector(512),  -- CLIP embeddings for image search
    metadata JSONB
);

-- Processing queue table
CREATE TABLE processing_queue (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'queued',
    attempts INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    worker_id TEXT
);

-- Create indexes for performance
CREATE INDEX idx_documents_status ON documents(conversion_status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_embeddings_document ON document_embeddings(document_id);
CREATE INDEX idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_images_vector ON document_images USING ivfflat (image_embedding vector_cosine_ops);
CREATE INDEX idx_content_fulltext ON document_content USING gin(to_tsvector('english', plain_text));
```

## MCP Tools API

### 1. `convert_single`
Converts a single PDF file to Markdown and stores in database.

```python
{
    "name": "convert_single",
    "description": "Convert a single PDF file to Markdown and store in database",
    "parameters": {
        "file_path": str,
        "output_dir": Optional[str],
        "store_embeddings": bool = True,
        "options": {
            "ocr_language": str = "eng",
            "preserve_layout": bool = True,
            "chunk_size": int = 1000,
            "chunk_overlap": int = 200
        }
    }
}
```

### 2. `batch_convert`
Batch converts multiple PDF files based on pattern matching.

```python
{
    "name": "batch_convert",
    "description": "Batch convert PDF files in a directory",
    "parameters": {
        "directory": str,
        "pattern": str = "**/*.pdf",
        "recursive": bool = True,
        "output_base": Optional[str],
        "store_embeddings": bool = True
    }
}
```

### 3. `semantic_search`
Search documents using natural language queries with vector similarity.

```python
{
    "name": "semantic_search",
    "description": "Search documents using semantic similarity",
    "parameters": {
        "query": str,
        "top_k": int = 10,
        "threshold": float = 0.7,
        "filter": Optional[Dict]
    }
}
```

### 4. `hybrid_search`
Combines vector semantic search with full-text search for best results.

```python
{
    "name": "hybrid_search",
    "description": "Hybrid search using both semantic and keyword matching",
    "parameters": {
        "query": str,
        "semantic_weight": float = 0.7,
        "keyword_weight": float = 0.3,
        "top_k": int = 10
    }
}
```

### 5. `get_status`
Returns current processing status and queue information.

```python
{
    "name": "get_status",
    "description": "Get processing status from database",
    "parameters": {
        "job_id": Optional[str],
        "include_stats": bool = True
    }
}
```

### 6. `configure`
Updates server configuration dynamically.

```python
{
    "name": "configure",
    "description": "Configure server settings",
    "parameters": {
        "watch_directories": Optional[List[str]],
        "embedding_config": Optional[Dict],
        "ocr_settings": Optional[Dict]
    }
}
```

### 7. `stream_progress`
Server-sent events endpoint for real-time progress updates.

```python
{
    "name": "stream_progress",
    "description": "Stream real-time progress updates",
    "parameters": {
        "filter": Optional[Dict]
    }
}
```

### 8. `find_similar`
Find documents similar to a given document.

```python
{
    "name": "find_similar",
    "description": "Find documents similar to a reference document",
    "parameters": {
        "document_id": int,
        "top_k": int = 5,
        "min_similarity": float = 0.6
    }
}
```

## Core Processing Pipeline

### PDF Processing with MinerU
MinerU provides advanced PDF processing capabilities:
- Layout-aware text extraction
- Table detection and extraction
- Formula recognition and extraction
- Built-in OCR for scanned content
- Automatic content chunking for embeddings
- Multi-language support

### Embedding Generation
Two options for generating embeddings:
1. **Ollama (Local)**: Running on localhost for privacy and cost efficiency
2. **OpenAI API**: For highest quality embeddings when needed

### Vector Search with PGVector
PGVector enables efficient similarity search:
- **IVFFlat Index**: For large datasets with good performance
- **HNSW Index**: For smaller datasets with highest recall
- **Hybrid Search**: Combines semantic and keyword search

## Project Structure

```
pdf-to-markdown-mcp/
├── src/
│   └── pdf_to_markdown_mcp/
│       ├── __init__.py
│       ├── main.py              # FastAPI application
│       ├── models/              # Pydantic models
│       │   ├── __init__.py
│       │   ├── document.py
│       │   ├── request.py
│       │   └── response.py
│       ├── api/                 # API endpoints
│       │   ├── __init__.py
│       │   ├── convert.py
│       │   ├── search.py
│       │   └── status.py
│       ├── core/                # Core business logic
│       │   ├── __init__.py
│       │   ├── processor.py
│       │   ├── chunker.py
│       │   └── watcher.py
│       ├── services/            # External services
│       │   ├── __init__.py
│       │   ├── mineru.py
│       │   ├── embeddings.py
│       │   └── database.py
│       ├── db/                  # Database operations
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── session.py
│       │   └── migrations/
│       ├── worker/              # Celery tasks
│       │   ├── __init__.py
│       │   ├── tasks.py
│       │   └── celery.py
│       └── config.py            # Settings management
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── alembic/
│   └── versions/
├── scripts/
├── docs/
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
├── ARCHITECTURE.md
└── CLAUDE.md
```

## Data Flow

1. **File Detection**: Watchdog monitors directories for new PDFs
2. **Validation**: File validated and hashed for deduplication
3. **Queue**: Task added to Celery queue for processing
4. **Processing**: MinerU extracts content, tables, formulas, images
5. **OCR**: Built-in OCR processes scanned content
6. **Chunking**: Content split into overlapping chunks
7. **Embeddings**: Generate embeddings via Ollama or OpenAI
8. **Storage**: Content and vectors stored in PostgreSQL
9. **Search**: Semantic and hybrid search available immediately

## Performance Considerations

### Resource Management
- Connection pooling for PostgreSQL
- Batch processing for embeddings
- Streaming for large PDF files
- Memory-mapped file reading
- Async I/O throughout

### Optimization Strategies
- Lazy loading of large objects
- Caching frequently accessed embeddings
- Index optimization based on query patterns
- Automatic vacuum and reindexing
- Query result caching

### Scalability
- Horizontal scaling via Celery workers
- Database read replicas for search
- PGVector index partitioning for large datasets

## Security Considerations

### Input Validation
- File type and size validation
- Path traversal prevention
- PDF malware scanning
- SQL injection prevention via parameterized queries

### Resource Limits
- Maximum file size: 500MB (configurable)
- Processing timeout: 5 minutes per file
- Queue depth limit: 1000 items
- Memory limit per worker: 4GB

### Data Protection
- Sensitive data never logged
- Credentials in environment variables
- Database encryption at rest
- TLS for all network connections

## Error Handling

### Error Categories
- `ValidationError`: Invalid input data
- `ProcessingError`: PDF processing failures
- `EmbeddingError`: Embedding generation issues
- `DatabaseError`: Database operation failures
- `SystemError`: System-level issues

### Recovery Strategies
- Automatic retries with exponential backoff
- Dead letter queue for failed tasks
- Graceful degradation for non-critical features
- Comprehensive error logging and monitoring

## Monitoring

### Health Checks
- `/health`: Overall system health
- `/ready`: Readiness for traffic
- `/metrics`: Prometheus metrics

### Key Metrics
- PDF processing rate
- Average processing time
- Queue depth
- Error rate
- Search latency
- Embedding generation time
- Database connection pool status

### Logging
- Structured logging with context
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Centralized log aggregation
- Correlation IDs for request tracking

## Dependencies

### Core Dependencies
- `mineru`: Advanced PDF processing
- `fastapi`: Web framework
- `pydantic`: Data validation
- `sqlalchemy`: ORM
- `pgvector`: Vector operations
- `celery`: Task queue
- `ollama`: Local embeddings
- `watchdog`: File monitoring

### Development Dependencies
- `pytest`: Testing framework
- `ruff`: Linting
- `black`: Code formatting
- `mypy`: Type checking
- `bandit`: Security scanning
- `pre-commit`: Git hooks