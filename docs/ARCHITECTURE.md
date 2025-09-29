# PDF to Markdown MCP Server - Architecture

## System Overview
A Python-based Model Context Protocol (MCP) server that converts PDFs to searchable Markdown with vector embeddings stored in PostgreSQL with PGVector. The system uses MinerU for advanced PDF processing, maintains directory structure mirroring, and provides semantic search capabilities.

## Core Features
- **Directory Structure Preservation**: Automatically mirrors PDF directory structures to Markdown outputs
- **Recursive Monitoring**: Watches multiple directories for new PDFs and maintains folder hierarchies
- **Vector Search**: Semantic and hybrid search with embeddings stored in PostgreSQL
- **Model Tracking**: Records embedding model versions for reproducibility
- **Advanced PDF Processing**: Layout-aware extraction with tables, formulas, and OCR

## Technology Stack
- **Language**: Python 3.11+
- **Package Manager**: uv
- **Web Framework**: FastAPI with Pydantic v2
- **Database**: PostgreSQL 17+ with PGVector extension
- **PDF Processing**: MinerU library
- **Embeddings**: Ollama (local) or OpenAI API
- **Testing**: pytest with TDD approach
- **Code Quality**: ruff, black, mypy, bandit
- **Development OS**: Manjaro Linux

## System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  File System    │────▶│  Watchdog    │────▶│  Task Queue    │
│  (PDF dirs)     │     │  (Recursive) │     │  (Celery)      │
└─────────────────┘     └──────────────┘     └────────────────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Directory      │     │  Converter   │◀────│  PDF Processor │
│  Mirror Engine  │────▶│  Orchestrator│     │  (MinerU)      │
└─────────────────┘     └──────────────┘     └────────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Markdown       │     │  Embedding   │     │  PostgreSQL    │
│  Output Dirs    │     │  Generator   │────▶│  + PGVector    │
└─────────────────┘     │  (Multi-Model)│    └────────────────┘
                        └──────────────┘              ▲
                                                      │
┌─────────────────────────────────────────────────────┘
│  MCP Server (FastAPI)                               
│  ├── convert_single                                 
│  ├── batch_convert (with structure preservation)    
│  ├── semantic_search                                
│  └── hybrid_search                                  
└──────────────────────────────────────────────────────
```

## Directory Structure Management

### Input/Output Mirroring
The system maintains exact directory structures between source PDFs and output Markdown:

```
pdfs/                              markdown/
├── agentic_research_results  →   ├── agentic_research_results
├── codex_articles            →   ├── codex_articles
│   ├── additional_papers     →   │   ├── additional_papers
│   ├── research_1            →   │   ├── research_1
│   └── research_2            →   │   └── research_2
└── scientific_articles       →   └── scientific_articles
    ├── air_pollution         →       ├── air_pollution
    └── ambient_particulate   →       └── ambient_particulate
```

### Configuration
```python
# Environment variables
WATCH_BASE_DIR=/data/pdfs
OUTPUT_BASE_DIR=/data/markdown
PRESERVE_STRUCTURE=true
MAX_DIRECTORY_DEPTH=10
IGNORE_PATTERNS=["*.tmp", "~*", "._*"]
```

## Database Schema

```sql
-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    source_relative_path TEXT NOT NULL,  -- Relative to watch directory
    output_path TEXT UNIQUE NOT NULL,
    output_relative_path TEXT NOT NULL,  -- Relative to output directory
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes BIGINT,
    directory_depth INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversion_status TEXT CHECK (conversion_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    metadata JSONB
);

-- Path mappings for directory structure
CREATE TABLE path_mappings (
    id SERIAL PRIMARY KEY,
    source_directory TEXT NOT NULL,
    output_directory TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    directory_level INTEGER,
    files_count INTEGER DEFAULT 0,
    last_scanned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_directory, relative_path)
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
    mineru_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings table with model tracking
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding vector(1536),  -- Dimensions based on model
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    model_version TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted images table with model tracking
CREATE TABLE document_images (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    image_index INTEGER,
    image_path TEXT,
    output_image_path TEXT,  -- Mirrored path in markdown directory
    ocr_text TEXT,
    ocr_confidence FLOAT,
    image_embedding vector(512),  -- CLIP embeddings
    embedding_model TEXT DEFAULT 'clip-vit-base-patch32',
    metadata JSONB
);

-- Embedding models registry
CREATE TABLE embedding_models (
    id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    model_type TEXT CHECK (model_type IN ('text', 'image', 'hybrid')),
    dimensions INTEGER NOT NULL,
    provider TEXT,  -- 'openai', 'ollama', 'huggingface'
    endpoint_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing queue table
CREATE TABLE processing_queue (
    id SERIAL PRIMARY KEY,
    source_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'queued',
    attempts INTEGER DEFAULT 0,
    preserve_structure BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    worker_id TEXT
);

-- Create optimized indexes
CREATE INDEX idx_documents_status ON documents(conversion_status);
CREATE INDEX idx_documents_relative_path ON documents(source_relative_path);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_path_mappings_source ON path_mappings(source_directory, relative_path);
CREATE INDEX idx_embeddings_document ON document_embeddings(document_id);
CREATE INDEX idx_embeddings_model ON document_embeddings(embedding_model);
CREATE INDEX idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_images_vector ON document_images USING ivfflat (image_embedding vector_cosine_ops);
CREATE INDEX idx_content_fulltext ON document_content USING gin(to_tsvector('english', plain_text));
```

## MCP Tools API

### 1. `convert_single`
Converts a single PDF file to Markdown while preserving directory structure.

```python
{
    "name": "convert_single",
    "description": "Convert a single PDF file to Markdown with structure preservation",
    "parameters": {
        "file_path": str,
        "output_dir": Optional[str],  # Auto-calculated if preserve_structure=True
        "preserve_structure": bool = True,
        "embedding_model": str = "text-embedding-3-small",
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
Batch converts PDF directories while maintaining folder hierarchies.

```python
{
    "name": "batch_convert",
    "description": "Batch convert PDFs with directory structure mirroring",
    "parameters": {
        "source_directory": str,
        "output_directory": str,
        "pattern": str = "**/*.pdf",
        "recursive": bool = True,
        "preserve_structure": bool = True,
        "embedding_model": str = "text-embedding-3-small",
        "store_embeddings": bool = True,
        "skip_existing": bool = True
    }
}
```

### 3. `watch_directories`
Configure and manage directory watching for automatic conversion.

```python
{
    "name": "watch_directories",
    "description": "Configure directory watching with structure preservation",
    "parameters": {
        "watch_configs": [
            {
                "source_dir": str,
                "output_dir": str,
                "recursive": bool = True,
                "patterns": List[str] = ["*.pdf"],
                "auto_convert": bool = True
            }
        ],
        "action": str = "add|remove|list"
    }
}
```

### 4. `semantic_search`
Search documents using natural language queries with model-aware filtering.

```python
{
    "name": "semantic_search",
    "description": "Search documents using semantic similarity",
    "parameters": {
        "query": str,
        "top_k": int = 10,
        "threshold": float = 0.7,
        "embedding_model": Optional[str],  # Filter by model
        "filter": {
            "directory_path": Optional[str],  # Search within specific directory
            "date_range": Optional[Tuple[datetime, datetime]],
            "has_images": Optional[bool],
            "has_tables": Optional[bool]
        }
    }
}
```

### 5. `hybrid_search`
Combines vector semantic search with full-text search.

```python
{
    "name": "hybrid_search",
    "description": "Hybrid search using both semantic and keyword matching",
    "parameters": {
        "query": str,
        "semantic_weight": float = 0.7,
        "keyword_weight": float = 0.3,
        "top_k": int = 10,
        "search_path": Optional[str]  # Limit to directory path
    }
}
```

### 6. `get_status`
Returns processing status with directory structure information.

```python
{
    "name": "get_status",
    "description": "Get processing status and directory statistics",
    "parameters": {
        "job_id": Optional[str],
        "directory_path": Optional[str],
        "include_tree": bool = False,
        "include_stats": bool = True
    }
}
```

### 7. `sync_directories`
Synchronize PDF source with Markdown output directories.

```python
{
    "name": "sync_directories",
    "description": "Sync source PDFs with output Markdown directories",
    "parameters": {
        "source_directory": str,
        "output_directory": str,
        "mode": str = "update|full|verify",
        "dry_run": bool = False
    }
}
```

### 8. `configure`
Updates server configuration including embedding models.

```python
{
    "name": "configure",
    "description": "Configure server settings and models",
    "parameters": {
        "watch_directories": Optional[List[Dict]],
        "embedding_config": {
            "text_model": str,
            "image_model": str,
            "provider": str,
            "endpoint": Optional[str]
        },
        "ocr_settings": Optional[Dict],
        "directory_settings": {
            "preserve_structure": bool,
            "max_depth": int,
            "ignore_patterns": List[str]
        }
    }
}
```

## Core Processing Pipeline

### Directory Mirror Watcher
```python
class DirectoryMirrorWatcher:
    """Watches source directories and maintains output structure"""
    
    def __init__(self, source_base: Path, output_base: Path):
        self.source_base = source_base
        self.output_base = output_base
        self.observer = Observer()
        
    def on_created(self, event):
        if event.src_path.endswith('.pdf'):
            # Calculate relative path from base
            relative_path = Path(event.src_path).relative_to(self.source_base)
            
            # Create mirrored output path
            output_path = self.output_base / relative_path.with_suffix('.md')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Queue conversion with structure preservation
            task = ConversionTask(
                source_path=event.src_path,
                output_path=str(output_path),
                relative_path=str(relative_path),
                preserve_structure=True,
                embedding_model=self.current_embedding_model
            )
            queue.add(task)
```

### PDF Processing with MinerU
MinerU provides advanced PDF processing:
- Layout-aware text extraction
- Table detection and structure preservation
- Mathematical formula recognition
- Built-in OCR for scanned content
- Image extraction with position metadata
- Multi-language support

### Embedding Generation Pipeline
Multi-model support with version tracking:

```python
class EmbeddingPipeline:
    def __init__(self):
        self.models = {
            'text-embedding-3-small': OpenAIEmbedder(dims=1536),
            'text-embedding-3-large': OpenAIEmbedder(dims=3072),
            'nomic-embed-text': OllamaEmbedder(dims=768),
            'clip-vit-base-patch32': CLIPEmbedder(dims=512)
        }
    
    def generate_embeddings(self, chunks: List[str], model_name: str):
        embedder = self.models[model_name]
        return [
            {
                'embedding': embedder.encode(chunk),
                'model': model_name,
                'dimensions': embedder.dimensions,
                'version': embedder.version
            }
            for chunk in chunks
        ]
```

### Vector Search with PGVector
Optimized similarity search with model awareness:
- **IVFFlat Index**: For datasets >1M documents
- **HNSW Index**: For <1M documents with higher recall
- **Hybrid Search**: Combines semantic and keyword matching
- **Model-specific search**: Filter by embedding model for consistency

## Data Flow

1. **File Detection**: Watchdog monitors configured PDF directories recursively
2. **Path Calculation**: Relative paths computed for structure preservation
3. **Directory Creation**: Output directory structure created to mirror source
4. **Validation**: File validated, hashed for deduplication
5. **Queue**: Task added to Celery queue with path mappings
6. **Processing**: MinerU extracts content, tables, formulas, images
7. **OCR**: Built-in OCR processes scanned content
8. **Chunking**: Content split into overlapping chunks for embedding
9. **Embeddings**: Generate embeddings with selected model, store model info
10. **Storage**: Content, vectors, and paths stored in PostgreSQL
11. **Output**: Markdown file written to mirrored directory structure
12. **Search**: Semantic and hybrid search available immediately

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
│       │   ├── directory.py     # Directory structure models
│       │   ├── embedding.py     # Embedding model configs
│       │   ├── request.py
│       │   └── response.py
│       ├── api/                 # API endpoints
│       │   ├── __init__.py
│       │   ├── convert.py
│       │   ├── search.py
│       │   ├── directory.py     # Directory management endpoints
│       │   └── status.py
│       ├── core/                # Core business logic
│       │   ├── __init__.py
│       │   ├── processor.py
│       │   ├── chunker.py
│       │   ├── watcher.py
│       │   └── mirror.py        # Directory mirroring logic
│       ├── services/            # External services
│       │   ├── __init__.py
│       │   ├── mineru.py
│       │   ├── embeddings/      # Multi-model embedding support
│       │   │   ├── __init__.py
│       │   │   ├── openai.py
│       │   │   ├── ollama.py
│       │   │   └── registry.py
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
│   ├── init_db.py
│   ├── sync_directories.py
│   └── migrate_embeddings.py
├── docs/
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
├── ARCHITECTURE.md
└── CLAUDE.md
```

## Performance Considerations

### Resource Management
- Connection pooling with pgbouncer for high concurrency
- Batch processing for embeddings (100 chunks/batch)
- Streaming for large PDF files (>50MB)
- Memory-mapped file reading for efficiency
- Async I/O throughout FastAPI endpoints

### Optimization Strategies
- Directory structure caching in Redis
- Lazy loading of large markdown files
- Embedding cache for frequently accessed documents
- Index optimization based on query patterns
- Automatic vacuum and reindexing schedules
- Parallel processing for directory trees

### Scalability
- Horizontal scaling via multiple Celery workers
- Database read replicas for search operations
- PGVector index partitioning by directory path
- CDN for serving converted Markdown files
- Queue prioritization based on directory depth

## Security Considerations

### Input Validation
- File type validation (PDF mime-type checking)
- File size limits (500MB default, configurable)
- Path traversal prevention in directory operations
- PDF malware scanning with ClamAV integration
- SQL injection prevention via parameterized queries

### Access Control
- Directory-level permissions
- API key authentication for MCP tools
- Rate limiting per API key
- Audit logging for all conversions

### Resource Limits
- Maximum file size: 500MB (configurable)
- Processing timeout: 5 minutes per file
- Queue depth limit: 1000 items
- Memory limit per worker: 4GB
- Maximum directory depth: 10 levels

### Data Protection
- Sensitive paths never logged
- Credentials in environment variables
- Database encryption at rest
- TLS for all network connections
- Secure temporary file handling

## Error Handling

### Error Categories
- `PathError`: Invalid paths or permission issues
- `ValidationError`: Invalid input data
- `ProcessingError`: PDF processing failures
- `EmbeddingError`: Embedding generation issues
- `DatabaseError`: Database operation failures
- `DirectoryError`: Directory structure issues
- `SystemError`: System-level issues

### Recovery Strategies
- Automatic retries with exponential backoff
- Dead letter queue for failed tasks
- Partial conversion recovery (resume from page)
- Directory structure repair utilities
- Graceful degradation for non-critical features
- Comprehensive error logging with context

## Monitoring

### Health Checks
- `/health`: Overall system health
- `/ready`: Readiness for traffic
- `/metrics`: Prometheus metrics
- `/directory/status`: Directory sync status

### Key Metrics
- PDF processing rate (docs/minute)
- Average processing time per page
- Queue depth and wait times
- Directory sync lag
- Error rate by category
- Search latency P50/P95/P99
- Embedding generation throughput
- Database connection pool utilization
- Disk usage for output directories

### Logging
- Structured logging with JSON format
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Directory operation audit trail
- Centralized log aggregation (ELK stack)
- Correlation IDs for request tracking
- Performance profiling for slow operations

## Configuration Examples

### Basic Setup
```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

directories:
  watch_base: /data/pdfs
  output_base: /data/markdown
  preserve_structure: true
  max_depth: 10
  scan_interval: 60  # seconds

embeddings:
  default_model: text-embedding-3-small
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    ollama:
      endpoint: http://localhost:11434

processing:
  max_file_size_mb: 500
  timeout_seconds: 300
  batch_size: 10
  concurrent_workers: 4
```

### Directory Watch Configuration
```json
{
  "watch_configs": [
    {
      "source_dir": "/data/pdfs/scientific_articles",
      "output_dir": "/data/markdown/scientific_articles",
      "recursive": true,
      "patterns": ["*.pdf"],
      "auto_convert": true,
      "embedding_model": "text-embedding-3-small"
    },
    {
      "source_dir": "/data/pdfs/codex_articles",
      "output_dir": "/data/markdown/codex_articles",
      "recursive": true,
      "patterns": ["*.pdf"],
      "auto_convert": true,
      "embedding_model": "nomic-embed-text"
    }
  ]
}
```

## Dependencies

### Core Dependencies
- `mineru`: Advanced PDF processing with layout preservation
- `fastapi`: High-performance web framework
- `pydantic`: Data validation and settings
- `sqlalchemy`: ORM with async support
- `pgvector`: Vector similarity operations
- `celery`: Distributed task queue
- `redis`: Caching and queue backend
- `ollama`: Local embedding generation
- `openai`: Cloud embedding API
- `watchdog`: File system monitoring

### Development Dependencies
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `ruff`: Fast Python linter
- `black`: Code formatting
- `mypy`: Static type checking
- `bandit`: Security vulnerability scanning
- `pre-commit`: Git hooks for code quality

## Migration Path

### From Existing System
1. **Database Migration**: Run Alembic migrations to add new columns
2. **Model Registry**: Populate embedding_models table
3. **Path Mapping**: Generate path_mappings for existing documents
4. **Directory Sync**: Run initial sync to create Markdown structure
5. **Embedding Update**: Optionally regenerate with model tracking

### Rollback Strategy
1. Keep original schema backup
2. Maintain conversion logs
3. Support dual-mode operation during transition
4. Gradual migration by directory

## Future Enhancements

### Planned Features
- Multi-format support (DOCX, EPUB, HTML)
- Incremental updates for modified PDFs
- WebSocket support for real-time progress
- Distributed processing across multiple nodes
- S3/MinIO backend for large deployments
- GraphQL API for flexible queries
- Web UI for monitoring and management
- Automatic quality assessment scores
- Language detection and multi-lingual embeddings
- Custom chunking strategies per document type