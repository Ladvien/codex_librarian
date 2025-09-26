# Advanced Usage

This guide covers advanced features and deployment scenarios for the PDF to Markdown MCP Server.

## Production Deployment

### Docker Deployment

Create a `docker-compose.yml` for production deployment:

```yaml
version: '3.8'

services:
  postgresql:
    image: pgvector/pgvector:0.5.1-pg15
    environment:
      POSTGRES_DB: pdf_to_markdown_mcp
      POSTGRES_USER: pdf_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  pdf-mcp-server:
    build: .
    environment:
      - DATABASE_URL=postgresql://pdf_user:secure_password@postgresql:5432/pdf_to_markdown_mcp
      - REDIS_URL=redis://redis:6379/0
      - EMBEDDING_PROVIDER=ollama
    ports:
      - "8000:8000"
    depends_on:
      - postgresql
      - redis
    volumes:
      - ./data:/app/data

  celery-worker:
    build: .
    command: celery -A pdf_to_markdown_mcp.worker.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://pdf_user:secure_password@postgresql:5432/pdf_to_markdown_mcp
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgresql
      - redis
    volumes:
      - ./data:/app/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

Example Kubernetes deployment with Helm chart structure:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pdf-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pdf-mcp-server
  template:
    metadata:
      labels:
        app: pdf-mcp-server
    spec:
      containers:
      - name: pdf-mcp-server
        image: pdf-to-markdown-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: connection-string
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
```

### Nginx Reverse Proxy

Configure Nginx for production:

```nginx
upstream pdf_mcp_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://pdf_mcp_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files (if serving docs)
    location /static/ {
        alias /var/www/pdf-mcp/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## High-Performance Configuration

### Database Optimization

PostgreSQL tuning for vector operations:

```sql
-- postgresql.conf optimizations
shared_preload_libraries = 'pg_stat_statements,pgvector'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 16MB

-- Create optimal indexes
CREATE INDEX CONCURRENTLY idx_document_embeddings_vector
ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Analyze table statistics
ANALYZE document_embeddings;
```

### Redis Configuration

Optimize Redis for Celery tasks:

```conf
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

### Celery Optimization

Configure Celery for high throughput:

```python
# celery_config.py
from celery import Celery

app = Celery('pdf_to_markdown_mcp')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_expires=3600,
    worker_concurrency=4,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)
```

## Monitoring and Observability

### Prometheus Metrics

Enable metrics collection:

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total',
                       'Total HTTP requests',
                       ['method', 'endpoint', 'status'])

REQUEST_LATENCY = Histogram('http_request_duration_seconds',
                           'HTTP request latency')

# PDF processing metrics
PDF_PROCESSING_TIME = Histogram('pdf_processing_duration_seconds',
                                'Time spent processing PDFs')

PDF_PROCESSED_TOTAL = Counter('pdfs_processed_total',
                             'Total PDFs processed')

# Database metrics
DB_CONNECTIONS = Gauge('database_connections_active',
                      'Active database connections')
```

### Health Checks

Implement comprehensive health checks:

```python
# health.py
from fastapi import APIRouter
from sqlalchemy import text
import redis
import ollama

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {}
    }

    # Database check
    try:
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        checks["services"]["database"] = "healthy"
    except Exception as e:
        checks["services"]["database"] = f"unhealthy: {e}"
        checks["status"] = "unhealthy"

    # Redis check
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        checks["services"]["redis"] = "healthy"
    except Exception as e:
        checks["services"]["redis"] = f"unhealthy: {e}"
        checks["status"] = "unhealthy"

    # Ollama check (if enabled)
    if settings.EMBEDDING_PROVIDER in ["ollama", "both"]:
        try:
            ollama.list()
            checks["services"]["ollama"] = "healthy"
        except Exception as e:
            checks["services"]["ollama"] = f"unhealthy: {e}"
            checks["status"] = "unhealthy"

    return checks
```

### Logging Configuration

Structured logging for production:

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger()
```

## Advanced Features

### Custom Embedding Models

Implement custom embedding providers:

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Load your custom model

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Custom implementation
        return embeddings
```

### Batch Processing

Process multiple PDFs efficiently:

```python
@app.post("/batch/convert")
async def batch_convert(
    batch_request: BatchConvertRequest,
    background_tasks: BackgroundTasks
):
    task_ids = []
    for file_path in batch_request.file_paths:
        task = process_pdf_async.delay(file_path)
        task_ids.append(task.id)

    return {"task_ids": task_ids, "status": "submitted"}
```

### Streaming Responses

Stream processing progress:

```python
from fastapi.responses import StreamingResponse

@app.get("/stream/process/{task_id}")
async def stream_progress(task_id: str):
    async def generate_progress():
        while True:
            result = celery_app.AsyncResult(task_id)
            if result.ready():
                yield f"data: {result.result}\n\n"
                break
            yield f"data: {{'status': 'processing', 'progress': {result.info}}}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate_progress(), media_type="text/plain")
```

## Security Hardening

### API Key Authentication

Implement API key security:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    if credentials.credentials not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return credentials.credentials
```

### Rate Limiting

Implement rate limiting:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/convert")
@limiter.limit("10/minute")
async def convert_pdf(request: Request, ...):
    # Implementation
```

## Performance Tuning

### Connection Pooling

Optimize database connections:

```python
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True
)
```

### Caching Strategy

Implement intelligent caching:

```python
from functools import lru_cache
import asyncio

class EmbeddingCache:
    def __init__(self):
        self._cache = {}

    async def get_or_compute(self, text: str, compute_func):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        embedding = await compute_func(text)
        self._cache[text_hash] = embedding
        return embedding
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large PDFs**:
   ```python
   # Process in chunks
   CHUNK_SIZE = 1024 * 1024  # 1MB chunks
   ```

2. **Database Connection Pool Exhaustion**:
   ```bash
   # Monitor connections
   SELECT count(*) FROM pg_stat_activity;
   ```

3. **Redis Memory Usage**:
   ```bash
   # Monitor Redis memory
   redis-cli INFO memory
   ```

### Performance Profiling

Profile your application:

```python
import cProfile
import pstats

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    # Your code here
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)
```

This guide provides the foundation for deploying and optimizing the PDF to Markdown MCP Server in production environments.
