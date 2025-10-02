# Feature: MCP Semantic Search Tool

## Overview

Add Model Context Protocol (MCP) tool to enable LLMs to semantically search the document library using vector embeddings. This allows AI assistants to retrieve relevant documents from the codex_librarian based on natural language queries.

## Business Value

- **LLM Integration**: Enables Claude and other MCP-compatible LLMs to search the document library
- **Semantic Understanding**: Goes beyond keyword matching to understand query intent
- **Research Acceleration**: Researchers can ask natural language questions and get relevant papers/documents
- **Knowledge Retrieval**: Powers RAG (Retrieval Augmented Generation) workflows

## User Story

```
As an LLM (Claude, GPT, etc.),
I want to search a document library using semantic similarity,
So that I can retrieve relevant documents to answer user questions.
```

**Example Usage:**
```
User: "Find papers about transformer architectures and attention mechanisms"
LLM: [calls search_library tool with query]
System: [returns top 10 most relevant documents with metadata]
LLM: "I found 3 highly relevant papers on transformers..."
```

## Technical Design

### 1. MCP Tool Specification

**Tool Name:** `search_library`

**Tool Description:**
"Semantically search the document library using vector embeddings. Returns markdown documents ranked by relevance to the query."

**Parameters:**
```json
{
  "query": {
    "type": "string",
    "description": "Natural language search query",
    "required": true
  },
  "limit": {
    "type": "integer",
    "description": "Maximum number of results to return (default: 10, max: 50)",
    "required": false,
    "default": 10
  },
  "min_similarity": {
    "type": "number",
    "description": "Minimum similarity threshold 0.0-1.0 (default: 0.7)",
    "required": false,
    "default": 0.7
  },
  "tags": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Filter by document tags/categories",
    "required": false
  },
  "date_from": {
    "type": "string",
    "format": "date",
    "description": "Filter documents processed after this date (ISO 8601)",
    "required": false
  },
  "date_to": {
    "type": "string",
    "format": "date",
    "description": "Filter documents processed before this date (ISO 8601)",
    "required": false
  }
}
```

**Response Format:**
```json
{
  "results": [
    {
      "document_id": "uuid",
      "filename": "transformer_paper.pdf",
      "source_path": "/mnt/codex_fs/research/papers/transformer_paper.pdf",
      "markdown_path": "/mnt/codex_fs/research/librarian_output/transformer_paper.md",
      "similarity_score": 0.92,
      "page_count": 15,
      "processed_at": "2025-09-29T10:30:00Z",
      "excerpt": "...relevant text snippet from the document...",
      "tags": ["nlp", "deep-learning"]
    }
  ],
  "query": "transformer architectures attention mechanisms",
  "total_results": 3,
  "search_duration_ms": 45
}
```

### 2. Search Strategy (State-of-the-Art 2025)

Based on recent research, we'll implement a **3-stage hybrid retrieval pipeline**:

#### Stage 1: Candidate Retrieval (Hybrid Search)
- **Vector Search** (pgvector): Semantic similarity using cosine distance
- **Keyword Search** (PostgreSQL FTS): BM25-style ranking for exact matches
- Retrieve top 100-200 candidates from each method

**Why Hybrid?**
- Vector search: Captures semantic meaning, handles synonyms
- BM25/keyword: Excellent for exact matches, technical terms, proper nouns
- Combined: 20-30% better recall than either alone (source: 2025 research)

#### Stage 2: Score Fusion
Use **Reciprocal Rank Fusion (RRF)** to combine rankings:
```
RRF_score = sum(1 / (k + rank_i))
```
Where k=60 (standard constant), rank_i is position in each result set.

**Benefits:**
- Normalization-free (no need to normalize scores)
- Robust to score distribution differences
- Simple, fast, effective

#### Stage 3: Reranking (Optional, Future Enhancement)
- Cross-encoder model (ColBERT-style) for precise relevance scoring
- Only needed for >10M documents or precision-critical applications
- Can be added later without breaking changes

### 3. Database Schema (Already Exists)

**Tables Used:**
```sql
-- documents: Main document metadata
documents (
  id UUID PRIMARY KEY,
  filename VARCHAR,
  source_path TEXT,
  output_path TEXT,  -- Path to markdown file
  conversion_status VARCHAR,
  page_count INTEGER,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

-- document_content: Chunked text with embeddings
document_content (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id),
  chunk_index INTEGER,
  chunk_text TEXT,
  markdown_content TEXT,
  page_number INTEGER
)

-- document_embeddings: Vector embeddings (PGVector)
document_embeddings (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id),
  content_id UUID REFERENCES document_content(id),
  embedding VECTOR(768),  -- nomic-embed-text dimensions
  created_at TIMESTAMP
)
```

**Required Indexes:**
```sql
-- HNSW index for fast vector similarity (pgvector 0.5+)
CREATE INDEX idx_embeddings_hnsw ON document_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_content_fts ON document_content
  USING gin(to_tsvector('english', chunk_text));

-- B-tree indexes for filtering
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_status ON documents(conversion_status);
```

### 4. Implementation Architecture

#### Option A: Standalone MCP Server (RECOMMENDED)
```
┌─────────────────────────────────────────────┐
│ MCP Client (Claude Desktop, etc.)          │
└────────────┬────────────────────────────────┘
             │ MCP Protocol (stdio/SSE)
             ▼
┌─────────────────────────────────────────────┐
│ MCP Server (Python)                         │
│ - search_library tool                       │
│ - Direct PostgreSQL connection              │
│ - Ollama embeddings                         │
└────────────┬────────────────────────────────┘
             │
             ├──▶ PostgreSQL + PGVector
             │   └─ Vector similarity search
             │
             └──▶ Ollama (localhost:11434)
                 └─ Query embedding generation
```

**Pros:**
- Simple deployment (single Python process)
- No FastAPI dependency overhead
- Direct database access (faster)
- Standard MCP protocol (works with any client)

**Cons:**
- Duplicates embedding logic from main app
- Separate credential management

#### Option B: FastAPI Endpoint + MCP Adapter
```
┌─────────────────────────────────────────────┐
│ MCP Client (Claude Desktop, etc.)          │
└────────────┬────────────────────────────────┘
             │ MCP Protocol
             ▼
┌─────────────────────────────────────────────┐
│ MCP Adapter (Thin wrapper)                  │
└────────────┬────────────────────────────────┘
             │ HTTP :8000
             ▼
┌─────────────────────────────────────────────┐
│ FastAPI Server (Existing)                   │
│ - /api/v1/search endpoint                   │
│ - Reuses existing services                  │
└─────────────────────────────────────────────┘
```

**Pros:**
- Reuses existing FastAPI services
- Single codebase for search logic
- Can also be called via HTTP directly

**Cons:**
- More complex (two processes)
- FastAPI must be running
- Extra network hop

**DECISION: Use Option A (Standalone MCP Server)**
- Simpler for end users
- Standard MCP pattern
- Can add FastAPI endpoint later if needed

### 5. MCP Server Implementation

**File Structure:**
```
src/pdf_to_markdown_mcp/
├── mcp/
│   ├── __init__.py
│   ├── server.py           # Main MCP server
│   ├── tools.py            # Tool definitions
│   ├── search.py           # Search logic
│   └── config.py           # MCP-specific config
├── services/
│   └── search_service.py   # Shared search service
└── ...
```

**MCP Server (server.py):**
```python
from mcp import Server, Tool
from .search import SearchService

server = Server("codex-librarian")

@server.tool()
async def search_library(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.7,
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None
) -> dict:
    """
    Semantically search the document library using vector embeddings.

    Returns markdown documents ranked by relevance to the query.
    """
    search_service = SearchService()
    results = await search_service.hybrid_search(
        query=query,
        limit=limit,
        min_similarity=min_similarity,
        tags=tags,
        date_from=date_from,
        date_to=date_to
    )
    return results
```

**Search Service (search.py):**
```python
from typing import List, Optional
import asyncio
from sqlalchemy import select, text
from pgvector.sqlalchemy import Vector
import httpx

class SearchService:
    def __init__(self, db_url: str, ollama_url: str):
        self.db_url = db_url
        self.ollama_url = ollama_url

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.7,
        tags: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> dict:
        """
        Performs hybrid search combining vector and keyword search.
        """
        start_time = time.time()

        # Step 1: Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Step 2: Run parallel searches
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(query_embedding, limit * 2),
            self._keyword_search(query, limit * 2)
        )

        # Step 3: Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )

        # Step 4: Apply filters
        filtered_results = self._apply_filters(
            fused_results,
            min_similarity=min_similarity,
            tags=tags,
            date_from=date_from,
            date_to=date_to
        )

        # Step 5: Format response
        duration_ms = (time.time() - start_time) * 1000
        return {
            "results": filtered_results[:limit],
            "query": query,
            "total_results": len(filtered_results),
            "search_duration_ms": round(duration_ms, 2)
        }

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama nomic-embed-text."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            return response.json()["embedding"]

    async def _vector_search(
        self,
        query_embedding: List[float],
        limit: int
    ) -> List[dict]:
        """Perform vector similarity search using pgvector."""
        async with get_async_session() as session:
            query = text("""
                SELECT
                    d.id,
                    d.filename,
                    d.source_path,
                    d.output_path,
                    d.page_count,
                    d.created_at,
                    dc.chunk_text,
                    dc.markdown_content,
                    dc.page_number,
                    1 - (de.embedding <=> :query_embedding::vector) as similarity
                FROM document_embeddings de
                JOIN document_content dc ON de.content_id = dc.id
                JOIN documents d ON de.document_id = d.id
                WHERE d.conversion_status = 'completed'
                ORDER BY de.embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            result = await session.execute(
                query,
                {"query_embedding": query_embedding, "limit": limit}
            )
            return [dict(row) for row in result.fetchall()]

    async def _keyword_search(self, query: str, limit: int) -> List[dict]:
        """Perform full-text keyword search using PostgreSQL FTS."""
        async with get_async_session() as session:
            query_sql = text("""
                SELECT
                    d.id,
                    d.filename,
                    d.source_path,
                    d.output_path,
                    d.page_count,
                    d.created_at,
                    dc.chunk_text,
                    dc.markdown_content,
                    dc.page_number,
                    ts_rank_cd(
                        to_tsvector('english', dc.chunk_text),
                        plainto_tsquery('english', :query)
                    ) as rank
                FROM document_content dc
                JOIN documents d ON dc.document_id = d.id
                WHERE
                    to_tsvector('english', dc.chunk_text) @@
                    plainto_tsquery('english', :query)
                    AND d.conversion_status = 'completed'
                ORDER BY rank DESC
                LIMIT :limit
            """)
            result = await session.execute(
                query_sql,
                {"query": query, "limit": limit}
            )
            return [dict(row) for row in result.fetchall()]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
        k: int = 60
    ) -> List[dict]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i))
        where k=60 is standard constant
        """
        scores = {}

        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result['id']
            if doc_id not in scores:
                scores[doc_id] = {'result': result, 'rrf_score': 0}
            scores[doc_id]['rrf_score'] += 1 / (k + rank)

        # Add keyword search scores
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result['id']
            if doc_id not in scores:
                scores[doc_id] = {'result': result, 'rrf_score': 0}
            scores[doc_id]['rrf_score'] += 1 / (k + rank)

        # Sort by RRF score
        ranked_results = sorted(
            scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        return [item['result'] for item in ranked_results]

    def _apply_filters(
        self,
        results: List[dict],
        min_similarity: float,
        tags: Optional[List[str]],
        date_from: Optional[str],
        date_to: Optional[str]
    ) -> List[dict]:
        """Apply post-search filters."""
        filtered = results

        # Similarity threshold
        if min_similarity:
            filtered = [
                r for r in filtered
                if r.get('similarity', 1.0) >= min_similarity
            ]

        # Date range
        if date_from:
            filtered = [
                r for r in filtered
                if r['created_at'] >= date_from
            ]
        if date_to:
            filtered = [
                r for r in filtered
                if r['created_at'] <= date_to
            ]

        # Tags (TODO: implement when tagging system exists)
        # if tags:
        #     filtered = [r for r in filtered if any(t in r.get('tags', []) for t in tags)]

        return filtered
```

### 6. MCP Configuration

**File: `~/.config/claude/mcp_config.json`** (or equivalent for other MCP clients)
```json
{
  "mcpServers": {
    "codex-librarian": {
      "command": "python",
      "args": [
        "-m",
        "pdf_to_markdown_mcp.mcp.server"
      ],
      "env": {
        "DATABASE_URL": "postgresql://codex_librarian:PASSWORD@192.168.1.104:5432/codex_librarian",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Security Note:** Database credentials are stored in MCP client config (Claude Desktop settings). Not exposed in code or logs.

### 7. Performance Optimization

**For Large Libraries (>1M documents):**

1. **Index Tuning:**
```sql
-- HNSW index parameters for scale
CREATE INDEX idx_embeddings_hnsw ON document_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (
    m = 16,              -- Higher = better recall, more memory
    ef_construction = 64 -- Higher = better index quality, slower build
  );

-- Runtime ef_search tuning
SET hnsw.ef_search = 100;  -- Higher = better recall, slower queries
```

2. **Caching:**
```python
# Cache frequently searched queries (LRU)
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(query: str) -> List[float]:
    return generate_embedding(query)
```

3. **Connection Pooling:**
```python
# Async connection pool for PostgreSQL
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

4. **Pagination for Large Result Sets:**
```python
# If limit > 50, use cursor-based pagination
if limit > 50:
    return paginated_search(query, page_size=50)
```

**Expected Performance (Based on 2025 Benchmarks):**
- **Small library** (<100K docs): <100ms per query
- **Medium library** (100K-1M docs): <300ms per query
- **Large library** (>1M docs): <500ms per query (with HNSW index)

### 8. Testing Strategy

**Unit Tests:**
```python
# tests/unit/test_mcp_search.py
async def test_search_library_basic():
    result = await search_library(query="neural networks", limit=5)
    assert len(result['results']) <= 5
    assert result['results'][0]['similarity_score'] > 0.7

async def test_search_with_filters():
    result = await search_library(
        query="transformers",
        limit=10,
        min_similarity=0.8,
        date_from="2024-01-01"
    )
    assert all(r['similarity_score'] >= 0.8 for r in result['results'])
```

**Integration Tests:**
```python
# tests/integration/test_mcp_integration.py
async def test_mcp_server_connection():
    # Test MCP protocol handshake
    client = MCPClient("codex-librarian")
    await client.connect()
    assert client.is_connected()

async def test_end_to_end_search():
    # Test full search pipeline
    result = await mcp_client.call_tool(
        "search_library",
        {"query": "attention mechanisms", "limit": 5}
    )
    assert len(result['results']) > 0
```

**Load Tests:**
```python
# tests/performance/test_search_performance.py
async def test_concurrent_searches():
    # Simulate 100 concurrent searches
    queries = ["query1", "query2", ...] * 100
    start = time.time()

    results = await asyncio.gather(*[
        search_library(q, limit=10) for q in queries
    ])

    duration = time.time() - start
    avg_latency = duration / len(queries)
    assert avg_latency < 0.5  # Must be under 500ms average
```

### 9. Deployment Plan

**Phase 1: MVP (Week 1)**
- [ ] Implement standalone MCP server
- [ ] Basic vector search (no hybrid)
- [ ] Direct PostgreSQL connection
- [ ] Simple tool definition
- [ ] Manual testing with Claude Desktop

**Phase 2: Hybrid Search (Week 2)**
- [ ] Add BM25 keyword search
- [ ] Implement RRF score fusion
- [ ] Add similarity threshold filtering
- [ ] Performance benchmarking

**Phase 3: Production Hardening (Week 3)**
- [ ] Add comprehensive error handling
- [ ] Implement connection pooling
- [ ] Add query result caching
- [ ] Create monitoring/logging
- [ ] Write documentation

**Phase 4: Advanced Features (Future)**
- [ ] Cross-encoder reranking
- [ ] Tag-based filtering
- [ ] Date range queries
- [ ] Search result snippets with highlighting
- [ ] Multi-lingual search support

### 10. Success Metrics

**Technical Metrics:**
- Search latency P50 < 200ms
- Search latency P95 < 500ms
- Query success rate > 99%
- Index build time < 1 hour (for 1M docs)

**Quality Metrics:**
- Recall@10 > 85% (user finds relevant doc in top 10)
- Precision@10 > 75% (10 results are relevant)
- User satisfaction score > 4.0/5.0

**Scalability Metrics:**
- Support 1M+ documents
- Handle 100 concurrent queries
- <500ms latency at 95th percentile

### 11. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PGVector index build too slow | High | Medium | Use parallel index builds, optimize HNSW parameters |
| Ollama embedding service unavailable | High | Low | Implement retry logic, fallback to cached embeddings |
| Large result sets cause memory issues | Medium | Medium | Implement streaming results, pagination |
| MCP protocol changes break compatibility | Medium | Low | Pin MCP SDK version, monitor spec updates |
| Database connection pool exhaustion | High | Medium | Tune pool size, implement graceful degradation |

### 12. Documentation Requirements

**User Documentation:**
- [ ] MCP setup guide for Claude Desktop
- [ ] Tool usage examples
- [ ] Query syntax best practices
- [ ] Troubleshooting common issues

**Developer Documentation:**
- [ ] Search algorithm explanation
- [ ] Database schema documentation
- [ ] API reference for search service
- [ ] Performance tuning guide

**Operations Documentation:**
- [ ] Deployment checklist
- [ ] Monitoring setup
- [ ] Backup/restore procedures
- [ ] Scaling guidelines

## Dependencies

**Required:**
- `mcp` - Official MCP SDK for Python
- `asyncpg` - Async PostgreSQL driver
- `pgvector` - Vector similarity extension
- `httpx` - Async HTTP client for Ollama
- `sqlalchemy[asyncio]` - Async ORM

**Optional:**
- `sentence-transformers` - Alternative embedding generation
- `rank-bm25` - Pure Python BM25 implementation
- `redis` - Query result caching

## References

- [MCP Specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [PGVector Performance Guide](https://github.com/pgvector/pgvector#performance)
- [Hybrid Search Best Practices 2025](https://www.analyticsvidhya.com/blog/2024/12/contextual-rag-systems-with-hybrid-search-and-reranking/)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)

---

**Document Status:** Draft v1.0
**Last Updated:** 2025-09-29
**Author:** System Design (based on user requirements + SOTA research)
**Review Status:** Pending stakeholder review