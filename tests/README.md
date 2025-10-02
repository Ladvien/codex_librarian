# PDF to Markdown MCP Server - Test Suite

Comprehensive test suite with **three distinct test categories**: unit tests (fast, mocked), integration tests (real services + GPU), and end-to-end tests (full pipeline validation).

## Test Categories

### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Fast, isolated component testing
- **Resources**: Mocked services, SQLite database, CPU-only
- **Duration**: ~10-30 seconds
- **Requirements**: None (no external services)
- **Use Case**: Development, CI/CD, pre-commit checks

### 2. Integration Tests (`tests/integration/`)
- **Purpose**: Validate components with real services
- **Resources**: Real PostgreSQL, GPU/CUDA, Redis, Ollama
- **Duration**: ~2-5 minutes
- **Requirements**: GPU, all services running (see below)
- **Use Case**: Pre-deployment validation, performance testing

### 3. End-to-End Tests (`tests/e2e/`)
- **Purpose**: Full pipeline validation
- **Resources**: All real services, complete workflows
- **Duration**: ~5-10 minutes
- **Requirements**: GPU, all services running, test data
- **Use Case**: Release validation, regression testing

## Quick Start

```bash
# Run fast unit tests only (default for development)
pytest -m unit

# Run integration tests (requires GPU + services)
pytest -m integration

# Run end-to-end tests (full pipeline)
pytest -m e2e

# Run all tests
pytest

# Run with coverage
pytest --cov=src/pdf_to_markdown_mcp --cov-report=html
```

## Prerequisites

### For Unit Tests
✅ No prerequisites - works out of the box

### For Integration + E2E Tests
- ✅ NVIDIA GPU with CUDA 12.4+ installed
- ✅ PostgreSQL 17+ with PGVector extension
- ✅ Redis server running (Docker or native)
- ✅ Ollama with `nomic-embed-text` model
- ✅ MinerU standalone service running
- ✅ Celery worker running (optional for some tests)

## Service Setup

### 1. Start PostgreSQL with PGVector
```bash
# Check if PostgreSQL is running
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "SELECT 1"

# Verify PGVector extension
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "SELECT * FROM pg_extension WHERE extname='vector'"
```

### 2. Start Redis
```bash
# Using Docker (recommended for testing)
docker run -d --name redis-test -p 6379:6379 redis:latest

# Or check existing Redis
redis-cli ping  # Should return PONG
```

### 3. Start Ollama and Pull Model
```bash
# Start Ollama service
sudo systemctl start ollama

# Pull embedding model
ollama pull nomic-embed-text

# Verify
curl http://localhost:11434/api/tags | jq '.models[].name'
```

### 4. Start MinerU Standalone Service
```bash
# Ensure GPU is available
nvidia-smi

# Start MinerU standalone
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 MINERU_DEVICE_MODE=cuda REDIS_PORT=6379 \
  python src/pdf_to_markdown_mcp/services/mineru_standalone.py &

# Verify (check /tmp/mineru.log)
tail -f /tmp/mineru.log
```

### 5. Start Celery Workers (Optional)
```bash
# Start worker
sudo systemctl start pdf-celery-worker

# Verify
celery -A pdf_to_markdown_mcp.worker inspect stats
```

## Validation Script

Run the validation script to check all prerequisites:

```bash
./scripts/validate_test_env.sh
```

Example output:
```
=== Test Environment Validation ===
PostgreSQL:     ✅ Available
PGVector:       ✅ Available
CUDA/GPU:       ✅ Available (RTX 3090, 24GB)
Redis:          ✅ Available
Ollama:         ✅ Available
Ollama Model:   ✅ nomic-embed-text available
MinerU Service: ✅ Running
Celery Worker:  ✅ Running

All prerequisites met for integration/e2e tests!
```

## Running Tests

### Unit Tests (Fast Development)
```bash
# Run all unit tests
pytest -m unit

# Run specific unit test file
pytest tests/unit/test_mineru_service.py

# Run with verbose output
pytest -m unit -v

# Run unit tests in parallel (faster)
pytest -m unit -n auto
```

### Integration Tests (Real Services)
```bash
# Run all integration tests
pytest -m integration

# Run database integration tests only
pytest -m "integration and database"

# Run GPU integration tests only
pytest -m "integration and gpu"

# Run with GPU performance monitoring
pytest -m "integration and gpu" -v -s
```

### End-to-End Tests (Full Pipeline)
```bash
# Run all e2e tests
pytest -m e2e

# Run with detailed logging
pytest -m e2e -v -s --log-cli-level=INFO
```

### Custom Test Combinations
```bash
# Run all tests except slow ones
pytest -m "not slow"

# Run integration + e2e (all real resource tests)
pytest -m "integration or e2e"

# Run only GPU-dependent tests
pytest -m gpu

# Run tests requiring specific services
pytest -m redis
pytest -m embeddings
pytest -m mineru
```

## Test Execution Scripts

### Quick Scripts
```bash
# Unit tests only
./scripts/test_unit.sh

# Integration tests with environment check
./scripts/test_integration.sh

# Validate environment before running
./scripts/validate_test_env.sh
```

## Test Markers Reference

| Marker | Description | Requirements |
|--------|-------------|--------------|
| `unit` | Fast unit tests | None |
| `integration` | Integration with real services | Services + GPU |
| `e2e` | End-to-end pipeline tests | Services + GPU + Data |
| `gpu` | Requires GPU/CUDA | NVIDIA GPU |
| `slow` | Slow tests (>30s) | Varies |
| `database` | Requires database | PostgreSQL |
| `redis` | Requires Redis | Redis server |
| `mineru` | Requires MinerU service | MinerU + GPU |
| `embeddings` | Requires embedding service | Ollama |
| `mcp` | MCP server functionality | Database + Ollama |
| `security` | Security-focused tests | Varies |

## Troubleshooting

### GPU Not Available
```bash
# Check GPU
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Set environment
export CUDA_VISIBLE_DEVICES=0
export MINERU_DEVICE_MODE=cuda
```

### PostgreSQL Connection Issues
```bash
# Test connection
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian

# Check DATABASE_URL in .env
grep DATABASE_URL .env

# Verify PGVector
psql -h 192.168.1.104 -U codex_librarian -d codex_librarian \
  -c "SELECT * FROM pg_extension WHERE extname='vector'"
```

### Redis Connection Issues
```bash
# Test Redis
redis-cli ping

# Check if port is correct
redis-cli -p 6379 ping

# Check Docker containers
docker ps | grep redis
```

### Ollama Issues
```bash
# Check Ollama running
curl http://localhost:11434/api/tags

# Pull model if missing
ollama pull nomic-embed-text

# Restart Ollama
sudo systemctl restart ollama
```

### MinerU Service Issues
```bash
# Check MinerU log
tail -100 /tmp/mineru.log

# Verify GPU usage
nvidia-smi | grep python

# Restart service
pkill -f mineru_standalone.py
CUDA_VISIBLE_DEVICES=0 MINERU_DEVICE_MODE=cuda REDIS_PORT=6379 \
  python src/pdf_to_markdown_mcp/services/mineru_standalone.py &
```

### Tests Skipped
If tests are being skipped, check the skip reason:
```bash
pytest -m integration -v -rs
```

Common skip reasons:
- `GPU/CUDA not available` - Install CUDA drivers
- `PostgreSQL not available` - Start PostgreSQL server
- `Redis not available` - Start Redis server
- `Ollama model not available` - Run `ollama pull nomic-embed-text`

## Test Organization

```
tests/
├── unit/              # Fast mocked tests (SQLite, CPU)
│   ├── services/      # Service unit tests
│   ├── core/          # Core functionality tests
│   ├── security/      # Security tests
│   └── mcp/           # MCP server tests
│
├── integration/       # Real service tests (PostgreSQL, GPU, Redis)
│   ├── test_mineru_gpu_processing.py
│   ├── test_database_operations.py
│   ├── test_redis_queues.py
│   ├── test_embedding_service.py
│   └── conftest.py
│
├── e2e/              # Full pipeline tests
│   ├── test_full_pipeline_gpu.py
│   ├── test_mcp_search_pipeline.py
│   └── conftest.py
│
├── fixtures/         # Shared test fixtures
│   ├── real_database.py    # Real PostgreSQL fixtures
│   ├── real_gpu.py          # GPU validation fixtures
│   └── real_services.py     # Real service fixtures
│
├── conftest.py       # Unit test fixtures (mocked)
├── integration_conftest.py  # Integration test fixtures (real)
└── pytest.ini        # Pytest configuration
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# Unit tests - runs on every commit
- name: Run Unit Tests
  run: pytest -m unit --cov=src

# Integration tests - runs on GPU runner
- name: Run Integration Tests
  if: github.event_name == 'pull_request'
  run: |
    ./scripts/validate_test_env.sh
    pytest -m integration -v
```

## Performance Expectations

| Test Category | Duration | Tests | Purpose |
|---------------|----------|-------|---------|
| Unit | 10-30s | ~100+ | Fast feedback |
| Integration | 2-5min | ~30+ | Service validation |
| E2E | 5-10min | ~10+ | Pipeline validation |
| **Total** | **7-15min** | **~140+** | **Full test suite** |

## Best Practices

1. **Development Workflow**:
   - Run unit tests frequently during development
   - Run integration tests before committing
   - Run e2e tests before creating PR

2. **Test Isolation**:
   - Each test should be independent
   - Use database transactions (auto-rollback)
   - Clean up test data after execution

3. **GPU Tests**:
   - Verify GPU usage with memory monitoring
   - Assert GPU memory increased during processing
   - Check CUDA kernel execution

4. **Integration Tests**:
   - Use real services, not mocks
   - Validate actual outputs (files, database entries)
   - Measure real performance metrics

## Additional Resources

- **CLAUDE.md**: Project overview and deployment guide
- **Architecture Documentation**: See `docs/` directory
- **MCP Setup Guide**: See `docs/MCP_SETUP.md`

## Questions?

If you have questions or encounter issues:
1. Check this README for troubleshooting steps
2. Run `./scripts/validate_test_env.sh` for diagnostics
3. Review test output for specific error messages
4. Check service logs (`/tmp/mineru.log`, `/var/log/celery-worker.log`)
