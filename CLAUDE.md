# CLAUDE.md - AI Development Guide

## Project Overview
You are helping develop a PDF to Markdown MCP (Model Context Protocol) server that converts PDFs to searchable Markdown with vector embeddings stored in PostgreSQL with PGVector.

## Core Technologies
- **Language**: Python 3.11+
- **Package Manager**: uv
- **Web Framework**: FastAPI with Pydantic v2
- **Database**: PostgreSQL 15+ with PGVector extension
- **Queue**: Celery with Redis
- **PDF Processing**: MinerU library
- **Testing**: pytest with TDD approach
- **Code Quality**: ruff, black, mypy, bandit
- **Development OS**: Manjaro Linux

## Rules
- DO NOT maintain legacy, deprecated, or backwards compatibility code.  This is a prototype.
- Use PDFs found in these folders recursively: `/mnt/codex_fs/research/`
- Save all test outpout to `/mnt/codex_fs/research/librarian_output/`

## Development Principles

### 1. Test-Driven Development (TDD)
**ALWAYS follow the TDD cycle:**
1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Improve the code while keeping tests green

```python
# ALWAYS start with a test
def test_feature_behavior():
    # Given (Arrange)
    # When (Act)  
    # Then (Assert)
    pass

# THEN implement the feature
def feature_implementation():
    pass
```

### 2. Type Safety
**Use Pydantic models and type hints everywhere:**
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class PDFDocument(BaseModel):
    path: Path
    size_bytes: int = Field(gt=0)
    hash: str
    
    class Config:
        frozen = True  # Immutable by default
```

### 3. Code Quality Standards
Before committing ANY code:
```bash
# Format with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/ --fix

# Type check with mypy
mypy src/

# Security check with bandit
bandit -r src/

# Run all tests
pytest -v --cov=pdf_to_markdown_mcp
```

## Project Structure
```
pdf-to-markdown-mcp/
├── src/
│   └── pdf_to_markdown_mcp/
│       ├── __init__.py
│       ├── main.py           # FastAPI app
│       ├── models/           # Pydantic models
│       ├── api/              # API endpoints
│       ├── core/             # Core business logic
│       ├── services/         # Services (MinerU, embeddings)
│       ├── db/              # Database operations
│       ├── worker/          # Celery tasks
│       └── config.py        # Settings with pydantic-settings
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── fixtures/           # Test fixtures and data
│   └── conftest.py        # pytest configuration
├── alembic/               # Database migrations
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── pyproject.toml        # Project configuration
├── .env.example          # Environment variables template
├── .gitignore
├── README.md
└── CLAUDE.md            # This file
```

## Key Implementation Details

### MinerU Integration
```python
from mineru.api import MinerUAPI
from mineru.config import MinerUConfig

# MinerU handles:
# - PDF parsing with layout preservation
# - Table extraction
# - Formula extraction  
# - Built-in OCR
# - Automatic chunking for embeddings
```

### Database Models (SQLAlchemy)
```python
from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True)
    embedding = Column(Vector(1536))  # PGVector column
    chunk_text = Column(Text)
```

### FastAPI Endpoints
```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PDF to Markdown MCP")

@app.post("/convert", response_model=ConversionResponse)
async def convert_pdf(
    request: ConversionRequest,
    db: Session = Depends(get_db)
) -> ConversionResponse:
    # Implementation
    pass
```

### Celery Tasks
```python
from celery import Celery

app = Celery('pdf_to_markdown_mcp')

@app.task
def process_pdf(document_id: int) -> dict:
    # Long-running PDF processing
    pass
```

## Testing Guidelines

### Unit Test Template
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestComponent:
    """Test ComponentName following TDD"""
    
    @pytest.fixture
    def component(self):
        """Setup component with mocked dependencies"""
        return Component(mock_dep1, mock_dep2)
    
    @pytest.mark.asyncio
    async def test_behavior_description(self, component):
        """Test that component behaves correctly when..."""
        # Given (Arrange)
        input_data = {...}
        expected_output = {...}
        
        # When (Act)
        result = await component.method(input_data)
        
        # Then (Assert)
        assert result == expected_output
```

### Integration Test Template
```python
@pytest.mark.integration
class TestIntegration:
    @pytest.fixture
    async def test_db(self):
        """Create test database"""
        # Setup test DB
        yield db
        # Teardown
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, test_db):
        """Test complete workflow"""
        pass
```

## Common Commands

### Development Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt
uv pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Database Operations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Running Services
```bash
# Start FastAPI (development)
uvicorn pdf_to_markdown_mcp.main:app --reload

# Start Celery worker
celery -A pdf_to_markdown_mcp.worker worker --loglevel=debug

# Start Celery beat (scheduled tasks)
celery -A pdf_to_markdown_mcp.worker beat --loglevel=debug
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pdf_to_markdown_mcp --cov-report=html

# Run specific test file
pytest tests/unit/test_processor.py

# Run tests in watch mode
pytest-watch

# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/
```

## AI Assistant Guidelines

When asked to implement a feature:

1. **ALWAYS start with tests**
   - Write comprehensive test cases first
   - Cover happy path, edge cases, and error conditions
   - Use mocks for external dependencies

2. **Implement incrementally**
   - Write minimal code to pass each test
   - Refactor after tests pass
   - Keep functions small and focused

3. **Use type hints and Pydantic**
   - Every function should have type hints
   - Use Pydantic for all data validation
   - Prefer composition over inheritance

4. **Follow Python best practices**
   - Use async/await for I/O operations
   - Use context managers for resources
   - Handle errors explicitly
   - Log important operations

5. **Document as you go**
   - Write clear docstrings
   - Add inline comments for complex logic
   - Update README if adding new features

## Error Handling Pattern
```python
from typing import Result
from structlog import get_logger

logger = get_logger()

async def process_document(doc_id: int) -> Result[Document, ProcessingError]:
    """Process document with proper error handling"""
    try:
        # Processing logic
        logger.info("document_processed", doc_id=doc_id)
        return Success(document)
    except ValidationError as e:
        logger.error("validation_failed", doc_id=doc_id, error=str(e))
        return Failure(ProcessingError.validation_error(str(e)))
    except Exception as e:
        logger.exception("unexpected_error", doc_id=doc_id)
        return Failure(ProcessingError.system_error(str(e)))
```

## Performance Considerations
- Use connection pooling for PostgreSQL
- Batch embedding generation
- Stream large files instead of loading into memory
- Use async I/O wherever possible
- Profile before optimizing

## Security Checklist
- [ ] Validate all inputs with Pydantic
- [ ] Use parameterized queries (SQLAlchemy handles this)
- [ ] Sanitize file paths
- [ ] Limit file sizes
- [ ] Use environment variables for secrets
- [ ] Run bandit security checks
- [ ] Implement rate limiting on API endpoints

## Questions to Ask When Implementing
1. Have I written tests first?
2. Are all functions type-hinted?
3. Is input validation handled by Pydantic?
4. Are errors handled gracefully?
5. Is the code formatted with black?
6. Does ruff report any issues?
7. Does mypy pass without errors?
8. Are there security implications?
9. Is the feature documented?
10. Will this scale with large PDFs?

## MinerU Specific Notes
- MinerU handles OCR internally - no need for separate Tesseract
- Supports multiple languages including Chinese
- Provides automatic chunking for embeddings
- Preserves document structure and formatting
- Extracts tables and formulas natively

## Vector Search Best Practices
- Normalize embeddings before storing
- Use appropriate index (IVFFlat for large datasets)
- Consider hybrid search (semantic + keyword)
- Monitor index performance
- Implement result re-ranking for better relevance

## Remember
- **TDD is not optional** - No code without tests
- **Type safety is mandatory** - Use Pydantic and mypy
- **Code quality tools must pass** - black, ruff, bandit
- **Performance matters** - Profile and optimize
- **Security is critical** - Validate everything

This guide ensures consistent, high-quality development. When in doubt, refer back to these principles.