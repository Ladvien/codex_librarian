# PDF to Markdown MCP - API Examples

This directory contains comprehensive examples demonstrating how to use the PDF to Markdown MCP system programmatically.

## Examples Overview

### 🚀 **complete_api_usage.py** - Full Pipeline Demonstration
**The definitive example** showing the entire system working together:
- Database connection and setup
- MinerU PDF processing service
- Text chunking for embeddings
- Embedding generation (Ollama/OpenAI)
- Vector database storage with PGVector
- Semantic similarity search
- Comprehensive error handling
- Performance monitoring

**Start here** to see the complete API in action.

### 💾 **basic_database_usage.py** - Simple Database Operations
Learn the database layer with basic operations:
- Database connection setup
- Document record creation
- Content storage and retrieval
- Basic statistics and querying

Perfect introduction to the data layer.

### 🎯 **manual_processing.py** - Interactive Processing
Step-by-step PDF processing with user control:
- Interactive PDF selection from research directory
- Direct service integration without Celery
- Rich CLI interface for processing progress
- Manual control over each processing step

Great for learning the API step-by-step.

### 🔍 **embedding_usage.py** - Embedding Services
Comprehensive embedding generation and search:
- Ollama (local) and OpenAI provider usage
- Batch embedding generation
- Vector similarity search
- Embedding normalization

Learn semantic search capabilities.

### 📄 **mineru_demo.py** - PDF Processing Service
Deep dive into MinerU PDF processing:
- PDF parsing with layout preservation
- Table and formula extraction
- OCR capabilities
- Concurrent processing
- Error handling strategies

Understand the core PDF processing engine.

### 👁️ **file_watcher_demo.py** - Directory Monitoring
File system monitoring without dependencies:
- Directory monitoring for PDF files
- Event handling patterns
- Mock task queue for learning
- Configurable watcher behavior

Perfect for understanding file monitoring before full integration.

## Running Examples

All examples are standalone and can be run directly:

```bash
# Run the complete API demonstration
python examples/complete_api_usage.py

# Try interactive manual processing
python examples/manual_processing.py

# Learn embedding services
python examples/embedding_usage.py

# Test basic database operations
python examples/basic_database_usage.py

# Explore PDF processing
python examples/mineru_demo.py

# Understand file monitoring
python examples/file_watcher_demo.py
```

## Prerequisites

- PostgreSQL with PGVector extension
- Environment variables configured (see `.env.example`)
- Dependencies installed via `uv pip install -e ".[dev]"`

## Learning Path

1. **Start with**: `basic_database_usage.py` - Learn data layer
2. **Then try**: `mineru_demo.py` - Understand PDF processing
3. **Next**: `embedding_usage.py` - Learn semantic search
4. **Advanced**: `manual_processing.py` - Interactive processing
5. **Integration**: `file_watcher_demo.py` - File monitoring
6. **Complete**: `complete_api_usage.py` - Full system integration

Each example is self-contained and demonstrates specific aspects of the system.