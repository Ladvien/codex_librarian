
# PDF to Markdown MCP Server Documentation

A comprehensive MCP (Model Context Protocol) server that converts PDFs to searchable Markdown with vector embeddings stored in PostgreSQL with PGVector extension.

## Overview

The PDF to Markdown MCP Server provides a production-ready solution for processing PDF documents into structured Markdown format while maintaining semantic search capabilities through vector embeddings. Built with modern Python technologies and following best practices for scalability and maintainability.

## Key Features

- **PDF Processing**: Advanced PDF parsing with OCR, table extraction, and formula recognition using MinerU
- **Vector Search**: Semantic similarity search powered by PostgreSQL with PGVector extension
- **Dual Embedding**: Support for both Ollama (local) and OpenAI (API) embedding providers
- **Background Processing**: Asynchronous PDF processing with Celery and Redis
- **File Monitoring**: Automatic processing of new PDFs with Watchdog
- **Production Ready**: Comprehensive logging, monitoring, and error handling
- **MCP Compatible**: Full Model Context Protocol implementation

## Quick Navigation

```{toctree}
:maxdepth: 2
:caption: User Guide

quick_start
config
advanced_usage
```

```{toctree}
:maxdepth: 2
:caption: API Reference

apidocs/index
```

## Architecture Overview

The server follows a microservices architecture with clear separation of concerns:

- **FastAPI**: Modern async web framework with automatic OpenAPI documentation
- **PostgreSQL + PGVector**: Vector database for semantic search capabilities
- **Celery + Redis**: Distributed task queue for background processing
- **MinerU**: Advanced PDF processing engine with multi-language OCR
- **Watchdog**: File system monitoring for automated workflows

## Getting Started

For a quick start, see our [Quick Start Guide](quick_start.md).

For detailed configuration options, check the [Configuration Guide](config.md).

For production deployment and advanced features, see [Advanced Usage](advanced_usage.md).

## API Documentation

Complete API reference is available in the [API Documentation](apidocs/index.html).

## Support

- **GitHub Repository**: [https://github.com/Ladvien/pdf-to-markdown-mcp](https://github.com/Ladvien/pdf-to-markdown-mcp)
- **Documentation**: [https://pdf-to-markdown-mcp.readthedocs.io](https://pdf-to-markdown-mcp.readthedocs.io)
- **Issues**: Report issues on [GitHub Issues](https://github.com/Ladvien/pdf-to-markdown-mcp/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Ladvien/pdf-to-markdown-mcp/blob/main/LICENSE) file for details.

