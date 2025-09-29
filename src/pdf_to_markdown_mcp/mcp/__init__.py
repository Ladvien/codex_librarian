"""
MCP (Model Context Protocol) server for codex_librarian.

Provides semantic search capabilities via the `search_library` tool.
All configuration comes from MCP client environment variables.

Usage:
    python -m pdf_to_markdown_mcp.mcp.server

Configuration:
    Set environment variables in your MCP client configuration:
    - DATABASE_URL (required): PostgreSQL connection string
    - OLLAMA_URL (optional): Ollama API endpoint (default: http://localhost:11434)
    - OLLAMA_MODEL (optional): Embedding model (default: nomic-embed-text)
    - See config.py for full list of options

Tools:
    - search_library: Semantic search with hybrid retrieval (vector + BM25)
"""

__version__ = "0.1.0"

from .config import MCPConfig
from .context import DatabasePool

__all__ = ["MCPConfig", "DatabasePool"]