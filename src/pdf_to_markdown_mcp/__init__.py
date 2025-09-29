"""
PDF to Markdown MCP Server

A Model Context Protocol server that converts PDFs to searchable Markdown
with vector embeddings stored in PostgreSQL with PGVector extension.

Author: FastAPI Specialist
Date: 2025-09-25
"""

__version__ = "0.1.0"
__author__ = "FastAPI Specialist"

# Package-level imports (using relative imports to avoid issues)
try:
    from .config import Settings
except ImportError:
    # Config module may not exist yet during development
    Settings = None

__all__ = [
    "Settings",
    "__author__",
    "__version__",
]
