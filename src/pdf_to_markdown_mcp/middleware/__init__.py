"""
Security middleware for PDF to Markdown MCP Server.

Provides comprehensive security headers and protections.
"""

from .security import SecurityHeadersMiddleware

__all__ = ["SecurityHeadersMiddleware"]
