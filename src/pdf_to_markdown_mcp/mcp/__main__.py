"""
Main entry point for MCP server.

Run with: python -m pdf_to_markdown_mcp.mcp.server
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")