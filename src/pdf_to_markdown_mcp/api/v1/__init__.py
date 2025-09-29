"""
V1 API router for PDF to Markdown MCP Server.

This module creates the complete V1 API router with all MCP tool endpoints
organized in a modular structure.
"""

from fastapi import APIRouter

from pdf_to_markdown_mcp.api import config, convert, search, status
from pdf_to_markdown_mcp.api.versioning import APIVersion, VersionedAPIRouter


def create_v1_router() -> APIRouter:
    """
    Create the complete V1 API router with all endpoints.

    This function assembles all the individual endpoint routers into
    a single V1 router with proper versioning and organization.
    """
    # Create main V1 router
    v1_router = VersionedAPIRouter(
        version=APIVersion.V1,
        tags=["v1", "PDF to Markdown MCP"],
    )

    # Include all endpoint routers
    v1_router.include_router(
        convert.router, prefix="/convert", tags=["conversion", "v1"]
    )

    v1_router.include_router(search.router, prefix="/search", tags=["search", "v1"])

    v1_router.include_router(status.router, prefix="/status", tags=["monitoring", "v1"])

    v1_router.include_router(
        config.router, prefix="/config", tags=["configuration", "v1"]
    )

    return v1_router


# Export the V1 router
v1_router = create_v1_router()

__all__ = ["create_v1_router", "v1_router"]
