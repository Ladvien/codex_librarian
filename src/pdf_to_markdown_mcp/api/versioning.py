"""
API versioning system for PDF to Markdown MCP Server.

This module implements API versioning functionality including version
detection, routing, and middleware for handling multiple API versions.
"""

import re
from enum import Enum
from typing import Any

from fastapi import APIRouter, Request
from starlette.middleware.base import BaseHTTPMiddleware


class APIVersion(str, Enum):
    """Supported API versions."""

    V1 = "v1"


class VersionedAPIRouter(APIRouter):
    """APIRouter with automatic version prefixing and tagging."""

    def __init__(
        self,
        version: APIVersion,
        prefix: str = "",
        tags: list | None = None,
        **kwargs: Any,
    ):
        """Initialize versioned router with version prefix and tags."""
        # Add version to prefix
        versioned_prefix = f"/api/{version.value}{prefix}"

        # Add version to tags
        versioned_tags = tags or []
        version_tag = f"v{version.value}"
        if version_tag not in versioned_tags:
            versioned_tags.append(version_tag)

        super().__init__(prefix=versioned_prefix, tags=versioned_tags, **kwargs)

        self.version = version


def get_api_version(request: Request) -> APIVersion:
    """
    Extract API version from request headers or URL path.

    Priority:
    1. Accept header: application/vnd.api+json;version=1
    2. URL path: /api/v1/...
    3. Default to latest version (V1)
    """
    # Check Accept header for version
    accept_header = request.headers.get("Accept", "")
    version_match = re.search(r"version=(\d+)", accept_header)
    if version_match:
        version_num = version_match.group(1)
        if version_num == "1":
            return APIVersion.V1

    # Check URL path for version
    path = request.url.path
    path_match = re.search(r"/api/v(\d+)/", path)
    if path_match:
        version_num = path_match.group(1)
        if version_num == "1":
            return APIVersion.V1

    # Default to latest version
    return APIVersion.V1


async def version_middleware(request: Request, call_next):
    """
    Middleware to handle API versioning.

    Adds version information to request state and response headers.
    """
    # Detect API version from request
    api_version = get_api_version(request)
    request.state.api_version = api_version

    # Process request
    response = await call_next(request)

    # Add version header to response
    response.headers["X-API-Version"] = api_version.value

    return response


class VersionMiddleware(BaseHTTPMiddleware):
    """Middleware class for API versioning."""

    async def dispatch(self, request: Request, call_next):
        """Process request with version handling."""
        return await version_middleware(request, call_next)


def create_versioned_openapi_schema(
    app,
    version: APIVersion,
    title: str = "PDF to Markdown MCP Server API",
    description: str = "API for converting PDFs to Markdown with vector search",
) -> dict[str, Any]:
    """
    Create version-specific OpenAPI schema.

    This ensures that each API version has proper documentation
    with version-specific information.
    """
    schema = app.openapi()

    # Update schema info with version
    schema["info"]["title"] = f"{title} - {version.value.upper()}"
    schema["info"]["version"] = version.value

    # Add version to description
    version_description = f"\n\n**API Version:** {version.value}"
    schema["info"]["description"] = (
        schema["info"].get("description", description) + version_description
    )

    # Filter paths to only include this version
    versioned_paths = {}
    version_prefix = f"/api/{version.value}"

    for path, methods in schema.get("paths", {}).items():
        if path.startswith(version_prefix):
            versioned_paths[path] = methods

    schema["paths"] = versioned_paths

    return schema


def add_version_tags_to_operations(router: APIRouter, version: APIVersion):
    """
    Add version tags to all operations in a router.

    This helps with API documentation organization.
    """
    version_tag = f"v{version.value}"

    for route in router.routes:
        if hasattr(route, "tags") and route.tags:
            if version_tag not in route.tags:
                route.tags.append(version_tag)
        else:
            route.tags = [version_tag]


# Version compatibility helpers


def is_version_supported(version: str) -> bool:
    """Check if a version string is supported."""
    return version in [v.value for v in APIVersion]


def get_latest_version() -> APIVersion:
    """Get the latest supported API version."""
    return APIVersion.V1  # Update this when new versions are added


def get_all_supported_versions() -> list[APIVersion]:
    """Get list of all supported API versions."""
    return list(APIVersion)


def version_comparison_key(version: APIVersion) -> tuple:
    """
    Generate comparison key for version ordering.

    This allows for proper version ordering when needed.
    """
    # For now, just use the version number
    version_num = int(version.value.replace("v", ""))
    return (version_num,)


# Deprecation helpers (for future use)


def is_version_deprecated(version: APIVersion) -> bool:
    """Check if a version is deprecated (for future use)."""
    # For now, no versions are deprecated
    return False


def get_deprecation_info(version: APIVersion) -> dict[str, Any] | None:
    """Get deprecation information for a version (for future use)."""
    if not is_version_deprecated(version):
        return None

    # Return deprecation info when needed
    return {
        "deprecated": True,
        "sunset_date": None,  # Set when deprecating
        "migration_guide": None,  # Link to migration guide
    }
