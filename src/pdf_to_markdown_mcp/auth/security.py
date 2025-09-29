"""
Security and authentication utilities for PDF to Markdown MCP Server.

This module provides API key authentication, rate limiting, input validation,
and other security measures to protect API endpoints.
"""

import hashlib
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Security configuration from environment variables
API_KEY = os.environ.get("API_KEY", "")
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"
ALLOWED_PATHS = [
    "/tmp/pdf_to_markdown_mcp",
    os.environ.get("OUTPUT_DIRECTORY", "/mnt/codex_fs/research/librarian_output/"),
]


class SecurityManager:
    """
    Comprehensive security manager for API protection.

    Handles authentication, authorization, input validation,
    and security monitoring.
    """

    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
        self.failed_attempts: dict[str, int] = {}
        self.last_attempt: dict[str, float] = {}

    def create_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def verify_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        provided_hash = self.hash_api_key(provided_key)
        return secrets.compare_digest(provided_hash, stored_hash)


# Global security manager instance
security_manager = SecurityManager()


def validate_path_security(path: Path) -> Path:
    """
    Validate file path to prevent directory traversal attacks.

    Args:
        path: Path to validate

    Returns:
        Validated and resolved path

    Raises:
        ValueError: If path is invalid or contains traversal attempts
    """
    try:
        # Resolve the path to eliminate any .. components
        resolved_path = path.resolve()

        # Check if the resolved path starts with any allowed directory
        allowed = False
        for allowed_dir in ALLOWED_PATHS:
            try:
                allowed_dir_resolved = Path(allowed_dir).resolve()
                if str(resolved_path).startswith(str(allowed_dir_resolved)):
                    allowed = True
                    break
            except Exception:
                continue

        if not allowed:
            logger.error(
                f"Path validation failed: {path} -> {resolved_path} not in allowed directories",
                extra={
                    "path": str(path),
                    "resolved": str(resolved_path),
                    "allowed": ALLOWED_PATHS,
                },
            )
            raise ValueError(f"Access denied to path: {path}")

        # Additional security checks
        path_str = str(resolved_path).lower()

        # Block common dangerous paths
        dangerous_patterns = [
            "/etc/",
            "/root/",
            "/home/",
            "/usr/bin/",
            "/bin/",
            "passwd",
            "shadow",
            ".ssh",
            ".env",
            "config",
        ]

        for pattern in dangerous_patterns:
            if pattern in path_str:
                logger.warning(
                    f"Blocked potentially dangerous path: {path}",
                    extra={"pattern": pattern, "path": str(path)},
                )
                raise ValueError(f"Access denied to dangerous path: {path}")

        return resolved_path

    except Exception as e:
        logger.error(f"Path validation error: {e}", extra={"path": str(path)})
        raise ValueError(f"Invalid path: {path}")


def validate_file_security(file_path: Path) -> dict[str, Any]:
    """
    Comprehensive file security validation.

    Args:
        file_path: Path to file to validate

    Returns:
        Dict with validation results

    Raises:
        ValueError: If file fails security validation
    """
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Validate file size
    max_size = int(os.environ.get("MAX_FILE_SIZE_MB", "500")) * 1024 * 1024
    file_size = file_path.stat().st_size

    if file_size > max_size:
        raise ValueError(f"File size {file_size} exceeds maximum {max_size} bytes")

    # Validate file type
    allowed_types = [".pdf"]
    if file_path.suffix.lower() not in allowed_types:
        raise ValueError(f"File type {file_path.suffix} not allowed")

    # Basic file content validation (PDF header check)
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                raise ValueError("Invalid PDF file format")
    except Exception as e:
        raise ValueError(f"File validation error: {e}")

    return {
        "path": str(file_path),
        "size_bytes": file_size,
        "type": file_path.suffix.lower(),
        "valid": True,
    }


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def verify_api_key_dependency(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        security_manager.security
    ),
) -> bool:
    """
    FastAPI dependency for API key authentication.

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials

    Returns:
        True if authenticated

    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication if not required
    if not REQUIRE_AUTH:
        return True

    # Check if API key is configured
    if not API_KEY:
        logger.error("Authentication required but no API_KEY configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication configuration error",
        )

    # Get client IP for logging
    client_ip = get_client_ip(request)

    # Check for credentials
    if not credentials:
        logger.warning(
            f"Authentication failed: No credentials provided from {client_ip}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate API key
    provided_key = credentials.credentials

    # Simple key comparison (in production, use hashed comparison)
    if not secrets.compare_digest(provided_key, API_KEY):
        # Track failed attempts
        security_manager.failed_attempts[client_ip] = (
            security_manager.failed_attempts.get(client_ip, 0) + 1
        )
        security_manager.last_attempt[client_ip] = time.time()

        logger.warning(
            f"Authentication failed: Invalid API key from {client_ip}",
            extra={
                "client_ip": client_ip,
                "failed_attempts": security_manager.failed_attempts[client_ip],
                "provided_key_hash": hashlib.sha256(provided_key.encode()).hexdigest()[
                    :8
                ],
            },
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Reset failed attempts on successful authentication
    if client_ip in security_manager.failed_attempts:
        del security_manager.failed_attempts[client_ip]
        del security_manager.last_attempt[client_ip]

    logger.info(f"Authentication successful from {client_ip}")
    return True


def rate_limit_check(client_ip: str, max_requests_per_minute: int = 60) -> bool:
    """
    Simple rate limiting check.

    Args:
        client_ip: Client IP address
        max_requests_per_minute: Maximum requests allowed per minute

    Returns:
        True if request is allowed, False if rate limited
    """
    current_time = time.time()

    # Clean old entries (older than 1 minute)
    cutoff_time = current_time - 60

    # Simple in-memory rate limiting (production should use Redis)
    if not hasattr(rate_limit_check, "requests"):
        rate_limit_check.requests = {}

    if client_ip not in rate_limit_check.requests:
        rate_limit_check.requests[client_ip] = []

    # Remove old requests
    rate_limit_check.requests[client_ip] = [
        req_time
        for req_time in rate_limit_check.requests[client_ip]
        if req_time > cutoff_time
    ]

    # Check rate limit
    if len(rate_limit_check.requests[client_ip]) >= max_requests_per_minute:
        return False

    # Add current request
    rate_limit_check.requests[client_ip].append(current_time)
    return True


def sanitize_error_message(error_message: str) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Args:
        error_message: Original error message

    Returns:
        Sanitized error message
    """
    # Remove potential sensitive information
    sensitive_patterns = [
        r"/[a-zA-Z0-9_/-]*passwd",
        r"/[a-zA-Z0-9_/-]*shadow",
        r"/home/[a-zA-Z0-9_/-]*",
        r"/root/[a-zA-Z0-9_/-]*",
        r"postgresql://[^@]*:[^@]*@",
        r"redis://[^@]*:[^@]*@",
        r"API_KEY=[a-zA-Z0-9_-]*",
        r"password=[a-zA-Z0-9_-]*",
    ]

    sanitized = error_message

    for pattern in sensitive_patterns:
        import re

        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

    # Generic fallback for any remaining sensitive-looking content
    if any(
        word in sanitized.lower() for word in ["password", "secret", "token", "key"]
    ):
        return "Operation failed due to security restrictions"

    return sanitized


# Authentication dependency for FastAPI routes
RequireAuth = Depends(verify_api_key_dependency)
