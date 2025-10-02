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

from pdf_to_markdown_mcp.config import settings

logger = logging.getLogger(__name__)

# Security configuration from settings (can be overridden for testing)
API_KEY = os.environ.get("API_KEY", "")
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"


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


def validate_path_security(path: Path | str) -> Path:
    """
    Validate file path to prevent directory traversal attacks.

    Args:
        path: Path to validate (Path object or string)

    Returns:
        Validated and resolved path

    Raises:
        HTTPException: If path is invalid or contains traversal attempts
    """
    # Convert to Path object if string
    if isinstance(path, str):
        path = Path(path)

    try:
        # Resolve the path to eliminate any .. components
        resolved_path = path.resolve()

        # Get allowed directories from settings
        allowed_paths = [
            Path(settings.INPUT_DIRECTORY).resolve() if hasattr(settings, 'INPUT_DIRECTORY') else None,
            Path(settings.OUTPUT_DIRECTORY).resolve(),
            Path("/tmp/pdf_to_markdown_mcp").resolve(),
        ]
        # Filter out None values
        allowed_paths = [p for p in allowed_paths if p is not None]

        # Check if the resolved path starts with any allowed directory
        allowed = False
        for allowed_dir in allowed_paths:
            try:
                if str(resolved_path).startswith(str(allowed_dir)):
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
                    "allowed": [str(p) for p in allowed_paths],
                },
            )
            raise HTTPException(
                status_code=400,
                detail=f"Path outside allowed directories"
            )

        # Additional security checks
        path_str = str(resolved_path).lower()
        filename = resolved_path.name.lower()

        # Block common dangerous directory paths
        dangerous_dir_patterns = [
            "/etc/",
            "/root/",
            "/home/",
            "/usr/bin/",
            "/bin/",
            "/proc/",
            "/dev/",
            "/sys/",
            "\\windows\\system32",
        ]

        for pattern in dangerous_dir_patterns:
            if pattern in path_str:
                logger.warning(
                    f"Blocked potentially dangerous path: {path}",
                    extra={"pattern": pattern, "path": str(path)},
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Access denied to dangerous path"
                )

        # Block dangerous filenames (check the actual filename, not the full path)
        dangerous_filenames = [
            "passwd", "shadow", "hosts", "resolv.conf", "fstab", "sudoers",
            ".env", ".env.local", "config.json", "database.yml", "secrets.json",
            "private.key", "boot.ini", "ntuser.dat", "system.ini", "win.ini",
            "id_rsa", "id_dsa", "authorized_keys", "known_hosts", ".ssh",
        ]

        for dangerous_name in dangerous_filenames:
            if dangerous_name in filename or filename == dangerous_name:
                logger.warning(
                    f"Blocked dangerous filename: {filename}",
                    extra={"filename": filename, "path": str(path)},
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Access denied to dangerous path"
                )

        return resolved_path

    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Path validation error: {e}", extra={"path": str(path)})
        raise HTTPException(
            status_code=400,
            detail=f"Invalid path provided"
        )


def validate_file_security(file_path: Path | str) -> dict[str, Any]:
    """
    Comprehensive file security validation.

    Args:
        file_path: Path to file to validate (Path object or string)

    Returns:
        Dict with validation results

    Raises:
        HTTPException: If file fails security validation
    """
    # Convert to Path object if string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found"
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file"
        )

    # Validate file size using settings
    max_size_mb = getattr(settings, 'MAX_FILE_SIZE_MB', 500)
    max_size = int(max_size_mb) * 1024 * 1024
    file_size = file_path.stat().st_size

    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size"
        )

    # Validate file type
    allowed_types = [".pdf"]
    if file_path.suffix.lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed"
        )

    # Basic file content validation (PDF header check)
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid PDF file format"
                )
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"File validation error: {str(e)}"
        )

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
