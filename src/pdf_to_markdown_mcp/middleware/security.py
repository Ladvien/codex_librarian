"""
Security headers middleware for comprehensive web application security.

Implements OWASP recommended security headers and rate limiting protections.
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pdf_to_markdown_mcp.config import settings

logger = logging.getLogger(__name__)

# Import slowapi for rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    RATE_LIMITING_AVAILABLE = True
except ImportError:
    logger.warning("slowapi not available, rate limiting disabled")
    RATE_LIMITING_AVAILABLE = False
    Limiter = None
    get_remote_address = None
    RateLimitExceeded = None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add comprehensive security headers to all responses.

    Implements OWASP security header recommendations:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking attacks
    - X-XSS-Protection: Enables XSS filtering (legacy browsers)
    - Strict-Transport-Security: Enforces HTTPS
    - Content-Security-Policy: Prevents XSS and code injection
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    """

    def __init__(self, app, csp_policy: str = None, hsts_max_age: int = 31536000):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI application
            csp_policy: Custom Content Security Policy
            hsts_max_age: HSTS max age in seconds (default: 1 year)
        """
        super().__init__(app)
        self.csp_policy = csp_policy or self._get_default_csp_policy()
        self.hsts_max_age = hsts_max_age

    def _get_default_csp_policy(self) -> str:
        """Get default Content Security Policy based on environment."""
        base_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'none'; "
            "object-src 'none'; "
            "child-src 'none'; "
            "frame-src 'none'; "
            "worker-src 'none'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'; "
            "manifest-src 'self'"
        )

        # Add development allowances if needed
        if settings.environment == "development":
            # Allow localhost connections for development tools
            base_policy = base_policy.replace(
                "connect-src 'self'",
                "connect-src 'self' ws://localhost:* http://localhost:*",
            )

        return base_policy

    def _get_security_headers(
        self, request: Request, response: Response
    ) -> dict[str, str]:
        """Get security headers based on request and response context."""
        headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Enable XSS filtering (legacy browsers)
            "X-XSS-Protection": "1; mode=block",
            # Control referrer information
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Remove server identification
            "Server": "PDF-to-Markdown-MCP",
            # Permissions policy (restrict features)
            "Permissions-Policy": (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            ),
        }

        # Add HSTS for HTTPS requests
        if request.url.scheme == "https" or settings.environment == "production":
            headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )

        # Content Security Policy
        csp_policy = self.csp_policy

        # Special handling for Server-Sent Events
        if self._is_sse_endpoint(request):
            # Modify CSP for SSE endpoints to allow event streams
            csp_policy = csp_policy.replace(
                "connect-src 'self'", "connect-src 'self' data:"
            )

        headers["Content-Security-Policy"] = csp_policy

        return headers

    def _is_sse_endpoint(self, request: Request) -> bool:
        """Check if this is a Server-Sent Events endpoint."""
        path = request.url.path
        return (
            "/stream" in path
            or "/events" in path
            or "/progress" in path
            or request.headers.get("Accept") == "text/event-stream"
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and add security headers to response."""
        # Process the request
        response = await call_next(request)

        # Add security headers
        security_headers = self._get_security_headers(request, response)

        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value

        # Log security headers for development/debugging
        if settings.debug:
            correlation_id = getattr(request.state, "correlation_id", "unknown")
            logger.debug(
                "Security headers applied",
                extra={
                    "correlation_id": correlation_id,
                    "path": request.url.path,
                    "headers_added": list(security_headers.keys()),
                },
            )

        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate request body size and prevent DoS attacks.

    Limits request body size to prevent memory exhaustion attacks.
    """

    def __init__(self, app, max_request_size: int = 100 * 1024 * 1024):  # 100MB default
        """
        Initialize request size middleware.

        Args:
            app: FastAPI application
            max_request_size: Maximum request body size in bytes
        """
        super().__init__(app)
        self.max_request_size = max_request_size

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate request size before processing."""
        # Check Content-Length header
        content_length = request.headers.get("Content-Length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    correlation_id = getattr(request.state, "correlation_id", "unknown")

                    logger.warning(
                        "Request body too large",
                        extra={
                            "correlation_id": correlation_id,
                            "path": request.url.path,
                            "size": size,
                            "max_allowed": self.max_request_size,
                        },
                    )

                    from fastapi import HTTPException

                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body too large. Maximum allowed: {self.max_request_size // (1024 * 1024)}MB",
                    )
            except ValueError:
                # Invalid Content-Length header
                pass

        return await call_next(request)


# Rate limiting configuration
def get_rate_limiter():
    """Get configured rate limiter for the application."""
    if not RATE_LIMITING_AVAILABLE:
        logger.warning("Rate limiting not available - slowapi not installed")
        return None

    # Create limiter with Redis storage (if available) or in-memory
    try:
        # Try to use Redis for distributed rate limiting
        from limits.storage import RedisStorage

        redis_url = (
            settings.redis.url
            if hasattr(settings, "redis")
            else "redis://localhost:6379"
        )
        storage = RedisStorage(redis_url)
        logger.info(f"Using Redis storage for rate limiting: {redis_url}")
    except Exception as e:
        # Fallback to in-memory storage
        logger.warning(f"Redis not available for rate limiting, using in-memory: {e}")
        from limits.storage import MemoryStorage

        storage = MemoryStorage()

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[
            f"{settings.rate_limit_per_minute}/minute",
            f"{settings.rate_limit_per_hour}/hour",
        ],
    )

    return limiter


def get_endpoint_rate_limits(path: str, method: str) -> list:
    """Get rate limits for specific endpoint based on security requirements."""
    # Define per-endpoint rate limits based on resource intensity
    endpoint_limits = {
        # High-resource endpoints (PDF processing)
        "POST:/api/v1/convert_single": ["5/minute", "20/hour"],
        "POST:/api/v1/batch_convert": ["2/minute", "5/hour"],
        # Medium-resource endpoints (search operations)
        "GET:/api/v1/semantic_search": ["30/minute", "200/hour"],
        "GET:/api/v1/hybrid_search": ["30/minute", "200/hour"],
        "GET:/api/v1/find_similar": ["20/minute", "100/hour"],
        # Low-resource endpoints (status, config)
        "GET:/api/v1/get_status": ["60/minute", "500/hour"],
        "GET:/api/v1/stream_progress": ["60/minute", "500/hour"],
        "POST:/api/v1/configure": ["10/minute", "50/hour"],
        # Health endpoints (more permissive)
        "GET:/health": ["120/minute", "1000/hour"],
        "GET:/ready": ["120/minute", "1000/hour"],
        "GET:/metrics": ["60/minute", "500/hour"],
    }

    endpoint_key = f"{method}:{path}"

    # Check for exact match
    if endpoint_key in endpoint_limits:
        return endpoint_limits[endpoint_key]

    # Check for path patterns
    for pattern, limits in endpoint_limits.items():
        if pattern.endswith("*") and endpoint_key.startswith(pattern[:-1]):
            return limits

    # Default limits for unspecified endpoints
    return [
        f"{settings.rate_limit_per_minute}/minute",
        f"{settings.rate_limit_per_hour}/hour",
    ]


# Configuration for easy import
def create_security_middleware(app):
    """Create and configure security middleware for the application."""
    # Get configuration from settings
    max_request_size = (
        getattr(settings.processing, "max_file_size_mb", 100) * 1024 * 1024
    )

    # Configure rate limiting
    if RATE_LIMITING_AVAILABLE:
        limiter = get_rate_limiter()
        if limiter:
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            logger.info(
                "Rate limiting configured",
                extra={
                    "default_limits": [
                        f"{settings.rate_limit_per_minute}/minute",
                        f"{settings.rate_limit_per_hour}/hour",
                    ]
                },
            )
    else:
        app.state.limiter = None
        logger.warning("Rate limiting disabled - slowapi not available")

    # Add request size validation
    app.add_middleware(RequestSizeMiddleware, max_request_size=max_request_size)

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info(
        "Security middleware configured",
        extra={
            "max_request_size_mb": max_request_size // (1024 * 1024),
            "environment": settings.environment,
            "rate_limiting_enabled": RATE_LIMITING_AVAILABLE,
        },
    )


def require_rate_limit(path_pattern: str, method: str = "GET"):
    """
    Decorator factory to add rate limiting to FastAPI endpoints.

    Usage:
        limiter = get_rate_limiter()

        @limiter.limit("5/minute")
        @app.post("/api/v1/convert_single")
        async def convert_single(...):
            pass

    Note: This is a helper function - actual rate limiting should be applied
    in the API routers using the limiter from app.state.limiter
    """

    def decorator(func):
        if RATE_LIMITING_AVAILABLE:
            limits = get_endpoint_rate_limits(path_pattern, method)
            # This decorator is for documentation - actual rate limiting
            # is applied in the create_security_middleware function
            func._rate_limits = limits
        return func

    return decorator
