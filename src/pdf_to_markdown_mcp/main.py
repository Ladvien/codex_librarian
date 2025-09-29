"""
FastAPI application for PDF to Markdown MCP Server.

Main application entry point with middleware, CORS, health checks,
and API route registration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_to_markdown_mcp.api import (
    config as config_api,
)
from pdf_to_markdown_mcp.api import (
    convert,
    health,
    search,
    status,
)
from pdf_to_markdown_mcp.config import configure_logging, settings

# Database connection imports removed for basic startup
from pdf_to_markdown_mcp.core.monitoring import (
    TracingManager,
    metrics_collector,
)
from pdf_to_markdown_mcp.middleware.security import create_security_middleware

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with correlation IDs and collect metrics."""

    async def dispatch(self, request: Request, call_next):
        """Process request with logging and metrics collection."""
        import time

        # Generate correlation ID using TracingManager
        correlation_id = TracingManager.generate_correlation_id()
        request.state.correlation_id = correlation_id
        TracingManager.set_correlation_id(correlation_id)

        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "process_time": f"{process_time:.3f}s",
                },
            )

            # Collect metrics for request
            self._collect_request_metrics(request, response, process_time)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "process_time": f"{process_time:.3f}s",
                },
                exc_info=True,
            )
            raise

    def _collect_request_metrics(
        self, request: Request, response: Response, process_time: float
    ):
        """Collect metrics for the request."""
        try:
            # Record API endpoint metrics
            path = request.url.path
            method = request.method
            status_code = response.status_code

            # Skip metrics collection for health endpoints to avoid noise
            if (
                path.startswith("/health")
                or path.startswith("/ready")
                or path.startswith("/metrics")
            ):
                return

            # Record search queries if this is a search endpoint
            if "/search" in path:
                # This would normally extract result count from response
                # For now, we'll record the query without result count
                metrics_collector.record_search_query(
                    search_type="api_request",
                    result_count=0,  # Would need to extract from response
                    response_time_ms=process_time * 1000,
                )

            # Record API response times
            if hasattr(metrics_collector, "record_api_request"):
                metrics_collector.record_api_request(
                    method=method,
                    path=path,
                    status_code=status_code,
                    response_time_ms=process_time * 1000,
                )

        except Exception as e:
            logger.error(f"Failed to collect request metrics: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting PDF to Markdown MCP Server")

    # Initialize database connections and load configuration
    try:
        from pdf_to_markdown_mcp.db.session import get_db
        from pdf_to_markdown_mcp.services.config_service import ConfigurationService

        # Load configuration from database (overrides .env if exists)
        with next(get_db()) as db:
            try:
                # Seed database from .env if first run
                ConfigurationService.seed_database_from_env(db)

                # Load config from database
                config_dict = ConfigurationService.load_from_database(db)

                # Apply to settings
                ConfigurationService.apply_database_config_to_settings(config_dict)

                logger.info("Configuration loaded from database")
            except Exception as e:
                logger.warning(f"Failed to load database config, using .env: {e}")

        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

    # Initialize monitoring
    logger.info("Starting monitoring systems")

    # Start system metrics collection in background
    asyncio.create_task(metrics_collector.collect_system_metrics())

    # NOTE: File watcher runs as a separate systemd service (pdf-file-watcher.service)
    # to avoid issues with uvicorn multiprocessing where Observer threads don't survive fork()
    logger.info("File watcher should be running as separate service: pdf-file-watcher.service")

    yield

    # Shutdown
    logger.info("Shutting down PDF to Markdown MCP Server")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Convert PDFs to searchable Markdown with vector embeddings",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure comprehensive security middleware (headers, rate limiting, request validation)
create_security_middleware(app)

# Add other middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.warning(
        "HTTP exception",
        extra={
            "correlation_id": correlation_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.warning(
        "Validation error",
        extra={
            "correlation_id": correlation_id,
            "errors": exc.errors(),
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid request data",
            "details": exc.errors(),
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        "Unhandled exception",
        extra={
            "correlation_id": correlation_id,
            "exception": str(exc),
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Internal server error",
            "correlation_id": correlation_id,
        },
    )


# Legacy health endpoints are now handled by the health router
# These are kept for backward compatibility but deprecated


@app.get("/health-legacy")
async def health_check_legacy() -> dict[str, Any]:
    """Legacy health check endpoint. Use /health instead."""
    logger.warning("Legacy /health-legacy endpoint used, consider migrating to /health")

    from pdf_to_markdown_mcp.api.health import get_health

    return await get_health()


@app.get("/ready-legacy")
async def readiness_check_legacy() -> dict[str, Any]:
    """Legacy readiness endpoint. Use /ready instead."""
    logger.warning("Legacy /ready-legacy endpoint used, consider migrating to /ready")

    from fastapi import Response

    from pdf_to_markdown_mcp.api.health import get_readiness

    response = Response()
    result = await get_readiness(response)
    return result


# Include API routers
app.include_router(convert.router, prefix="/api/v1", tags=["conversion"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(config_api.router, prefix="/api/v1", tags=["configuration"])

# Include health and monitoring endpoints (no prefix for standard health endpoints)
app.include_router(health.router, tags=["health"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "disabled",
        "health": "/health",
    }


if __name__ == "__main__":
    """Run the application directly."""
    uvicorn.run(
        "pdf_to_markdown_mcp.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.logging.level.lower(),
    )
