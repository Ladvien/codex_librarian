"""
Configuration API endpoints.

Implements the configure MCP tool for dynamic server configuration.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.config import settings
from pdf_to_markdown_mcp.core.watcher_service import WatcherManager
from pdf_to_markdown_mcp.db.session import get_db
from pdf_to_markdown_mcp.models.request import ConfigurationRequest
from pdf_to_markdown_mcp.models.response import (
    ConfigurationResponse,
    ErrorResponse,
    ErrorType,
)
from pdf_to_markdown_mcp.services.config_service import ConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/configure", response_model=ConfigurationResponse)
async def update_configuration(
    request: ConfigurationRequest, db: Session = Depends(get_db)
) -> ConfigurationResponse:
    """
    Update server configuration dynamically.

    This endpoint implements the configure MCP tool functionality.
    """
    try:
        logger.info("Configuration update requested")

        services_restarted = []
        validation_errors = []

        # Update output directory
        if request.output_directory is not None:
            try:
                output_path = request.output_directory

                # Validate directory exists or can be created
                from pathlib import Path
                output_dir = Path(output_path)

                if not output_dir.exists():
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created output directory: {output_dir}")
                    except Exception as e:
                        validation_errors.append(f"Cannot create output directory: {e}")

                if output_dir.exists() and not output_dir.is_dir():
                    validation_errors.append(f"Output path is not a directory: {output_path}")

                if not validation_errors:
                    # Update settings
                    settings.watcher.output_directory = str(output_dir)

                    # Save to database for persistence
                    try:
                        ConfigurationService.save_to_database(
                            "output_directory",
                            str(output_dir),
                            db,
                            updated_by="api",
                        )
                        logger.info(f"Saved output_directory to database: {output_dir}")
                    except Exception as e:
                        logger.error(f"Failed to save output_directory to database: {e}")
                        validation_errors.append(f"Failed to persist output_directory: {e}")

            except Exception as e:
                validation_errors.append(f"Error updating output directory: {e!s}")

        # Update watch directories
        if request.watch_directories is not None:
            try:
                # Validate all directories exist and are accessible
                for directory in request.watch_directories:
                    if not directory.exists():
                        validation_errors.append(
                            f"Directory does not exist: {directory}"
                        )
                    elif not directory.is_dir():
                        validation_errors.append(
                            f"Path is not a directory: {directory}"
                        )

                if not validation_errors:
                    # Update settings
                    old_directories = settings.watcher.watch_directories.copy()
                    settings.watcher.watch_directories = [
                        str(d) for d in request.watch_directories
                    ]

                    # Save to database for persistence
                    try:
                        ConfigurationService.save_to_database(
                            "watch_directories",
                            [str(d) for d in request.watch_directories],
                            db,
                            updated_by="api",
                        )
                        logger.info("Saved watch_directories to database")
                    except Exception as e:
                        logger.error(f"Failed to save watch_directories to database: {e}")
                        validation_errors.append(f"Failed to persist configuration: {e}")

                    # Restart file watcher if requested
                    if request.restart_watcher:
                        try:
                            # Restart file watcher service
                            watcher_manager = WatcherManager()

                            # Stop existing watchers
                            for old_dir in old_directories:
                                watcher_manager.stop_watcher(old_dir)

                            # Start watchers for new directories
                            for new_dir in request.watch_directories:
                                watcher_config = {
                                    "directory": str(new_dir),
                                    "patterns": settings.watcher.file_patterns,
                                    "recursive": settings.watcher.recursive,
                                    "enable_deduplication": settings.watcher.enable_deduplication,
                                }
                                watcher_manager.start_watcher(
                                    name=f"watcher_{new_dir.name}",
                                    config=watcher_config,
                                )

                            services_restarted.append("file_watcher")
                            logger.info(
                                "File watcher restarted with new directories",
                                extra={
                                    "old_directories": old_directories,
                                    "new_directories": [
                                        str(d) for d in request.watch_directories
                                    ],
                                },
                            )

                        except Exception as e:
                            validation_errors.append(
                                f"Failed to restart file watcher: {e!s}"
                            )
                            logger.error(f"Failed to restart file watcher: {e}")

            except Exception as e:
                validation_errors.append(f"Error updating watch directories: {e!s}")

        # Update embedding configuration
        if request.embedding_config is not None:
            try:
                # Validate embedding configuration
                if "provider" in request.embedding_config:
                    provider = request.embedding_config["provider"]
                    if provider not in ["ollama", "openai"]:
                        validation_errors.append(
                            f"Invalid embedding provider: {provider}"
                        )
                    else:
                        settings.embedding.provider = provider

                if "model" in request.embedding_config:
                    settings.embedding.model = request.embedding_config["model"]

                if "batch_size" in request.embedding_config:
                    batch_size = request.embedding_config["batch_size"]
                    if not isinstance(batch_size, int) or batch_size < 1:
                        validation_errors.append(
                            "Batch size must be a positive integer"
                        )
                    else:
                        settings.embedding.batch_size = batch_size

                if "dimensions" in request.embedding_config:
                    dimensions = request.embedding_config["dimensions"]
                    if not isinstance(dimensions, int) or dimensions < 1:
                        validation_errors.append(
                            "Embedding dimensions must be a positive integer"
                        )
                    else:
                        settings.embedding.dimensions = dimensions

                logger.info("Embedding configuration updated")

            except Exception as e:
                validation_errors.append(
                    f"Error updating embedding configuration: {e!s}"
                )

        # Update OCR settings
        if request.ocr_settings is not None:
            try:
                if "language" in request.ocr_settings:
                    settings.processing.ocr_language = request.ocr_settings["language"]

                if "dpi" in request.ocr_settings:
                    dpi = request.ocr_settings["dpi"]
                    if not isinstance(dpi, int) or dpi < 72:
                        validation_errors.append("DPI must be at least 72")

                if "preserve_layout" in request.ocr_settings:
                    settings.processing.preserve_layout = bool(
                        request.ocr_settings["preserve_layout"]
                    )

                logger.info("OCR settings updated")

            except Exception as e:
                validation_errors.append(f"Error updating OCR settings: {e!s}")

        # Update processing limits
        if request.processing_limits is not None:
            try:
                if "max_file_size_mb" in request.processing_limits:
                    max_size = request.processing_limits["max_file_size_mb"]
                    if not isinstance(max_size, (int, float)) or max_size <= 0:
                        validation_errors.append("Max file size must be positive")
                    else:
                        settings.processing.max_file_size_mb = int(max_size)

                if "processing_timeout_seconds" in request.processing_limits:
                    timeout = request.processing_limits["processing_timeout_seconds"]
                    if not isinstance(timeout, int) or timeout <= 0:
                        validation_errors.append("Processing timeout must be positive")
                    else:
                        settings.processing.processing_timeout_seconds = timeout

                if "chunk_size" in request.processing_limits:
                    chunk_size = request.processing_limits["chunk_size"]
                    if not isinstance(chunk_size, int) or chunk_size < 100:
                        validation_errors.append("Chunk size must be at least 100")
                    else:
                        settings.processing.chunk_size = chunk_size

                if "chunk_overlap" in request.processing_limits:
                    chunk_overlap = request.processing_limits["chunk_overlap"]
                    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                        validation_errors.append("Chunk overlap must be non-negative")
                    elif chunk_overlap >= settings.processing.chunk_size:
                        validation_errors.append(
                            "Chunk overlap must be less than chunk size"
                        )
                    else:
                        settings.processing.chunk_overlap = chunk_overlap

                logger.info("Processing limits updated")

            except Exception as e:
                validation_errors.append(f"Error updating processing limits: {e!s}")

        # Determine success status
        success = len(validation_errors) == 0

        if success:
            message = "Configuration updated successfully"
            if services_restarted:
                message += f". Services restarted: {', '.join(services_restarted)}"
        else:
            message = (
                f"Configuration update completed with {len(validation_errors)} errors"
            )

        return ConfigurationResponse(
            success=success,
            message=message,
            watch_directories=[str(d) for d in settings.watcher.watch_directories],
            embedding_provider=settings.embedding.provider,
            services_restarted=services_restarted,
            validation_errors=validation_errors,
        )

    except Exception as e:
        logger.exception("Unexpected error during configuration update")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to update configuration: {e!s}",
            ).dict(),
        )


@router.get("/configuration")
async def get_current_configuration() -> dict[str, Any]:
    """
    Get the current server configuration.
    """
    try:
        return ConfigurationService.get_current_config()

    except Exception as e:
        logger.exception("Error retrieving current configuration")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to retrieve configuration: {e!s}",
            ).dict(),
        )


@router.post("/configuration/validate")
async def validate_configuration(config_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate configuration without applying changes.
    """
    try:
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Validate watch directories
        if "watch_directories" in config_data:
            for directory in config_data["watch_directories"]:
                from pathlib import Path

                path = Path(directory)
                if not path.exists():
                    validation_results["errors"].append(
                        f"Directory does not exist: {directory}"
                    )
                    validation_results["valid"] = False
                elif not path.is_dir():
                    validation_results["errors"].append(
                        f"Path is not a directory: {directory}"
                    )
                    validation_results["valid"] = False

        # Validate embedding configuration
        if "embedding_config" in config_data:
            embedding_config = config_data["embedding_config"]
            if "provider" in embedding_config:
                if embedding_config["provider"] not in ["ollama", "openai"]:
                    validation_results["errors"].append(
                        "Provider must be 'ollama' or 'openai'"
                    )
                    validation_results["valid"] = False

            if "batch_size" in embedding_config:
                batch_size = embedding_config["batch_size"]
                if not isinstance(batch_size, int) or batch_size < 1:
                    validation_results["errors"].append(
                        "Batch size must be a positive integer"
                    )
                    validation_results["valid"] = False

        # Validate processing limits
        if "processing_limits" in config_data:
            limits = config_data["processing_limits"]
            if "max_file_size_mb" in limits:
                max_size = limits["max_file_size_mb"]
                if not isinstance(max_size, (int, float)) or max_size <= 0:
                    validation_results["errors"].append(
                        "Max file size must be positive"
                    )
                    validation_results["valid"] = False
                elif max_size > 1000:
                    validation_results["warnings"].append(
                        "Large max file size may impact performance"
                    )

        return validation_results

    except Exception as e:
        logger.exception("Error validating configuration")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Configuration validation failed: {e!s}",
            ).dict(),
        )


@router.post("/configuration/reset")
async def reset_configuration() -> ConfigurationResponse:
    """
    Reset configuration to default values.
    """
    try:
        from pdf_to_markdown_mcp.config import Settings

        # Get default settings
        default_settings = Settings()
        services_restarted = []

        # Reset to defaults
        old_watch_dirs = settings.watcher.watch_directories.copy()
        old_embedding_provider = settings.embedding.provider

        # Update all settings to defaults
        settings.watcher.watch_directories = default_settings.watcher.watch_directories
        settings.embedding.provider = default_settings.embedding.provider
        settings.embedding.model = default_settings.embedding.model
        settings.embedding.batch_size = default_settings.embedding.batch_size
        settings.embedding.dimensions = default_settings.embedding.dimensions
        settings.processing.max_file_size_mb = (
            default_settings.processing.max_file_size_mb
        )
        settings.processing.processing_timeout_seconds = (
            default_settings.processing.processing_timeout_seconds
        )
        settings.processing.chunk_size = default_settings.processing.chunk_size
        settings.processing.chunk_overlap = default_settings.processing.chunk_overlap
        settings.processing.ocr_language = default_settings.processing.ocr_language
        settings.processing.preserve_layout = (
            default_settings.processing.preserve_layout
        )

        # Restart file watcher if directories changed
        if old_watch_dirs != settings.watcher.watch_directories:
            try:
                watcher_manager = WatcherManager()

                # Stop old watchers
                for old_dir in old_watch_dirs:
                    watcher_manager.stop_watcher(old_dir)

                # Start default watchers
                for new_dir in settings.watcher.watch_directories:
                    watcher_config = {
                        "directory": new_dir,
                        "patterns": settings.watcher.file_patterns,
                        "recursive": settings.watcher.recursive,
                        "enable_deduplication": settings.watcher.enable_deduplication,
                    }
                    watcher_manager.start_watcher(
                        name=f"default_watcher_{Path(new_dir).name}",
                        config=watcher_config,
                    )

                services_restarted.append("file_watcher")
            except Exception as e:
                logger.error(f"Failed to restart file watcher during reset: {e}")

        # Restart embedding service if provider changed
        if old_embedding_provider != settings.embedding.provider:
            services_restarted.append("embedding_service")

        logger.info(
            "Configuration reset to defaults",
            extra={"services_restarted": services_restarted},
        )

        return ConfigurationResponse(
            success=True,
            message="Configuration reset to default values",
            watch_directories=settings.watcher.watch_directories,
            embedding_provider=settings.embedding.provider,
            services_restarted=services_restarted,
            validation_errors=[],
        )

    except Exception as e:
        logger.exception("Error resetting configuration")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorType.SYSTEM,
                message=f"Failed to reset configuration: {e!s}",
            ).dict(),
        )
