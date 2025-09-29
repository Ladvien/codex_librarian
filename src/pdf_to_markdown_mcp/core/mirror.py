"""
Directory mirroring logic for preserving PDF directory structure in Markdown output.

This module provides the core functionality for mapping source PDF directories
to output Markdown directories while preserving the exact folder hierarchy.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError

from ..db.models import PathMapping
from ..db.session import get_db_session

logger = logging.getLogger(__name__)


@dataclass
class MirrorConfig:
    """Configuration for directory mirroring."""

    watch_base_dir: Path
    output_base_dir: Path
    preserve_structure: bool = True
    max_directory_depth: int = 10
    create_missing_dirs: bool = True
    safe_filenames: bool = True


class PathMapper:
    """Utility class for safe path operations and calculations."""

    @staticmethod
    def is_safe_path(path: Path, base_path: Path) -> bool:
        """Check if path is safe (no directory traversal).

        Args:
            path: Path to validate
            base_path: Base directory path

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve to absolute paths
            abs_path = path.resolve()
            abs_base = base_path.resolve()

            # Check if path is within base directory
            return abs_base in abs_path.parents or abs_path == abs_base
        except (OSError, ValueError):
            return False

    @staticmethod
    def calculate_relative_path(file_path: Path, base_path: Path) -> Path:
        """Calculate relative path from base directory.

        Args:
            file_path: Absolute path to file
            base_path: Base directory path

        Returns:
            Relative path from base directory

        Raises:
            ValueError: If path is not under base directory
        """
        try:
            abs_file = file_path.resolve()
            abs_base = base_path.resolve()
            return abs_file.relative_to(abs_base)
        except ValueError as e:
            raise ValueError(
                f"File path {file_path} is not under base directory {base_path}"
            ) from e

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem storage.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        unsafe_chars = '<>:"/\\|?*'
        sanitized = filename
        for char in unsafe_chars:
            sanitized = sanitized.replace(char, "_")

        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(". ")

        # Ensure filename is not empty
        if not sanitized:
            sanitized = "untitled"

        # Truncate if too long (255 is typical filesystem limit)
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            max_name_len = 255 - len(ext)
            sanitized = name[:max_name_len] + ext

        return sanitized

    @staticmethod
    def calculate_directory_depth(path: Path, base_path: Path) -> int:
        """Calculate directory depth relative to base.

        Args:
            path: Path to calculate depth for
            base_path: Base directory

        Returns:
            Directory depth (0 = directly in base directory)
        """
        try:
            relative_path = PathMapper.calculate_relative_path(path, base_path)
            return len(relative_path.parent.parts)
        except ValueError:
            return 0


class DirectoryMirror:
    """Main class for directory structure mirroring."""

    def __init__(self, config: MirrorConfig, db_session_factory=None):
        """Initialize directory mirror.

        Args:
            config: Mirror configuration
            db_session_factory: Database session factory function
        """
        self.config = config
        self.db_session_factory = db_session_factory or get_db_session
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate mirror configuration."""
        if not self.config.watch_base_dir.exists():
            raise ValueError(
                f"Watch base directory does not exist: {self.config.watch_base_dir}"
            )

        if not self.config.watch_base_dir.is_dir():
            raise ValueError(
                f"Watch base path is not a directory: {self.config.watch_base_dir}"
            )

        if self.config.max_directory_depth < 1:
            raise ValueError("Maximum directory depth must be at least 1")

    def get_mirror_paths(self, source_file_path: Path) -> dict[str, Any]:
        """Calculate all mirrored paths for a source PDF file.

        Args:
            source_file_path: Absolute path to source PDF file

        Returns:
            Dictionary with mirrored path information

        Raises:
            ValueError: If source path is invalid or unsafe
        """
        # Validate source path
        if not source_file_path.is_absolute():
            raise ValueError("Source file path must be absolute")

        if not PathMapper.is_safe_path(source_file_path, self.config.watch_base_dir):
            raise ValueError(
                f"Source path {source_file_path} is not safe or not under watch directory"
            )

        try:
            # Calculate relative path from watch base
            source_relative = PathMapper.calculate_relative_path(
                source_file_path, self.config.watch_base_dir
            )

            # Calculate directory depth
            directory_depth = PathMapper.calculate_directory_depth(
                source_file_path, self.config.watch_base_dir
            )

            # Validate directory depth
            if directory_depth > self.config.max_directory_depth:
                raise ValueError(
                    f"Directory depth {directory_depth} exceeds maximum {self.config.max_directory_depth}"
                )

            # Create output path by changing extension to .md
            output_relative = source_relative.with_suffix(".md")

            # Sanitize filename if required
            if self.config.safe_filenames:
                sanitized_name = PathMapper.sanitize_filename(output_relative.name)
                output_relative = output_relative.with_name(sanitized_name)

            # Calculate absolute output path
            output_absolute = self.config.output_base_dir / output_relative

            return {
                "source_path": source_file_path,
                "source_relative_path": source_relative,
                "output_path": output_absolute,
                "output_relative_path": output_relative,
                "directory_depth": directory_depth,
                "output_directory": output_absolute.parent,
            }

        except ValueError as e:
            logger.error(
                f"Failed to calculate mirror paths for {source_file_path}: {e}"
            )
            raise

    def create_output_directory(self, output_directory: Path) -> bool:
        """Create output directory structure if it doesn't exist.

        Args:
            output_directory: Directory to create

        Returns:
            True if directory was created or already exists, False on failure
        """
        try:
            if output_directory.exists():
                if not output_directory.is_dir():
                    logger.error(
                        f"Output path exists but is not a directory: {output_directory}"
                    )
                    return False
                return True

            if not self.config.create_missing_dirs:
                logger.warning(
                    f"Output directory does not exist and creation is disabled: {output_directory}"
                )
                return False

            # Create directory with parents
            output_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_directory}")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory {output_directory}: {e}")
            return False

    def store_path_mapping(
        self, source_file_path: Path, mirror_info: dict[str, Any]
    ) -> bool:
        """Store path mapping information in database.

        Args:
            source_file_path: Source PDF file path
            mirror_info: Mirror path information from get_mirror_paths()

        Returns:
            True if mapping was stored successfully, False otherwise
        """
        try:
            with self.db_session_factory() as session:
                # Check if mapping already exists
                existing_mapping = (
                    session.query(PathMapping)
                    .filter_by(
                        source_directory=str(self.config.watch_base_dir),
                        relative_path=str(mirror_info["source_relative_path"].parent),
                    )
                    .first()
                )

                if existing_mapping:
                    # Update existing mapping
                    existing_mapping.files_count += 1
                    existing_mapping.last_scanned = datetime.utcnow()
                    existing_mapping.updated_at = datetime.utcnow()
                else:
                    # Create new mapping
                    path_mapping = PathMapping(
                        source_directory=str(self.config.watch_base_dir),
                        output_directory=str(self.config.output_base_dir),
                        relative_path=str(mirror_info["source_relative_path"].parent),
                        directory_level=mirror_info["directory_depth"],
                        files_count=1,
                    )
                    session.add(path_mapping)

                session.commit()
                logger.debug(f"Stored path mapping for {source_file_path}")
                return True

        except IntegrityError as e:
            logger.warning(f"Path mapping already exists for {source_file_path}: {e}")
            return True  # Not a failure, just duplicate
        except Exception as e:
            logger.error(f"Failed to store path mapping for {source_file_path}: {e}")
            return False

    def process_file_for_mirroring(
        self, source_file_path: Path
    ) -> dict[str, Any] | None:
        """Process a file for directory mirroring.

        This is the main entry point that calculates paths, creates directories,
        and stores mappings.

        Args:
            source_file_path: Absolute path to source PDF file

        Returns:
            Mirror information dictionary if successful, None on failure
        """
        try:
            logger.debug(f"Processing file for mirroring: {source_file_path}")

            # Calculate mirror paths
            mirror_info = self.get_mirror_paths(source_file_path)

            # Create output directory
            if not self.create_output_directory(mirror_info["output_directory"]):
                logger.error(
                    f"Failed to create output directory for {source_file_path}"
                )
                return None

            # Store path mapping
            if not self.store_path_mapping(source_file_path, mirror_info):
                logger.warning(f"Failed to store path mapping for {source_file_path}")
                # Don't fail the whole process for mapping storage failure

            logger.info(
                f"Successfully processed file for mirroring: "
                f"{source_file_path} -> {mirror_info['output_path']}"
            )

            return mirror_info

        except Exception as e:
            logger.error(
                f"Failed to process file for mirroring {source_file_path}: {e}"
            )
            return None

    def get_directory_mappings(
        self, source_directory: str | None = None
    ) -> list[dict[str, Any]]:
        """Get directory mappings from database.

        Args:
            source_directory: Filter by specific source directory (optional)

        Returns:
            List of directory mapping dictionaries
        """
        try:
            with self.db_session_factory() as session:
                query = session.query(PathMapping)

                if source_directory:
                    query = query.filter_by(source_directory=source_directory)

                mappings = query.order_by(
                    PathMapping.directory_level, PathMapping.relative_path
                ).all()

                return [
                    {
                        "id": mapping.id,
                        "source_directory": mapping.source_directory,
                        "output_directory": mapping.output_directory,
                        "relative_path": mapping.relative_path,
                        "directory_level": mapping.directory_level,
                        "files_count": mapping.files_count,
                        "last_scanned": mapping.last_scanned,
                        "created_at": mapping.created_at,
                        "updated_at": mapping.updated_at,
                    }
                    for mapping in mappings
                ]

        except Exception as e:
            logger.error(f"Failed to get directory mappings: {e}")
            return []

    def sync_directory_structure(self, scan_existing: bool = False) -> dict[str, Any]:
        """Synchronize directory structure mappings.

        Args:
            scan_existing: Whether to scan existing files for mappings

        Returns:
            Dictionary with sync statistics
        """
        stats = {
            "directories_scanned": 0,
            "mappings_created": 0,
            "mappings_updated": 0,
            "files_processed": 0,
            "errors": 0,
        }

        try:
            # Walk through watch directory
            for root, dirs, files in os.walk(self.config.watch_base_dir):
                root_path = Path(root)
                stats["directories_scanned"] += 1

                # Process PDF files if scan_existing is True
                if scan_existing:
                    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
                    for pdf_file in pdf_files:
                        file_path = root_path / pdf_file
                        try:
                            mirror_info = self.get_mirror_paths(file_path)
                            if self.store_path_mapping(file_path, mirror_info):
                                stats["mappings_created"] += 1
                            stats["files_processed"] += 1
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            stats["errors"] += 1

                # Limit directory depth
                relative_depth = PathMapper.calculate_directory_depth(
                    root_path, self.config.watch_base_dir
                )
                if relative_depth >= self.config.max_directory_depth:
                    dirs.clear()  # Don't descend further

            logger.info(f"Directory sync completed: {stats}")

        except Exception as e:
            logger.error(f"Failed to sync directory structure: {e}")
            stats["errors"] += 1

        return stats


def create_directory_mirror(
    watch_base_dir: str,
    output_base_dir: str,
    max_depth: int = 10,
    db_session_factory=None,
) -> DirectoryMirror:
    """Factory function to create a DirectoryMirror with standard configuration.

    Args:
        watch_base_dir: Base directory to watch for PDFs
        output_base_dir: Base directory for markdown output
        max_depth: Maximum directory depth to process
        db_session_factory: Database session factory

    Returns:
        Configured DirectoryMirror instance
    """
    config = MirrorConfig(
        watch_base_dir=Path(watch_base_dir),
        output_base_dir=Path(output_base_dir),
        max_directory_depth=max_depth,
        preserve_structure=True,
        create_missing_dirs=True,
        safe_filenames=True,
    )

    return DirectoryMirror(config, db_session_factory)
