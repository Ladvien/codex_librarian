"""
Tests for directory mirroring functionality.

Tests the core DirectoryMirror class and related path mapping logic.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from pdf_to_markdown_mcp.core.mirror import (
    DirectoryMirror,
    MirrorConfig,
    PathMapper,
    create_directory_mirror,
)
from pdf_to_markdown_mcp.models.document import DirectoryMirrorInfo


class TestPathMapper:
    """Test PathMapper utility functions."""

    def test_is_safe_path_valid(self):
        """Test that valid paths are considered safe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            test_path = base_path / "subdir" / "file.pdf"

            assert PathMapper.is_safe_path(test_path, base_path)

    def test_is_safe_path_invalid_traversal(self):
        """Test that directory traversal paths are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            traversal_path = base_path / ".." / ".." / "etc" / "passwd"

            assert not PathMapper.is_safe_path(traversal_path, base_path)

    def test_calculate_relative_path(self):
        """Test relative path calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            file_path = base_path / "documents" / "research" / "paper.pdf"

            relative = PathMapper.calculate_relative_path(file_path, base_path)
            assert relative == Path("documents/research/paper.pdf")

    def test_calculate_relative_path_invalid(self):
        """Test relative path calculation with invalid path."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            base_path = Path(temp_dir1)
            file_path = Path(temp_dir2) / "file.pdf"

            with pytest.raises(ValueError, match="not under base directory"):
                PathMapper.calculate_relative_path(file_path, base_path)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test basic sanitization
        assert PathMapper.sanitize_filename("file<>:name.pdf") == "file___name.pdf"

        # Test empty filename
        assert PathMapper.sanitize_filename("") == "untitled"

        # Test whitespace cleanup
        assert PathMapper.sanitize_filename("  file.pdf  ") == "file.pdf"

        # Test long filename truncation
        long_name = "a" * 300 + ".pdf"
        sanitized = PathMapper.sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".pdf")

    def test_calculate_directory_depth(self):
        """Test directory depth calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # File directly in base
            file_path = base_path / "file.pdf"
            assert PathMapper.calculate_directory_depth(file_path, base_path) == 0

            # File in subdirectory
            file_path = base_path / "docs" / "research" / "file.pdf"
            assert PathMapper.calculate_directory_depth(file_path, base_path) == 2


class TestMirrorConfig:
    """Test MirrorConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            watch_dir = Path(temp_dir) / "watch"
            output_dir = Path(temp_dir) / "output"
            watch_dir.mkdir()
            output_dir.mkdir()

            config = MirrorConfig(watch_base_dir=watch_dir, output_base_dir=output_dir)

            assert config.watch_base_dir == watch_dir
            assert config.output_base_dir == output_dir
            assert config.preserve_structure is True
            assert config.max_directory_depth == 10


class TestDirectoryMirror:
    """Test DirectoryMirror core functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            watch_dir = base_path / "watch"
            output_dir = base_path / "output"
            watch_dir.mkdir()
            output_dir.mkdir()

            yield watch_dir, output_dir

    @pytest.fixture
    def mirror_config(self, temp_dirs):
        """Create test mirror configuration."""
        watch_dir, output_dir = temp_dirs
        return MirrorConfig(
            watch_base_dir=watch_dir,
            output_base_dir=output_dir,
            max_directory_depth=5,
        )

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return Mock(return_value=session)

    @pytest.fixture
    def directory_mirror(self, mirror_config, mock_db_session):
        """Create DirectoryMirror instance for testing."""
        return DirectoryMirror(mirror_config, mock_db_session)

    def test_config_validation_invalid_watch_dir(self, temp_dirs):
        """Test configuration validation with invalid watch directory."""
        watch_dir, output_dir = temp_dirs
        nonexistent_dir = watch_dir / "nonexistent"

        config = MirrorConfig(
            watch_base_dir=nonexistent_dir, output_base_dir=output_dir
        )

        with pytest.raises(ValueError, match="Watch base directory does not exist"):
            DirectoryMirror(config)

    def test_get_mirror_paths_basic(self, directory_mirror, temp_dirs):
        """Test basic mirror path calculation."""
        watch_dir, output_dir = temp_dirs

        # Create test file structure
        test_subdir = watch_dir / "research" / "papers"
        test_subdir.mkdir(parents=True)
        test_file = test_subdir / "document.pdf"
        test_file.write_text("test content")

        mirror_info = directory_mirror.get_mirror_paths(test_file)

        assert mirror_info["source_path"] == test_file
        assert mirror_info["source_relative_path"] == Path(
            "research/papers/document.pdf"
        )
        assert (
            mirror_info["output_path"]
            == output_dir / "research" / "papers" / "document.md"
        )
        assert mirror_info["output_relative_path"] == Path(
            "research/papers/document.md"
        )
        assert mirror_info["directory_depth"] == 2

    def test_get_mirror_paths_unsafe_path(self, directory_mirror, temp_dirs):
        """Test rejection of unsafe paths."""
        watch_dir, _ = temp_dirs

        # Try to use path outside watch directory
        unsafe_path = watch_dir.parent / "outside.pdf"
        unsafe_path.write_text("test")

        with pytest.raises(ValueError, match="not safe or not under watch directory"):
            directory_mirror.get_mirror_paths(unsafe_path)

    def test_get_mirror_paths_too_deep(self, directory_mirror, temp_dirs):
        """Test rejection of paths that are too deep."""
        watch_dir, _ = temp_dirs

        # Create deeply nested structure (deeper than max_directory_depth=5)
        deep_path = watch_dir
        for i in range(7):  # Create 7 levels deep
            deep_path = deep_path / f"level_{i}"
        deep_path.mkdir(parents=True)

        test_file = deep_path / "document.pdf"
        test_file.write_text("test content")

        with pytest.raises(ValueError, match="Directory depth .* exceeds maximum"):
            directory_mirror.get_mirror_paths(test_file)

    def test_create_output_directory(self, directory_mirror, temp_dirs):
        """Test output directory creation."""
        _, output_dir = temp_dirs

        target_dir = output_dir / "new" / "nested" / "directory"

        result = directory_mirror.create_output_directory(target_dir)

        assert result is True
        assert target_dir.exists()
        assert target_dir.is_dir()

    def test_create_output_directory_exists(self, directory_mirror, temp_dirs):
        """Test output directory creation when directory exists."""
        _, output_dir = temp_dirs

        existing_dir = output_dir / "existing"
        existing_dir.mkdir()

        result = directory_mirror.create_output_directory(existing_dir)

        assert result is True
        assert existing_dir.exists()

    def test_create_output_directory_disabled(
        self, mirror_config, temp_dirs, mock_db_session
    ):
        """Test behavior when directory creation is disabled."""
        mirror_config.create_missing_dirs = False
        mirror = DirectoryMirror(mirror_config, mock_db_session)

        _, output_dir = temp_dirs
        nonexistent_dir = output_dir / "nonexistent"

        result = mirror.create_output_directory(nonexistent_dir)

        assert result is False
        assert not nonexistent_dir.exists()

    def test_store_path_mapping_new(self, directory_mirror, temp_dirs, mock_db_session):
        """Test storing new path mapping."""
        watch_dir, output_dir = temp_dirs

        mirror_info = {
            "source_relative_path": Path("docs/test.pdf"),
            "directory_depth": 1,
        }

        # Mock database operations
        mock_session = mock_db_session()
        mock_session.query().filter_by().first.return_value = (
            None  # No existing mapping
        )

        test_file = watch_dir / "docs" / "test.pdf"
        test_file.parent.mkdir()
        test_file.write_text("test")

        result = directory_mirror.store_path_mapping(test_file, mirror_info)

        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_store_path_mapping_existing(
        self, directory_mirror, temp_dirs, mock_db_session
    ):
        """Test updating existing path mapping."""
        watch_dir, _ = temp_dirs

        mirror_info = {
            "source_relative_path": Path("docs/test.pdf"),
            "directory_depth": 1,
        }

        # Mock existing mapping
        existing_mapping = Mock()
        existing_mapping.files_count = 1

        mock_session = mock_db_session()
        mock_session.query().filter_by().first.return_value = existing_mapping

        test_file = watch_dir / "docs" / "test.pdf"
        test_file.parent.mkdir()
        test_file.write_text("test")

        result = directory_mirror.store_path_mapping(test_file, mirror_info)

        assert result is True
        assert existing_mapping.files_count == 2
        mock_session.commit.assert_called_once()

    def test_process_file_for_mirroring_success(
        self, directory_mirror, temp_dirs, mock_db_session
    ):
        """Test complete file mirroring process."""
        watch_dir, output_dir = temp_dirs

        # Create test file
        test_subdir = watch_dir / "research"
        test_subdir.mkdir()
        test_file = test_subdir / "paper.pdf"
        test_file.write_text("test content")

        # Mock database operations
        mock_session = mock_db_session()
        mock_session.query().filter_by().first.return_value = None

        result = directory_mirror.process_file_for_mirroring(test_file)

        assert result is not None
        assert result["source_path"] == test_file
        assert result["output_path"] == output_dir / "research" / "paper.md"
        assert (output_dir / "research").exists()

    def test_process_file_for_mirroring_failure(self, directory_mirror, temp_dirs):
        """Test file mirroring process with invalid file."""
        watch_dir, _ = temp_dirs

        # Use file outside watch directory
        invalid_file = watch_dir.parent / "invalid.pdf"
        invalid_file.write_text("test")

        result = directory_mirror.process_file_for_mirroring(invalid_file)

        assert result is None


class TestDirectoryMirrorFactory:
    """Test factory function for creating DirectoryMirror."""

    def test_create_directory_mirror(self):
        """Test factory function creates proper DirectoryMirror."""
        with tempfile.TemporaryDirectory() as temp_dir:
            watch_dir = Path(temp_dir) / "watch"
            output_dir = Path(temp_dir) / "output"
            watch_dir.mkdir()
            output_dir.mkdir()

            mirror = create_directory_mirror(
                str(watch_dir), str(output_dir), max_depth=15
            )

            assert isinstance(mirror, DirectoryMirror)
            assert mirror.config.watch_base_dir == watch_dir
            assert mirror.config.output_base_dir == output_dir
            assert mirror.config.max_directory_depth == 15


class TestDirectoryMirrorInfo:
    """Test DirectoryMirrorInfo Pydantic model."""

    def test_valid_mirror_info(self):
        """Test creation of valid DirectoryMirrorInfo."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            mirror_info = DirectoryMirrorInfo(
                source_path=base_path / "input" / "file.pdf",
                source_relative_path=Path("input/file.pdf"),
                output_path=base_path / "output" / "file.md",
                output_relative_path=Path("output/file.md"),
                directory_depth=1,
                output_directory=base_path / "output",
            )

            assert mirror_info.source_relative_path == Path("input/file.pdf")
            assert mirror_info.directory_depth == 1

    def test_invalid_relative_path(self):
        """Test validation of non-absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            with pytest.raises(ValueError, match="Absolute paths must be absolute"):
                DirectoryMirrorInfo(
                    source_path=Path("relative/path.pdf"),  # Invalid - not absolute
                    source_relative_path=Path("input/file.pdf"),
                    output_path=base_path / "output" / "file.md",
                    output_relative_path=Path("output/file.md"),
                    directory_depth=1,
                    output_directory=base_path / "output",
                )

    def test_negative_directory_depth(self):
        """Test validation of directory depth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            with pytest.raises(
                ValueError, match="ensure this value is greater than or equal to 0"
            ):
                DirectoryMirrorInfo(
                    source_path=base_path / "input" / "file.pdf",
                    source_relative_path=Path("input/file.pdf"),
                    output_path=base_path / "output" / "file.md",
                    output_relative_path=Path("output/file.md"),
                    directory_depth=-1,  # Invalid
                    output_directory=base_path / "output",
                )
