"""Unit tests for file system monitoring with Watchdog."""

import hashlib
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.core.watcher import (
    DirectoryWatcher,
    FileValidator,
    PDFFileHandler,
    SmartFileDetector,
    WatcherConfig,
)


class TestWatcherConfig:
    """Test WatcherConfig following TDD"""

    def test_config_initialization_with_defaults(self):
        """Test that WatcherConfig initializes with proper defaults"""
        # Given (Arrange)
        config = WatcherConfig()

        # Then (Assert)
        assert config.watch_directories == []
        assert config.recursive is True
        assert config.patterns == ["*.pdf", "*.PDF"]
        assert config.ignore_patterns == ["**/.*", "**/tmp/*", "**/temp/*"]
        assert config.stability_timeout == 5.0
        assert config.max_file_size_mb == 500
        assert config.enable_deduplication is True

    def test_config_initialization_with_custom_values(self):
        """Test that WatcherConfig accepts custom configuration values"""
        # Given (Arrange)
        watch_dirs = ["/path/to/watch1", "/path/to/watch2"]
        patterns = ["*.pdf", "*.docx"]
        ignore_patterns = ["**/ignore/*"]

        # When (Act)
        config = WatcherConfig(
            watch_directories=watch_dirs,
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            recursive=False,
            stability_timeout=10.0,
            max_file_size_mb=100,
        )

        # Then (Assert)
        assert config.watch_directories == watch_dirs
        assert config.patterns == patterns
        assert config.ignore_patterns == ignore_patterns
        assert config.recursive is False
        assert config.stability_timeout == 10.0
        assert config.max_file_size_mb == 100


class TestFileValidator:
    """Test FileValidator following TDD"""

    @pytest.fixture
    def validator(self):
        """Setup FileValidator with mocked dependencies"""
        return FileValidator()

    def test_validate_pdf_with_valid_file(self, validator):
        """Test that validator correctly validates a proper PDF file"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.flush()
            file_path = Path(temp_file.name)

            # When (Act)
            result = validator.validate_pdf(file_path)

            # Then (Assert)
            assert result["valid"] is True
            assert result["mime_type"] == "application/pdf"
            assert result["size_bytes"] > 0
            assert result["hash"] is not None
            assert result["error"] is None

    def test_validate_pdf_with_invalid_file_extension(self, validator):
        """Test that validator rejects files with invalid file extension"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"This is not a PDF")
            temp_file.flush()
            file_path = Path(temp_file.name)

            # When (Act)
            result = validator.validate_pdf(file_path)

            # Then (Assert)
            assert result["valid"] is False
            assert result["error"] == "Invalid file extension: .txt"

    def test_validate_pdf_with_invalid_magic_bytes(self, validator):
        """Test that validator rejects PDF files with invalid magic bytes"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"This is not a real PDF file")
            temp_file.flush()
            file_path = Path(temp_file.name)

            # When (Act)
            result = validator.validate_pdf(file_path)

            # Then (Assert)
            assert result["valid"] is False
            assert result["error"] == "File does not start with PDF magic bytes"

    def test_validate_pdf_with_nonexistent_file(self, validator):
        """Test that validator handles nonexistent files gracefully"""
        # Given (Arrange)
        file_path = Path("/nonexistent/file.pdf")

        # When (Act)
        result = validator.validate_pdf(file_path)

        # Then (Assert)
        assert result["valid"] is False
        assert result["error"] is not None

    def test_calculate_file_hash(self, validator):
        """Test that file hash calculation works correctly"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile() as temp_file:
            content = b"test content for hashing"
            temp_file.write(content)
            temp_file.flush()
            file_path = Path(temp_file.name)

            expected_hash = hashlib.sha256(content).hexdigest()

            # When (Act)
            result_hash = validator.calculate_file_hash(file_path)

            # Then (Assert)
            assert result_hash == expected_hash


class TestSmartFileDetector:
    """Test SmartFileDetector following TDD"""

    @pytest.fixture
    def detector(self):
        """Setup SmartFileDetector with short timeout for testing"""
        return SmartFileDetector(stability_timeout=0.1)

    def test_file_stability_detection_unstable_file(self, detector):
        """Test that detector correctly identifies changing files as unstable"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = Path(temp_file.name)

            # Write initial content
            temp_file.write(b"initial content")
            temp_file.flush()

            # When (Act) - First check
            is_stable_1 = detector.is_file_stable(file_path)

            # Modify file
            temp_file.write(b" more content")
            temp_file.flush()

            # Second check immediately after modification
            is_stable_2 = detector.is_file_stable(file_path)

            # Then (Assert)
            assert is_stable_1 is False  # First check always returns False
            assert is_stable_2 is False  # File changed, so unstable

    def test_file_stability_detection_stable_file(self, detector):
        """Test that detector identifies stable files after timeout"""
        # Given (Arrange)
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = Path(temp_file.name)
            temp_file.write(b"stable content")
            temp_file.flush()

            # When (Act)
            # First check
            is_stable_1 = detector.is_file_stable(file_path)

            # Wait for stability timeout
            time.sleep(0.15)  # Slightly longer than timeout

            # Second check
            is_stable_2 = detector.is_file_stable(file_path)

            # Then (Assert)
            assert is_stable_1 is False  # First check always returns False
            assert is_stable_2 is True  # File stable after timeout


class TestPDFFileHandler:
    """Test PDFFileHandler following TDD"""

    @pytest.fixture
    def mock_task_queue(self):
        """Create mock task queue"""
        return Mock()

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return WatcherConfig(
            patterns=["*.pdf"], stability_timeout=0.1, max_file_size_mb=1
        )

    @pytest.fixture
    def handler(self, mock_task_queue, config):
        """Setup PDFFileHandler with mocked dependencies"""
        return PDFFileHandler(mock_task_queue, config)

    def test_is_pdf_file_with_valid_pdf(self, handler):
        """Test that handler correctly identifies PDF files"""
        # Given (Arrange)
        pdf_path = "/path/to/document.pdf"

        # When (Act)
        result = handler.is_pdf_file(pdf_path)

        # Then (Assert)
        assert result is True

    def test_is_pdf_file_with_uppercase_extension(self, handler):
        """Test that handler handles uppercase PDF extensions"""
        # Given (Arrange)
        pdf_path = "/path/to/document.PDF"

        # When (Act)
        result = handler.is_pdf_file(pdf_path)

        # Then (Assert)
        assert result is True

    def test_is_pdf_file_with_non_pdf_file(self, handler):
        """Test that handler rejects non-PDF files"""
        # Given (Arrange)
        txt_path = "/path/to/document.txt"

        # When (Act)
        result = handler.is_pdf_file(txt_path)

        # Then (Assert)
        assert result is False

    def test_on_created_with_valid_pdf(self, handler, mock_task_queue):
        """Test that handler processes new PDF files correctly"""
        # Given (Arrange)
        mock_detector = Mock()
        mock_detector.is_file_stable.return_value = True

        mock_validator = Mock()
        mock_validator.validate_pdf.return_value = {
            "valid": True,
            "mime_type": "application/pdf",
            "size_bytes": 1000,
            "hash": "test_hash",
            "error": None,
        }

        # Replace instances in handler
        handler.detector = mock_detector
        handler.validator = mock_validator

        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/test.pdf"

        # When (Act)
        handler.on_created(mock_event)

        # Then (Assert)
        mock_task_queue.queue_pdf_processing.assert_called_once()
        args = mock_task_queue.queue_pdf_processing.call_args[0]
        assert args[0] == "/path/to/test.pdf"

    def test_on_created_with_invalid_pdf(self, handler, mock_task_queue):
        """Test that handler skips invalid PDF files"""
        # Given (Arrange)
        mock_validator = Mock()
        mock_validator.validate_pdf.return_value = {
            "valid": False,
            "error": "Invalid MIME type",
        }
        handler.validator = mock_validator

        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/fake.pdf"

        # When (Act)
        handler.on_created(mock_event)

        # Then (Assert)
        mock_task_queue.queue_pdf_processing.assert_not_called()

    def test_on_created_ignores_directories(self, handler, mock_task_queue):
        """Test that handler ignores directory creation events"""
        # Given (Arrange)
        mock_event = Mock()
        mock_event.is_directory = True
        mock_event.src_path = "/path/to/directory"

        # When (Act)
        handler.on_created(mock_event)

        # Then (Assert)
        mock_task_queue.queue_pdf_processing.assert_not_called()


class TestDirectoryWatcher:
    """Test DirectoryWatcher following TDD"""

    @pytest.fixture
    def mock_task_queue(self):
        """Create mock task queue"""
        return Mock()

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return WatcherConfig(watch_directories=["/tmp/test"], recursive=True)

    @pytest.fixture
    def watcher(self, mock_task_queue, config):
        """Setup DirectoryWatcher with mocked dependencies"""
        return DirectoryWatcher(mock_task_queue, config)

    @patch("pdf_to_markdown_mcp.core.watcher.Observer")
    def test_start_watching(self, mock_observer_class, watcher):
        """Test that watcher starts monitoring directories correctly"""
        # Given (Arrange)
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer

        # When (Act)
        watcher.start()

        # Then (Assert)
        mock_observer.start.assert_called_once()
        assert watcher.observer is mock_observer

    @patch("pdf_to_markdown_mcp.core.watcher.Observer")
    def test_stop_watching(self, mock_observer_class, watcher):
        """Test that watcher stops monitoring gracefully"""
        # Given (Arrange)
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        watcher.start()

        # When (Act)
        watcher.stop()

        # Then (Assert)
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    @patch("pdf_to_markdown_mcp.core.watcher.Observer")
    def test_add_watch_directory(self, mock_observer_class, watcher):
        """Test that watcher can add new directories dynamically"""
        # Given (Arrange)
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        watcher.start()

        new_directory = "/tmp/new_watch_dir"

        # When (Act)
        watcher.add_watch_directory(new_directory)

        # Then (Assert)
        assert new_directory in watcher.config.watch_directories
        # Observer.schedule should be called for the new directory
        assert mock_observer.schedule.call_count >= 1

    def test_is_running_when_stopped(self, watcher):
        """Test that is_running returns False when watcher is stopped"""
        # When (Act)
        result = watcher.is_running()

        # Then (Assert)
        assert result is False

    @patch("pdf_to_markdown_mcp.core.watcher.Observer")
    def test_is_running_when_started(self, mock_observer_class, watcher):
        """Test that is_running returns True when watcher is active"""
        # Given (Arrange)
        mock_observer = Mock()
        mock_observer.is_alive.return_value = True
        mock_observer_class.return_value = mock_observer
        watcher.start()

        # When (Act)
        result = watcher.is_running()

        # Then (Assert)
        assert result is True

    def test_update_config(self, watcher):
        """Test that configuration can be updated dynamically"""
        # Given (Arrange)
        new_config = WatcherConfig(
            watch_directories=["/tmp/updated"],
            patterns=["*.pdf", "*.docx"],
            stability_timeout=10.0,
        )

        # When (Act)
        watcher.update_config(new_config)

        # Then (Assert)
        assert watcher.config == new_config
        assert watcher.config.watch_directories == ["/tmp/updated"]
        assert watcher.config.patterns == ["*.pdf", "*.docx"]
        assert watcher.config.stability_timeout == 10.0
