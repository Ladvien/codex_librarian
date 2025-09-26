#!/usr/bin/env python3
"""
Simple file watcher demo using only core watcher functionality.

This demo shows the watchdog file monitoring without database dependencies.
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from pdf_to_markdown_mcp.core.watcher import (
        DirectoryWatcher,
        WatcherConfig,
        PDFFileHandler,
        FileValidator,
        SmartFileDetector
    )
except ImportError as e:
    logger.error(f"Failed to import watcher modules: {e}")
    logger.info("This demo requires the watchdog library. Install it with: pip install watchdog")
    sys.exit(1)


class SimpleDemoTaskQueue:
    """Simple demo task queue that logs operations."""

    def __init__(self):
        """Initialize simple demo task queue."""
        self.processed_files = []
        self.call_count = 0

    def queue_pdf_processing(
        self,
        file_path: str,
        validation_result: Dict[str, Any],
        priority: int = 5,
        processing_options: Dict[str, Any] = None
    ) -> int:
        """Simple queue_pdf_processing method.

        Args:
            file_path: Path to PDF file
            validation_result: File validation metadata
            priority: Processing priority
            processing_options: Optional processing configuration

        Returns:
            Mock document ID
        """
        self.call_count += 1
        document_id = self.call_count

        # Log the queuing operation
        logger.info(f"🔄 QUEUED PDF #{document_id}: {Path(file_path).name}")
        logger.info(f"   📁 Path: {file_path}")
        logger.info(f"   📊 Size: {validation_result.get('size_bytes', 0)} bytes")

        if validation_result.get('hash'):
            logger.info(f"   🔗 Hash: {validation_result['hash'][:16]}...")

        logger.info(f"   ✅ Valid: {validation_result.get('valid', False)}")

        if validation_result.get('error'):
            logger.warning(f"   ⚠️ Error: {validation_result['error']}")

        # Store for statistics
        self.processed_files.append({
            'document_id': document_id,
            'file_path': file_path,
            'timestamp': time.time(),
            'validation_result': validation_result
        })

        return document_id


def create_sample_pdf(directory: Path, filename: str = "sample.pdf") -> Path:
    """Create a sample PDF file for testing.

    Args:
        directory: Directory to create file in
        filename: Name of file to create

    Returns:
        Path to created file
    """
    file_path = directory / filename

    # Create a minimal PDF file
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
182
%%EOF"""

    with open(file_path, 'wb') as f:
        f.write(pdf_content)

    logger.info(f"📄 Created sample PDF: {file_path}")
    return file_path


def test_file_validator():
    """Test the FileValidator functionality."""
    logger.info("🧪 Testing FileValidator...")

    validator = FileValidator()

    # Create temporary test file
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test with valid PDF
        pdf_file = create_sample_pdf(temp_path, "test_validation.pdf")
        result = validator.validate_pdf(pdf_file)

        logger.info(f"   ✅ PDF Validation Result:")
        logger.info(f"      Valid: {result['valid']}")
        logger.info(f"      MIME Type: {result['mime_type']}")
        logger.info(f"      Size: {result['size_bytes']} bytes")
        logger.info(f"      Hash: {result['hash'][:16] if result['hash'] else 'None'}...")

        # Test with non-PDF file
        txt_file = temp_path / "test.txt"
        with open(txt_file, 'w') as f:
            f.write("This is not a PDF")

        result2 = validator.validate_pdf(txt_file)
        logger.info(f"   ❌ TXT Validation Result:")
        logger.info(f"      Valid: {result2['valid']}")
        logger.info(f"      Error: {result2['error']}")


def test_smart_file_detector():
    """Test the SmartFileDetector functionality."""
    logger.info("🧪 Testing SmartFileDetector...")

    detector = SmartFileDetector(stability_timeout=1.0)

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "stability_test.pdf"

        # Create initial file
        with open(test_file, 'w') as f:
            f.write("initial content")

        # First check - should be unstable
        stable1 = detector.is_file_stable(test_file)
        logger.info(f"   First check (should be unstable): {stable1}")

        # Modify file
        with open(test_file, 'a') as f:
            f.write(" - more content")

        # Second check - should still be unstable
        stable2 = detector.is_file_stable(test_file)
        logger.info(f"   After modification (should be unstable): {stable2}")

        # Wait and check again
        time.sleep(1.2)
        stable3 = detector.is_file_stable(test_file)
        logger.info(f"   After timeout (should be stable): {stable3}")


def main():
    """Main entry point for simple demo."""
    logger.info("🎬 Simple PDF File Watcher Demo")
    logger.info("=" * 50)

    # Test components individually first
    test_file_validator()
    test_smart_file_detector()

    logger.info("🚀 Starting File Watcher Demo...")

    # Setup
    demo_dir = Path("./simple_demo_watch")
    demo_dir.mkdir(exist_ok=True)

    task_queue = SimpleDemoTaskQueue()

    # Configuration
    config = WatcherConfig(
        watch_directories=[str(demo_dir.absolute())],
        recursive=True,
        patterns=["*.pdf", "*.PDF"],
        stability_timeout=2.0,  # 2 seconds for demo
        max_file_size_mb=10,
        enable_deduplication=True
    )

    logger.info(f"⚙️ Configuration:")
    logger.info(f"   📂 Watch directory: {demo_dir.absolute()}")
    logger.info(f"   🎯 Patterns: {config.patterns}")
    logger.info(f"   ⏱️ Stability timeout: {config.stability_timeout}s")

    # Create watcher
    watcher = DirectoryWatcher(task_queue, config)

    # Signal handler
    running = True
    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"📡 Received signal {signum}, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start watcher
        watcher.start()
        logger.info("✅ Watcher started!")
        logger.info(f"👀 Drop PDF files into: {demo_dir.absolute()}")

        # Create sample file after 3 seconds
        def create_sample_later():
            time.sleep(3)
            if running:
                create_sample_pdf(demo_dir, f"auto_sample_{int(time.time())}.pdf")

        import threading
        threading.Thread(target=create_sample_later, daemon=True).start()

        # Monitor
        logger.info("🔄 Monitoring loop started (Ctrl+C to stop)")
        while running:
            time.sleep(5)

            status = watcher.get_status()
            stats = len(task_queue.processed_files)

            logger.info(f"📊 Status: Running={status['is_running']}, "
                      f"Files processed={stats}, "
                      f"Tracked={status['stable_files_tracked']}")

    except KeyboardInterrupt:
        logger.info("⏹️ Demo interrupted")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
    finally:
        logger.info("🛑 Shutting down...")
        watcher.stop()

        # Show final statistics
        logger.info(f"📈 Final Statistics:")
        logger.info(f"   📁 Total files processed: {len(task_queue.processed_files)}")

        if task_queue.processed_files:
            logger.info(f"   📋 Processed files:")
            for file_info in task_queue.processed_files:
                logger.info(f"      - {Path(file_info['file_path']).name} (#{file_info['document_id']})")

    logger.info("🏁 Simple Watcher Demo completed")


if __name__ == '__main__':
    main()