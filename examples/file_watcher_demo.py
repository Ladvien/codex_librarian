#!/usr/bin/env python3
"""
File Watcher Demo Example - Directory Monitoring Without Dependencies

This example demonstrates the file watcher system in standalone mode:
- Directory monitoring for PDF files
- Event handling without database dependencies
- Mock task queue for learning the watcher API
- Configurable watcher behavior

Perfect for understanding how file monitoring works before full integration.
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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

try:
    from pdf_to_markdown_mcp.core.watcher import DirectoryWatcher, WatcherConfig
except ImportError as e:
    logger.error(f"Failed to import watcher modules: {e}")
    logger.info(
        "This demo requires the watchdog library. Install it with: pip install watchdog"
    )
    sys.exit(1)


class DemoTaskQueue:
    """Demo task queue that logs operations instead of queuing to Celery."""

    def __init__(self):
        """Initialize demo task queue."""
        self.processed_files = []
        self.call_count = 0

    def queue_pdf_processing(
        self,
        file_path: str,
        validation_result: Dict[str, Any],
        priority: int = 5,
        processing_options: Dict[str, Any] = None,
    ) -> int:
        """Demo queue_pdf_processing method.

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
        logger.info(f"   🔍 MIME: {validation_result.get('mime_type', 'unknown')}")
        logger.info(f"   🎯 Priority: {priority}")
        logger.info(f"   ✅ Valid: {validation_result.get('valid', False)}")

        if validation_result.get("error"):
            logger.warning(f"   ⚠️ Error: {validation_result['error']}")

        # Store for statistics
        self.processed_files.append(
            {
                "document_id": document_id,
                "file_path": file_path,
                "timestamp": time.time(),
                "validation_result": validation_result,
                "priority": priority,
            }
        )

        return document_id

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_files": len(self.processed_files),
            "last_processed": (
                self.processed_files[-1] if self.processed_files else None
            ),
        }


class DemoWatcherService:
    """Demo service for running PDF file watcher."""

    def __init__(self):
        """Initialize demo service."""
        self.watcher = None
        self.task_queue = DemoTaskQueue()
        self.running = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def create_demo_directory(self, base_dir: str = "./demo_watch") -> Path:
        """Create demo directory for testing.

        Args:
            base_dir: Base directory path

        Returns:
            Path to created demo directory
        """
        demo_path = Path(base_dir)
        demo_path.mkdir(exist_ok=True)

        # Create some subdirectories
        (demo_path / "pdfs").mkdir(exist_ok=True)
        (demo_path / "input").mkdir(exist_ok=True)

        logger.info(f"📁 Created demo directory: {demo_path.absolute()}")
        logger.info(f"   You can drop PDF files into: {demo_path.absolute()}")
        logger.info(
            f"   Or into subdirectories: {demo_path / 'pdfs'}, {demo_path / 'input'}"
        )

        return demo_path

    def create_sample_pdf(self, directory: Path, filename: str = "sample.pdf") -> Path:
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

        with open(file_path, "wb") as f:
            f.write(pdf_content)

        logger.info(f"📄 Created sample PDF: {file_path}")
        return file_path

    def start_watcher(self, watch_directories: list[str]):
        """Start the file watcher.

        Args:
            watch_directories: List of directories to watch
        """
        # Create configuration
        config = WatcherConfig(
            watch_directories=watch_directories,
            recursive=True,
            patterns=["*.pdf", "*.PDF"],
            ignore_patterns=["**/.*", "**/tmp/*", "**/temp/*"],
            stability_timeout=2.0,  # Short timeout for demo
            max_file_size_mb=100,  # Reasonable limit for demo
            enable_deduplication=True,
        )

        logger.info(f"⚙️ Configuration:")
        logger.info(f"   📂 Watch directories: {config.watch_directories}")
        logger.info(f"   🔄 Recursive: {config.recursive}")
        logger.info(f"   🎯 Patterns: {config.patterns}")
        logger.info(f"   ⏱️ Stability timeout: {config.stability_timeout}s")
        logger.info(f"   📏 Max file size: {config.max_file_size_mb}MB")

        # Create watcher
        self.watcher = DirectoryWatcher(self.task_queue, config)

        try:
            self.watcher.start()
            self.running = True
            logger.info("🚀 Watcher started successfully!")
            logger.info("👀 Monitoring for PDF files...")

        except Exception as e:
            logger.error(f"❌ Failed to start watcher: {e}")
            return False

        return True

    def run_demo_loop(self):
        """Run demo monitoring loop."""
        logger.info("🔄 Demo monitoring loop started (Ctrl+C to stop)")
        logger.info(
            "💡 Try adding PDF files to the watch directories to see them detected!"
        )

        try:
            while self.running:
                time.sleep(10)  # Status update every 10 seconds

                # Get watcher status
                if self.watcher:
                    status = self.watcher.get_status()
                    stats = self.task_queue.get_stats()

                    logger.info(
                        f"📊 Status: Running={status['is_running']}, "
                        f"Processed={status['processed_files_count']}, "
                        f"Tracked={status['stable_files_tracked']}, "
                        f"Total Queued={stats['total_files']}"
                    )

        except KeyboardInterrupt:
            logger.info("⏹️ Demo loop interrupted")
        except Exception as e:
            logger.error(f"❌ Error in demo loop: {e}")

    def shutdown(self):
        """Shutdown watcher gracefully."""
        logger.info("🛑 Shutting down demo watcher service...")
        self.running = False

        if self.watcher:
            try:
                self.watcher.stop()
                logger.info("✅ Watcher stopped successfully")
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}")

        # Show final statistics
        stats = self.task_queue.get_stats()
        logger.info(f"📈 Final Statistics:")
        logger.info(f"   📁 Total files processed: {stats['total_files']}")

        if stats["total_files"] > 0:
            logger.info(f"   📋 Processed files:")
            for i, file_info in enumerate(
                self.task_queue.processed_files[-5:], 1
            ):  # Show last 5
                logger.info(
                    f"      {i}. {Path(file_info['file_path']).name} (#{file_info['document_id']})"
                )


def main():
    """Main entry point for demo."""
    logger.info("🎬 PDF File Watcher Demo")
    logger.info("=" * 50)

    # Create demo service
    service = DemoWatcherService()
    service.setup_signal_handlers()

    # Get or create demo directory
    if len(sys.argv) > 1:
        watch_dirs = sys.argv[1:]
        logger.info(f"📂 Using provided directories: {watch_dirs}")
    else:
        # Create demo directory
        demo_dir = service.create_demo_directory()
        watch_dirs = [str(demo_dir)]

        # Create a sample PDF after a short delay
        def create_sample_later():
            time.sleep(5)
            service.create_sample_pdf(demo_dir, f"demo_sample_{int(time.time())}.pdf")

        import threading

        threading.Thread(target=create_sample_later, daemon=True).start()

    # Validate directories
    valid_dirs = []
    for dir_path in watch_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            valid_dirs.append(str(path.absolute()))
            logger.info(f"✅ Valid directory: {path.absolute()}")
        else:
            logger.warning(f"⚠️ Invalid directory: {dir_path}")

    if not valid_dirs:
        logger.error("❌ No valid directories to watch!")
        sys.exit(1)

    # Start watcher
    if not service.start_watcher(valid_dirs):
        logger.error("❌ Failed to start watcher")
        sys.exit(1)

    # Run demo
    try:
        service.run_demo_loop()
    finally:
        service.shutdown()

    logger.info("🏁 PDF File Watcher Demo completed")


if __name__ == "__main__":
    main()
