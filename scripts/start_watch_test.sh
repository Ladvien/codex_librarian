#!/bin/bash
# ==============================================================================
# Start PDF Watch Test Service
# ==============================================================================
# This script starts the directory watcher for the test folder with proper
# directory mirroring enabled.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WATCH_DIR="$PROJECT_ROOT/watch_pdf_test"
OUTPUT_DIR="$PROJECT_ROOT/watch_pdf_test_output"
PID_FILE="$PROJECT_ROOT/watch_test.pid"
LOG_FILE="$PROJECT_ROOT/watch_test.log"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}Starting PDF Watch Test Service${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Watch service is already running (PID: $PID)${NC}"
        echo -e "Use ${BLUE}./scripts/stop_watch_test.sh${NC} to stop it first"
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Validate directories
echo -e "${YELLOW}Step 1: Validating directories...${NC}"
if [ ! -d "$WATCH_DIR" ]; then
    echo -e "${RED}✗ Watch directory not found: $WATCH_DIR${NC}"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Creating output directory: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
fi

echo -e "${GREEN}✓ Watch directory: $WATCH_DIR${NC}"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"

# Check database connection
echo -e "${YELLOW}Step 2: Checking database connection...${NC}"
if [ -f "$PROJECT_ROOT/.env.test" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env.test" | xargs)
    echo -e "${GREEN}✓ Loaded test environment${NC}"
elif [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
    echo -e "${GREEN}✓ Loaded production environment${NC}"
else
    echo -e "${YELLOW}⚠ No environment file found, using defaults${NC}"
fi

# Test database connection
echo -e "${YELLOW}Step 3: Testing database connection...${NC}"
cd "$PROJECT_ROOT"

# Activate virtual environment and test DB connection
source .venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'src')

try:
    from pdf_to_markdown_mcp.db.session import get_db_session

    with get_db_session() as session:
        from sqlalchemy import text
        result = session.execute(text('SELECT 1')).scalar()
        print('✓ Database connection successful')

    # Check for tables
    from pdf_to_markdown_mcp.db.models import Document, PathMapping
    with get_db_session() as session:
        session.query(Document).first()
        session.query(PathMapping).first()
        print('✓ Required tables exist')

except Exception as e:
    print(f'✗ Database error: {e}')
    print('Run database setup first: ./scripts/setup_test_database.sh')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Database connection failed. Exiting.${NC}"
    exit 1
fi

# Create watch service configuration
echo -e "${YELLOW}Step 4: Creating watch service...${NC}"

# Create Python script to run the watcher
cat > "$PROJECT_ROOT/watch_test_service.py" << 'EOF'
#!/usr/bin/env python3
"""
PDF Watch Test Service

This script runs the directory watcher with directory mirroring enabled
for manual testing of the PDF to Markdown conversion pipeline.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_to_markdown_mcp.core.watcher_service import create_watcher_service, WatcherConfig
from pdf_to_markdown_mcp.core.mirror import create_directory_mirror

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watch_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuration
WATCH_DIR = str(Path(__file__).parent / "watch_pdf_test")
OUTPUT_DIR = str(Path(__file__).parent / "watch_pdf_test_output")

class WatchTestService:
    def __init__(self):
        self.watcher = None
        self.running = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def start(self):
        """Start the watch service."""
        try:
            logger.info("Starting PDF Watch Test Service")
            logger.info(f"Watch directory: {WATCH_DIR}")
            logger.info(f"Output directory: {OUTPUT_DIR}")

            # Create watcher configuration
            config = WatcherConfig(
                watch_directories=[WATCH_DIR],
                recursive=True,
                patterns=["*.pdf", "*.PDF"],
                ignore_patterns=[".*", "*/tmp/*", "*/_samples/*"],
                stability_timeout=2.0,
                max_file_size_mb=100,
                enable_deduplication=True,
            )

            # Create watcher service with directory mirroring
            self.watcher = create_watcher_service(
                config=config,
                enable_directory_mirroring=True,
                output_base_dir=OUTPUT_DIR
            )

            # Start watching
            self.watcher.start()
            self.running = True

            logger.info("✅ PDF Watch Test Service started successfully!")
            logger.info("Drop PDF files into watch_pdf_test/ subdirectories to test")
            logger.info("Press Ctrl+C to stop the service")

            # Keep running
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the watch service."""
        if self.watcher:
            logger.info("Stopping watcher...")
            self.watcher.stop()

        self.running = False
        logger.info("PDF Watch Test Service stopped")

if __name__ == "__main__":
    service = WatchTestService()
    service.setup_signal_handlers()
    service.start()
EOF

chmod +x "$PROJECT_ROOT/watch_test_service.py"

# Start the service
echo -e "${YELLOW}Step 5: Starting watch service...${NC}"
cd "$PROJECT_ROOT"

# Start in background and capture PID
source .venv/bin/activate && python watch_test_service.py > "$LOG_FILE" 2>&1 &
WATCH_PID=$!
echo $WATCH_PID > "$PID_FILE"

# Wait a moment and check if it started successfully
sleep 2
if ps -p "$WATCH_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Watch service started successfully!${NC}"
    echo -e "${GREEN}   PID: $WATCH_PID${NC}"
    echo -e "${GREEN}   Log file: $LOG_FILE${NC}"
else
    echo -e "${RED}✗ Watch service failed to start${NC}"
    echo -e "${RED}Check log file: $LOG_FILE${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🎉 PDF Watch Test Service is now running!${NC}"
echo -e "${BLUE}==============================================================================${NC}"

echo -e "\n${BLUE}Testing Instructions:${NC}"
echo -e "1. Drop PDF files into subdirectories of: ${YELLOW}$WATCH_DIR${NC}"
echo -e "2. Check markdown output in: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "3. Monitor activity: ${YELLOW}tail -f $LOG_FILE${NC}"
echo -e "4. Check status: ${YELLOW}./scripts/check_watch_status.sh${NC}"
echo -e "5. Stop service: ${YELLOW}./scripts/stop_watch_test.sh${NC}"

echo -e "\n${BLUE}Example Test Files:${NC}"
echo -e "  $WATCH_DIR/research/papers/test_paper.pdf"
echo -e "  $WATCH_DIR/reports/2024/monthly_report.pdf"
echo -e "  $WATCH_DIR/books/technical_guide.pdf"

echo -e "\n${YELLOW}Service Details:${NC}"
echo -e "  Process ID: $WATCH_PID"
echo -e "  Log File: $LOG_FILE"
echo -e "  PID File: $PID_FILE"
echo -e "  Watch Directory: $WATCH_DIR"
echo -e "  Output Directory: $OUTPUT_DIR"

echo -e "\n${GREEN}Ready for testing! 📁➡️📄${NC}"