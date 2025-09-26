#!/bin/bash
# ==============================================================================
# Check PDF Watch Test Service Status
# ==============================================================================

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
PID_FILE="$PROJECT_ROOT/watch_test.pid"
LOG_FILE="$PROJECT_ROOT/watch_test.log"
WATCH_DIR="$PROJECT_ROOT/watch_pdf_test"
OUTPUT_DIR="$PROJECT_ROOT/watch_pdf_test_output"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}PDF Watch Test Service Status${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Check if service is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service Status: RUNNING${NC}"
        echo -e "   Process ID: $PID"

        # Get process info
        PROCESS_INFO=$(ps -p "$PID" -o pid,ppid,start,etime,cmd --no-headers)
        echo -e "   Process Info: $PROCESS_INFO"
    else
        echo -e "${RED}❌ Service Status: STOPPED (stale PID file)${NC}"
        echo -e "   Stale PID: $PID"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}⏹️  Service Status: NOT RUNNING${NC}"
fi

echo -e "\n${YELLOW}Directory Status:${NC}"
echo -e "   Watch Directory: $WATCH_DIR"
if [ -d "$WATCH_DIR" ]; then
    PDF_COUNT=$(find "$WATCH_DIR" -name "*.pdf" -o -name "*.PDF" 2>/dev/null | wc -l)
    echo -e "   ${GREEN}✓ Exists ($PDF_COUNT PDFs)${NC}"
else
    echo -e "   ${RED}✗ Not found${NC}"
fi

echo -e "   Output Directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    MD_COUNT=$(find "$OUTPUT_DIR" -name "*.md" 2>/dev/null | wc -l)
    echo -e "   ${GREEN}✓ Exists ($MD_COUNT Markdown files)${NC}"
else
    echo -e "   ${YELLOW}⚠ Not created yet${NC}"
fi

# Check recent activity
echo -e "\n${YELLOW}Recent Activity:${NC}"
if [ -f "$LOG_FILE" ]; then
    echo -e "   Log File: $LOG_FILE"
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo "unknown")
    echo -e "   Log Size: $LOG_SIZE bytes"

    echo -e "\n   ${BLUE}Last 5 log entries:${NC}"
    tail -n 5 "$LOG_FILE" | sed 's/^/   /'
else
    echo -e "   ${YELLOW}No log file found${NC}"
fi

# Database status
echo -e "\n${YELLOW}Database Status:${NC}"
cd "$PROJECT_ROOT"

# Load environment
if [ -f ".env.test" ]; then
    export $(grep -v '^#' .env.test | xargs) 2>/dev/null || true
elif [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

# Check database connection and document count
python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from pdf_to_markdown_mcp.db.session import get_db_session
    from pdf_to_markdown_mcp.db.models import Document, PathMapping
    from sqlalchemy import func

    with get_db_session() as session:
        # Test connection
        session.execute(__import__('sqlalchemy').text('SELECT 1'))
        print('   ✅ Database: Connected')

        # Count documents
        doc_count = session.query(func.count(Document.id)).scalar() or 0
        print(f'   📄 Documents: {doc_count} total')

        # Count processing statuses
        pending = session.query(func.count(Document.id)).filter(Document.conversion_status == 'pending').scalar() or 0
        processing = session.query(func.count(Document.id)).filter(Document.conversion_status == 'processing').scalar() or 0
        completed = session.query(func.count(Document.id)).filter(Document.conversion_status == 'completed').scalar() or 0
        failed = session.query(func.count(Document.id)).filter(Document.conversion_status == 'failed').scalar() or 0

        print(f'      • Pending: {pending}')
        print(f'      • Processing: {processing}')
        print(f'      • Completed: {completed}')
        print(f'      • Failed: {failed}')

        # Count path mappings
        mapping_count = session.query(func.count(PathMapping.id)).scalar() or 0
        print(f'   🗂️  Path Mappings: {mapping_count} total')

except Exception as e:
    print(f'   ❌ Database: Error - {e}')
" 2>/dev/null

echo -e "\n${BLUE}==============================================================================${NC}"

# Quick actions
echo -e "${YELLOW}Quick Actions:${NC}"
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
    echo -e "   Stop Service: ${BLUE}./scripts/stop_watch_test.sh${NC}"
    echo -e "   View Live Log: ${BLUE}tail -f $LOG_FILE${NC}"
    echo -e "   Watch Activity: ${BLUE}./scripts/watch_activity.sh${NC}"
else
    echo -e "   Start Service: ${BLUE}./scripts/start_watch_test.sh${NC}"
    echo -e "   View Last Log: ${BLUE}cat $LOG_FILE${NC}"
fi

echo -e "   Cleanup Test: ${BLUE}./scripts/cleanup_watch_test.sh${NC}"
echo -e "   Add Test Files: ${BLUE}cp your_pdfs/*.pdf $WATCH_DIR/research/papers/${NC}"