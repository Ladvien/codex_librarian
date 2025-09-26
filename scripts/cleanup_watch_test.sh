#!/bin/bash
# ==============================================================================
# Cleanup PDF Watch Test Environment
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
WATCH_DIR="$PROJECT_ROOT/watch_pdf_test"
OUTPUT_DIR="$PROJECT_ROOT/watch_pdf_test_output"
PID_FILE="$PROJECT_ROOT/watch_test.pid"
LOG_FILE="$PROJECT_ROOT/watch_test.log"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}Cleaning up PDF Watch Test Environment${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Parse command line arguments
CLEAN_DB=false
KEEP_STRUCTURE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-db)
            CLEAN_DB=true
            shift
            ;;
        --remove-dirs)
            KEEP_STRUCTURE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean-db      Also clean database test entries"
            echo "  --remove-dirs   Remove directory structure (not just files)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Default behavior: Remove files but keep directory structure"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Step 1: Stop the service if running
echo -e "${YELLOW}Step 1: Stopping watch service...${NC}"
if [ -f "$PID_FILE" ]; then
    ./scripts/stop_watch_test.sh
else
    echo -e "${GREEN}✓ Service not running${NC}"
fi

# Step 2: Clean watch directory
echo -e "${YELLOW}Step 2: Cleaning watch directory...${NC}"
if [ -d "$WATCH_DIR" ]; then
    if [ "$KEEP_STRUCTURE" = true ]; then
        # Remove only PDF files, keep directory structure
        find "$WATCH_DIR" -name "*.pdf" -type f -delete 2>/dev/null || true
        find "$WATCH_DIR" -name "*.PDF" -type f -delete 2>/dev/null || true
        echo -e "${GREEN}✓ Removed PDF files (kept directory structure)${NC}"
    else
        # Remove entire directory structure
        rm -rf "$WATCH_DIR"
        echo -e "${GREEN}✓ Removed entire watch directory${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Watch directory not found${NC}"
fi

# Step 3: Clean output directory
echo -e "${YELLOW}Step 3: Cleaning output directory...${NC}"
if [ -d "$OUTPUT_DIR" ]; then
    if [ "$KEEP_STRUCTURE" = true ]; then
        # Remove only markdown files, keep directory structure
        find "$OUTPUT_DIR" -name "*.md" -type f -delete 2>/dev/null || true
        find "$OUTPUT_DIR" -type d -empty -delete 2>/dev/null || true
        echo -e "${GREEN}✓ Removed markdown files (cleaned empty dirs)${NC}"
    else
        # Remove entire output directory
        rm -rf "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Removed entire output directory${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Output directory not found${NC}"
fi

# Step 4: Clean service files
echo -e "${YELLOW}Step 4: Cleaning service files...${NC}"
FILES_CLEANED=0

if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
    FILES_CLEANED=$((FILES_CLEANED + 1))
fi

if [ -f "$PROJECT_ROOT/watch_test_service.py" ]; then
    rm -f "$PROJECT_ROOT/watch_test_service.py"
    FILES_CLEANED=$((FILES_CLEANED + 1))
fi

if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
    FILES_CLEANED=$((FILES_CLEANED + 1))
fi

echo -e "${GREEN}✓ Cleaned $FILES_CLEANED service files${NC}"

# Step 5: Clean database (if requested)
if [ "$CLEAN_DB" = true ]; then
    echo -e "${YELLOW}Step 5: Cleaning test database entries...${NC}"
    cd "$PROJECT_ROOT"

    # Load environment
    if [ -f ".env.test" ]; then
        export $(grep -v '^#' .env.test | xargs) 2>/dev/null || true
    elif [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs) 2>/dev/null || true
    fi

    # Clean database entries for test files
    python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from pdf_to_markdown_mcp.db.session import get_db_session
    from pdf_to_markdown_mcp.db.models import Document, PathMapping, DocumentContent, DocumentEmbedding
    from sqlalchemy import func

    with get_db_session() as session:
        # Count entries before cleanup
        doc_count_before = session.query(func.count(Document.id)).scalar() or 0
        mapping_count_before = session.query(func.count(PathMapping.id)).scalar() or 0

        # Delete test-related documents and their dependencies
        watch_pattern = '%watch_pdf_test%'
        test_docs = session.query(Document).filter(Document.source_path.like(watch_pattern)).all()
        doc_ids = [doc.id for doc in test_docs]

        if doc_ids:
            # Delete related records first (foreign key constraints)
            session.query(DocumentEmbedding).filter(DocumentEmbedding.document_id.in_(doc_ids)).delete(synchronize_session=False)
            session.query(DocumentContent).filter(DocumentContent.document_id.in_(doc_ids)).delete(synchronize_session=False)

            # Delete documents
            session.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)

        # Delete test path mappings
        session.query(PathMapping).filter(PathMapping.source_directory.like(watch_pattern)).delete(synchronize_session=False)

        session.commit()

        # Count after cleanup
        doc_count_after = session.query(func.count(Document.id)).scalar() or 0
        mapping_count_after = session.query(func.count(PathMapping.id)).scalar() or 0

        docs_deleted = doc_count_before - doc_count_after
        mappings_deleted = mapping_count_before - mapping_count_after

        print(f'   📄 Deleted {docs_deleted} test documents')
        print(f'   🗂️  Deleted {mappings_deleted} test path mappings')

except Exception as e:
    print(f'   ❌ Database cleanup error: {e}')
    sys.exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Database cleanup completed${NC}"
    else
        echo -e "${RED}✗ Database cleanup failed${NC}"
    fi
else
    echo -e "${YELLOW}Step 5: Skipping database cleanup (use --clean-db to enable)${NC}"
fi

# Step 6: Recreate directory structure if needed
if [ "$KEEP_STRUCTURE" = true ] && ([ ! -d "$WATCH_DIR" ] || [ ! -d "$OUTPUT_DIR" ]); then
    echo -e "${YELLOW}Step 6: Recreating directory structure...${NC}"

    # Recreate watch directories
    mkdir -p "$WATCH_DIR"/{research/{papers,archive,drafts},reports/{2024,2023},books,manuals}
    mkdir -p "$OUTPUT_DIR"

    # Recreate README
    if [ ! -f "$WATCH_DIR/README.md" ]; then
        cp "$SCRIPT_DIR/../watch_pdf_test/README.md" "$WATCH_DIR/" 2>/dev/null || true
    fi

    echo -e "${GREEN}✓ Recreated directory structure${NC}"
else
    echo -e "${YELLOW}Step 6: Directory structure preserved${NC}"
fi

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}🧹 Cleanup completed!${NC}"
echo -e "${BLUE}==============================================================================${NC}"

echo -e "\n${BLUE}Cleanup Summary:${NC}"
if [ "$KEEP_STRUCTURE" = true ]; then
    echo -e "   • Removed test PDF and markdown files"
    echo -e "   • Preserved directory structure"
else
    echo -e "   • Removed entire test directories"
fi

echo -e "   • Cleaned service files"

if [ "$CLEAN_DB" = true ]; then
    echo -e "   • Cleaned database test entries"
else
    echo -e "   • Preserved database entries (use --clean-db to clean)"
fi

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "   1. Start fresh testing: ${BLUE}./scripts/start_watch_test.sh${NC}"
echo -e "   2. Add your PDFs to: ${BLUE}$WATCH_DIR/<subdirectory>/${NC}"
echo -e "   3. Monitor activity: ${BLUE}./scripts/watch_activity.sh${NC}"

if [ "$KEEP_STRUCTURE" = true ]; then
    echo -e "\n${GREEN}Ready for new tests! 🎯${NC}"
else
    echo -e "\n${GREEN}Environment fully reset! 🔄${NC}"
fi