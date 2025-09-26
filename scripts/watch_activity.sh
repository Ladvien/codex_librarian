#!/bin/bash
# ==============================================================================
# Watch PDF Test Service Activity (Live Monitoring)
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
LOG_FILE="$PROJECT_ROOT/watch_test.log"
WATCH_DIR="$PROJECT_ROOT/watch_pdf_test"
OUTPUT_DIR="$PROJECT_ROOT/watch_pdf_test_output"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}PDF Watch Test Service - Live Activity Monitor${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}No log file found. Service may not be running.${NC}"
    echo -e "Start the service first: ${BLUE}./scripts/start_watch_test.sh${NC}"
    exit 1
fi

echo -e "${YELLOW}Monitoring activity... (Press Ctrl+C to stop)${NC}"
echo -e "${BLUE}Log File: $LOG_FILE${NC}"
echo -e "${BLUE}Watch Directory: $WATCH_DIR${NC}"
echo -e "${BLUE}Output Directory: $OUTPUT_DIR${NC}"
echo -e ""

# Function to colorize log output
colorize_logs() {
    while IFS= read -r line; do
        case "$line" in
            *"ERROR"* | *"CRITICAL"*)
                echo -e "${RED}$line${NC}"
                ;;
            *"WARNING"*)
                echo -e "${YELLOW}$line${NC}"
                ;;
            *"SUCCESS"* | *"completed"* | *"✓"*)
                echo -e "${GREEN}$line${NC}"
                ;;
            *"INFO"*)
                echo -e "${BLUE}$line${NC}"
                ;;
            *)
                echo "$line"
                ;;
        esac
    done
}

# Show last few lines first
echo -e "${YELLOW}Recent activity:${NC}"
tail -n 10 "$LOG_FILE" | colorize_logs
echo -e ""
echo -e "${YELLOW}Live activity:${NC}"

# Follow the log file
tail -f "$LOG_FILE" | colorize_logs