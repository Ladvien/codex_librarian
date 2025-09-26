#!/bin/bash
# ==============================================================================
# Stop PDF Watch Test Service
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

echo -e "${BLUE}Stopping PDF Watch Test Service...${NC}"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}No PID file found. Service may not be running.${NC}"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}Service not running (PID $PID not found)${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

# Stop the service
echo -e "${YELLOW}Stopping service (PID: $PID)...${NC}"
kill "$PID"

# Wait for graceful shutdown
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service stopped gracefully${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo -e "${YELLOW}Force stopping service...${NC}"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo -e "${GREEN}✓ Service stopped${NC}"