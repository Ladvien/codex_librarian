#!/bin/bash
# ==============================================================================
# Test Database Cleanup Script
# ==============================================================================
# This script cleans up the test database and removes test data.

set -e

# Configuration
PG_HOST="four"
TEST_DB_NAME="pdf_to_markdown_mcp_test"
TEST_USER="test_user"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}Cleaning up test database${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Function to run SQL commands via SSH
run_sql() {
    local sql="$1"
    local database="${2:-postgres}"
    ssh "$PG_HOST" "sudo -u postgres psql -d '$database' -c \"$sql\""
}

echo -e "${YELLOW}Step 1: Terminating active connections...${NC}"
run_sql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$TEST_DB_NAME' AND pid <> pg_backend_pid();" || true

echo -e "${YELLOW}Step 2: Dropping test database...${NC}"
run_sql "DROP DATABASE IF EXISTS $TEST_DB_NAME;" || true
echo -e "${GREEN}✓ Dropped test database${NC}"

echo -e "${YELLOW}Step 3: Dropping test user...${NC}"
run_sql "DROP USER IF EXISTS $TEST_USER;" || true
echo -e "${GREEN}✓ Dropped test user${NC}"

echo -e "${YELLOW}Step 4: Removing test environment file...${NC}"
if [ -f ".env.test" ]; then
    rm -f .env.test
    echo -e "${GREEN}✓ Removed .env.test file${NC}"
else
    echo -e "${YELLOW}  No .env.test file found${NC}"
fi

echo -e "\n${GREEN}✓ Test database cleanup completed!${NC}"