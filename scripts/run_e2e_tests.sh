#!/bin/bash
# ==============================================================================
# End-to-End Test Runner Script
# ==============================================================================
# This script sets up the test database, runs E2E tests, and provides cleanup.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SETUP_DB=true
CLEANUP_DB=false
VERBOSE=false
TEST_PATTERN="test_directory_mirroring_e2e.py"
PYTEST_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-setup)
            SETUP_DB=false
            shift
            ;;
        --cleanup)
            CLEANUP_DB=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            PYTEST_ARGS="$PYTEST_ARGS -v -s"
            shift
            ;;
        --pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-setup     Skip database setup (use existing test DB)"
            echo "  --cleanup      Clean up test database after tests"
            echo "  --verbose, -v  Run tests with verbose output"
            echo "  --pattern      Test file pattern (default: test_directory_mirroring_e2e.py)"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Setup DB, run tests, keep DB"
            echo "  $0 --cleanup          # Setup DB, run tests, cleanup DB"
            echo "  $0 --no-setup         # Skip setup, run tests with existing DB"
            echo "  $0 --verbose --cleanup # Verbose output with cleanup"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}Running End-to-End Tests for Directory Mirroring${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Step 1: Set up test database (if requested)
if [ "$SETUP_DB" = true ]; then
    echo -e "\n${YELLOW}Step 1: Setting up test database...${NC}"
    if ! ./scripts/setup_test_database.sh; then
        echo -e "${RED}✗ Database setup failed!${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}Step 1: Skipping database setup (using existing)${NC}"
    if [ ! -f ".env.test" ]; then
        echo -e "${RED}✗ No .env.test file found! Run with database setup first.${NC}"
        exit 1
    fi
fi

# Step 2: Load test environment
echo -e "\n${YELLOW}Step 2: Loading test environment...${NC}"
if [ -f ".env.test" ]; then
    export $(grep -v '^#' .env.test | xargs)
    echo -e "${GREEN}✓ Loaded test environment variables${NC}"
else
    echo -e "${RED}✗ .env.test file not found!${NC}"
    exit 1
fi

# Step 3: Run database migrations
echo -e "\n${YELLOW}Step 3: Running database migrations...${NC}"
if command -v alembic &> /dev/null; then
    if alembic upgrade head; then
        echo -e "${GREEN}✓ Database migrations completed${NC}"
    else
        echo -e "${YELLOW}⚠ Migration failed - may be expected for test DB${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Alembic not found, skipping migrations${NC}"
fi

# Step 4: Run the tests
echo -e "\n${YELLOW}Step 4: Running E2E tests...${NC}"
echo -e "${BLUE}Test command: pytest tests/integration/$TEST_PATTERN $PYTEST_ARGS${NC}"

# Set pytest environment variables
export PYTHONPATH="src:$PYTHONPATH"
export TEST_DATABASE_URL="$TEST_DATABASE_URL"

# Run the tests
TEST_EXIT_CODE=0
if [ "$VERBOSE" = true ]; then
    pytest tests/integration/$TEST_PATTERN $PYTEST_ARGS --tb=short || TEST_EXIT_CODE=$?
else
    pytest tests/integration/$TEST_PATTERN $PYTEST_ARGS || TEST_EXIT_CODE=$?
fi

# Step 5: Display results
echo -e "\n${YELLOW}Step 5: Test Results${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All E2E tests passed successfully!${NC}"
else
    echo -e "${RED}❌ Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

# Step 6: Cleanup (if requested)
if [ "$CLEANUP_DB" = true ]; then
    echo -e "\n${YELLOW}Step 6: Cleaning up test database...${NC}"
    if ./scripts/cleanup_test_database.sh; then
        echo -e "${GREEN}✓ Test database cleaned up${NC}"
    else
        echo -e "${YELLOW}⚠ Cleanup failed, but tests completed${NC}"
    fi
else
    echo -e "\n${YELLOW}Step 6: Keeping test database for future runs${NC}"
    echo -e "${BLUE}  To cleanup later: ./scripts/cleanup_test_database.sh${NC}"
    echo -e "${BLUE}  To run tests again: $0 --no-setup${NC}"
fi

# Final status
echo -e "\n${BLUE}==============================================================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 E2E Testing completed successfully!${NC}"
    if [ -f ".env.test" ] && [ "$CLEANUP_DB" = false ]; then
        echo -e "\n${BLUE}Database Info:${NC}"
        echo -e "  Connection: $(grep TEST_DATABASE_URL .env.test | cut -d'=' -f2)"
        echo -e "  Use this database for additional testing or debugging"
    fi
else
    echo -e "${RED}💥 E2E Testing failed!${NC}"
    echo -e "\n${BLUE}Debugging Tips:${NC}"
    echo -e "  1. Check test database connection: $(grep TEST_DATABASE_URL .env.test | cut -d'=' -f2)"
    echo -e "  2. Run with verbose output: $0 --verbose"
    echo -e "  3. Check PostgreSQL logs on server 'four'"
    echo -e "  4. Verify SSH connection to 'four' is working"
fi

echo -e "${BLUE}==============================================================================${NC}"

exit $TEST_EXIT_CODE