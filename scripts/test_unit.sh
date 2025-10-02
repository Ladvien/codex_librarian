#!/usr/bin/env bash
#
# Run unit tests for PDF to Markdown MCP Server
#
# Unit tests are fast, use mocked services, and require no external dependencies.
# They use SQLite database and CPU-only processing.
#
# Usage:
#   ./scripts/test_unit.sh              # Run all unit tests
#   ./scripts/test_unit.sh --cov        # Run with coverage report
#   ./scripts/test_unit.sh --parallel   # Run tests in parallel

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}Error: .venv not found. Run: uv venv && uv sync${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running Unit Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Parse command line arguments
COVERAGE=false
PARALLEL=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cov|--coverage)
            COVERAGE=true
            shift
            ;;
        --parallel|-n)
            PARALLEL=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest -m unit"

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Running with coverage report...${NC}"
    PYTEST_CMD="$PYTEST_CMD --cov=src/pdf_to_markdown_mcp --cov-report=term --cov-report=html"
fi

# Add parallel execution if requested
if [ "$PARALLEL" = true ]; then
    echo -e "${YELLOW}Running tests in parallel...${NC}"
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add any extra arguments
PYTEST_CMD="$PYTEST_CMD $EXTRA_ARGS"

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

if eval "$PYTEST_CMD"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Unit Tests Passed${NC}"
    echo -e "${GREEN}========================================${NC}"

    if [ "$COVERAGE" = true ]; then
        echo -e "${YELLOW}Coverage report saved to: htmlcov/index.html${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Unit Tests Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
