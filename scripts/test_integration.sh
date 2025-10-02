#!/usr/bin/env bash
#
# Run integration tests for PDF to Markdown MCP Server
#
# Integration tests use real services (PostgreSQL, GPU, Redis, Ollama).
# Prerequisites:
#   - NVIDIA GPU with CUDA
#   - PostgreSQL + PGVector
#   - Redis server
#   - Ollama with nomic-embed-text model
#
# Usage:
#   ./scripts/test_integration.sh              # Run all integration tests
#   ./scripts/test_integration.sh --skip-env   # Skip environment validation
#   ./scripts/test_integration.sh --verbose    # Verbose output

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

# Parse command line arguments
SKIP_ENV_CHECK=false
VERBOSE=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-env)
            SKIP_ENV_CHECK=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Integration Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Validate environment unless skipped
if [ "$SKIP_ENV_CHECK" = false ]; then
    echo -e "${YELLOW}Validating test environment...${NC}"

    if [ -f "$SCRIPT_DIR/validate_test_env.sh" ]; then
        if bash "$SCRIPT_DIR/validate_test_env.sh" --quiet; then
            echo -e "${GREEN}✅ Environment validation passed${NC}"
        else
            echo -e "${RED}❌ Environment validation failed${NC}"
            echo -e "${YELLOW}Run './scripts/validate_test_env.sh' for detailed diagnostics${NC}"
            echo -e "${YELLOW}Or use --skip-env to skip validation (not recommended)${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Warning: validate_test_env.sh not found, skipping validation${NC}"
    fi

    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running Integration Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Using real services:${NC}"
echo "  • PostgreSQL + PGVector"
echo "  • NVIDIA GPU (CUDA)"
echo "  • Redis"
echo "  • Ollama (nomic-embed-text)"
echo ""

# Build pytest command
PYTEST_CMD="pytest -m integration"

# Add verbose output if requested
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v -s --log-cli-level=INFO"
fi

# Add any extra arguments
PYTEST_CMD="$PYTEST_CMD $EXTRA_ARGS"

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

# Capture start time
START_TIME=$(date +%s)

if eval "$PYTEST_CMD"; then
    # Capture end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Integration Tests Passed${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Duration: ${DURATION} seconds${NC}"
    exit 0
else
    # Capture end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Integration Tests Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}Duration: ${DURATION} seconds${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo "  1. Check service status: ./scripts/validate_test_env.sh"
    echo "  2. Review GPU usage: nvidia-smi"
    echo "  3. Check MinerU logs: tail -100 /tmp/mineru.log"
    echo "  4. Check Celery logs: tail -100 /var/log/celery-worker.log"
    echo "  5. See tests/README.md for detailed troubleshooting"
    exit 1
fi
