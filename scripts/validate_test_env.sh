#!/usr/bin/env bash
#
# Validate test environment for integration and e2e tests
#
# Checks all prerequisites:
#   - PostgreSQL + PGVector
#   - GPU/CUDA
#   - Redis
#   - Ollama + model
#   - MinerU service
#   - Celery workers
#
# Usage:
#   ./scripts/validate_test_env.sh         # Full validation with colors
#   ./scripts/validate_test_env.sh --quiet # Minimal output for scripts

set +e  # Don't exit on error (we want to check all services)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUIET=false
if [[ "$1" == "--quiet" ]]; then
    QUIET=true
fi

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
OPTIONAL_FAILED=0

# Load .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Helper function to check service
check_service() {
    local name="$1"
    local command="$2"
    local is_optional="${3:-false}"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    if eval "$command" > /dev/null 2>&1; then
        if [ "$QUIET" = false ]; then
            echo -e "${GREEN}✅ $name${NC}"
        fi
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "$QUIET" = false ]; then
            if [ "$is_optional" = "true" ]; then
                echo -e "${YELLOW}⚠️  $name (optional)${NC}"
                OPTIONAL_FAILED=$((OPTIONAL_FAILED + 1))
            else
                echo -e "${RED}❌ $name${NC}"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
            fi
        fi
        return 1
    fi
}

if [ "$QUIET" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Environment Validation${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
fi

# 1. Check PostgreSQL
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking PostgreSQL...${NC}"
fi

DB_HOST="${DB_HOST:-192.168.1.104}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-codex_librarian}"
DB_USER="${DB_USER:-codex_librarian}"

check_service "PostgreSQL connection" \
    "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c 'SELECT 1' 2>&1 | grep -q '1 row'"

# Check PGVector extension
if check_service "PGVector extension" \
    "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \"SELECT * FROM pg_extension WHERE extname='vector'\" 2>&1 | grep -q 'vector'"; then
    if [ "$QUIET" = false ]; then
        # Get version info
        PGVECTOR_VERSION=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT extversion FROM pg_extension WHERE extname='vector'" 2>/dev/null | tr -d ' ')
        echo "  Version: $PGVECTOR_VERSION"
    fi
fi

# Check required tables
check_service "Required tables exist" \
    "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \"SELECT tablename FROM pg_tables WHERE tablename IN ('documents', 'document_embeddings')\" 2>&1 | grep -q 'documents'"

echo ""

# 2. Check GPU/CUDA
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking GPU/CUDA...${NC}"
fi

check_service "nvidia-smi available" "nvidia-smi"

if nvidia-smi > /dev/null 2>&1; then
    # Get GPU info
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    if [ "$QUIET" = false ]; then
        echo "  GPU: $GPU_NAME"
        echo "  Memory: $GPU_MEMORY"
    fi

    # Check CUDA via Python
    if check_service "CUDA available in Python" \
        "python -c 'import torch; assert torch.cuda.is_available()'"; then
        if [ "$QUIET" = false ]; then
            CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)
            echo "  CUDA Version: $CUDA_VERSION"
        fi
    fi

    # Check environment variables
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        if [ "$QUIET" = false ]; then
            echo -e "${YELLOW}  ⚠️  CUDA_VISIBLE_DEVICES not set${NC}"
        fi
    else
        if [ "$QUIET" = false ]; then
            echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        fi
    fi

    DEVICE_MODE="${MINERU_DEVICE_MODE:-cpu}"
    if [ "$DEVICE_MODE" != "cuda" ]; then
        if [ "$QUIET" = false ]; then
            echo -e "${YELLOW}  ⚠️  MINERU_DEVICE_MODE is '$DEVICE_MODE', should be 'cuda'${NC}"
        fi
    else
        if [ "$QUIET" = false ]; then
            echo "  MINERU_DEVICE_MODE: $DEVICE_MODE"
        fi
    fi
fi

echo ""

# 3. Check Redis
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking Redis...${NC}"
fi

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

check_service "Redis server" "redis-cli -h $REDIS_HOST -p $REDIS_PORT ping | grep -q 'PONG'"

if redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; then
    if [ "$QUIET" = false ]; then
        REDIS_VERSION=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT INFO | grep redis_version | cut -d: -f2 | tr -d '\r')
        echo "  Version: $REDIS_VERSION"

        # Check queue lengths
        MINERU_REQ=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT LLEN mineru_requests 2>/dev/null || echo "0")
        MINERU_RES=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT LLEN mineru_results 2>/dev/null || echo "0")
        echo "  mineru_requests queue: $MINERU_REQ"
        echo "  mineru_results queue: $MINERU_RES"
    fi
fi

echo ""

# 4. Check Ollama
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking Ollama...${NC}"
fi

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-nomic-embed-text}"

check_service "Ollama service" "curl -s $OLLAMA_URL/api/tags | grep -q 'models'"

if curl -s $OLLAMA_URL/api/tags > /dev/null 2>&1; then
    if [ "$QUIET" = false ]; then
        OLLAMA_VERSION=$(curl -s $OLLAMA_URL/api/version 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        echo "  Version: $OLLAMA_VERSION"
    fi

    # Check for nomic-embed-text model
    if curl -s $OLLAMA_URL/api/tags | grep -q "$OLLAMA_MODEL"; then
        check_service "Ollama model ($OLLAMA_MODEL)" "true"
    else
        check_service "Ollama model ($OLLAMA_MODEL)" "false"
        if [ "$QUIET" = false ]; then
            echo -e "  ${YELLOW}Run: ollama pull $OLLAMA_MODEL${NC}"
        fi
    fi
fi

echo ""

# 5. Check MinerU Service
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking MinerU Service...${NC}"
fi

# Check if MinerU process is running
check_service "MinerU process running" "pgrep -f 'mineru_standalone.py'" true

if pgrep -f "mineru_standalone.py" > /dev/null 2>&1; then
    if [ "$QUIET" = false ]; then
        # Check GPU usage
        if nvidia-smi | grep -q "mineru\|python"; then
            echo -e "  ${GREEN}✅ Using GPU${NC}"
            GPU_MEM=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
            echo "  GPU Memory: $GPU_MEM"
        else
            echo -e "  ${YELLOW}⚠️  Not using GPU (may be idle)${NC}"
        fi

        # Check log file
        if [ -f "/tmp/mineru.log" ]; then
            LAST_LOG=$(tail -1 /tmp/mineru.log 2>/dev/null || echo "")
            if [ -n "$LAST_LOG" ]; then
                echo "  Last log: ${LAST_LOG:0:80}..."
            fi
        fi
    fi
else
    if [ "$QUIET" = false ]; then
        echo -e "  ${YELLOW}Start with: python src/pdf_to_markdown_mcp/services/mineru_standalone.py${NC}"
    fi
fi

echo ""

# 6. Check Celery Workers (optional)
if [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Checking Celery Workers (optional)...${NC}"
fi

check_service "Celery worker" "pgrep -f 'celery worker'" true

if pgrep -f "celery worker" > /dev/null 2>&1; then
    if [ "$QUIET" = false ]; then
        WORKER_COUNT=$(pgrep -f "celery worker" | wc -l)
        echo "  Workers running: $WORKER_COUNT"
    fi
else
    if [ "$QUIET" = false ]; then
        echo "  Start with: sudo systemctl start pdf-celery-worker"
    fi
fi

# Check Celery beat (optional)
check_service "Celery beat" "pgrep -f 'celery beat'" true

echo ""

# Summary
if [ "$QUIET" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"

    if [ $FAILED_CHECKS -gt 0 ]; then
        echo -e "${RED}Failed: $FAILED_CHECKS (required)${NC}"
    fi

    if [ $OPTIONAL_FAILED -gt 0 ]; then
        echo -e "${YELLOW}Failed: $OPTIONAL_FAILED (optional)${NC}"
    fi

    echo ""

    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "${GREEN}✅ All required prerequisites met!${NC}"
        echo -e "${GREEN}You can run integration and e2e tests.${NC}"
        exit 0
    else
        echo -e "${RED}❌ Some required prerequisites are missing.${NC}"
        echo -e "${YELLOW}Integration/e2e tests may fail or be skipped.${NC}"
        echo ""
        echo -e "${YELLOW}Setup instructions:${NC}"
        echo "  • PostgreSQL: See CLAUDE.md for database setup"
        echo "  • GPU/CUDA: Install NVIDIA drivers and CUDA toolkit"
        echo "  • Redis: docker run -d -p 6379:6379 redis"
        echo "  • Ollama: See docs for installation and model pull"
        echo "  • MinerU: Start with scripts/start_mineru.sh"
        echo ""
        echo "See tests/README.md for detailed troubleshooting."
        exit 1
    fi
else
    # Quiet mode - just return exit code
    if [ $FAILED_CHECKS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
fi
