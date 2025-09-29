#!/bin/bash

# Quick Health Check Script for PDF to Markdown MCP System
# This script provides a rapid check of all critical components

echo "======================================================================"
echo "🏥 QUICK SYSTEM HEALTH CHECK"
echo "======================================================================"
echo "Timestamp: $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize counters
FAILURES=0
WARNINGS=0

# Function to check service
check_service() {
    local name=$1
    local check_command=$2

    if eval $check_command > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name: Running${NC}"
    else
        echo -e "${RED}❌ $name: Not running${NC}"
        ((FAILURES++))
    fi
}

# Function to check port
check_port() {
    local service=$1
    local port=$2

    if ss -tln | grep -q ":$port "; then
        echo -e "${GREEN}✅ $service (port $port): Open${NC}"
    else
        echo -e "${RED}❌ $service (port $port): Closed${NC}"
        ((FAILURES++))
    fi
}

echo "1️⃣ CHECKING CORE SERVICES"
echo "----------------------------"
check_port "PostgreSQL" 5432
check_port "Redis" 6379
check_port "Ollama" 11434
check_port "FastAPI" 8000
check_service "Celery Worker" "systemctl is-active pdf-celery-worker"
check_service "Celery Beat" "systemctl is-active pdf-celery-beat"

echo ""
echo "2️⃣ CHECKING GPU & MINERU"
echo "----------------------------"
# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✅ GPU: Available (${GPU_MEM} MB in use)${NC}"

    # Check for MinerU process using GPU
    if nvidia-smi | grep -q "python.*mineru\|mineru_standalone"; then
        echo -e "${GREEN}✅ MinerU: Using GPU${NC}"
    else
        if ps aux | grep -q "[m]ineru_standalone.py"; then
            echo -e "${YELLOW}⚠️ MinerU: Running but not using GPU${NC}"
            ((WARNINGS++))
        else
            echo -e "${RED}❌ MinerU: Not running${NC}"
            ((FAILURES++))
        fi
    fi
else
    echo -e "${RED}❌ GPU: Not available${NC}"
    ((FAILURES++))
fi

echo ""
echo "3️⃣ CHECKING DATABASE"
echo "----------------------------"
export PGPASSWORD='$DB_PASSWORD'
if psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "SELECT 1" > /dev/null 2>&1; then
    # Get stats
    STATS=$(psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -t -c "
        SELECT
            (SELECT COUNT(*) FROM documents) as total_docs,
            (SELECT COUNT(*) FROM documents WHERE conversion_status = 'pending') as pending,
            (SELECT COUNT(*) FROM documents WHERE conversion_status = 'completed') as completed,
            (SELECT COUNT(*) FROM document_embeddings) as embeddings
    ")

    # Parse stats
    read -r TOTAL PENDING COMPLETED EMBEDDINGS <<< $(echo $STATS | tr -d '|')

    echo -e "${GREEN}✅ Database: Connected${NC}"
    echo "   📊 Documents: $TOTAL total, $COMPLETED completed, $PENDING pending"
    echo "   🧮 Embeddings: $EMBEDDINGS"

    if [ "$PENDING" -gt 100 ]; then
        echo -e "${YELLOW}   ⚠️ High number of pending documents${NC}"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}❌ Database: Connection failed${NC}"
    ((FAILURES++))
fi

echo ""
echo "4️⃣ CHECKING OUTPUT"
echo "----------------------------"
OUTPUT_DIR="/mnt/codex_fs/research/librarian_output"
if [ -d "$OUTPUT_DIR" ]; then
    # Count recent markdown files (last hour)
    RECENT_FILES=$(find "$OUTPUT_DIR" -name "*.md" -mmin -60 2>/dev/null | wc -l)
    TOTAL_FILES=$(find "$OUTPUT_DIR" -name "*.md" 2>/dev/null | wc -l)

    echo -e "${GREEN}✅ Output Directory: Exists${NC}"
    echo "   📁 Files: $TOTAL_FILES total, $RECENT_FILES in last hour"

    if [ "$RECENT_FILES" -eq 0 ] && [ "$PENDING" -gt 0 ]; then
        echo -e "${YELLOW}   ⚠️ No recent output despite pending documents${NC}"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}❌ Output Directory: Not found${NC}"
    ((FAILURES++))
fi

echo ""
echo "5️⃣ CHECKING REDIS QUEUES"
echo "----------------------------"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis: Connected${NC}"

    # Check queue lengths
    CELERY_Q=$(redis-cli llen celery 2>/dev/null || echo 0)
    PDF_Q=$(redis-cli llen pdf_processing 2>/dev/null || echo 0)
    EMB_Q=$(redis-cli llen embeddings 2>/dev/null || echo 0)
    MINERU_Q=$(redis-cli llen mineru_requests 2>/dev/null || echo 0)

    if [ "$CELERY_Q" -gt 0 ] || [ "$PDF_Q" -gt 0 ] || [ "$EMB_Q" -gt 0 ] || [ "$MINERU_Q" -gt 0 ]; then
        echo "   📨 Queue lengths:"
        [ "$CELERY_Q" -gt 0 ] && echo "      - celery: $CELERY_Q"
        [ "$PDF_Q" -gt 0 ] && echo "      - pdf_processing: $PDF_Q"
        [ "$EMB_Q" -gt 0 ] && echo "      - embeddings: $EMB_Q"
        [ "$MINERU_Q" -gt 0 ] && echo "      - mineru_requests: $MINERU_Q"
    else
        echo "   📨 All queues empty"
    fi
else
    echo -e "${RED}❌ Redis: Not connected${NC}"
    ((FAILURES++))
fi

echo ""
echo "6️⃣ CHECKING RECENT ACTIVITY"
echo "----------------------------"
# Check MinerU log for recent activity
if [ -f /tmp/mineru.log ]; then
    RECENT_JOBS=$(tail -100 /tmp/mineru.log | grep "Completed job.*success: True" | wc -l)
    LAST_ACTIVITY=$(tail -1 /tmp/mineru.log | cut -d' ' -f1-2)

    if [ "$RECENT_JOBS" -gt 0 ]; then
        echo -e "${GREEN}✅ MinerU Activity: $RECENT_JOBS successful jobs in log${NC}"
        echo "   Last activity: $LAST_ACTIVITY"
    else
        echo -e "${YELLOW}⚠️ MinerU: No recent successful jobs${NC}"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠️ MinerU log not found${NC}"
    ((WARNINGS++))
fi

# Check Celery worker log
if [ -f /var/log/celery-worker.log ]; then
    RECENT_ERRORS=$(tail -100 /var/log/celery-worker.log | grep -i "error\|failed" | wc -l)
    if [ "$RECENT_ERRORS" -gt 10 ]; then
        echo -e "${YELLOW}⚠️ Celery: $RECENT_ERRORS errors in recent log${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✅ Celery: Few errors in recent log${NC}"
    fi
fi

echo ""
echo "======================================================================"
echo "📊 SUMMARY"
echo "======================================================================"

if [ $FAILURES -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✅ SYSTEM STATUS: HEALTHY${NC}"
        echo "All systems operational!"
    else
        echo -e "${YELLOW}⚠️ SYSTEM STATUS: WARNING${NC}"
        echo "$WARNINGS warning(s) detected"
    fi
else
    echo -e "${RED}❌ SYSTEM STATUS: FAILED${NC}"
    echo "$FAILURES failure(s) detected"
    echo "$WARNINGS warning(s) detected"
fi

echo ""
echo "For detailed diagnostics, run:"
echo "  python /mnt/datadrive_m2/codex_librarian/scripts/system_diagnostic.py"
echo ""

# Exit with appropriate code
if [ $FAILURES -gt 0 ]; then
    exit 1
else
    exit 0
fi