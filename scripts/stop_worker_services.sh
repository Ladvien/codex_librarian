#!/bin/bash
# Stop worker services for PDF to Markdown MCP Server
# This script stops Redis broker and Celery workers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Stopping PDF to Markdown MCP Server Worker Services${NC}"
echo "====================================================="

# Function to check if process is running
process_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to stop process gracefully
stop_process() {
    local pid_file="$1"
    local process_name="$2"
    local timeout="${3:-10}"

    if process_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        echo -e "${YELLOW}Stopping $process_name (PID: $pid)...${NC}"

        # Send TERM signal
        kill -TERM "$pid" 2>/dev/null || true

        # Wait for graceful shutdown
        for i in $(seq 1 $timeout); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}$process_name stopped gracefully${NC}"
                rm -f "$pid_file"
                return 0
            fi
            sleep 1
        done

        # Force kill if still running
        echo -e "${YELLOW}Force killing $process_name...${NC}"
        kill -KILL "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        echo -e "${GREEN}$process_name stopped forcefully${NC}"
    else
        echo -e "${YELLOW}$process_name is not running${NC}"
        rm -f "$pid_file"
    fi
}

# Function to stop Celery worker
stop_celery_worker() {
    echo -e "${YELLOW}Stopping Celery worker...${NC}"

    # Stop using celery control if possible
    if command -v celery >/dev/null 2>&1; then
        cd "$PROJECT_DIR"
        celery -A src.pdf_to_markdown_mcp.worker.celery control shutdown 2>/dev/null || true
        sleep 2
    fi

    # Stop using PID file
    stop_process "/tmp/celery_worker.pid" "Celery Worker" 15

    # Clean up log file if desired
    if [ -f "/tmp/celery_worker.log" ]; then
        echo -e "${YELLOW}Celery worker log available at: /tmp/celery_worker.log${NC}"
    fi
}

# Function to stop Celery beat
stop_celery_beat() {
    echo -e "${YELLOW}Stopping Celery beat scheduler...${NC}"
    stop_process "/tmp/celery_beat.pid" "Celery Beat" 10

    # Clean up beat schedule file
    if [ -f "/tmp/celerybeat-schedule" ]; then
        echo -e "${YELLOW}Removing beat schedule file...${NC}"
        rm -f /tmp/celerybeat-schedule
    fi

    # Clean up log file if desired
    if [ -f "/tmp/celery_beat.log" ]; then
        echo -e "${YELLOW}Celery beat log available at: /tmp/celery_beat.log${NC}"
    fi
}

# Function to stop Flower
stop_flower() {
    echo -e "${YELLOW}Stopping Flower monitoring...${NC}"
    stop_process "/tmp/flower.pid" "Flower" 5
}

# Function to stop Redis (if started by our script)
stop_redis() {
    echo -e "${YELLOW}Checking Redis...${NC}"

    # Try to stop Docker Redis first
    if [ -f "$PROJECT_DIR/docker-compose.redis.yml" ]; then
        if command -v docker-compose >/dev/null 2>&1; then
            cd "$PROJECT_DIR"
            if docker-compose -f docker-compose.redis.yml ps | grep -q "redis"; then
                echo -e "${YELLOW}Stopping Docker Redis...${NC}"
                docker-compose -f docker-compose.redis.yml down
                echo -e "${GREEN}Docker Redis stopped${NC}"
                return 0
            fi
        fi
    fi

    # Check for local Redis processes that might be ours
    if pgrep -f "redis-server.*redis.conf" >/dev/null; then
        echo -e "${YELLOW}Found local Redis process using our config...${NC}"
        pkill -f "redis-server.*redis.conf" || true
        echo -e "${GREEN}Local Redis process stopped${NC}"
    else
        echo -e "${YELLOW}No Redis processes started by our script found${NC}"
        echo -e "${YELLOW}Note: If Redis is running independently, it will continue running${NC}"
    fi
}

# Function to clean up temporary files
cleanup_temp_files() {
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"

    # Remove PID files
    rm -f /tmp/celery_worker.pid
    rm -f /tmp/celery_beat.pid
    rm -f /tmp/flower.pid
    rm -f /tmp/celeryworker.pid
    rm -f /tmp/celerybeat.pid

    # Optional: Clean up log files (commented out to preserve logs)
    # rm -f /tmp/celery_worker.log
    # rm -f /tmp/celery_beat.log

    echo -e "${GREEN}Temporary files cleaned${NC}"
}

# Function to show final status
show_final_status() {
    echo
    echo -e "${GREEN}Final Status:${NC}"
    echo "============="

    # Check if any processes are still running
    local still_running=false

    # Check Celery processes
    if pgrep -f "celery.*worker" >/dev/null; then
        echo -e "Celery Workers: ${RED}Still Running${NC}"
        still_running=true
    else
        echo -e "Celery Workers: ${GREEN}Stopped${NC}"
    fi

    if pgrep -f "celery.*beat" >/dev/null; then
        echo -e "Celery Beat: ${RED}Still Running${NC}"
        still_running=true
    else
        echo -e "Celery Beat: ${GREEN}Stopped${NC}"
    fi

    if pgrep -f "flower" >/dev/null; then
        echo -e "Flower: ${RED}Still Running${NC}"
        still_running=true
    else
        echo -e "Flower: ${GREEN}Stopped${NC}"
    fi

    # Check Redis
    if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
        echo -e "Redis: ${YELLOW}Running${NC} (may be independent instance)"
    else
        echo -e "Redis: ${GREEN}Stopped${NC}"
    fi

    if $still_running; then
        echo
        echo -e "${YELLOW}Some processes may still be running. Use 'ps aux | grep celery' to check.${NC}"
        echo -e "${YELLOW}To force kill all Celery processes: pkill -f celery${NC}"
    else
        echo
        echo -e "${GREEN}All worker services stopped successfully!${NC}"
    fi

    # Show log locations
    echo
    echo -e "${YELLOW}Log files (preserved):${NC}"
    if [ -f "/tmp/celery_worker.log" ]; then
        echo "- Celery Worker: /tmp/celery_worker.log"
    fi
    if [ -f "/tmp/celery_beat.log" ]; then
        echo "- Celery Beat: /tmp/celery_beat.log"
    fi
}

# Main execution
main() {
    case "${1:-stop}" in
        "stop")
            stop_flower
            stop_celery_beat
            stop_celery_worker

            read -p "Stop Redis as well? (y/N): " stop_redis_choice
            if [[ $stop_redis_choice =~ ^[Yy]$ ]]; then
                stop_redis
            fi

            cleanup_temp_files
            show_final_status
            ;;
        "force")
            echo -e "${RED}Force stopping all services...${NC}"

            # Force kill all related processes
            pkill -f "celery.*worker" 2>/dev/null || true
            pkill -f "celery.*beat" 2>/dev/null || true
            pkill -f "flower" 2>/dev/null || true

            # Stop Redis if started by us
            stop_redis

            cleanup_temp_files
            echo -e "${GREEN}Force stop completed${NC}"
            show_final_status
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 [stop|force|help]"
            echo
            echo "Commands:"
            echo "  stop    - Gracefully stop all worker services (default)"
            echo "  force   - Force kill all worker services"
            echo "  help    - Show this help message"
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"