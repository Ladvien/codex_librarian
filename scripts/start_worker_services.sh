#!/bin/bash
# Start worker services for PDF to Markdown MCP Server
# This script starts Redis broker and Celery workers with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REDIS_COMPOSE_FILE="$PROJECT_DIR/docker-compose.redis.yml"
VENV_PATH="$PROJECT_DIR/.venv"

echo -e "${GREEN}Starting PDF to Markdown MCP Server Worker Services${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Redis is running
check_redis() {
    if command_exists redis-cli; then
        redis-cli ping >/dev/null 2>&1
    else
        return 1
    fi
}

# Function to start Redis with Docker
start_redis_docker() {
    echo -e "${YELLOW}Starting Redis with Docker...${NC}"

    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    if ! command_exists docker-compose; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi

    if [ ! -f "$REDIS_COMPOSE_FILE" ]; then
        echo -e "${RED}Error: Redis compose file not found at $REDIS_COMPOSE_FILE${NC}"
        exit 1
    fi

    cd "$PROJECT_DIR"
    docker-compose -f docker-compose.redis.yml up -d

    echo -e "${GREEN}Redis started successfully${NC}"

    # Wait for Redis to be ready
    echo "Waiting for Redis to be ready..."
    for i in {1..30}; do
        if check_redis; then
            echo -e "${GREEN}Redis is ready!${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo -e "${RED}Timeout waiting for Redis to start${NC}"
            exit 1
        fi
    done
}

# Function to start Redis locally
start_redis_local() {
    echo -e "${YELLOW}Starting Redis locally...${NC}"

    if ! command_exists redis-server; then
        echo -e "${RED}Error: Redis is not installed locally${NC}"
        echo "Please install Redis or use Docker option"
        exit 1
    fi

    # Check if Redis is already running
    if check_redis; then
        echo -e "${GREEN}Redis is already running${NC}"
        return 0
    fi

    # Start Redis with our configuration
    redis-server "$PROJECT_DIR/redis.conf" &
    REDIS_PID=$!

    # Wait for Redis to start
    echo "Waiting for Redis to start..."
    for i in {1..15}; do
        if check_redis; then
            echo -e "${GREEN}Redis started successfully (PID: $REDIS_PID)${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 15 ]; then
            echo -e "${RED}Timeout waiting for Redis to start${NC}"
            kill $REDIS_PID 2>/dev/null || true
            exit 1
        fi
    done
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source "$VENV_PATH/bin/activate"
        echo -e "${GREEN}Virtual environment activated${NC}"
    else
        echo -e "${RED}Warning: Virtual environment not found at $VENV_PATH${NC}"
        echo "Make sure dependencies are installed globally or create a virtual environment"
    fi
}

# Function to start Celery worker
start_celery_worker() {
    echo -e "${YELLOW}Starting Celery worker...${NC}"

    cd "$PROJECT_DIR"

    # Set environment variables if .env exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    fi

    # Start Celery worker with appropriate queues
    celery -A src.pdf_to_markdown_mcp.worker.celery worker \
        --loglevel=info \
        --concurrency=4 \
        --queues=pdf_processing,embeddings,maintenance,monitoring \
        --hostname=worker@%h \
        --pidfile=/tmp/celeryworker.pid \
        --logfile=/tmp/celery_worker.log &

    WORKER_PID=$!
    echo -e "${GREEN}Celery worker started (PID: $WORKER_PID)${NC}"

    # Save PID for cleanup
    echo $WORKER_PID > /tmp/celery_worker.pid
}

# Function to start Celery beat scheduler
start_celery_beat() {
    echo -e "${YELLOW}Starting Celery beat scheduler...${NC}"

    cd "$PROJECT_DIR"

    # Set environment variables if .env exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    fi

    # Start Celery beat
    celery -A src.pdf_to_markdown_mcp.worker.celery beat \
        --loglevel=info \
        --pidfile=/tmp/celerybeat.pid \
        --logfile=/tmp/celery_beat.log &

    BEAT_PID=$!
    echo -e "${GREEN}Celery beat scheduler started (PID: $BEAT_PID)${NC}"

    # Save PID for cleanup
    echo $BEAT_PID > /tmp/celery_beat.pid
}

# Function to start Flower monitoring
start_flower() {
    echo -e "${YELLOW}Starting Flower monitoring (optional)...${NC}"

    if ! command_exists flower; then
        echo -e "${YELLOW}Flower not installed, skipping monitoring dashboard${NC}"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Set environment variables if .env exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    fi

    flower -A src.pdf_to_markdown_mcp.worker.celery \
        --port=5555 \
        --basic_auth=admin:admin \
        --persistent=True &

    FLOWER_PID=$!
    echo -e "${GREEN}Flower monitoring started at http://localhost:5555 (PID: $FLOWER_PID)${NC}"
    echo "Login: admin / Password: admin"

    # Save PID for cleanup
    echo $FLOWER_PID > /tmp/flower.pid
}

# Function to show status
show_status() {
    echo
    echo -e "${GREEN}Service Status:${NC}"
    echo "==============="

    # Check Redis
    if check_redis; then
        echo -e "Redis: ${GREEN}Running${NC}"
    else
        echo -e "Redis: ${RED}Not Running${NC}"
    fi

    # Check Celery worker
    if [ -f /tmp/celery_worker.pid ] && kill -0 $(cat /tmp/celery_worker.pid) 2>/dev/null; then
        echo -e "Celery Worker: ${GREEN}Running${NC} (PID: $(cat /tmp/celery_worker.pid))"
    else
        echo -e "Celery Worker: ${RED}Not Running${NC}"
    fi

    # Check Celery beat
    if [ -f /tmp/celery_beat.pid ] && kill -0 $(cat /tmp/celery_beat.pid) 2>/dev/null; then
        echo -e "Celery Beat: ${GREEN}Running${NC} (PID: $(cat /tmp/celery_beat.pid))"
    else
        echo -e "Celery Beat: ${RED}Not Running${NC}"
    fi

    # Check Flower
    if [ -f /tmp/flower.pid ] && kill -0 $(cat /tmp/flower.pid) 2>/dev/null; then
        echo -e "Flower: ${GREEN}Running${NC} (PID: $(cat /tmp/flower.pid)) - http://localhost:5555"
    else
        echo -e "Flower: ${RED}Not Running${NC}"
    fi

    echo
    echo -e "${GREEN}Logs:${NC}"
    echo "======"
    echo "Celery Worker: /tmp/celery_worker.log"
    echo "Celery Beat: /tmp/celery_beat.log"
    echo
    echo -e "${YELLOW}To stop services, run: ./scripts/stop_worker_services.sh${NC}"
}

# Main execution
main() {
    case "${1:-start}" in
        "start")
            echo -e "${YELLOW}Choose Redis startup method:${NC}"
            echo "1. Docker (recommended)"
            echo "2. Local installation"
            echo "3. Skip Redis (already running)"
            read -p "Enter choice (1-3): " redis_choice

            case $redis_choice in
                1)
                    start_redis_docker
                    ;;
                2)
                    start_redis_local
                    ;;
                3)
                    if ! check_redis; then
                        echo -e "${RED}Error: Redis is not running${NC}"
                        exit 1
                    fi
                    echo -e "${GREEN}Using existing Redis instance${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice${NC}"
                    exit 1
                    ;;
            esac

            activate_venv
            start_celery_worker
            start_celery_beat

            read -p "Start Flower monitoring dashboard? (y/N): " start_flower_choice
            if [[ $start_flower_choice =~ ^[Yy]$ ]]; then
                start_flower
            fi

            show_status
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 [start|status|help]"
            echo
            echo "Commands:"
            echo "  start   - Start all worker services (default)"
            echo "  status  - Show service status"
            echo "  help    - Show this help message"
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap for cleanup
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    ./scripts/stop_worker_services.sh 2>/dev/null || true
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Run main function
main "$@"