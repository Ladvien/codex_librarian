#!/bin/bash
#
# Configuration Reload Trigger Script
#
# This script helps trigger configuration reload for the PDF file watcher service.
# It finds the running watcher process and sends SIGUSR1 signal for immediate reload.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_NAME="run_file_watcher.py"
SIGNAL="USR1"

echo -e "${GREEN}PDF File Watcher Configuration Reload${NC}"
echo "======================================"

# Function to find the watcher process
find_watcher_process() {
    local pids=$(pgrep -f "$SCRIPT_NAME" 2>/dev/null || true)
    echo "$pids"
}

# Function to send reload signal
send_reload_signal() {
    local pid=$1
    echo -e "${YELLOW}Sending SIGUSR1 to process $pid...${NC}"

    if kill -$SIGNAL "$pid" 2>/dev/null; then
        echo -e "${GREEN}✓ Configuration reload signal sent successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to send signal to process $pid${NC}"
        return 1
    fi
}

# Function to show process information
show_process_info() {
    local pid=$1
    echo "Process Information:"
    echo "  PID: $pid"
    echo "  Command: $(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo 'Unknown')"
    echo "  Started: $(ps -p "$pid" -o lstart --no-headers 2>/dev/null || echo 'Unknown')"
    echo ""
}

# Main script logic
main() {
    echo "Looking for running file watcher processes..."

    local pids=$(find_watcher_process)

    if [[ -z "$pids" ]]; then
        echo -e "${RED}✗ No running file watcher processes found${NC}"
        echo ""
        echo "To start the file watcher:"
        echo "  python scripts/run_file_watcher.py"
        exit 1
    fi

    # Convert to array to handle multiple processes
    local pid_array=($pids)
    local pid_count=${#pid_array[@]}

    if [[ $pid_count -eq 1 ]]; then
        local pid=${pid_array[0]}
        echo -e "${GREEN}✓ Found file watcher process${NC}"
        show_process_info "$pid"

        if send_reload_signal "$pid"; then
            echo ""
            echo -e "${GREEN}Configuration reload triggered successfully!${NC}"
            echo ""
            echo "Check the watcher logs for reload confirmation:"
            echo "  tail -f /var/log/pdf-mcp-watcher.log"
            echo "  # Or wherever your logs are configured"
        else
            exit 1
        fi

    elif [[ $pid_count -gt 1 ]]; then
        echo -e "${YELLOW}Found multiple file watcher processes:${NC}"
        for i in "${!pid_array[@]}"; do
            echo "  [$((i+1))] PID: ${pid_array[i]}"
        done
        echo ""

        read -p "Select process number to reload (1-$pid_count): " selection

        if [[ "$selection" =~ ^[0-9]+$ ]] && [[ $selection -ge 1 ]] && [[ $selection -le $pid_count ]]; then
            local selected_pid=${pid_array[$((selection-1))]}
            echo ""
            show_process_info "$selected_pid"

            if send_reload_signal "$selected_pid"; then
                echo -e "${GREEN}Configuration reload triggered for PID $selected_pid!${NC}"
            else
                exit 1
            fi
        else
            echo -e "${RED}✗ Invalid selection${NC}"
            exit 1
        fi
    fi
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -p, --pid PID  Send signal to specific process ID"
    echo "  -l, --list     List running watcher processes only"
    echo ""
    echo "Examples:"
    echo "  $0                    # Find and reload watcher automatically"
    echo "  $0 --pid 12345        # Reload specific process"
    echo "  $0 --list             # List running processes"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    -p|--pid)
        if [[ -n "${2:-}" ]]; then
            pid="$2"
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo -e "${GREEN}Sending reload signal to PID $pid${NC}"
                    if send_reload_signal "$pid"; then
                        echo -e "${GREEN}Configuration reload triggered!${NC}"
                    else
                        exit 1
                    fi
                else
                    echo -e "${RED}✗ Process $pid not found${NC}"
                    exit 1
                fi
            else
                echo -e "${RED}✗ Invalid PID: $pid${NC}"
                exit 1
            fi
        else
            echo -e "${RED}✗ PID required with --pid option${NC}"
            exit 1
        fi
        ;;
    -l|--list)
        echo "Running file watcher processes:"
        local pids=$(find_watcher_process)
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                echo "  PID: $pid - $(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo 'Unknown command')"
            done
        else
            echo -e "${YELLOW}  No running file watcher processes found${NC}"
        fi
        ;;
    "")
        main
        ;;
    *)
        echo -e "${RED}✗ Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
esac