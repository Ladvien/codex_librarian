#!/bin/bash
# View logs for PDF to Markdown services

SERVICE="${1:-all}"

case "$SERVICE" in
    api|fastapi|server)
        echo "=== API Server Logs (last 50 lines) ==="
        sudo tail -n 50 /var/log/pdf-api-server.log
        ;;
    worker|celery)
        echo "=== Celery Worker Logs (last 50 lines) ==="
        sudo tail -n 50 /var/log/celery-worker.log
        ;;
    beat|scheduler)
        echo "=== Celery Beat Logs (last 50 lines) ==="
        sudo tail -n 50 /var/log/celery-beat.log
        ;;
    all)
        echo "=== API Server Logs (last 20 lines) ==="
        sudo tail -n 20 /var/log/pdf-api-server.log
        echo ""
        echo "=== Celery Worker Logs (last 20 lines) ==="
        sudo tail -n 20 /var/log/celery-worker.log
        echo ""
        echo "=== Celery Beat Logs (last 20 lines) ==="
        sudo tail -n 20 /var/log/celery-beat.log
        ;;
    follow)
        echo "Following all logs (Ctrl+C to stop)..."
        sudo tail -f /var/log/pdf-api-server.log /var/log/celery-worker.log /var/log/celery-beat.log
        ;;
    *)
        echo "Usage: $0 [all|api|worker|beat|follow]"
        echo ""
        echo "Examples:"
        echo "  $0           # Show last 20 lines from all logs"
        echo "  $0 api       # Show last 50 lines from API server"
        echo "  $0 worker    # Show last 50 lines from Celery worker"
        echo "  $0 beat      # Show last 50 lines from Celery beat"
        echo "  $0 follow    # Follow all logs in real-time"
        exit 1
        ;;
esac