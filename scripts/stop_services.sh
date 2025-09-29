#!/bin/bash
# Stop all PDF to Markdown services

echo "🛑 Stopping all PDF to Markdown services..."
echo ""

services=("pdf-api-server" "pdf-celery-beat" "pdf-celery-worker")

for service in "${services[@]}"; do
    echo "Stopping $service..."
    sudo systemctl stop "$service.service"

    if ! systemctl is-active --quiet "$service.service"; then
        echo "✅ $service stopped"
    else
        echo "❌ $service still running"
    fi
done

echo ""
echo "✅ All services stopped"