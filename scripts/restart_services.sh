#!/bin/bash
# Restart all PDF to Markdown services

echo "🔄 Restarting all PDF to Markdown services..."
echo ""

services=("pdf-celery-worker" "pdf-celery-beat" "pdf-api-server")

for service in "${services[@]}"; do
    echo "Restarting $service..."
    sudo systemctl restart "$service.service"
    sleep 2

    if systemctl is-active --quiet "$service.service"; then
        echo "✅ $service restarted successfully"
    else
        echo "❌ $service failed to restart"
        sudo systemctl status "$service.service" --no-pager
    fi
    echo ""
done

echo "✅ All services restarted"
echo ""
echo "Run './scripts/check_services.sh' to verify status"