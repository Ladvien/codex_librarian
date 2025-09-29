#!/bin/bash
# Check status of all PDF to Markdown services

echo "======================================================================"
echo "PDF to Markdown MCP Server - Service Status"
echo "======================================================================"
echo ""

# Get local IP
LOCAL_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d'/' -f1)

echo "🌐 Network Information"
echo "----------------------"
echo "Local IP: $LOCAL_IP"
echo "API URL: http://$LOCAL_IP:8000"
echo "API Docs: http://$LOCAL_IP:8000/docs"
echo ""

echo "📊 Service Status"
echo "----------------------"
services=("pdf-api-server" "pdf-celery-worker" "pdf-celery-beat")

for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service.service"; then
        echo "✅ $service: RUNNING"
    else
        echo "❌ $service: STOPPED"
    fi
done
echo ""

echo "🏥 Health Check"
echo "----------------------"
if curl -s --max-time 5 http://localhost:8000/health > /tmp/health.json 2>/dev/null; then
    echo "API Status: $(jq -r '.status' /tmp/health.json 2>/dev/null || echo 'unknown')"
    echo "Database: $(jq -r '.services.database' /tmp/health.json 2>/dev/null || echo 'unknown')"
    echo "Redis: $(jq -r '.services.redis' /tmp/health.json 2>/dev/null || echo 'unknown')"
    echo "Celery: $(jq -r '.services.celery' /tmp/health.json 2>/dev/null || echo 'unknown')"
else
    echo "❌ API not responding"
fi
echo ""

echo "⚙️  Current Configuration"
echo "----------------------"
if curl -s --max-time 5 http://localhost:8000/api/v1/configuration > /tmp/config.json 2>/dev/null; then
    echo "Watch Directories: $(jq -r '.watch_directories' /tmp/config.json 2>/dev/null || echo 'unknown')"
    echo "Output Directory: $(jq -r '.output_directory' /tmp/config.json 2>/dev/null || echo 'unknown')"
else
    echo "❌ Cannot retrieve configuration"
fi
echo ""

echo "🖥️  GPU Status"
echo "----------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Devices:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | while IFS=',' read -r name memory_used memory_total util temp; do
        echo "  📊 $name"
        echo "    Memory: $memory_used / $memory_total"
        echo "    Utilization: $util%"
        echo "    Temperature: $temp°C"
    done
else
    echo "❌ nvidia-smi not available"
fi
echo ""

echo "📈 Model Loading Status"
echo "----------------------"
if curl -s --max-time 5 http://localhost:8000/metrics/json > /tmp/metrics.json 2>/dev/null; then
    echo "Model Singleton: $(jq -r '.model_status.singleton_initialized // "unknown"' /tmp/metrics.json 2>/dev/null)"
    echo "GPU Fallbacks: $(jq -r '.gpu.fallback_count // "0"' /tmp/metrics.json 2>/dev/null)"
    echo "Processing Queue: $(jq -r '.database.processing_queue_depth // "0"' /tmp/metrics.json 2>/dev/null)"
else
    echo "❌ Metrics not available"
fi
echo ""

echo "======================================================================"