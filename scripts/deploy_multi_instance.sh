#!/usr/bin/env bash
#
# Deployment Script - Enable Multi-Instance MinerU
#
# This script deploys the multi-instance configuration for 2-3x throughput improvement.
#

set -e

echo "================================================"
echo "  MinerU Multi-Instance Deployment Script      "
echo "================================================"
echo ""
echo "This will deploy multi-instance configuration:"
echo "  - 3 parallel MinerU GPU instances"
echo "  - Batch database writes"
echo "  - Async file I/O"
echo "  - 3 concurrent Celery workers"
echo ""
echo "Expected improvement: 2-3x throughput"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Step 1: Pre-deployment checks..."
echo "=================================="

# Check if running as ladvien user
if [ "$(whoami)" != "ladvien" ]; then
    echo "✗ Error: This script must be run as user 'ladvien'"
    exit 1
fi

# Check Redis is running
if ! redis-cli ping &>/dev/null; then
    echo "✗ Error: Redis is not running"
    echo "  Start with: sudo systemctl start redis"
    exit 1
fi
echo "✓ Redis is running"

# Check PostgreSQL is accessible
export PGPASSWORD='PdfMcp2025Secure'
if ! psql -h 192.168.1.104 -U codex_librarian -d codex_librarian -c "SELECT 1" &>/dev/null; then
    echo "✗ Error: Cannot connect to PostgreSQL"
    echo "  Check connection settings in .env"
    exit 1
fi
echo "✓ PostgreSQL is accessible"

# Check GPU is available
if ! nvidia-smi &>/dev/null; then
    echo "✗ Error: GPU not detected"
    echo "  Ensure NVIDIA drivers are installed"
    exit 1
fi
echo "✓ GPU detected"

# Check Python virtual environment
if [ ! -d ".venv" ]; then
    echo "✗ Error: Python virtual environment not found"
    echo "  Create with: python -m venv .venv"
    exit 1
fi
echo "✓ Python virtual environment found"

echo ""
echo "Step 2: Stopping existing services..."
echo "======================================="

# Stop existing MinerU instances
for i in 0 1 2; do
    if systemctl is-active --quiet mineru-gpu-service-$i.service; then
        echo "Stopping MinerU instance $i..."
        sudo systemctl stop mineru-gpu-service-$i.service
    fi
done

# Stop Celery services
if systemctl is-active --quiet pdf-celery-worker.service; then
    echo "Stopping Celery worker..."
    sudo systemctl stop pdf-celery-worker.service
fi

if systemctl is-active --quiet pdf-celery-beat.service; then
    echo "Stopping Celery beat..."
    sudo systemctl stop pdf-celery-beat.service
fi

echo "✓ All services stopped"
echo ""

echo "Step 3: Installing systemd service files..."
echo "============================================"

# Install MinerU instance service files
for i in 0 1 2; do
    if [ -f "systemd/mineru-gpu-service-$i.service" ]; then
        echo "Installing mineru-gpu-service-$i.service..."
        sudo cp systemd/mineru-gpu-service-$i.service /etc/systemd/system/
    else
        echo "✗ Error: systemd/mineru-gpu-service-$i.service not found"
        exit 1
    fi
done

# Install pool service
if [ -f "systemd/mineru-gpu-pool.service" ]; then
    echo "Installing mineru-gpu-pool.service..."
    sudo cp systemd/mineru-gpu-pool.service /etc/systemd/system/
else
    echo "✗ Error: systemd/mineru-gpu-pool.service not found"
    exit 1
fi

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "✓ Service files installed"
echo ""

echo "Step 4: Creating log files..."
echo "=============================="

# Create log files for each instance
for i in 0 1 2; do
    sudo touch /var/log/mineru-gpu-service-$i.log
    sudo chown ladvien:ladvien /var/log/mineru-gpu-service-$i.log
    sudo chmod 644 /var/log/mineru-gpu-service-$i.log
    echo "✓ Created /var/log/mineru-gpu-service-$i.log"
done

echo ""

echo "Step 5: Verifying configuration..."
echo "===================================="

# Check if .env has correct settings
if ! grep -q "MINERU_INSTANCE_COUNT=3" .env; then
    echo "⚠ Warning: MINERU_INSTANCE_COUNT not set to 3 in .env"
    echo "  This should be set to 3 for multi-instance mode"
fi

if ! grep -q "ENABLE_BATCH_WRITES=true" .env; then
    echo "⚠ Warning: ENABLE_BATCH_WRITES not enabled in .env"
fi

if ! grep -q "ENABLE_ASYNC_FILE_IO=true" .env; then
    echo "⚠ Warning: ENABLE_ASYNC_FILE_IO not enabled in .env"
fi

echo "✓ Configuration checked"
echo ""

echo "Step 6: Clearing Redis queues..."
echo "=================================="

# Clear all queues for fresh start
redis-cli FLUSHDB
echo "✓ Redis queues cleared"
echo ""

echo "Step 7: Starting MinerU instances..."
echo "======================================"

# Start instances one by one with delay
for i in 0 1 2; do
    echo "Starting MinerU instance $i..."
    sudo systemctl start mineru-gpu-service-$i.service
    sleep 3

    if systemctl is-active --quiet mineru-gpu-service-$i.service; then
        echo "✓ Instance $i started successfully"
    else
        echo "✗ Failed to start instance $i"
        echo "  Check logs: sudo journalctl -u mineru-gpu-service-$i -n 50"
        exit 1
    fi
done

echo ""
echo "Waiting for GPU models to load (10 seconds)..."
sleep 10

echo ""

echo "Step 8: Starting Celery services..."
echo "===================================="

# Start Celery worker
echo "Starting Celery worker..."
sudo systemctl start pdf-celery-worker.service
sleep 3

if systemctl is-active --quiet pdf-celery-worker.service; then
    echo "✓ Celery worker started successfully"
else
    echo "✗ Failed to start Celery worker"
    echo "  Check logs: sudo journalctl -u pdf-celery-worker -n 50"
    exit 1
fi

# Start Celery beat
echo "Starting Celery beat..."
sudo systemctl start pdf-celery-beat.service
sleep 2

if systemctl is-active --quiet pdf-celery-beat.service; then
    echo "✓ Celery beat started successfully"
else
    echo "✗ Failed to start Celery beat"
    echo "  Check logs: sudo journalctl -u pdf-celery-beat -n 50"
    exit 1
fi

echo ""

echo "Step 9: Verifying deployment..."
echo "================================"

echo ""
echo "Service Status:"
echo "---------------"
for i in 0 1 2; do
    systemctl is-active mineru-gpu-service-$i.service && echo "✓ MinerU instance $i: ACTIVE" || echo "✗ MinerU instance $i: INACTIVE"
done
systemctl is-active pdf-celery-worker.service && echo "✓ Celery worker: ACTIVE" || echo "✗ Celery worker: INACTIVE"
systemctl is-active pdf-celery-beat.service && echo "✓ Celery beat: ACTIVE" || echo "✗ Celery beat: INACTIVE"

echo ""
echo "GPU Status:"
echo "-----------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s: %s MB / %s MB (%.1f%%)\n", $1, $2, $3, ($2/$3)*100}'

echo ""
echo "Redis Queue Status:"
echo "-------------------"
for i in 0 1 2; do
    queue_len=$(redis-cli llen mineru_requests_$i 2>/dev/null || echo "0")
    echo "  mineru_requests_$i: $queue_len items"
done

echo ""
echo "================================================"
echo "  Deployment Complete!                          "
echo "================================================"
echo ""
echo "Multi-instance mode is now active."
echo ""
echo "Next steps:"
echo "  1. Monitor GPU: watch -n 1 nvidia-smi"
echo "  2. View logs: ./scripts/view_logs.sh all"
echo "  3. Check health: ./scripts/check_services.sh"
echo "  4. Test with PDFs in: /mnt/codex_fs/research/"
echo ""
echo "Expected performance:"
echo "  - 2-3x throughput improvement"
echo "  - GPU utilization >80%"
echo "  - Reduced idle time between PDFs"
echo ""
echo "To rollback:"
echo "  ./scripts/rollback_to_single_instance.sh"
echo ""
