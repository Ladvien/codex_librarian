#!/usr/bin/env bash
#
# Rollback Script - Revert to Single MinerU Instance
#
# This script quickly reverts all performance optimizations and returns
# to the stable single-instance configuration.
#

set -e

echo "=========================================="
echo "  MinerU Multi-Instance Rollback Script  "
echo "=========================================="
echo ""
echo "This will revert to single-instance mode:"
echo "  - Stop all 3 MinerU instances"
echo "  - Start single MinerU instance (ID 0)"
echo "  - Disable batch writes"
echo "  - Disable async file I/O"
echo "  - Reset Celery to single worker"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled."
    exit 0
fi

echo ""
echo "Step 1: Stopping all services..."
echo "=================================="

# Stop the pool service (which stops all instances)
if systemctl is-active --quiet mineru-gpu-pool.service; then
    echo "Stopping MinerU GPU pool..."
    sudo systemctl stop mineru-gpu-pool.service
fi

# Stop individual instances if still running
for i in 0 1 2; do
    if systemctl is-active --quiet mineru-gpu-service-$i.service; then
        echo "Stopping MinerU instance $i..."
        sudo systemctl stop mineru-gpu-service-$i.service
    fi
done

# Stop Celery worker
if systemctl is-active --quiet pdf-celery-worker.service; then
    echo "Stopping Celery worker..."
    sudo systemctl stop pdf-celery-worker.service
fi

# Stop Celery beat
if systemctl is-active --quiet pdf-celery-beat.service; then
    echo "Stopping Celery beat..."
    sudo systemctl stop pdf-celery-beat.service
fi

echo "✓ All services stopped"
echo ""

echo "Step 2: Updating configuration..."
echo "==================================="

# Backup current .env
if [ -f .env ]; then
    echo "Backing up current .env to .env.backup.$(date +%Y%m%d_%H%M%S)"
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# Update .env to disable optimizations
if [ -f .env ]; then
    echo "Updating .env settings..."

    # Disable multi-instance mode
    sed -i 's/^MINERU_INSTANCE_COUNT=.*/MINERU_INSTANCE_COUNT=1/' .env

    # Disable batch writes
    sed -i 's/^ENABLE_BATCH_WRITES=.*/ENABLE_BATCH_WRITES=false/' .env

    # Disable async file I/O
    sed -i 's/^ENABLE_ASYNC_FILE_IO=.*/ENABLE_ASYNC_FILE_IO=false/' .env

    # Reset Celery worker concurrency
    sed -i 's/^CELERY_WORKER_CONCURRENCY=.*/CELERY_WORKER_CONCURRENCY=1/' .env

    # Reset Celery worker pool
    sed -i 's/^CELERY_WORKER_POOL=.*/CELERY_WORKER_POOL=solo/' .env

    echo "✓ Configuration updated"
else
    echo "⚠ Warning: .env file not found!"
fi

echo ""

echo "Step 3: Clearing Redis queues..."
echo "=================================="

# Clear all MinerU queues
redis-cli FLUSHDB || echo "⚠ Warning: Could not clear Redis (is it running?)"

echo "✓ Redis queues cleared"
echo ""

echo "Step 4: Starting single-instance services..."
echo "=============================================="

# Start single MinerU instance (ID 0)
echo "Starting MinerU instance 0..."
sudo systemctl start mineru-gpu-service-0.service

# Wait for MinerU to initialize
echo "Waiting for MinerU to initialize (5 seconds)..."
sleep 5

# Check if MinerU started successfully
if systemctl is-active --quiet mineru-gpu-service-0.service; then
    echo "✓ MinerU instance 0 started successfully"
else
    echo "✗ Failed to start MinerU instance 0"
    echo "Check logs: sudo journalctl -u mineru-gpu-service-0 -n 50"
    exit 1
fi

# Start Celery worker
echo "Starting Celery worker..."
sudo systemctl start pdf-celery-worker.service
sleep 3

if systemctl is-active --quiet pdf-celery-worker.service; then
    echo "✓ Celery worker started successfully"
else
    echo "✗ Failed to start Celery worker"
    echo "Check logs: sudo journalctl -u pdf-celery-worker -n 50"
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
    echo "Check logs: sudo journalctl -u pdf-celery-beat -n 50"
    exit 1
fi

echo ""
echo "Step 5: Verifying rollback..."
echo "=============================="

echo ""
echo "Service Status:"
echo "---------------"
systemctl is-active mineru-gpu-service-0.service && echo "✓ MinerU instance 0: ACTIVE" || echo "✗ MinerU instance 0: INACTIVE"
systemctl is-active mineru-gpu-service-1.service && echo "  MinerU instance 1: ACTIVE (should be inactive)" || echo "✓ MinerU instance 1: INACTIVE"
systemctl is-active mineru-gpu-service-2.service && echo "  MinerU instance 2: ACTIVE (should be inactive)" || echo "✓ MinerU instance 2: INACTIVE"
systemctl is-active pdf-celery-worker.service && echo "✓ Celery worker: ACTIVE" || echo "✗ Celery worker: INACTIVE"
systemctl is-active pdf-celery-beat.service && echo "✓ Celery beat: ACTIVE" || echo "✗ Celery beat: INACTIVE"

echo ""
echo "Configuration:"
echo "--------------"
grep "^MINERU_INSTANCE_COUNT=" .env || echo "MINERU_INSTANCE_COUNT not set"
grep "^ENABLE_BATCH_WRITES=" .env || echo "ENABLE_BATCH_WRITES not set"
grep "^ENABLE_ASYNC_FILE_IO=" .env || echo "ENABLE_ASYNC_FILE_IO not set"

echo ""
echo "=========================================="
echo "  Rollback Complete!                     "
echo "=========================================="
echo ""
echo "System is now running in single-instance mode."
echo ""
echo "To monitor:"
echo "  - View logs: ./scripts/view_logs.sh all"
echo "  - Check health: ./scripts/check_services.sh"
echo "  - GPU status: nvidia-smi"
echo ""
echo "To re-enable multi-instance mode:"
echo "  1. Update .env: MINERU_INSTANCE_COUNT=3"
echo "  2. Enable optimizations: ENABLE_BATCH_WRITES=true, ENABLE_ASYNC_FILE_IO=true"
echo "  3. Restart: sudo systemctl start mineru-gpu-pool"
echo ""
