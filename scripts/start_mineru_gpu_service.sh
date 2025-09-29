#!/usr/bin/env bash
set -e

echo "🚀 Starting MinerU GPU Service..."

# Copy systemd service file if needed
if [ ! -f /etc/systemd/system/mineru-gpu-service.service ]; then
    echo "Installing systemd service..."
    sudo cp systemd/mineru-gpu-service.service /etc/systemd/system/
    sudo systemctl daemon-reload
fi

# Stop existing service if running
echo "Stopping any existing MinerU GPU service..."
sudo systemctl stop mineru-gpu-service 2>/dev/null || true

# Enable and start the service
echo "Starting MinerU GPU service..."
sudo systemctl enable mineru-gpu-service
sudo systemctl start mineru-gpu-service

# Check status
sleep 2
if sudo systemctl is-active --quiet mineru-gpu-service; then
    echo "✅ MinerU GPU service started successfully"
    echo ""
    echo "View logs with: tail -f /var/log/mineru-gpu-service.log"
    echo "Check status with: sudo systemctl status mineru-gpu-service"
else
    echo "❌ Failed to start MinerU GPU service"
    echo "Check logs: sudo journalctl -u mineru-gpu-service -n 50"
    exit 1
fi