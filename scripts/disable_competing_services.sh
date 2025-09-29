#!/usr/bin/env bash
set -e

echo "🛑 Disabling competing services that interfere with GPU processing..."

# Kill all watch_and_mirror processes
echo "Killing watch_and_mirror processes..."
pkill -f "watch_and_mirror.py" 2>/dev/null || true

# Stop and disable the services that auto-restart watch_and_mirror
for service in codex-watcher pdf-file-watcher; do
    if systemctl list-units --all | grep -q "$service.service"; then
        echo "Stopping $service..."
        systemctl --user stop "$service.service" 2>/dev/null || true
        echo "Disabling $service..."
        systemctl --user disable "$service.service" 2>/dev/null || true
    fi
done

# Also try system-level services in case they're not user services
for service in codex-watcher pdf-file-watcher; do
    echo "Attempting to stop system-level $service..."
    sudo systemctl stop "$service.service" 2>/dev/null || true
    sudo systemctl disable "$service.service" 2>/dev/null || true
done

# Final check
remaining=$(ps aux | grep -c "watch_and_mirror.py" | grep -v grep || echo "0")
if [ "$remaining" -eq "0" ]; then
    echo "✅ All competing services disabled successfully"
else
    echo "⚠️  Some processes might still be running. Forcing kill..."
    ps aux | grep "watch_and_mirror" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    echo "✅ Force killed remaining processes"
fi

echo "✅ Competing services disabled. GPU processing should now work correctly."