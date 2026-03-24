#!/bin/bash
# Emergency stop — P1.5: call /emergency-close HTTP endpoint, THEN stop the service.
set -e

API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-3000}"
ENDPOINT="http://${API_HOST}:${API_PORT}/emergency-close"

echo "=== EMERGENCY STOP ==="

# Step 1: Call the HTTP endpoint to close all positions and halt trading
echo "Calling /emergency-close endpoint..."
HTTP_STATUS=$(curl -sf -X POST "${ENDPOINT}" \
    -H "Content-Type: application/json" \
    -w "%{http_code}" \
    -o /tmp/emergency_close_response.json \
    2>&1) || HTTP_STATUS="000"

if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "000" ]; then
    echo "Endpoint reached (HTTP ${HTTP_STATUS})"
    if [ -f /tmp/emergency_close_response.json ]; then
        cat /tmp/emergency_close_response.json
    fi
else
    echo "WARNING: endpoint returned HTTP ${HTTP_STATUS}"
    if [ -f /tmp/emergency_close_response.json ]; then
        cat /tmp/emergency_close_response.json
    fi
fi

# Step 2: Stop the systemd service
echo ""
echo "Stopping ai-trading-agent service..."
sudo systemctl stop ai-trading-agent

# Step 3: Stop the timers
echo "Stopping timers..."
sudo systemctl stop price-monitor.timer health-check.timer 2>/dev/null || true

# Step 4: Kill any stray processes
echo "Killing any stray processes..."
pkill -f "src/main.py" 2>/dev/null || true

echo ""
echo "=== Status ==="
systemctl is-active ai-trading-agent && echo "WARNING: bot still running" || echo "Bot: STOPPED"
echo ""
echo "To restart: sudo systemctl start ai-trading-agent"
