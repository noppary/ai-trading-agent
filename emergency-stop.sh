#!/bin/bash
# Emergency stop — halts the trading bot and all related timers immediately.
set -e

echo "=== EMERGENCY STOP ==="
echo "Stopping ai-trading-agent service..."
sudo systemctl stop ai-trading-agent

echo "Stopping timers..."
sudo systemctl stop price-monitor.timer health-check.timer 2>/dev/null || true

echo "Killing any stray processes..."
pkill -f "src/main.py" 2>/dev/null || true

echo ""
echo "=== Status ==="
systemctl is-active ai-trading-agent && echo "WARNING: bot still running" || echo "Bot: STOPPED"

echo ""
echo "To restart: sudo systemctl start ai-trading-agent"
