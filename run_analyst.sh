#!/usr/bin/env bash
set -euo pipefail

cd /root/ai-trading-agent
source .env
python3 data/trade_logger.py
python3 src/analyst.py
