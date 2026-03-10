#!/usr/bin/env bash
set -euo pipefail

cd /root/ai-trading-agent
set -a
source .env
set +a
python3 data/trade_logger.py
python3 src/analyst.py
