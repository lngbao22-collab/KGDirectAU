#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/DaBR_WN18RR.json"
LOG_DIR="logs/nohup"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/DaBR_WN18RR_${STAMP}.out"
PID_FILE="${LOG_DIR}/DaBR_WN18RR_${STAMP}.pid"

mkdir -p "$LOG_DIR"

echo "Config:  ${CONFIG}"
echo "Log:     ${LOG_FILE}"
echo "PID file: ${PID_FILE}"

nohup python main.py --config-path "${CONFIG}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started (PID $(cat "${PID_FILE}"))"
echo "Tail: tail -f ${LOG_FILE}"
