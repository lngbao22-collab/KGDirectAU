#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Error: no python interpreter found" >&2
  exit 1
fi

CONFIG="configs/SimKGC-AU_WN18RR_chunked.json"
LOG_DIR="logs/nohup"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/SimKGC-AU_WN18RR_chunked_${STAMP}.out"
PID_FILE="${LOG_DIR}/SimKGC-AU_WN18RR_chunked_${STAMP}.pid"

mkdir -p "$LOG_DIR"

echo "Python:  ${PYTHON}"
echo "Config:  ${CONFIG}"
echo "Log:     ${LOG_FILE}"
echo "PID file: ${PID_FILE}"

nohup "${PYTHON}" main.py --config-path "${CONFIG}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started (PID $(cat "${PID_FILE}"))"
echo "Tail: tail -f ${LOG_FILE}"
