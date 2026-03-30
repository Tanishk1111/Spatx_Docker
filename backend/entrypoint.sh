#!/bin/bash
set -e

echo "========================================"
echo " SpatX Backend — Docker"
echo "========================================"

# Ensure persistent directories exist
mkdir -p /app/data /app/uploads /app/pratyaksha_sessions

# Initialise DB + admin user (idempotent — skips if already done)
python /app/init_db.py

echo "[OK] Starting uvicorn on 0.0.0.0:9001 ..."
exec uvicorn app_enhanced:app \
    --host 0.0.0.0 \
    --port 9001 \
    --workers "${UVICORN_WORKERS:-1}" \
    --log-level info
