#!/bin/sh
# ── Backend entrypoint — resets demo data on each container start ──
#
# Removes the persisted SQLite database so the FastAPI startup handler
# auto-seeds fresh 4-robot data.  Named volumes no longer prevent reseeding.
set -e

DB_PATH="${DATA_DIR:-/app/data}/demo.db"

if [ -f "$DB_PATH" ]; then
    echo "entrypoint: removing stale $DB_PATH"
    rm -f "$DB_PATH"
fi

echo "entrypoint: starting uvicorn..."
exec uvicorn app.robot_agent.main:app --host 0.0.0.0 --port 8000
