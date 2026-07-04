#!/usr/bin/env bash
# ── start.sh ────────────────────────────────────────────────────────────
# Start all services in detached mode.
# Usage:  ./scripts/start.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== Starting services ==="
docker compose up -d

echo "=== Services started ==="
docker compose ps
