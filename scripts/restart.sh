#!/usr/bin/env bash
# ── restart.sh ──────────────────────────────────────────────────────────
# Restart all services (stop + start).
# Usage:  ./scripts/restart.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Restarting services ==="
docker compose restart

echo "=== Services restarted ==="
docker compose ps
