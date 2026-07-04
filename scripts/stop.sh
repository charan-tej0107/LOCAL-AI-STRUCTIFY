#!/usr/bin/env bash
# ── stop.sh ─────────────────────────────────────────────────────────────
# Gracefully stop all services.
# Usage:  ./scripts/stop.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== Stopping services ==="
docker compose down

echo "=== Services stopped ==="
