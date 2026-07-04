#!/usr/bin/env bash
# ── deploy.sh ───────────────────────────────────────────────────────────
# Full deployment: pull latest code, rebuild, restart services.
# Usage:  ./scripts/deploy.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== Pulling latest code ==="
git pull

echo "=== Rebuilding images ==="
docker compose build --pull

echo "=== Restarting services ==="
docker compose down --remove-orphans
docker compose up -d

echo "=== Deployment complete ==="
docker compose ps
