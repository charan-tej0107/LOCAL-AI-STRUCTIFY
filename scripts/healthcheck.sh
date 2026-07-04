#!/usr/bin/env bash
# ── healthcheck.sh ──────────────────────────────────────────────────────
# Check whether all docker-compose services are healthy.
# Returns:
#   0  — all services are running (healthy)
#   1  — one or more services are not running (unhealthy)
# Usage:  ./scripts/healthcheck.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Service Status ==="
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Determine overall health
UNHEALTHY=$(docker compose ps --format "{{.Status}}" | grep -civ "Up" || true)

if [ "${UNHEALTHY}" -gt 0 ]; then
    echo ""
    echo "WARNING: ${UNHEALTHY} service(s) are not running."
    exit 1
fi

echo ""
echo "All services are healthy."
