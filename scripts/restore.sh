#!/usr/bin/env bash
# ── restore.sh ──────────────────────────────────────────────────────────
# Restore persistent data from a backup archive.
# Usage:  ./scripts/restore.sh [path-to-backup.tar.gz]
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-backup.tar.gz>"
    echo ""
    echo "Available backups:"
    ls -lh ./backups/ 2>/dev/null || echo "  (no backups found)"
    exit 1
fi

ARCHIVE="$1"

if [ ! -f "${ARCHIVE}" ]; then
    echo "Error: file not found — ${ARCHIVE}"
    exit 1
fi

echo "=== Restoring from: ${ARCHIVE} ==="

# Stop services before restoring data
echo "Stopping services..."
docker compose down 2>/dev/null || true

# Extract the archive, overwriting existing files
tar -xzf "${ARCHIVE}"

echo "=== Restore complete ==="
echo "Run ./scripts/start.sh to bring services back up."
