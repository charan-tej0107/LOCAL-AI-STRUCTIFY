#!/usr/bin/env bash
# ── backup.sh ───────────────────────────────────────────────────────────
# Backup persistent data (database, uploads, cache) to a timestamped
# archive in the backups/ directory.
# Usage:  ./scripts/backup.sh
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE="${BACKUP_DIR}/structify_backup_${TIMESTAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}"

echo "=== Creating backup: ${ARCHIVE} ==="

# Archive the data directories that are mounted as volumes
tar -czf "${ARCHIVE}" \
    --exclude='backups' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    ./data/ \
    ./uploads/ \
    ./cache/ \
    ./models/ \
    ./nginx/ \
    ./docker-compose.yml \
    ./.env 2>/dev/null || true

echo "=== Backup complete: $(du -sh "${ARCHIVE}" | cut -f1) ==="
echo "File: ${ARCHIVE}"

# Keep only the last 14 backups
find "${BACKUP_DIR}" -name "structify_backup_*.tar.gz" -mtime +14 -delete 2>/dev/null || true
