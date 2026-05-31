#!/usr/bin/env bash
# Format logs: move .err and .out into err/ and out/, then archive all folders.
# Usage: ./scripts/shell/format_and_archive_logs.sh [LOGS_DIR]
# Default LOGS_DIR: repo root / logs

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="${1:-${REPO_ROOT}/logs}"
cd "$LOGS_DIR"

# 1. Create err/ and out/ folders
mkdir -p err out

# 2. Move top-level .err and .out files into respective folders (skip err/ and out/ and archive/)
for f in *.err; do
  [ -f "$f" ] && mv -f "$f" err/
done
for f in *.out; do
  [ -f "$f" ] && mv -f "$f" out/
done

# 3. Create timestamped archive folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_SUBDIR="archive/${TIMESTAMP}"
mkdir -p "$ARCHIVE_SUBDIR"

# 4. Move err/ and out/ into the archive
[ -d "err" ] && mv err "$ARCHIVE_SUBDIR/" 2>/dev/null || true
[ -d "out" ] && mv out "$ARCHIVE_SUBDIR/" 2>/dev/null || true

# 5. Move all other non-archive directories (RUN_TAG dirs, etc.) into the archive
for d in */; do
  [ -z "$d" ] && continue
  dirname="${d%/}"
  if [ "$dirname" != "archive" ]; then
    mv -f "$d" "$ARCHIVE_SUBDIR/"
  fi
done

echo "✓ Logs formatted and archived to ${LOGS_DIR}/${ARCHIVE_SUBDIR}/"
echo "  - err/ and out/ (moved .err and .out files)"
echo "  - RUN_TAG and other job dirs"
ls -la "$ARCHIVE_SUBDIR" 2>/dev/null || true
