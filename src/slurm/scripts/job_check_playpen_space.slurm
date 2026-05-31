#!/usr/bin/bash
# Check /playpen disk usage; delete /playpen/omnom; then recheck space.
# Run on a node where /playpen (or playpen-shared) is available.

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --time=0:15:00
#SBATCH --job-name=check_playpen_space
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || true
mkdir -p logs

run_space_check() {
  local label="$1"
  local PREFIX
  echo ""
  echo "########## $label ##########"
  echo ""
  for PREFIX in /playpen /playpen-shared; do
    if [[ ! -d "$PREFIX" ]]; then
      echo "[SKIP] $PREFIX not found on this node."
      echo ""
      continue
    fi
    echo "========== $PREFIX =========="
    df -h "$PREFIX" 2>/dev/null || true
    echo ""
    echo "Total size of $PREFIX:"
    du -sh "$PREFIX" 2>/dev/null || true
    echo ""
    echo "Top-level folders (sorted by size, largest first):"
    du --max-depth=1 -h "$PREFIX" 2>/dev/null | sort -hr | head -30
    echo ""
    MAX_TOP=$(du --max-depth=1 -s "$PREFIX" 2>/dev/null | sort -rn | head -2 | tail -1)
    if [[ -n "$MAX_TOP" ]]; then
      echo ">>> LARGEST TOP-LEVEL FOLDER: $MAX_TOP"
    fi
    echo ""
    echo "Largest directories under $PREFIX (depth 2, top 25):"
    du -h --max-depth=2 "$PREFIX" 2>/dev/null | sort -hr | head -25
    echo ""
  done
}

echo "=== Playpen space check at $(date) ==="
echo "Node: $(hostname)"

# --- BEFORE: initial space check ---
run_space_check "BEFORE (initial state)"

# --- DELETE /playpen/omnom (and /playpen/omnon if present) ---
for DIR in /playpen/omnom /playpen/omnon; do
  if [[ -d "$DIR" ]]; then
    echo "========== Deleting $DIR =========="
    du -sh "$DIR" 2>/dev/null || true
    rm -rf "$DIR"
    if [[ -d "$DIR" ]]; then
      echo "WARNING: Failed to remove $DIR"
    else
      echo "Removed: $DIR"
    fi
    echo ""
  fi
done

# --- AFTER: recheck space ---
run_space_check "AFTER (after deleting omnom/omnon)"

echo "=== Done at $(date) ==="
