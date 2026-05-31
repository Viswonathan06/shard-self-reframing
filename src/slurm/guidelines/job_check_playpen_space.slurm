#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --job-name=check_playpen_space
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=gpu1
# Check /playpen space and how much you're using.

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || exit 1
mkdir -p logs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo ""

if [ -d "/playpen" ]; then
    echo "========== /playpen filesystem (df) =========="
    df -h /playpen 2>/dev/null || true
    echo ""

    echo "========== Space used in /playpen (total) =========="
    du -sh /playpen 2>/dev/null || true
    echo ""

    echo "========== Space used per top-level dir in /playpen =========="
    for d in /playpen/*/; do
        [ -d "$d" ] && du -sh "$d" 2>/dev/null || true
    done
    echo ""

    echo "========== Usage by vmanoran =========="
    if [ -d "/playpen/vmanoran" ]; then
        echo "/playpen/vmanoran:"
        du -sh /playpen/vmanoran 2>/dev/null || true
        ls -la /playpen/vmanoran 2>/dev/null || true
    else
        echo "/playpen/vmanoran not found."
    fi
    echo "Direct children of /playpen owned by vmanoran:"
    find /playpen -maxdepth 1 -user vmanoran 2>/dev/null | while read p; do
        [ -e "$p" ] && du -sh "$p" 2>/dev/null
    done
    echo "Total (vmanoran's trees under /playpen, no double-count):"
    find /playpen -maxdepth 1 -user vmanoran 2>/dev/null -exec du -s {} \; | awk '{s+=$1} END {printf "%.1f GB\n", s/1024/1024}'
    echo ""

    if [ -d "/playpen/$(whoami)" ] && [ "$(whoami)" != "vmanoran" ]; then
        echo "========== Your dir: /playpen/$(whoami) =========="
        du -sh "/playpen/$(whoami)" 2>/dev/null || true
        ls -la "/playpen/$(whoami)" 2>/dev/null || true
        echo ""
    fi

    echo "========== Contents of omnom =========="
    if [ -d "/playpen/omnom" ]; then
        du -sh /playpen/omnom 2>/dev/null || true
        echo ""
        echo "/playpen/omnom:"
        ls -la /playpen/omnom 2>/dev/null || true
        echo ""
        echo "First-level subdirs (size):"
        for d in /playpen/omnom/*/; do
            [ -d "$d" ] && du -sh "$d" 2>/dev/null || true
        done
    else
        echo "/playpen/omnom not found."
    fi
    echo ""

    if [ -d "/playpen/huggingface" ]; then
        echo "========== /playpen/huggingface (HF cache) =========="
        du -sh /playpen/huggingface 2>/dev/null || true
        ls -la /playpen/huggingface 2>/dev/null || true
        PLAYPEN_HUB="$HF_HOME"
        if [ -d "$PLAYPEN_HUB" ]; then
            echo ""
            echo "========== Models under Hugging Face ($PLAYPEN_HUB) =========="
            for d in "$PLAYPEN_HUB"/models--*; do
                [ -d "$d" ] || continue
                size=$(du -sh "$d" 2>/dev/null | cut -f1)
                snap="$d/snapshots"
                repo=$(basename "$d" | sed 's/^models--//;s/--/\//')
                if [ -d "$snap" ]; then
                    revs=$(ls -1 "$snap" 2>/dev/null | tr '\n' ' ')
                    echo "  $repo  ($size)  revisions: $revs"
                else
                    echo "  $repo  ($size)  (no snapshots)"
                fi
            done
            echo "========================================"
        fi
        echo ""
    fi

    echo "========================================"
else
    echo "/playpen not found on this node."
fi

echo ""
echo "Done."
exit 0
