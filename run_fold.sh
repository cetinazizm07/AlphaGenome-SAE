#!/usr/bin/env bash
# Run one model fold end to end.
#
#   1. fetch the fold weights if they are not on disk yet
#   2. extract all six taps for every split
#   3. train one SAE per tap per seed
#   4. copy the activations to Drive, verify them, then free the local copy
#
# The local copy is deleted only after rclone check passes. If the check fails
# the script stops and leaves everything in place.
#
# Usage: run_fold.sh <0|1|2>
set -euo pipefail

FOLD="${1:?usage: run_fold.sh <fold>}"
ROOT=/mnt/ag
CODE="$HOME/ag"
PY="$CODE/.venv/bin/python"
REMOTE="gdrive:alphagenome_sae"

TAPS=(bin_size_4 bin_size_16 bin_size_64 resid_pre_b0 resid_pre_b4 resid_pre_b8)
SEEDS=(0 1 2)

# Frozen design: dictionary width and row budget are equal across taps, so the
# depth comparison is not confounded by dictionary size.
N_FEATURES=8192
POSITIONS=8192
# Measured on a one-tap pilot: FVU was still falling ~3% per 100 steps at 3000.
# The budget is part of the recipe, so it cannot be raised after the fact, and
# all three folds must share it to stay comparable. Chosen once, deliberately.
STEPS=6000

ACTS="$ROOT/acts/fold$FOLD"
SAEDIR="$ROOT/sae/fold$FOLD"
LOGDIR="$ROOT/logs/fold$FOLD"
mkdir -p "$ACTS" "$SAEDIR" "$LOGDIR"

WEIGHTS="$ROOT/weights/model_fold_$FOLD.safetensors"
MANIFEST="$CODE/manifests/window_manifest_fold$FOLD.parquet"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- 0. prerequisites -----------------------------------------------------
[ -f "$MANIFEST" ] || { echo "missing manifest $MANIFEST"; exit 1; }

if [ ! -f "$WEIGHTS" ]; then
  say "fetching weights for fold $FOLD"
  "$PY" - "$FOLD" <<'PY'
import sys
from huggingface_hub import hf_hub_download
f = sys.argv[1]
print(hf_hub_download("gtca/alphagenome_pytorch",
                      f"model_fold_{f}.safetensors", local_dir="/mnt/ag/weights"))
PY
fi

FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
say "free on $ROOT: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 250 ]; then
  echo "need ~230 GB for one fold, only ${FREE_GB} GB free"; exit 1
fi

# --- 1. extract -----------------------------------------------------------
# run_extraction is idempotent: it records each shard in index.json as it goes,
# returns immediately when the run is already complete, and resumes from the
# last finished shard otherwise. A preemption costs at most one shard.
say "extracting fold $FOLD"
{
  "$PY" -m ag_sae.extract \
    --manifest "$MANIFEST" \
    --weights "$WEIGHTS" \
    --fasta "$ROOT/data/GRCh38.primary_assembly.genome.fa" \
    --out "$ACTS" \
    --positions-per-window "$POSITIONS" \
    --windows-per-shard 8 \
    --seed "$FOLD"
} 2>&1 | tee -a "$LOGDIR/extract.log"
du -sh "$ACTS"

# --- 2. train -------------------------------------------------------------
for tap in "${TAPS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$SAEDIR/${tap}_seed$seed"
    if [ -f "$out/training.json" ] && grep -q '"status": "complete"' "$out/training.json"; then
      say "skip $tap seed $seed, already complete"
      continue
    fi
    say "training $tap seed $seed"
    "$PY" -m ag_sae.train \
      --activations "$ACTS" --tap "$tap" --out "$out" \
      --n-features "$N_FEATURES" --steps "$STEPS" --seed "$seed" \
      2>&1 | tee "$LOGDIR/train_${tap}_seed$seed.log"
  done
done

# --- 3. push the checkpoints ---------------------------------------------
# latest.pt is Adam resume state, ~300 MB per run and useless once the run is
# finished. best.pt is the analysis checkpoint and is what gets archived.
say "uploading SAE checkpoints"
rclone copy "$SAEDIR" "$REMOTE/sae/fold$FOLD" --exclude "**/latest.pt" \
  --transfers 4 --checkers 8 --drive-stop-on-upload-limit --stats 30s

# --- 4. push the activations, verify, then free the local copy ------------
say "uploading activations (this is the long one)"
rclone copy "$ACTS" "$REMOTE/acts/fold$FOLD" \
  --transfers 8 --checkers 16 --drive-chunk-size 128M \
  --drive-stop-on-upload-limit --stats 60s 2>&1 | tee "$LOGDIR/upload.log"

say "verifying the upload"
if rclone check "$ACTS" "$REMOTE/acts/fold$FOLD" --one-way --checkers 16 \
     2>&1 | tee "$LOGDIR/check.log"; then
  # Everything is archived, so only free what the analysis stage does not need.
  # test and dev stay on disk; without them concept matching would mean pulling
  # 219 GB back from Drive. index.json is kept and ShardStore opens only the
  # shards of the split it is asked for, so the remaining splits still load.
  say "check passed, freeing the archived splits"
  find "$ACTS" -maxdepth 1 -type f \
    \( -name '*_train_*' -o -name '*_val_*' -o -name '*_test_trained_*' \) -delete
  du -sh "$ACTS"
else
  echo "UPLOAD CHECK FAILED for fold $FOLD; local copy kept at $ACTS"; exit 1
fi

say "fold $FOLD done"
df -h "$ROOT" | tail -1
