#!/usr/bin/env bash
# Retrain one fold's residual-stream SAEs with the v2 recipe on one A100, then
# rerun the downstream experiments on them.
#
#   0. pilot (PILOT=1): b8 seed0 only, legacy vs v2, k=409 and k=64; stops there
#   1. activations: pull the train/val shards of the three resid taps from Drive
#      (or re-extract with REEXTRACT=1, see the extraction notes in the README)
#   2. train: 3 seeds per tap in one process, k~409 into sae_v2/, k=64 into sae_v2_k64/
#   3. push the checkpoints
#   4. triad ablation + MANE injection (python -m ag_sae matrix --sae-pool ...)
#   5. CTCF 2x2 motif mediation (python -m ag_sae.motif_intervene --campaign ...)
#
# Usage: run_fold_v2.sh <0|1|2|3>
# Env:   PILOT=1, REEXTRACT=1, MAX_BATCH (conditions per forward, default 4;
#        use 2 on a 40 GB A100), PRECISION (tf32|bf16, default tf32)
set -euo pipefail

FOLD="${1:?usage: run_fold_v2.sh <fold>}"
ROOT=/mnt/ag
CODE="$HOME/ag"
PY="${PY:-$CODE/.venv/bin/python}"
REMOTE="gdrive:alphagenome_sae"
TAPS=(resid_pre_b0 resid_pre_b4 resid_pre_b8)
MAX_BATCH="${MAX_BATCH:-4}"
PRECISION="${PRECISION:-tf32}"

ACTS="$ROOT/acts/fold$FOLD"
LOGDIR="$ROOT/logs_v2/fold$FOLD"
mkdir -p "$ACTS" "$LOGDIR"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- 1. activations ---------------------------------------------------------
if [ "${REEXTRACT:-0}" = 1 ]; then
  say "re-extracting fold $FOLD (resid taps only, 4 windows per forward)"
  "$PY" -m ag_sae.extract \
    --manifest "$CODE/manifests/window_manifest_fold$FOLD.parquet" \
    --weights "$ROOT/weights/model_fold_$FOLD.safetensors" \
    --fasta "$ROOT/data/GRCh38.primary_assembly.genome.fa" \
    --out "$ACTS" --taps "${TAPS[@]}" \
    --positions-per-window "${POSITIONS:-8192}" --windows-per-shard 8 \
    --batch-windows "$MAX_BATCH" --seed "$FOLD" 2>&1 | tee -a "$LOGDIR/extract.log"
else
  say "pulling train/val resid shards of fold $FOLD from Drive"
  rclone copy "$REMOTE/acts/fold$FOLD" "$ACTS" --include "index.json" \
    --include "*_train_resid_pre_b*" --include "*_val_resid_pre_b*" \
    --transfers 16 --checkers 16 --drive-chunk-size 128M --stats 60s
fi

train() {   # train <tap> <pool> <extra args...>
  local tap="$1" pool="$2"; shift 2
  "$PY" -m ag_sae.train --activations "$ACTS" --tap "$tap" \
    --out "$ROOT/$pool/fold$FOLD/${tap}_seed{seed}" --n-features 8192 \
    --precision "$PRECISION" "$@" 2>&1 | tee -a "$LOGDIR/train_${pool}_${tap}.log"
}

# --- 0. pilot -------------------------------------------------------------
if [ "${PILOT:-0}" = 1 ]; then
  say "pilot: resid_pre_b8 seed0, legacy vs v2, k~409 and k=64"
  # Legacy runs use exact fp32 matmuls, as the first campaign did.
  PRECISION=fp32 train resid_pre_b8 pilot_legacy --seeds 0 --recipe legacy
  train resid_pre_b8 pilot_v2 --seeds 0
  PRECISION=fp32 train resid_pre_b8 pilot_legacy_k64 --seeds 0 --recipe legacy --topk 64
  train resid_pre_b8 pilot_v2_k64 --seeds 0 --topk 64
  for p in pilot_legacy pilot_v2 pilot_legacy_k64 pilot_v2_k64; do
    "$PY" - "$ROOT/$p/fold$FOLD/resid_pre_b8_seed0/training.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
h = json.load(open(sys.argv[1].replace("training.json", "history.json")))[-1]["validation"]
print(f"{sys.argv[1].split('/')[3]:>18}: fvu={r['best_validation_fvu']:.4f} "
      f"never_fired={h['features_never_fired']} dead={r['dead_latents']} {r['seconds']/60:.1f} min")
PY
  done
  exit 0
fi

# --- 2. train ---------------------------------------------------------------
for tap in "${TAPS[@]}"; do
  say "training $tap (v2, k~409, seeds 0-2)"
  train "$tap" sae_v2 --seeds 0 1 2
  say "training $tap (v2, k=64, seeds 0-2)"
  train "$tap" sae_v2_k64 --seeds 0 1 2 --topk 64
done

# --- 3. push the checkpoints ------------------------------------------------
for pool in sae_v2 sae_v2_k64; do
  rclone copy "$ROOT/$pool/fold$FOLD" "$REMOTE/$pool/fold$FOLD" --exclude "**/latest.pt" \
    --transfers 4 --checkers 8 --stats 30s
done

# --- 4/5. downstream --------------------------------------------------------
for pool in sae_v2 sae_v2_k64; do
  say "triad ablation + MANE injection ($pool)"
  "$PY" -u -m ag_sae matrix --fold "fold$FOLD" --sae-pool "$pool" --max-batch "$MAX_BATCH" 2>&1 | tee -a "$LOGDIR/matrix_$pool.log"
  say "CTCF 2x2 motif mediation ($pool)"
  "$PY" -u -m ag_sae.motif_intervene --campaign --fold "fold$FOLD" --sae-pool "$pool" \
    --max-batch "$MAX_BATCH" 2>&1 | tee -a "$LOGDIR/motif_$pool.log"
done
say "fold $FOLD done"
