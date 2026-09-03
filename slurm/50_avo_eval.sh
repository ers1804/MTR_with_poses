#!/bin/bash -l
#SBATCH --job-name=mtrpose_avo   # unique: a sibling project also names jobs avo_eval
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/avo_%j.txt
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
# AVO scoring job: train ONE candidate (or the no-pose baseline) under the frozen 25k
# proxy recipe and report best-epoch minADE on the DEV half of the validation scenes.
#
#   sbatch 50_avo_eval.sh <eval_id> <seed> <overlay_dir_remote|NONE>
#
# Frozen-repo guarantee: trains from $WS/MTR_avo (pinned commit, verified below), never
# from the mutable working copy. Result JSON -> $WS/avo_results/<eval_id>_s<seed>.json.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
EVAL_ID=$1; SEED=$2; OVERLAY=$3
RES="$WS/avo_results/${EVAL_ID}_s${SEED}.json"
mkdir -p "$WS/avo_results"

fail() { echo "{\"status\": \"failed\", \"reason\": \"$1\"}" > "$RES"; exit 1; }

if ! type -t module >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025 python/3.12-conda cuda/12.6.2

# frozen repo check — the search must not run on drifted code
cd "$WS/MTR_avo"
PIN=$(cat "$WS/MTR_avo.commit")
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || fail "frozen repo at $HEAD, expected $PIN"
[ -z "$(git status --porcelain -uno)" ] || fail "frozen repo has modified tracked files"

mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
set +u; source "$TMPDIR/env/bin/activate"; conda-unpack; set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24

tar xf "$WS/data/poses_25k.tar" -C "$TMPDIR"
tar xf "$WS/data/scenarios_25k.tar" -C "$TMPDIR"

# overlay -> mode-dependent config overrides
EXTRA=()
if [ "$OVERLAY" != "NONE" ]; then
  export POSE_OVERLAY_DIR="$OVERLAY"
  read -r MODE FDIM < <(python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('o', '$OVERLAY/overlay.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.MODE, m.FEATURE_DIM)") || fail "overlay does not import"
  if [ "$MODE" = "early" ]; then
    EXTRA=(MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_AGENT $((29 + FDIM)))
  else
    EXTRA=(MODEL.CONTEXT_ENCODER.USE_POSE_ENCODER True
           MODEL.CONTEXT_ENCODER.USE_OVERLAY_FUSION True
           MODEL.CONTEXT_ENCODER.POSE_INPUT_DIM "$FDIM")
  fi
fi

TAG="AVO_${EVAL_ID}_s${SEED}"
cd tools
python train.py \
  --cfg_file cfgs/waymo/mtr+pose_data_avo_proxy.yaml \
  --extra_tag "$TAG" --random_seed "$SEED" --workers 8 --max_ckpt_save_num 1 \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_avo/models" \
        OPTIMIZATION.LR "${AVO_LR:-0.0001}" \
        "${EXTRA[@]}" \
  || fail "training crashed"

OUT="../output/waymo/mtr+pose_data_avo_proxy/$TAG"
python - "$OUT" "$RES" "$SEED" <<'PY'
import glob, json, pickle, sys
import numpy as np
out, res, seed = sys.argv[1], sys.argv[2], sys.argv[3]
dev = set(open('../data/waymo/avo_eval_dev_scenes.txt').read().split())
best = None
for p in sorted(glob.glob(f'{out}/eval/eval_with_train/metrics_epoch_*.pkl')):
    recs = pickle.load(open(p, 'rb'))
    vals = [r['minADE'] for r in recs if r['scenario_id'] in dev]
    if not vals:
        continue
    m = float(np.mean(vals))
    ep = int(p.rsplit('_', 1)[1].split('.')[0])
    if not np.isfinite(m):
        continue
    if best is None or m < best['minADE_dev']:
        best = {'minADE_dev': m, 'epoch': ep, 'n_dev_agents': len(vals)}
if best is None:
    json.dump({'status': 'failed', 'reason': 'no finite dev metrics'}, open(res, 'w'))
else:
    best.update(status='ok', seed=int(seed))
    json.dump(best, open(res, 'w'))
print('wrote', res, best)
PY

rm -f "$OUT/ckpt/best_model.pth" "$OUT/eval/eval_with_train/result.pkl" 2>/dev/null || true
