#!/bin/bash -l
#SBATCH --job-name=mtrpose_conf
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/conf_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
# Confirmation run: 30-epoch recipe, reports minADE on BOTH the dev and the
# held-out FINAL split (the loop has never scored FINAL).
#   sbatch 52_avo_confirm.sh <tag> <seed> <overlay|NONE> <config_basename>
unset SLURM_EXPORT_ENV
set -eu
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
ARM=$1; SEED=$2; OVERLAY=$3; CFG=$4
# TAG must include the seed: --extra_tag sets the output dir, and without the seed all
# seeds of an arm write into ONE directory and overwrite each other (2026-09-10 loss).
TAG="${ARM}_s${SEED}"
RES="$WS/avo_confirm/${TAG}.json"
mkdir -p "$WS/avo_confirm"
fail() { echo "{\"status\": \"failed\", \"reason\": \"$1\"}" > "$RES"; exit 1; }

if ! type -t module >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025 python/3.12-conda cuda/12.6.2

cd "$WS/MTR_avo"
PIN=$(cat "$WS/MTR_avo.commit"); HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || fail "frozen repo at $HEAD, expected $PIN"
[ -z "$(git status --porcelain -uno)" ] || fail "frozen repo has modified tracked files"

mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
set +u; source "$TMPDIR/env/bin/activate"; conda-unpack; set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24
tar xf "$WS/data/poses_25k.tar" -C "$TMPDIR"
tar xf "$WS/data/scenarios_25k.tar" -C "$TMPDIR"

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

cd tools
python train.py \
  --cfg_file "cfgs/waymo/${CFG}.yaml" \
  --extra_tag "$TAG" --random_seed "$SEED" --workers 8 --max_ckpt_save_num 1 \
  --ckpt_save_interval 100 \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_avo/models" \
        OPTIMIZATION.LR "${AVO_LR:-0.0001}" \
        "${EXTRA[@]}" \
  || fail "training crashed"

# output dir DERIVED from the config name (the 2026-09-10 extraction bug)
OUT="../output/waymo/${CFG}/$TAG"
python - "$OUT" "$RES" "$SEED" "$TAG" <<'PY'
import glob, json, pickle, sys
import numpy as np
out, res, seed, tag = sys.argv[1:5]
dev   = set(open('../data/waymo/avo_eval_dev_scenes.txt').read().split())
final = set(open('../data/waymo/avo_eval_final_scenes.txt').read().split())
per_epoch = {}
for p in sorted(glob.glob(f'{out}/**/metrics_epoch_*.pkl', recursive=True)):
    try: recs = pickle.load(open(p, 'rb'))
    except Exception: continue
    ep = int(p.rsplit('_', 1)[1].split('.')[0])
    d = [r['minADE'] for r in recs if r['scenario_id'] in dev]
    f = [r['minADE'] for r in recs if r['scenario_id'] in final]
    if not d or not f: continue
    md, mf = float(np.mean(d)), float(np.mean(f))
    if not (np.isfinite(md) and np.isfinite(mf)): continue
    per_epoch[ep] = {'dev': md, 'final': mf, 'n_dev': len(d), 'n_final': len(f)}
if not per_epoch:
    json.dump({'status':'failed','reason':f'no finite metrics under {out}'}, open(res,'w'))
else:
    last = max(per_epoch)
    bd = min(per_epoch, key=lambda e: per_epoch[e]['dev'])
    json.dump({'status':'ok','seed':int(seed),'tag':tag,
               'last_epoch':last,'best_dev_epoch':bd,
               'minADE_dev_last':per_epoch[last]['dev'],
               'minADE_final_last':per_epoch[last]['final'],
               'minADE_dev_bestdev':per_epoch[bd]['dev'],
               'minADE_final_bestdev':per_epoch[bd]['final'],
               'per_epoch':per_epoch}, open(res,'w'), indent=1)
print('wrote', res)
PY
rm -f "$OUT/ckpt/best_model.pth" "$OUT/eval/eval_with_train/result.pkl" 2>/dev/null || true
echo "$TAG done $(date -Is)" >> "$WS/logs/status.txt"
