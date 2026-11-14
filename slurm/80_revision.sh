#!/bin/bash -l
#SBATCH --job-name=mtr_rev
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/rev_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
#
# Runs for the revision of the IV paper (pre-submission review of 2026-11-13). None of
# them is needed before the reviews come back, and the account's other jobs have a
# deadline, so this array is submitted at the back of the queue and throttled:
#
#   sbatch --nice=4000 --array=0-32%3 slurm/80_revision.sh
#
# --nice keeps every task below the account's other pending jobs, %3 keeps the array to
# three GPUs at a time. To pause it: scontrol hold <jobid>. To change the throttle:
# scontrol update JobId=<jobid> ArrayTaskThrottle=<n>.
#
# The grid, in the order the review weighs it (cfg|lr|seed|poses tar|tag):
#   0-9    pre-fit poses on the OFFBOARD track, own against stranger, five seeds.
#          The null on WOMD's own tracks was measured with poses fitted to those tracks
#          (review, flaw 1). Completes the 2x2 of track by pose.
#   10-18  oracle levels at a median error of 1, 2 and 3 deg, three seeds. The ladder had
#          no level between 0 and 5 deg, so its fast half-life was only bounded (flaw 4).
#          Tags follow the workstation's ladder (MS_oracledir<k>_25k_s<seed>) so that
#          tools/scripts/detection_floor.py reads them once its level list is extended.
#   19-28  pre-fit poses through EARLY fusion under the 30 cm tracker, own against
#          stranger, five seeds. The pre-fit gain was measured through the branch alone.
#   29-32  pre-fit poses at 10 cm, seeds 404 and 505. Seeds 101 to 303 are the
#          workstation's (tools/scripts/run_prefit_local.sh). Its queue also lists 404
#          and 505, so whichever platform gets there first makes the other redundant.
#
# Every task is one of the existing 25k recipes with one thing changed. No code is
# needed beyond what the 30 cm pre-fit runs used (slurm/60_heading_noise.sh, prefit).
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
STATUS="$WS/logs/status_rev.txt"

TASKS=(
  "cross_attn_pe_with_map_prefit_scale25k|0.000025|101|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_scale25k|0.000025|101|prefit|PF"
  "cross_attn_pe_with_map_prefit_scale25k|0.000025|202|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_scale25k|0.000025|202|prefit|PF"
  "cross_attn_pe_with_map_prefit_scale25k|0.000025|303|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_scale25k|0.000025|303|prefit|PF"
  "cross_attn_pe_with_map_prefit_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_prefit_scale25k|0.000025|505|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_scale25k|0.000025|505|prefit|PF"
  "cross_attn_pe_with_map_oracledir01_scale25k|0.000025|101|fitted|MS_oracledir01_25k_s101"
  "cross_attn_pe_with_map_oracledir02_scale25k|0.000025|101|fitted|MS_oracledir02_25k_s101"
  "cross_attn_pe_with_map_oracledir03_scale25k|0.000025|101|fitted|MS_oracledir03_25k_s101"
  "cross_attn_pe_with_map_oracledir01_scale25k|0.000025|202|fitted|MS_oracledir01_25k_s202"
  "cross_attn_pe_with_map_oracledir02_scale25k|0.000025|202|fitted|MS_oracledir02_25k_s202"
  "cross_attn_pe_with_map_oracledir03_scale25k|0.000025|202|fitted|MS_oracledir03_25k_s202"
  "cross_attn_pe_with_map_oracledir01_scale25k|0.000025|303|fitted|MS_oracledir01_25k_s303"
  "cross_attn_pe_with_map_oracledir02_scale25k|0.000025|303|fitted|MS_oracledir02_25k_s303"
  "cross_attn_pe_with_map_oracledir03_scale25k|0.000025|303|fitted|MS_oracledir03_25k_s303"
  "cross_attn_pe_with_map_derived_prefit_posnoise30_scale25k|0.000025|101|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_derived_prefit_posnoise30_scale25k|0.000025|101|prefit|PF"
  "cross_attn_pe_with_map_derived_prefit_posnoise30_scale25k|0.000025|202|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_derived_prefit_posnoise30_scale25k|0.000025|202|prefit|PF"
  "cross_attn_pe_with_map_derived_prefit_posnoise30_scale25k|0.000025|303|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_derived_prefit_posnoise30_scale25k|0.000025|303|prefit|PF"
  "cross_attn_pe_with_map_derived_prefit_posnoise30_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_derived_prefit_posnoise30_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_derived_prefit_posnoise30_scale25k|0.000025|505|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_derived_prefit_posnoise30_scale25k|0.000025|505|prefit|PF"
  "cross_attn_pe_with_map_prefit_posnoise10_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_posnoise10_scale25k|0.000025|404|prefit|PF"
  "cross_attn_pe_with_map_prefit_posnoise10_scale25k|0.000025|505|prefit|PF"
  "cross_attn_pe_with_map_shufflepose_prefit_posnoise10_scale25k|0.000025|505|prefit|PF"
)

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#TASKS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is beyond the grid (${#TASKS[@]} tasks)" >&2; exit 2
fi
IFS='|' read -r cfg lr seed poses tag <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
lrtag=$(echo "$lr" | sed 's/0\.0*//')
[ "$tag" = PF ] && tag="PF_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
echo "task $SLURM_ARRAY_TASK_ID -> cfg=$cfg lr=$lr seed=$seed poses=$poses tag=$tag"

[ -f "$WS/MTR_with_poses/tools/cfgs/waymo/mtr+pose_data_${cfg}.yaml" ] || { echo "no config for $cfg" >&2; exit 2; }
out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cfg}/$tag"
if [ -f "$out/DONE" ]; then echo "already done: $tag"; exit 0; fi

if ! type -t module >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025
module add python/3.12-conda
module add cuda/12.6.2

echo "unpacking env to $TMPDIR/env ..."
mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
set +u
source "$TMPDIR/env/bin/activate"
conda-unpack
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_CENTERS=24

# $HOME is close to its quota, so every cache goes to the node-local disk (see 60).
export XDG_CACHE_HOME="$TMPDIR/cache"
export MPLCONFIGDIR="$TMPDIR/cache/mpl"
export TORCH_HOME="$TMPDIR/cache/torch"
export TRITON_CACHE_DIR="$TMPDIR/cache/triton"
export CUDA_CACHE_PATH="$TMPDIR/cache/nv"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "staging data to $TMPDIR ..."
case "$poses" in
  prefit) POSES_TAR="poses_${RUNG}_prefit.tar"; POSES_DIR="$TMPDIR/poses_prefit";;
  fitted) POSES_TAR="poses_${RUNG}.tar";        POSES_DIR="$TMPDIR/poses";;
  *) echo "unknown pose set: $poses" >&2; exit 2;;
esac
tar xf "$WS/data/$POSES_TAR"            -C "$TMPDIR"
tar xf "$WS/data/scenarios_${RUNG}.tar" -C "$TMPDIR"
[ -d "$POSES_DIR/training" ] || { echo "$POSES_TAR did not unpack to $POSES_DIR" >&2; exit 2; }

note() { flock "$STATUS.lock" -c "echo \"$(date +%F_%H:%M:%S) $tag $*\" >> '$STATUS'"; }
note "START"
trap 'note "EXIT rc=$?"' EXIT

cd "$WS/MTR_with_poses/tools"
python train.py \
  --cfg_file "cfgs/waymo/mtr+pose_data_${cfg}.yaml" \
  --extra_tag "$tag" --random_seed "$seed" --workers 8 --max_ckpt_save_num 5 \
  --set DATA_CONFIG.DATA_ROOT "$POSES_DIR" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_with_poses/models" \
        OPTIMIZATION.LR "$lr"

# metrics_epoch pickles are what every later analysis reads, so they stay. The raw
# result.pkl is big and not needed again.
rm -f "$out/eval/eval_with_train/result.pkl" 2>/dev/null || true
touch "$out/DONE"
note "DONE best=$(grep -hE '^minADE' "$out"/log_train_*.txt 2>/dev/null | awk '{print $2}' | sort -g | head -1)"

# Checkpoints of the last five epochs stay, weights only. A comparison between
# architectures needs its epoch chosen on one set of scenes and scored on another
# (held-out replication of 2026-11-13: the gain over no pose is there at the best
# validation epoch and not at the final one), and that needs more than epoch 30. With the
# optimizer state a checkpoint is 1.3 GB, without it a third, so five of them and the
# best model take what epoch 30 and latest_model.pth took before. Stripping comes after
# DONE: train.py resumes from the newest file in ckpt/, and a stripped file cannot resume.
python - "$out/ckpt" <<'PY' || echo "WARNING: checkpoints not stripped, they keep their optimizer state"
import glob, os, sys
import torch
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.pth'))):
    if os.path.basename(f) == 'latest_model.pth':
        continue
    c = torch.load(f, map_location='cpu')
    assert c['model_state'], f
    if c.get('optimizer_state') is None:
        continue
    c['optimizer_state'] = None
    torch.save(c, f + '.tmp')
    os.replace(f + '.tmp', f)
    print('weights only:', os.path.basename(f), os.path.getsize(f) >> 20, 'MiB')
PY
rm -f "$out/ckpt/latest_model.pth"
echo "DONE $tag"
