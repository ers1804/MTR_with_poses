#!/bin/bash -l
#SBATCH --job-name=mtr_rev2
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/rev2_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
#
# Second part of the revision queue (see 80_revision.sh): the runs that need dataset code
# the cluster's main checkout does not have. They run from their own checkout, $WS/MTR_rev,
# so that no file another job reads is replaced. Submitted behind the first part:
#
#   sbatch --nice=4000 --dependency=afterany:<job id of 80_revision.sh> --array=0-29%3 \
#          slurm/82_revision_code.sh
#
# The grid (cfg|lr|seed|poses tar|tag):
#   0-14   the three arms with FULL AGENT CONTEXT, five seeds (review, flaw 3). Every other
#          run gives the model pedestrians and map alone. Here vehicles, cyclists, the ego
#          and pedestrians without a pose join the scene as context (CONTEXT_AGENTS all).
#          Offboard track, fitted poses, the recipe of the clean 25k cells.
#   15-29  the three arms under a WARM-STARTED TRACKER WITH A DEGRADED HEADING, pre-fit
#          poses, 30 cm, five seeds (flaw 2). TRACK_WARM_START gives every track that is
#          valid at the first observed frame 3 s of history before it. TRACK_HEADING_NOISE_DEG
#          puts one yaw error of s.d. 25 deg on each track (see below for the level). The
#          pose arms keep the paper's encoding (root in world coordinates), so this asks
#          whether the pre-fit gain survives, not whether a body can repair the heading.
#          Stratum fixed in advance: at least 200 LiDAR returns, scored on held-out
#          validation scenes (tools/scripts/heldout_replication.py).
#   30-39  the two pose arms of 15-29 with the body's root in the AGENT FRAME
#          (AGENT_CENTRIC_POSE_ROT), five seeds. Their no-pose arm is that of 15-29. This
#          is the test of whether a body can repair a wrong heading.
#   40-54  the three arms of 15-29 at 10 cm of detection noise, five seeds.
# Tasks 30-54 were added after the first submission and go in as a third array:
#
#   sbatch --nice=4000 --job-name=mtr_rev3 --dependency=afterany:<job id of 0-29> \
#          --array=30-54%3 slurm/82_revision_code.sh
#
# The heading level: s.d. 25 deg per track is a mean absolute error of 20 deg. Published AP
# and heading-weighted APH for pedestrians on the Waymo Open Dataset give a mean absolute
# heading error of pi (1 - APH / AP): 14 deg (CenterPoint-Voxel), 23 deg (PV-RCNN), 26 deg
# (CenterPoint-Pillar), 35 deg (PointPillars). PointPillars on nuScenes: 0.37 rad.
#
# Tests of the three dataset options: tools/scripts/test_revision_dataset.py.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
STATUS="$WS/logs/status_rev2.txt"
REPO="$WS/MTR_rev"

TASKS=(
  "no_pose_with_map_allctx_scale25k|0.000025|101|fitted|CX"
  "cross_attn_pe_with_map_allctx_scale25k|0.000025|101|fitted|CX"
  "cross_attn_pe_with_map_shufflepose_allctx_scale25k|0.000025|101|fitted|CX"
  "no_pose_with_map_allctx_scale25k|0.000025|202|fitted|CX"
  "cross_attn_pe_with_map_allctx_scale25k|0.000025|202|fitted|CX"
  "cross_attn_pe_with_map_shufflepose_allctx_scale25k|0.000025|202|fitted|CX"
  "no_pose_with_map_allctx_scale25k|0.000025|303|fitted|CX"
  "cross_attn_pe_with_map_allctx_scale25k|0.000025|303|fitted|CX"
  "cross_attn_pe_with_map_shufflepose_allctx_scale25k|0.000025|303|fitted|CX"
  "no_pose_with_map_allctx_scale25k|0.000025|404|fitted|CX"
  "cross_attn_pe_with_map_allctx_scale25k|0.000025|404|fitted|CX"
  "cross_attn_pe_with_map_shufflepose_allctx_scale25k|0.000025|404|fitted|CX"
  "no_pose_with_map_allctx_scale25k|0.000025|505|fitted|CX"
  "cross_attn_pe_with_map_allctx_scale25k|0.000025|505|fitted|CX"
  "cross_attn_pe_with_map_shufflepose_allctx_scale25k|0.000025|505|fitted|CX"
  "no_pose_with_map_warmtrk30_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk30_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk30_head25_scale25k|0.000025|101|prefit|TW"
  "no_pose_with_map_warmtrk30_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk30_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk30_head25_scale25k|0.000025|202|prefit|TW"
  "no_pose_with_map_warmtrk30_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk30_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk30_head25_scale25k|0.000025|303|prefit|TW"
  "no_pose_with_map_warmtrk30_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk30_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk30_head25_scale25k|0.000025|404|prefit|TW"
  "no_pose_with_map_warmtrk30_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk30_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk30_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_agentrot_warmtrk30_head25_scale25k|0.000025|505|prefit|TW"
  "no_pose_with_map_warmtrk10_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk10_head25_scale25k|0.000025|101|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk10_head25_scale25k|0.000025|101|prefit|TW"
  "no_pose_with_map_warmtrk10_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk10_head25_scale25k|0.000025|202|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk10_head25_scale25k|0.000025|202|prefit|TW"
  "no_pose_with_map_warmtrk10_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk10_head25_scale25k|0.000025|303|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk10_head25_scale25k|0.000025|303|prefit|TW"
  "no_pose_with_map_warmtrk10_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk10_head25_scale25k|0.000025|404|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk10_head25_scale25k|0.000025|404|prefit|TW"
  "no_pose_with_map_warmtrk10_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_prefit_warmtrk10_head25_scale25k|0.000025|505|prefit|TW"
  "cross_attn_pe_with_map_shufflepose_prefit_warmtrk10_head25_scale25k|0.000025|505|prefit|TW"
)

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#TASKS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is beyond the grid (${#TASKS[@]} tasks)" >&2; exit 2
fi
IFS='|' read -r cfg lr seed poses tag <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
lrtag=$(echo "$lr" | sed 's/0\.0*//')
tag="${tag}_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
echo "task $SLURM_ARRAY_TASK_ID -> cfg=$cfg lr=$lr seed=$seed poses=$poses tag=$tag"

[ -f "$REPO/tools/cfgs/waymo/mtr+pose_data_${cfg}.yaml" ] || { echo "no config for $cfg" >&2; exit 2; }
out="$REPO/output/waymo/mtr+pose_data_${cfg}/$tag"
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

cd "$REPO/tools"
# These runs differ from earlier ones only through options of this checkout's dataset code.
# If python picked up another checkout's mtr package, the options would be ignored without
# an error and the run would silently repeat an old condition. Refuse to start instead.
python - "$REPO" <<'PY'
import os, sys
import _init_path  # noqa: F401
from mtr.datasets.waymo import waymo_pose_dataset as d
# mtr has no __init__.py, so mtr.__file__ is None. The dataset module is what matters.
here, repo = os.path.realpath(d.__file__), os.path.realpath(sys.argv[1])
assert here.startswith(repo + os.sep), f'the dataset is imported from {here}, not from {repo}'
src = open(d.__file__).read()
for key in ('TRACK_WARM_START', 'TRACK_HEADING_NOISE_DEG', 'CONTEXT_AGENTS'):
    assert key in src, f'{d.__file__} does not know {key}'
print('code check ok:', here)
PY
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
