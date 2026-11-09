#!/bin/bash -l
#SBATCH --job-name=mtr_headnoise
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/hn_%A_%a.txt
#SBATCH --time=24:00:00
# 25k-rung runs need ~16.5 h; a 12 h cap killed 18 tasks at epoch ~21/30 on 2026-08-26.
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
#
# Onboard-heading ladder (reviewer objection, Flohr 2026-11-04).
#
# WOMD box states come from an offboard auto-labeller whose refinement net consumes
# every frame of a track, so the heading, position and velocity the baseline consumes
# are better than an onboard tracker could supply. The objection: pose may only pay
# once that state is as wrong as a causal tracker's. We cannot measure what a tracker
# would give, so we sweep it and report the pose content effect as a function of
# heading quality.
#
# Each rung corrupts the heading ONCE, where center_objects is assembled, so the
# agent-centric frame, the intention anchors defined in that frame, the map and the
# pose root all inherit the same per-agent error. This is what ZERO_HEADING_CHANNELS
# did not do: it removed the sin/cos channel only, leaving the frame heading-aligned.
#
#   sbatch --array=0-26 --job-name=mtr_hn_ladder  slurm/60_heading_noise.sh          # 3 sigma x 3 arms x 3 seeds
#   sbatch --array=0-2  --job-name=mtr_hn_lrcheck slurm/60_heading_noise.sh lrcheck  # LR re-tune at sigma=30
#   sbatch --array=0-17 --job-name=mtr_hn_posladder slurm/60_heading_noise.sh posladder
#
# posladder is the same question for POSITION (POSITION_NOISE_M: per-frame detection
# noise through a causal constant-velocity Kalman filter, heading left clean). Its
# clean rung is the ladder's sigma=0 cells, so only the two noisy rungs run here.
#
#   sbatch --array=0-8 --job-name=mtr_hn_standft slurm/60_heading_noise.sh standft
#
# standft fine-tunes each arm's sigma=0 model (its final epoch-30 checkpoint, same seed)
# for 5 epochs on pedestrians STANDING at prediction time only (TRAIN_CENTER_SPEED_MAX),
# to test whether full-dataset training drowns out a stationary-only pose benefit.
# Eval stays on every agent; the stationary subset is scored from the pickles.
#
# The job name is what lets the watchdog tell a queued ladder task 0 from lrcheck task 0.
#
# sigma=00 is included deliberately although the workstation already has those cells:
# it keeps every comparison inside one platform, and it should reproduce the 25k
# numbers (no-pose 0.3711 / pose 0.3774 / shuffled 0.3770), which is a free check that
# cluster runs are comparable at all.
#
# Task list is explicit (cfg|lr|seed) so the grid stays auditable and the watchdog can
# parse it back out. Read results on CORRECT vs ITS OWN CONTROL at the same sigma.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
MODE=${1:-ladder}

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

# $HOME is at 98% of its space quota and within ~9k files of the file quota, so keep
# every cache that would otherwise land in ~ on the node-local disk instead. Without
# this, matplotlib, the CUDA JIT cache and triton write into $HOME from all 27 tasks.
export XDG_CACHE_HOME="$TMPDIR/cache"
export MPLCONFIGDIR="$TMPDIR/cache/mpl"
export TORCH_HOME="$TMPDIR/cache/torch"
export TRITON_CACHE_DIR="$TMPDIR/cache/triton"
export CUDA_CACHE_PATH="$TMPDIR/cache/nv"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "staging data to $TMPDIR ..."
tar xf "$WS/data/poses_${RUNG}.tar"     -C "$TMPDIR"
tar xf "$WS/data/scenarios_${RUNG}.tar" -C "$TMPDIR"

PFX=HN
case "$MODE" in
  ladder) TASKS=(
    "no_pose_with_map_headnoise00_scale25k|0.000025|101"
    "no_pose_with_map_headnoise00_scale25k|0.000025|202"
    "no_pose_with_map_headnoise00_scale25k|0.000025|303"
    "cross_attn_pe_with_map_headnoise00_scale25k|0.000025|101"
    "cross_attn_pe_with_map_headnoise00_scale25k|0.000025|202"
    "cross_attn_pe_with_map_headnoise00_scale25k|0.000025|303"
    "cross_attn_pe_with_map_shufflepose_headnoise00_scale25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_headnoise00_scale25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_headnoise00_scale25k|0.000025|303"
    "no_pose_with_map_headnoise15_scale25k|0.000025|101"
    "no_pose_with_map_headnoise15_scale25k|0.000025|202"
    "no_pose_with_map_headnoise15_scale25k|0.000025|303"
    "cross_attn_pe_with_map_headnoise15_scale25k|0.000025|101"
    "cross_attn_pe_with_map_headnoise15_scale25k|0.000025|202"
    "cross_attn_pe_with_map_headnoise15_scale25k|0.000025|303"
    "cross_attn_pe_with_map_shufflepose_headnoise15_scale25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_headnoise15_scale25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_headnoise15_scale25k|0.000025|303"
    "no_pose_with_map_headnoise30_scale25k|0.000025|101"
    "no_pose_with_map_headnoise30_scale25k|0.000025|202"
    "no_pose_with_map_headnoise30_scale25k|0.000025|303"
    "cross_attn_pe_with_map_headnoise30_scale25k|0.000025|101"
    "cross_attn_pe_with_map_headnoise30_scale25k|0.000025|202"
    "cross_attn_pe_with_map_headnoise30_scale25k|0.000025|303"
    "cross_attn_pe_with_map_shufflepose_headnoise30_scale25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_headnoise30_scale25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_headnoise30_scale25k|0.000025|303"
  ); STATUS="$WS/logs/status_hn.txt";;
  # lrcheck: the 25k LR of 2.5e-5 was swept per arm at sigma=0
  # (slurm/10_scale_sweep.sh). The degraded condition is a different task, so confirm
  # the optimum has not moved before reading the ladder: if 5e-5 beats 2.5e-5 here, the
  # ladder is handicapped and the grid needs the second LR throughout.
  lrcheck) TASKS=(
    "no_pose_with_map_headnoise30_scale25k|0.00005|101"
    "no_pose_with_map_headnoise30_scale25k|0.00005|202"
    "no_pose_with_map_headnoise30_scale25k|0.00005|303"
  ); STATUS="$WS/logs/status_hn_lr.txt";;
  posladder) TASKS=(
    "no_pose_with_map_posnoise10_scale25k|0.000025|101"
    "no_pose_with_map_posnoise10_scale25k|0.000025|202"
    "no_pose_with_map_posnoise10_scale25k|0.000025|303"
    "cross_attn_pe_with_map_posnoise10_scale25k|0.000025|101"
    "cross_attn_pe_with_map_posnoise10_scale25k|0.000025|202"
    "cross_attn_pe_with_map_posnoise10_scale25k|0.000025|303"
    "cross_attn_pe_with_map_shufflepose_posnoise10_scale25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_posnoise10_scale25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_posnoise10_scale25k|0.000025|303"
    "no_pose_with_map_posnoise30_scale25k|0.000025|101"
    "no_pose_with_map_posnoise30_scale25k|0.000025|202"
    "no_pose_with_map_posnoise30_scale25k|0.000025|303"
    "cross_attn_pe_with_map_posnoise30_scale25k|0.000025|101"
    "cross_attn_pe_with_map_posnoise30_scale25k|0.000025|202"
    "cross_attn_pe_with_map_posnoise30_scale25k|0.000025|303"
    "cross_attn_pe_with_map_shufflepose_posnoise30_scale25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_posnoise30_scale25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_posnoise30_scale25k|0.000025|303"
    # 18-29 (2026-11-09): seeds 404/505, because n=3 put the first content effect
    # in the project at 30 cm and an earlier 3-seed win did not survive n=5.
    "no_pose_with_map_posnoise10_scale25k|0.000025|404"
    "no_pose_with_map_posnoise10_scale25k|0.000025|505"
    "cross_attn_pe_with_map_posnoise10_scale25k|0.000025|404"
    "cross_attn_pe_with_map_posnoise10_scale25k|0.000025|505"
    "cross_attn_pe_with_map_shufflepose_posnoise10_scale25k|0.000025|404"
    "cross_attn_pe_with_map_shufflepose_posnoise10_scale25k|0.000025|505"
    "no_pose_with_map_posnoise30_scale25k|0.000025|404"
    "no_pose_with_map_posnoise30_scale25k|0.000025|505"
    "cross_attn_pe_with_map_posnoise30_scale25k|0.000025|404"
    "cross_attn_pe_with_map_posnoise30_scale25k|0.000025|505"
    "cross_attn_pe_with_map_shufflepose_posnoise30_scale25k|0.000025|404"
    "cross_attn_pe_with_map_shufflepose_posnoise30_scale25k|0.000025|505"
  ); STATUS="$WS/logs/status_pn.txt"; PFX=PN;;
  standft) TASKS=(
    "no_pose_with_map_standft_scale25k|0.00000625|101"
    "no_pose_with_map_standft_scale25k|0.00000625|202"
    "no_pose_with_map_standft_scale25k|0.00000625|303"
    "cross_attn_pe_with_map_standft_scale25k|0.00000625|101"
    "cross_attn_pe_with_map_standft_scale25k|0.00000625|202"
    "cross_attn_pe_with_map_standft_scale25k|0.00000625|303"
    "cross_attn_pe_with_map_shufflepose_standft_scale25k|0.00000625|101"
    "cross_attn_pe_with_map_shufflepose_standft_scale25k|0.00000625|202"
    "cross_attn_pe_with_map_shufflepose_standft_scale25k|0.00000625|303"
  ); STATUS="$WS/logs/status_ft.txt"; PFX=FT;;
  *) echo "unknown mode: $MODE (expected ladder|lrcheck|posladder|standft)" >&2; exit 2;;
esac

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#TASKS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is beyond the grid (${#TASKS[@]} tasks)" >&2; exit 2
fi
spec=${TASKS[$SLURM_ARRAY_TASK_ID]}
cfg=${spec%%|*}; rest=${spec#*|}; lr=${rest%%|*}; seed=${rest##*|}
lrtag=$(echo "$lr" | sed 's/0\.0*//')
tag="${PFX}_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
echo "task $SLURM_ARRAY_TASK_ID -> cfg=$cfg lr=$lr seed=$seed tag=$tag"

INIT=""
if [ "$MODE" = standft ]; then
  arm=${cfg%_standft_scale25k}
  INIT="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${arm}_headnoise00_scale25k/HN_25k_${arm}_headnoise00_scale25k_lr25_s${seed}/ckpt/checkpoint_epoch_30.pth"
  [ -f "$INIT" ] || { echo "missing init checkpoint $INIT" >&2; exit 2; }
  echo "init from $INIT"
fi

note() { flock "$STATUS.lock" -c "echo \"$(date +%H:%M:%S) $tag $*\" >> '$STATUS'"; }
note "START"
trap 'note "EXIT rc=$?"' EXIT

cd "$WS/MTR_with_poses/tools"
python train.py \
  --cfg_file "cfgs/waymo/mtr+pose_data_${cfg}.yaml" \
  --extra_tag "$tag" --random_seed "$seed" --workers 8 --max_ckpt_save_num 1 \
  ${INIT:+--pretrained_model "$INIT"} \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_with_poses/models" \
        OPTIMIZATION.LR "$lr"

out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cfg}/$tag"
# metrics_epoch pickles are what every later analysis reads, so they stay; the
# checkpoint and the raw result.pkl are the big files and are not needed again.
rm -f "$out/ckpt/best_model.pth" "$out/eval/eval_with_train/result.pkl" 2>/dev/null || true
touch "$out/DONE"
note "DONE best=$(grep -hE '^minADE' "$out"/log_train_*.txt 2>/dev/null | awk '{print $2}' | sort -g | head -1)"
echo "DONE $tag"
