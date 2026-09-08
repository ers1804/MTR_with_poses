#!/bin/bash -l
#SBATCH --job-name=mtrpose_sweep
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/sweep_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-26%12
#SBATCH --export=NONE
# Scale-ladder sweep: 3 cells x 3 LRs x 3 seeds = 27 independent single-GPU runs.
# Rationale (2026-08-21): a single LR inherited from the 579-scene recipe manufactured a
# +37% fake "pose hurts" effect at 4.9x data; each arm must be trained where IT converges,
# so LR is swept per arm and the per-arm best is compared.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}                     # which ladder rung (payload dir name)

# Make the module system available regardless of how this script is invoked. A login shell
# (`bash -l`, or sbatch's `#!/bin/bash -l`) defines `module` as a function; a plain `ssh host
# "cmd"` does not, and the script then dies with "module: command not found".
if ! type -t module >/dev/null 2>&1; then
  set +u                      # site profile scripts reference unset vars (e.g. $GROUP)
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025
module add python/3.12-conda
module add cuda/12.6.2

# The env lives in the workspace as ONE conda-pack archive (file quota: ~82k soft, 33k used,
# a torch env alone is 100k+ files). Unpack it node-local per job.
echo "unpacking env to $TMPDIR/env ..."
mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
# `set -u` must be off across activation: conda-pack's activate script reads $CONDA_PREFIX
# before defining it, which aborts the job under strict mode.
set +u
source "$TMPDIR/env/bin/activate"
conda-unpack
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_CENTERS=24           # dense clean-data scenes reach 153 peds; cap for memory

# ---- stage data to node-local NVMe -----------------------------------------
echo "staging payload to $TMPDIR ..."
tar xf "$WS/data/poses_${RUNG}.tar"      -C "$TMPDIR"   # -> poses/{training,validation}
tar xf "$WS/data/scenarios_${RUNG}.tar"  -C "$TMPDIR"   # -> scenarios/processed_scenarios_*
echo "staged: $(ls "$TMPDIR/poses/training" | wc -l) train scenes, $(ls "$TMPDIR/poses/validation" | wc -l) val scenes"

# ---- sweep grid -------------------------------------------------------------
CELLS=(no_pose_with_map cross_attn_pe_with_map cross_attn_pe_with_map_shufflepose)
# Seed list is also a parameter: `sbatch 10_scale_sweep.sh 0.000025 404,505` extends an
# existing rung with more seeds (the CI on correct-vs-shuffled is seed-dominated).
IFS=',' read -r -a SEEDS <<< "${2:-101,202,303}"
# LR grid is a parameter: `sbatch 10_scale_sweep.sh 0.0000125` sweeps a single extra LR
# (array 0-8), which is how boundary optima get checked without redoing the whole grid.
IFS=',' read -r -a LRS <<< "${1:-0.0001,0.00005,0.000025}"
nlr=${#LRS[@]}; nseed=${#SEEDS[@]}
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$(( i / (nlr * nseed) ))]}
lr=${LRS[$(( (i / nseed) % nlr ))]}
seed=${SEEDS[$(( i % nseed ))]}
lrtag=$(echo "$lr" | sed 's/0\.0*//')
tag="HELMA_${RUNG}_${cell}_lr${lrtag}_s${seed}"
echo "task $i -> cell=$cell lr=$lr seed=$seed tag=$tag"

# Self-reported status: one shared file so progress can be read with a SINGLE cheap ssh for
# the whole array, instead of polling Slurm per task (which rate-limited the jump host once).
STATUS="$WS/logs/status.txt"
note() { flock "$STATUS.lock" -c "echo \"$(date +%H:%M:%S) task$i $tag $*\" >> '$STATUS'"; }
note "START"
trap 'note "EXIT rc=$?"' EXIT

cd "$WS/MTR_with_poses/tools"
python train.py \
  --cfg_file "cfgs/waymo/mtr+pose_data_${cell}_scale5k_lr5.yaml" \
  --extra_tag "$tag" --random_seed "$seed" --workers 8 --max_ckpt_save_num 1 \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_with_poses/models" \
        OPTIMIZATION.LR "$lr"

# keep logs/metrics, drop bulky checkpoints
out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cell}_scale5k_lr5/$tag"
rm -f "$out/ckpt/best_model.pth" "$out/eval/eval_with_train/result.pkl" 2>/dev/null || true
note "DONE best=$(grep -hE "^minADE" "$out"/log_train_*.txt 2>/dev/null | awk "{print \$2}" | sort -g | head -1)"
touch "$out/DONE"          # completion marker the watchdog checks
echo "DONE $tag"
