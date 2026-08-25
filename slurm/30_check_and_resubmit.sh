#!/bin/bash
# Watchdog: find sweep tasks that are neither finished nor running, and resubmit exactly those.
#
# A task counts as finished when its output dir contains a DONE marker (written by the sweep
# script only after training AND eval succeeded). Anything missing whose index is not currently
# queued/running is considered dead (node failure, timeout, preemption, OOM) and gets resubmitted.
#
# Usage:  bash slurm/30_check_and_resubmit.sh [--apply]
#         without --apply it only reports (safe to run from cron for visibility)
set -u
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
APPLY=${1:-}

CELLS=(no_pose_with_map cross_attn_pe_with_map cross_attn_pe_with_map_shufflepose)
SEEDS=(101 202 303)
declare -A GRIDS=( ["0.0001,0.00005,0.000025"]="main" ["0.0000125"]="ext" )

running_idx() {   # array indices currently queued or running, per job name
  squeue -u "$USER" -h -o "%i" 2>/dev/null | sed 's/.*_//' | tr '\n' ' '
}
RUNNING=" $(squeue -u "$USER" -h -o '%K' 2>/dev/null | tr '\n' ' ') "

for grid in "${!GRIDS[@]}"; do
  IFS=',' read -r -a LRS <<< "$grid"
  nlr=${#LRS[@]}; nseed=${#SEEDS[@]}
  total=$(( ${#CELLS[@]} * nlr * nseed ))
  missing=()
  for (( i=0; i<total; i++ )); do
    cell=${CELLS[$(( i / (nlr * nseed) ))]}
    lr=${LRS[$(( (i / nseed) % nlr ))]}
    seed=${SEEDS[$(( i % nseed ))]}
    lrtag=$(echo "$lr" | sed 's/0\.0*//')
    tag="HELMA_${RUNG}_${cell}_lr${lrtag}_s${seed}"
    out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cell}_scale5k_lr5/$tag"
    [ -f "$out/DONE" ] && continue
    # tasks submitted before the DONE marker existed only report completion in status.txt
    grep -q "$tag DONE" "$WS/logs/status.txt" 2>/dev/null && continue
    missing+=("$i")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "grid ${GRIDS[$grid]} ($grid): complete ($total/$total)"
    continue
  fi
  # drop indices that are still legitimately running
  todo=()
  for m in "${missing[@]}"; do
    case "$RUNNING" in *" $m "*) ;; *) todo+=("$m");; esac
  done
  echo "grid ${GRIDS[$grid]} ($grid): $(( total - ${#missing[@]} ))/$total done, ${#missing[@]} missing, ${#todo[@]} dead"
  if [ ${#todo[@]} -gt 0 ]; then
    list=$(IFS=,; echo "${todo[*]}")
    if [ "$APPLY" = "--apply" ]; then
      cd "$WS/MTR_with_poses" && sbatch --array="$list" slurm/10_scale_sweep.sh "$grid" \
        && echo "  resubmitted indices $list"
    else
      echo "  would resubmit indices $list (run with --apply)"
    fi
  fi
done
