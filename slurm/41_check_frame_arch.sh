#!/bin/bash
# Watchdog for the frame/architecture waves (40_frame_arch_sweep.sh).
#
# The task grid is PARSED OUT of the sweep script rather than duplicated here, so the two can
# never drift apart. A task is complete when its output dir has a DONE marker; anything missing
# whose array index is not currently queued/running is resubmitted for that wave.
#
# Usage: bash slurm/41_check_frame_arch.sh [--apply]
set -u
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
APPLY=${1:-}
SWEEP="$(dirname "$0")/40_frame_arch_sweep.sh"
RUNNING=" $(squeue -u "$USER" -h -o '%K' 2>/dev/null | tr '\n' ' ') "

for wave in wave1 wave2; do
  mapfile -t TASKS < <(sed -n "/^  ${wave}) TASKS=(/,/^  );;/p" "$SWEEP" | grep -oE '"[^"]+\|[^"]+\|[0-9]+"' | tr -d '"')
  if [ ${#TASKS[@]} -eq 0 ]; then echo "$wave: could not parse task list from $SWEEP" >&2; continue; fi
  missing=(); todo=()
  for i in "${!TASKS[@]}"; do
    spec=${TASKS[$i]}
    cfg=${spec%%|*}; rest=${spec#*|}; lr=${rest%%|*}; seed=${rest##*|}
    lrtag=$(echo "$lr" | sed 's/0\.0*//')
    tag="FA_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
    out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cfg}/$tag"
    [ -f "$out/DONE" ] && continue
    grep -q "$tag DONE" "$WS/logs/status_fa.txt" 2>/dev/null && continue
    missing+=("$i")
    case "$RUNNING" in *" $i "*) ;; *) todo+=("$i");; esac
  done
  echo "$wave: $(( ${#TASKS[@]} - ${#missing[@]} ))/${#TASKS[@]} done, ${#missing[@]} missing, ${#todo[@]} dead"
  if [ ${#todo[@]} -gt 0 ]; then
    list=$(IFS=,; echo "${todo[*]}")
    if [ "$APPLY" = "--apply" ]; then
      cd "$WS/MTR_with_poses" && sbatch --array="$list" slurm/40_frame_arch_sweep.sh "$wave" \
        && echo "  resubmitted $wave indices $list"
    else
      echo "  would resubmit $wave indices $list (run with --apply)"
    fi
  fi
done
