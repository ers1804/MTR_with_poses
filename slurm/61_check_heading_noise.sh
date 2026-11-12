#!/bin/bash
# Watchdog for the onboard-heading ladder (60_heading_noise.sh).
#
# The task grid is PARSED OUT of the sweep script rather than duplicated here, so the
# two cannot drift apart. A task is complete when its output dir has a DONE marker;
# anything missing whose array index is not currently queued or running counts as dead
# (node failure, timeout, preemption, OOM) and is resubmitted by index, so nothing
# already finished is recomputed. Safe to run repeatedly.
#
# Watch for the same index dying twice: after two resubmits, diagnose rather than
# spending more allocation.
#
# Usage: bash slurm/61_check_heading_noise.sh [--apply]
set -u
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
APPLY=${1:-}
SWEEP="$(dirname "$0")/60_heading_noise.sh"
# -r expands a pending range like [12-26%12] into one row per index; without it the
# pending tasks look dead and --apply would launch duplicates. Both modes index from 0,
# so the mode travels in the job name, mtr_hn_<mode>. Not the comment: Helma overwrites
# a running task's comment with its GPU bus id.
QUEUE=$(squeue -u "$USER" -h -r -o '%j %K' 2>/dev/null)

for mode in ladder lrcheck posladder standft prefit; do
  mapfile -t TASKS < <(sed -n "/^  ${mode}) TASKS=(/,/^  );/p" "$SWEEP" \
    | grep -oE '"[^"]+\|[^"]+\|[0-9]+"' | tr -d '"')
  if [ ${#TASKS[@]} -eq 0 ]; then
    echo "$mode: could not parse task list from $SWEEP" >&2; continue
  fi
  case "$mode" in
    ladder)    STATUS="$WS/logs/status_hn.txt";    PFX=HN;;
    lrcheck)   STATUS="$WS/logs/status_hn_lr.txt"; PFX=HN;;
    posladder) STATUS="$WS/logs/status_pn.txt";    PFX=PN;;
    standft)   STATUS="$WS/logs/status_ft.txt";    PFX=FT;;
    prefit)    STATUS="$WS/logs/status_pf.txt";    PFX=PF;;
  esac

  RUNNING=" $(awk -v m="mtr_hn_$mode" '$1 == m {print $2}' <<<"$QUEUE" | tr '\n' ' ') "
  missing=(); todo=()
  for i in "${!TASKS[@]}"; do
    spec=${TASKS[$i]}
    cfg=${spec%%|*}; rest=${spec#*|}; lr=${rest%%|*}; seed=${rest##*|}
    lrtag=$(echo "$lr" | sed 's/0\.0*//')
    tag="${PFX}_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
    out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cfg}/$tag"
    [ -f "$out/DONE" ] && continue
    grep -q "$tag DONE" "$STATUS" 2>/dev/null && continue
    missing+=("$i")
    case "$RUNNING" in *" $i "*) ;; *) todo+=("$i");; esac
  done
  echo "$mode: $(( ${#TASKS[@]} - ${#missing[@]} ))/${#TASKS[@]} done, ${#missing[@]} missing, ${#todo[@]} dead"

  if [ ${#todo[@]} -gt 0 ]; then
    list=$(IFS=,; echo "${todo[*]}")
    if [ "$APPLY" = "--apply" ]; then
      cd "$WS/MTR_with_poses" && sbatch --array="$list" --job-name="mtr_hn_$mode" slurm/60_heading_noise.sh "$mode" \
        && echo "  resubmitted $mode indices $list"
    else
      echo "  would resubmit $mode indices $list (run with --apply)"
    fi
  fi
done
