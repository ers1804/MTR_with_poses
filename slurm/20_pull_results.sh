#!/bin/bash
# Pull sweep results back from Helma and register the cells for hierarchical-bootstrap
# analysis. Run from the workstation, after the array finishes.
set -eu
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
REPO=/home/erik/ssd2/gitprojects/MTR_with_poses
RUNG=${RUNG:-25k}

echo "[1/3] syncing outputs (logs + metrics only; checkpoints are deleted cluster-side)"
rsync -avh --info=progress2 \
  --include '*/' --include 'log_train_*.txt' --include 'metrics_epoch_*.pkl' \
  --include 'best_eval_record.txt' --include 'DONE' --exclude '*' \
  helma:"$WS/MTR_with_poses/output/waymo/" "$REPO/output/waymo/"

echo "[2/3] runs pulled:"
find "$REPO/output/waymo" -maxdepth 2 -name "HELMA_${RUNG}_*" -type d | sed 's|.*/||' | sort

echo "[3/3] next: add the winning per-arm LR cells to tools/scripts/analyze_multiseed.py"
cat <<'NOTE'
  Cell naming from the sweep: HELMA_<rung>_<cell>_lr<lrtag>_s<seed>
  analyze_multiseed.py expects MS_<cell>_s<seed>; symlink or rename per-arm winners, e.g.
    for s in 101 202 303; do
      ln -s HELMA_25k_no_pose_with_map_lr5_s$s  MS_scale25k_nopose_s$s
    done
  Then register cells+pairs and run: python tools/scripts/analyze_multiseed.py
  Compare PER-ARM BEST across LRs — never a single shared LR (see slurm/README.md).
NOTE
