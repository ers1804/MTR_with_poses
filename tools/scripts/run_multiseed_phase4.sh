#!/bin/bash
# Phase 4 experiments (action plan): the stable-winner encoder (xattn+PE) in the
# realistic conditions it was never run in, plus the root-orientation ablation.
#   map_xattn_pe   xattn+PE + HD map (from scratch)      -> pairs with map_nopose
#   ft_xattn_pe    xattn+PE + map + H7 pretrain finetune -> pairs with ft_nopose
#   norootorient   xattn+PE, root-orientation channel zeroed (circularity control)
# 3 cells x 3 seeds (101/202/303), identical protocol to run_multiseed_matrix.sh.
set -u
cd "$(dirname "$0")/.."   # tools/

PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
H7_CKPT=../output/waymo/mtr+full_ped_pretrain/H7_full_pretrain/ckpt/checkpoint_epoch_30.pth
SEEDS="101 202 303"
MASTER_LOG=/tmp/multiseed_phase4.log

# cell_name:cfg_file:pretrained(0/1)
CELLS=(
  "map_xattn_pe:mtr+pose_data_cross_attn_pe_with_map:0"
  "ft_xattn_pe:mtr+full_ped_finetune_xattn_pe:1"
  "norootorient:mtr+pose_data_cross_attn_pe_norootorient:0"
)

echo "=== phase4 started $(date) ===" >> "$MASTER_LOG"
for cell in "${CELLS[@]}"; do
  name="${cell%%:*}"
  rest="${cell#*:}"
  cfg="${rest%%:*}"
  pretrained="${rest##*:}"
  for seed in $SEEDS; do
    tag="MS_${name}_s${seed}"
    outdir="../output/waymo/${cfg}/${tag}"
    if [ -f "${outdir}/DONE" ]; then
      echo "skip ${tag} (already done)" >> "$MASTER_LOG"
      continue
    fi
    echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
    extra_args=""
    if [ "$pretrained" = "1" ]; then
      extra_args="--pretrained_model ${H7_CKPT}"
    fi
    WANDB_MODE=offline "$PY" train.py \
      --cfg_file "cfgs/waymo/${cfg}.yaml" \
      --extra_tag "${tag}" \
      --random_seed "${seed}" \
      --workers 8 \
      --max_ckpt_save_num 1 \
      ${extra_args} \
      > "/tmp/${tag}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "${outdir}/DONE"
      rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
      best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
      echo "OK  ${tag} rc=0 best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
    else
      echo "FAIL ${tag} rc=${rc} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
    fi
  done
done
echo "=== phase4 finished $(date) ===" >> "$MASTER_LOG"
