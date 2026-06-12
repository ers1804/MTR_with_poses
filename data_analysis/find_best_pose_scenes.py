"""
Find scenes where pose conditioning (H3_geo_only) improves trajectory
prediction the most vs the no-pose baseline (H2_v2_real_traj).

Outputs a ranked CSV and summary list for qualitative visualization.
"""

import pickle
import numpy as np
import csv
import os

POSE_PKL   = "output/cfgs/waymo/mtr+pose_data_geo_only/H3_geo_only/eval/eval_with_train/result.pkl"
NPOSE_PKL  = "output/waymo/mtr+pose_data_no_pose/H2_v2_real_traj/eval/eval_with_train/result.pkl"
OUT_CSV    = "data_analysis/scene_pose_improvement.csv"
OUT_LIST   = "data_analysis/top_improvement_scenes.txt"

FUTURE_START = 11   # gt_trajs[11:] = future 80 steps
VALID_COL    = 9    # column 9 is the valid flag in gt_trajs
XY_COLS      = [0, 1]  # x, y positions


def compute_min_ade(pred_trajs, gt_xy, valid_mask):
    """
    pred_trajs: (K, T, 2)  K modes, T future steps, xy
    gt_xy:      (T, 2)
    valid_mask: (T,) bool
    Returns scalar minADE over valid future steps, or nan if no valid steps.
    """
    if valid_mask.sum() == 0:
        return np.nan
    # displacement per mode per step
    diff = pred_trajs[:, valid_mask, :] - gt_xy[valid_mask, :]  # (K, V, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (K, V)
    ade_per_mode = dist.mean(axis=-1)     # (K,)
    return ade_per_mode.min()


def compute_min_fde(pred_trajs, gt_xy, valid_mask):
    """Final displacement error at the last valid future step."""
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.nan
    last_idx = valid_indices[-1]
    diff = pred_trajs[:, last_idx, :] - gt_xy[last_idx, :]  # (K, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (K,)
    return dist.min()


def load_flat(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return {
        (str(item[0]['scenario_id']), str(item[0]['object_id'])): item[0]
        for item in data
    }


def main():
    print("Loading pkl files...")
    pose_dict  = load_flat(POSE_PKL)
    npose_dict = load_flat(NPOSE_PKL)

    keys = sorted(pose_dict.keys())
    print(f"Matched agents: {len(keys)}")

    rows = []
    for key in keys:
        p = pose_dict[key]
        n = npose_dict[key]

        scenario_id = str(p['scenario_id'])
        object_id   = str(p['object_id'])

        gt       = p['gt_trajs']               # (91, 10)
        gt_future = gt[FUTURE_START:, :]       # (80, 10)
        gt_xy     = gt_future[:, XY_COLS]      # (80, 2)
        valid     = gt_future[:, VALID_COL].astype(bool)  # (80,)
        n_valid   = valid.sum()

        pose_ade  = compute_min_ade(p['pred_trajs'], gt_xy, valid)
        npose_ade = compute_min_ade(n['pred_trajs'], gt_xy, valid)

        pose_fde  = compute_min_fde(p['pred_trajs'], gt_xy, valid)
        npose_fde = compute_min_fde(n['pred_trajs'], gt_xy, valid)

        if np.isnan(pose_ade) or np.isnan(npose_ade):
            continue

        ade_delta     = npose_ade - pose_ade       # positive = pose is better
        ade_delta_rel = ade_delta / npose_ade * 100

        fde_delta     = npose_fde - pose_fde
        fde_delta_rel = fde_delta / npose_fde * 100 if npose_fde > 0 else 0.0

        rows.append({
            'scenario_id':   scenario_id,
            'object_id':     object_id,
            'n_valid_steps': int(n_valid),
            'pose_minADE':   pose_ade,
            'npose_minADE':  npose_ade,
            'ade_delta':     ade_delta,       # pose - npose: positive = improvement
            'ade_delta_pct': ade_delta_rel,
            'pose_minFDE':   pose_fde,
            'npose_minFDE':  npose_fde,
            'fde_delta':     fde_delta,
            'fde_delta_pct': fde_delta_rel,
        })

    # Sort by ADE improvement (largest first)
    rows.sort(key=lambda r: r['ade_delta'], reverse=True)

    print(f"\nTotal valid agents: {len(rows)}")
    print(f"Agents where pose improves (ade_delta > 0): {sum(1 for r in rows if r['ade_delta'] > 0)}")
    print(f"Agents where pose hurts (ade_delta < 0): {sum(1 for r in rows if r['ade_delta'] < 0)}")

    # Overall stats
    deltas = np.array([r['ade_delta'] for r in rows])
    print(f"\nADE delta stats:")
    print(f"  Mean:   {deltas.mean():.4f}")
    print(f"  Median: {np.median(deltas):.4f}")
    print(f"  P75:    {np.percentile(deltas, 75):.4f}")
    print(f"  P90:    {np.percentile(deltas, 90):.4f}")
    print(f"  P95:    {np.percentile(deltas, 95):.4f}")
    print(f"  Max:    {deltas.max():.4f}")
    print(f"  Min:    {deltas.min():.4f}")

    # Write CSV
    os.makedirs("data_analysis", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nWrote {OUT_CSV}")

    # Write top-50 scene list
    top_n = 50
    with open(OUT_LIST, 'w') as f:
        f.write("# Top scenes by pose improvement (minADE, H3_geo_only vs H2_v2_real_traj)\n")
        f.write("# Format: rank | scenario_id | object_id | pose_ADE | npose_ADE | delta | delta_pct | valid_steps\n\n")
        for i, r in enumerate(rows[:top_n]):
            f.write(
                f"{i+1:3d} | {r['scenario_id']:<20} | obj={r['object_id']:<6} | "
                f"pose={r['pose_minADE']:.4f} | base={r['npose_minADE']:.4f} | "
                f"Δ={r['ade_delta']:+.4f} ({r['ade_delta_pct']:+.1f}%) | "
                f"steps={r['n_valid_steps']}\n"
            )
        f.write(f"\n# Bottom 20 (pose HURTS most)\n")
        for i, r in enumerate(rows[-20:]):
            f.write(
                f"  {len(rows)-19+i:3d} | {r['scenario_id']:<20} | obj={r['object_id']:<6} | "
                f"pose={r['pose_minADE']:.4f} | base={r['npose_minADE']:.4f} | "
                f"Δ={r['ade_delta']:+.4f} ({r['ade_delta_pct']:+.1f}%)\n"
            )
    print(f"Wrote {OUT_LIST}")

    # Print top 20 to console
    print(f"\nTop 20 agents by ADE improvement:")
    print(f"{'Rank':4} {'scenario_id':20} {'obj':6} {'pose_ADE':9} {'base_ADE':9} {'Δ':8} {'Δ%':7} {'steps':5}")
    print("-" * 85)
    for i, r in enumerate(rows[:20]):
        print(
            f"{i+1:4d} {r['scenario_id']:20s} {r['object_id']:6s} "
            f"{r['pose_minADE']:9.4f} {r['npose_minADE']:9.4f} "
            f"{r['ade_delta']:+8.4f} {r['ade_delta_pct']:+7.1f}% "
            f"{r['n_valid_steps']:5d}"
        )


if __name__ == "__main__":
    main()
