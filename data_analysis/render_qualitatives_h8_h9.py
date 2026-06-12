"""
Discover best scenes and render qualitative trajectory visualizations for
the H7-pretrain ablation: H8 (geo+pose+map, H7 fine-tune) vs H9 (no pose+map,
H7 fine-tune).

Same backbone (H7 pretrain on full Waymo), same data (579-scene subset),
isolates the contribution of the pose encoder against the strong trajectory
prior from H7. Both runs use the residual zero-init pose_fuser path
(H9 has USE_POSE_ENCODER=False; H8 has it True).

Discovers tiers automatically by computing per-scene minADE improvement,
then renders one PNG per scene under qualitatives_h8_h9/.

Layout: single panel with
  - past trajectory (gray)
  - GT future (black dashed)
  - H9 no-pose modes (red, faded) + top mode (red, solid)
  - H8 geo+pose modes (blue, faded) + top mode (blue, solid)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POSE_PKL  = "output/waymo/mtr+full_ped_finetune_geo/H8_finetune_geo/eval/eval_with_train/epoch_30/result.pkl"
NPOSE_PKL = "output/waymo/mtr+full_ped_finetune_no_pose/H9_finetune_no_pose/eval/eval_with_train/epoch_30/result.pkl"
OUT_DIR   = "qualitatives_h8_h9"
PAST_END  = 11
VALID_COL = 9
XY        = [0, 1]


def min_ade(pred, gt, valid):
    if valid.sum() == 0:
        return float('nan')
    return float(np.linalg.norm(
        pred[:, valid, :] - gt[valid, :], axis=-1
    ).mean(axis=-1).min())


def load_flat(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return {
        (str(item[0]['scenario_id']), str(item[0]['object_id'])): item[0]
        for item in data
    }


def discover_scenes(pose_dict, npose_dict):
    common = set(pose_dict.keys()) & set(npose_dict.keys())
    records = []
    for key in common:
        sid, oid = key
        p_item = pose_dict[key]
        n_item = npose_dict[key]

        gt        = p_item['gt_trajs']
        fut_valid = gt[PAST_END:, VALID_COL].astype(bool)
        fut_xy    = gt[PAST_END:, XY]
        origin    = gt[PAST_END - 1, XY]
        fut_c     = fut_xy - origin

        p_pred = p_item['pred_trajs'] - origin
        n_pred = n_item['pred_trajs'] - origin

        p_ade = min_ade(p_pred, fut_c, fut_valid)
        n_ade = min_ade(n_pred, fut_c, fut_valid)

        if np.isnan(p_ade) or np.isnan(n_ade) or n_ade == 0:
            continue

        abs_imp = n_ade - p_ade
        rel_imp = abs_imp / n_ade

        records.append((sid, oid, rel_imp, abs_imp, p_ade, n_ade))

    records.sort(key=lambda r: r[2], reverse=True)

    tier1, tier2, tier3 = [], [], []
    tier2_seen, tier3_seen = set(), set()
    for sid, oid, rel, absi, p_ade, n_ade in records:
        k = (sid, oid)
        if rel >= 0.50 and absi >= 0.10:
            tier1.append((sid, oid, rel, absi))
        elif absi >= 0.30 and k not in tier2_seen:
            tier2.append((sid, oid, rel, absi))
            tier2_seen.add(k)
        elif rel >= 0.20 and absi >= 0.05 and k not in tier2_seen and k not in tier3_seen:
            tier3.append((sid, oid, rel, absi))
            tier3_seen.add(k)

    tier1 = tier1[:12]
    tier2 = tier2[:20]
    tier3 = tier3[:20]

    print(f"\n=== Tier 1 (rel >=50%, top {len(tier1)}) ===")
    for s, o, r, a in tier1:
        print(f"  ({s!r:30s}, {o!r:6s})  rel={r*100:+.1f}%  abs={a:+.4f}")
    print(f"\n=== Tier 2 (|abs| >=0.30 m, top {len(tier2)}) ===")
    for s, o, r, a in tier2:
        print(f"  ({s!r:30s}, {o!r:6s})  rel={r*100:+.1f}%  abs={a:+.4f}")
    print(f"\n=== Tier 3 (rel >=20%, top {len(tier3)}) ===")
    for s, o, r, a in tier3:
        print(f"  ({s!r:30s}, {o!r:6s})  rel={r*100:+.1f}%  abs={a:+.4f}")

    scenes = (
        [("tier1", s, o) for s, o, *_ in tier1] +
        [("tier2", s, o) for s, o, *_ in tier2] +
        [("tier3", s, o) for s, o, *_ in tier3]
    )
    return scenes


def render_scene(ax, p_item, n_item, title):
    gt         = p_item['gt_trajs']
    past_xy    = gt[:PAST_END, XY]
    past_valid = gt[:PAST_END, VALID_COL].astype(bool)
    fut_xy     = gt[PAST_END:, XY]
    fut_valid  = gt[PAST_END:, VALID_COL].astype(bool)

    origin = gt[PAST_END - 1, XY]
    past_c = past_xy - origin
    fut_c  = fut_xy  - origin

    p_pred   = p_item['pred_trajs'] - origin
    n_pred   = n_item['pred_trajs'] - origin
    p_scores = p_item['pred_scores']
    n_scores = n_item['pred_scores']

    top_p = int(np.argmax(p_scores))
    top_n = int(np.argmax(n_scores))

    for k in range(n_pred.shape[0]):
        vxy = n_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_n:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=1.0,
                    linewidth=1.6, zorder=4, label='H9 no-pose (H7 finetune)')
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#d62728',
                    markersize=5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=0.18,
                    linewidth=0.8, zorder=2)

    for k in range(p_pred.shape[0]):
        vxy = p_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_p:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=1.0,
                    linewidth=1.6, zorder=4, label='H8 geo+pose (H7 finetune)')
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#1f77b4',
                    markersize=5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=0.18,
                    linewidth=0.8, zorder=2)

    gt_fut_c = fut_c[fut_valid]
    if len(gt_fut_c):
        ax.plot(gt_fut_c[:, 0], gt_fut_c[:, 1], 'k--', linewidth=1.3,
                label='GT future', zorder=6)
        ax.plot(gt_fut_c[-1, 0], gt_fut_c[-1, 1], 'k*', markersize=8, zorder=7)

    past_c_v = past_c[past_valid]
    if len(past_c_v):
        ax.plot(past_c_v[:, 0], past_c_v[:, 1], color='#555555', linewidth=1.4,
                label='Past (obs.)', zorder=6)
        ax.plot(0, 0, 's', color='#555555', markersize=6, zorder=7,
                label='Pred. origin')

    p_ade = min_ade(p_pred, fut_c, fut_valid)
    n_ade = min_ade(n_pred, fut_c, fut_valid)
    delta = n_ade - p_ade
    ax.set_title(
        f"{title}\n"
        f"H8 ADE={p_ade:.3f}  H9 ADE={n_ade:.3f}  "
        f"D={delta:+.3f} ({delta/n_ade*100:+.1f}%)",
        fontsize=7, pad=3
    )

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=7)
    ax.set_ylabel('y (m)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='best', framealpha=0.6)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading pkl files...")
    pose_dict  = load_flat(POSE_PKL)
    npose_dict = load_flat(NPOSE_PKL)
    print(f"  H8 (geo+pose):  {len(pose_dict)} items")
    print(f"  H9 (no pose):   {len(npose_dict)} items")

    scenes = discover_scenes(pose_dict, npose_dict)
    print(f"\nTotal scenes to render: {len(scenes)}")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 8,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    rendered = 0
    for tier, sid, oid in scenes:
        key = (sid, oid)
        if key not in pose_dict or key not in npose_dict:
            print(f"  SKIP {sid}/{oid} - not in both pkls")
            continue

        fig, ax = plt.subplots(figsize=(5, 4.5))
        title = f"[{tier.upper()}] scene={sid[:12]} | obj={oid}"
        render_scene(ax, pose_dict[key], npose_dict[key], title)
        plt.tight_layout(pad=0.5)

        fname = f"{OUT_DIR}/{tier}_{sid}_obj{oid}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        rendered += 1
        if rendered % 10 == 0:
            print(f"  {rendered}/{len(scenes)} rendered...")

    print(f"\nDone. {rendered} scenes -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
