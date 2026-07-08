"""
Generate 4-panel qualitative figure for the H7-pretrain ablation:
H8 (geo+pose, H7 pretrain) vs H9 (no pose, H7 pretrain).

Output: fig_qualitative.pdf (single 4-panel figure for the paper).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POSE_PKL = os.path.join(REPO, "output/waymo/mtr+full_ped_finetune_geo/H8_finetune_geo/eval/eval_with_train/epoch_30/result.pkl")
NPOSE_PKL = os.path.join(REPO, "output/waymo/mtr+full_ped_finetune_no_pose/H9_finetune_no_pose/eval/eval_with_train/epoch_30/result.pkl")
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_qualitative.pdf")
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_qualitative.png")

PAST_END  = 11
VALID_COL = 9
XY        = [0, 1]

# Hand-picked scenes representing different failure modes of H9 that H8 fixes
SCENES = [
    ("80e6f4ea96bdc4d4", "695",  "Long-distance walker"),
    ("aac9a2adca4e5406", "880",  "Curved trajectory"),
    ("42e3bc41d3c0c1e2", "6163", "Mode coverage"),
    ("ef3aa4413133b53f", "451",  "Stationary"),
]


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


def render_panel(ax, p_item, n_item, title):
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

    # H9 modes (red)
    for k in range(n_pred.shape[0]):
        vxy = n_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_n:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=1.0, linewidth=1.8, zorder=4)
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#d62728', markersize=4.5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=0.20, linewidth=0.8, zorder=2)

    # H8 modes (blue)
    for k in range(p_pred.shape[0]):
        vxy = p_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_p:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=1.0, linewidth=1.8, zorder=4)
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#1f77b4', markersize=4.5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=0.20, linewidth=0.8, zorder=2)

    # GT future (black dashed)
    gt_fut_c = fut_c[fut_valid]
    if len(gt_fut_c):
        ax.plot(gt_fut_c[:, 0], gt_fut_c[:, 1], 'k--', linewidth=1.4, zorder=6)
        ax.plot(gt_fut_c[-1, 0], gt_fut_c[-1, 1], 'k*', markersize=8, zorder=7)

    # Past (gray)
    past_c_v = past_c[past_valid]
    if len(past_c_v):
        ax.plot(past_c_v[:, 0], past_c_v[:, 1], color='#555555', linewidth=1.6, zorder=6)
        ax.plot(0, 0, 's', color='#555555', markersize=5, zorder=7)

    p_ade = min_ade(p_pred, fut_c, fut_valid)
    n_ade = min_ade(n_pred, fut_c, fut_valid)
    delta_pct = (n_ade - p_ade) / n_ade * 100

    ax.set_title(
        f"{title}\n"
        f"pose ADE={p_ade:.2f}  no-pose ADE={n_ade:.2f}  ($-${delta_pct:.0f}%)",
        fontsize=8, pad=2
    )
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)


def main():
    print("Loading pkl files...")
    pose_dict = load_flat(POSE_PKL)
    npose_dict = load_flat(NPOSE_PKL)
    print(f"  H8 (pose):     {len(pose_dict)} items")
    print(f"  H9 (no pose):  {len(npose_dict)} items")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))

    for ax, (sid, oid, label) in zip(axes, SCENES):
        key = (sid, oid)
        assert key in pose_dict and key in npose_dict, f"Missing scene {sid}/{oid}"
        render_panel(ax, pose_dict[key], npose_dict[key], label)

    # Shared legend on top
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#555555', linewidth=1.6, label='Past'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.4, label='GT future'),
        Line2D([0], [0], color='#d62728', linewidth=1.8, label='no-pose (top mode)'),
        Line2D([0], [0], color='#1f77b4', linewidth=1.8, label='pose (top mode)'),
    ]
    fig.legend(handles=legend_elems, loc='lower center',
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01),
               frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT_PDF, bbox_inches='tight')
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
