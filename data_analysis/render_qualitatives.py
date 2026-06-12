"""
Render qualitative trajectory visualizations for scenes where pose
conditioning (H3_geo_only) beats the no-pose baseline (H2_v2_real_traj).

Produces one PNG per scene under qualitatives/.
Layout: single panel with
  - past trajectory (gray)
  - GT future (black dashed)
  - baseline modes (red, faded) + best mode (red, solid)
  - pose modes (blue, faded) + best mode (blue, solid)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

POSE_PKL  = "output/cfgs/waymo/mtr+pose_data_geo_only/H3_geo_only/eval/eval_with_train/result.pkl"
NPOSE_PKL = "output/waymo/mtr+pose_data_no_pose/H2_v2_real_traj/eval/eval_with_train/result.pkl"
OUT_DIR   = "qualitatives"
PAST_END  = 11   # gt_trajs[0:11] = past, gt_trajs[11:] = future
VALID_COL = 9
XY        = [0, 1]

# ── scene lists by tier ─────────────────────────────────────────────────────

TIER1 = [  # near-perfect pose, >82% improvement
    ("790c12ffed169b7d", "268"),
    ("246caa1b0ffeca8d", "5001"),
    ("2963c3607b8019b9", "3249"),
    ("11a35dc852dd3ff9", "2790"),
    ("26597a5a95c142ef", "1549"),
    ("48a383ad74aa979c", "498"),
    ("b5e5ece83279cb91", "3446"),
    ("c54ec77eae49cc98", "1908"),
    ("43d8c298ce70093",  "4822"),
    ("729a979ba8cbe536", "4273"),
]

TIER2 = [  # large absolute Δ > 1.0m
    ("df23b6550b40a0b6", "2620"),
    ("e1d8ec4bf1152325", "5200"),
    ("b7d2dd91dcd8f0d",  "1524"),
    ("18c0f0fe66950a38", "4081"),
    ("87a9a5bb1fd7cd17", "36"),
    ("f445cb876402f3a4", "1154"),
    ("74925b002db2fce3", "710"),
    ("13ceed86f58effc3", "846"),
    ("5425f0c481d7d97c", "2840"),
    ("6786c97c43bc3189", "1243"),
    ("25def949c619625d", "3770"),
    ("11afdcdd1a5f91c4", "400"),
    ("d41b067dc64a3342", "3326"),
    ("bf929ba7ba2daa5a", "5419"),
    ("32c9cfa3b5c7080f", "2186"),
    ("b1528553c99957c9", "2491"),
    ("938996bf114271cf", "1542"),
    ("199cd31f14c89ee1", "2854"),
    ("fafa1b8fdeaef613", "676"),
    ("bbe9639bffe40da5", "6622"),
]

TIER3 = [  # 80-step, >60% relative improvement
    ("ab69a4132dbc3a47", "1756"),
    ("85d6ef715077c105", "4037"),
    ("497c59c0beee7859", "1474"),
    ("1c3ddd4b0e5af46d", "1357"),
    ("78ad573283fb5a49", "2670"),
    ("1441b2233c6e5146", "5363"),
    ("7d1368aa79ab7914", "4069"),
    ("c058d5c920a886c0", "1977"),
    ("7eb7d2549f248a83", "454"),
    ("7f1775686712791a", "433"),
    ("9d601141b0363feb", "3334"),
    ("18ac1d042cd1e81",  "4662"),
    ("64f73dc1cc0db186", "13"),
    ("fcd0472aa3fb5380", "75"),
    ("387bc7cfce10067d", "26"),
    ("864ba8308eeec24c", "2651"),
    ("effe653b8f2acd8d", "5682"),
    ("3b42cd06908e38c2", "2961"),
    ("97a0571d3800d4c0", "3476"),
    ("c215579c44e19ead", "867"),
]

SCENES = (
    [("tier1", s, o) for s, o in TIER1] +
    [("tier2", s, o) for s, o in TIER2] +
    [("tier3", s, o) for s, o in TIER3]
)


def load_flat(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return {
        (str(item[0]['scenario_id']), str(item[0]['object_id'])): item[0]
        for item in data
    }


def render_scene(ax, p_item, n_item, title, tag):
    gt        = p_item['gt_trajs']              # (91,10)
    past_xy   = gt[:PAST_END, XY]               # (11,2)
    past_valid= gt[:PAST_END, VALID_COL].astype(bool)
    fut_xy    = gt[PAST_END:, XY]               # (80,2)
    fut_valid = gt[PAST_END:, VALID_COL].astype(bool)

    # centre on last observed position
    origin = gt[PAST_END - 1, XY]
    past_c = past_xy - origin
    fut_c  = fut_xy  - origin

    p_pred   = p_item['pred_trajs'] - origin    # (6,80,2)
    n_pred   = n_item['pred_trajs'] - origin
    p_scores = p_item['pred_scores']            # (6,) — model confidence
    n_scores = n_item['pred_scores']

    top_p = int(np.argmax(p_scores))
    top_n = int(np.argmax(n_scores))

    # ── all modes faded, top-score mode opaque ───────────────────────────
    for k in range(6):
        vxy = n_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_n:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=1.0,
                    linewidth=1.6, zorder=4,
                    label='Baseline (no pose)')
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#d62728',
                    markersize=5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#d62728', alpha=0.18,
                    linewidth=0.8, zorder=2)

    for k in range(6):
        vxy = p_pred[k, fut_valid, :]
        if not len(vxy):
            continue
        if k == top_p:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=1.0,
                    linewidth=1.6, zorder=4,
                    label='GRU+geo (ours)')
            ax.plot(vxy[-1, 0], vxy[-1, 1], 'o', color='#1f77b4',
                    markersize=5, zorder=5)
        else:
            ax.plot(vxy[:, 0], vxy[:, 1], color='#1f77b4', alpha=0.18,
                    linewidth=0.8, zorder=2)

    # ── GT future ────────────────────────────────────────────────────────
    gt_fut_c = fut_c[fut_valid]
    if len(gt_fut_c):
        ax.plot(gt_fut_c[:, 0], gt_fut_c[:, 1], 'k--', linewidth=1.3,
                label='GT future', zorder=6)
        ax.plot(gt_fut_c[-1, 0], gt_fut_c[-1, 1], 'k*', markersize=8, zorder=7)

    # ── past trajectory ──────────────────────────────────────────────────
    past_c_v = past_c[past_valid]
    if len(past_c_v):
        ax.plot(past_c_v[:, 0], past_c_v[:, 1], color='#555555', linewidth=1.4,
                label='Past (obs.)', zorder=6)
        ax.plot(0, 0, 's', color='#555555', markersize=6, zorder=7,
                label='Pred. origin')

    # ── ADE annotations (minADE over all modes) ──────────────────────────
    def ade(pred, gt, valid):
        if valid.sum() == 0:
            return float('nan')
        return float(np.linalg.norm(pred[:, valid, :] - gt[valid, :], axis=-1).mean(axis=-1).min())
    p_ade = ade(p_pred, fut_c, fut_valid)
    n_ade = ade(n_pred, fut_c, fut_valid)
    delta = n_ade - p_ade
    ax.set_title(
        f"{title}\n"
        f"Pose ADE={p_ade:.3f}  Baseline ADE={n_ade:.3f}  Δ={delta:+.3f} ({delta/n_ade*100:+.1f}%)",
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

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 8,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    for tier, sid, oid in SCENES:
        key = (sid, oid)
        if key not in pose_dict or key not in npose_dict:
            print(f"  SKIP {sid}/{oid} — not found in pkl")
            continue

        p_item = pose_dict[key]
        n_item = npose_dict[key]

        fig, ax = plt.subplots(figsize=(5, 4.5))
        title = f"[{tier.upper()}] scene={sid[:12]}… | obj={oid}"
        render_scene(ax, p_item, n_item, title, tier)
        plt.tight_layout(pad=0.5)

        fname = f"{OUT_DIR}/{tier}_{sid}_obj{oid}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved {fname}")

    print(f"\nDone. {len(SCENES)} scenes rendered → {OUT_DIR}/")


if __name__ == "__main__":
    main()
