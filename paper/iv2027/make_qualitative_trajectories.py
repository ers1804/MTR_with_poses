"""
Qualitative trajectory figure for the IV paper.

Three arms on the same scenes, same seed: no-pose / pose / oracle.  The point is
the contrast, not a per-scene win: the oracle bends the predicted modes onto the
true future through the *same* branch that carries pose, while the pose arm is
visually indistinguishable from no-pose.  That is the paper's mechanism -- the
pathway is live, the body does not supply the information.

Scene selection is on BASELINE DIFFICULTY (high no-pose minADE), never on the
treatment effect.  Selecting on the effect would cherry-pick from a null; this
selects where headroom exists and then shows what each arm does with it.

Usage:  python3 make_qualitative_trajectories.py [--seed 101] [--n 3]
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/home/erik/ssd2/gitprojects/MTR_with_poses"
ARMS = {
    "no pose":        "mtr+pose_data_no_pose_with_map_fixdata/MS_map_nopose_fix_s{s}",
    "shuffled pose":  "mtr+pose_data_cross_attn_pe_with_map_shufflepose_fixdata/MS_map_shufflepose_fix_s{s}",
    "correct pose":   "mtr+pose_data_cross_attn_pe_with_map_fixdata/MS_map_xattn_pe_fix_s{s}",
    "oracle":         "mtr+pose_data_cross_attn_pe_with_map_oracle_fixdata/MS_map_oracle_fix_s{s}",
}
ORDER = ["no pose", "shuffled pose", "correct pose", "oracle"]
COLOURS = {"no pose": "#8A8A8A", "shuffled pose": "#B9752F",
           "correct pose": "#D48A2A", "oracle": "#4F82BE"}
PAST_END, VALID_COL, XY = 11, 9, [0, 1]


def load(arm, seed):
    path = os.path.join(REPO, "output/waymo", ARMS[arm].format(s=seed),
                        "eval/eval_with_train/epoch_30/result.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    # one entry per scenario, each a LIST of predicted agents -- keep them all
    out = {}
    for item in data:
        for rec in (item if isinstance(item, list) else [item]):
            out[(str(rec["scenario_id"]), str(rec["object_id"]))] = rec
    return out


def min_ade(rec):
    gt = rec["gt_trajs"]
    valid = gt[PAST_END:, VALID_COL].astype(bool)
    if valid.sum() == 0:
        return np.nan
    fut = gt[PAST_END:, XY][valid]
    pred = rec["pred_trajs"][:, valid, :]
    return float(np.linalg.norm(pred - fut, axis=-1).mean(axis=-1).min())


def scene_bounds(recs):
    """One square view shared by every arm in a row: same framing, equal boxes."""
    pts = []
    for rec in recs:
        gt = rec["gt_trajs"]
        v = gt[PAST_END:, VALID_COL].astype(bool)
        pts.append(gt[:PAST_END, XY])
        pts.append(gt[PAST_END:, XY][v])
        pts.append(rec["pred_trajs"][:, v, :].reshape(-1, 2))
    pts = np.concatenate(pts, axis=0)
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    r = max((hi - lo).max() / 2, 2.0) * 1.12
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def panel(ax, rec, arm, show_title, row_label, xlim, ylim):
    gt = rec["gt_trajs"]
    valid = gt[PAST_END:, VALID_COL].astype(bool)
    past = gt[:PAST_END, XY]
    fut = gt[PAST_END:, XY][valid]
    c = COLOURS[arm]

    for k in range(rec["pred_trajs"].shape[0]):
        ax.plot(*rec["pred_trajs"][k][valid].T, color=c, alpha=0.30, lw=0.9, zorder=2)
    best = np.argmin(np.linalg.norm(
        rec["pred_trajs"][:, valid, :] - fut, axis=-1).mean(axis=-1))
    ax.plot(*rec["pred_trajs"][best][valid].T, color=c, lw=1.9, zorder=4)

    ax.plot(*past.T, color="0.45", lw=1.6, zorder=3)
    ax.plot(*fut.T, color="black", ls="--", lw=1.4, zorder=5)
    ax.plot(*past[-1], "o", color="black", ms=3.4, zorder=6)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("0.65")
    ax.text(0.03, 0.95, f"{min_ade(rec):.2f} m", transform=ax.transAxes,
            fontsize=7, va="top", ha="left", color=c, weight="bold")
    if show_title:
        ax.set_title(arm, fontsize=8.5, pad=3)
    if row_label:
        ax.set_ylabel(row_label, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="101")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "paper/iv2027/fig_qualitative_traj"))
    args = ap.parse_args()

    arms = {a: load(a, args.seed) for a in ARMS}
    keys = set.intersection(*[set(d) for d in arms.values()])
    print(f"seed {args.seed}: {len(keys)} agents common to all three arms")

    ade = {a: {k: min_ade(arms[a][k]) for k in keys} for a in ARMS}
    for a in ARMS:
        vals = np.array([ade[a][k] for k in keys])
        print(f"  {a:8s} mean minADE {np.nanmean(vals):.4f}")

    # Selection is on the BASELINE only, stratified across percentiles of
    # no-pose minADE.  Taking the worst-k instead lands on pathological agents
    # where nothing helps (and nothing is learnable); percentiles give scenes
    # with headroom that are still representative.
    def usable(k):
        if not all(np.isfinite(ade[a][k]) for a in ARMS):
            return False
        gt = arms["no pose"][k]["gt_trajs"]
        v = gt[PAST_END:, VALID_COL].astype(bool)
        if v.sum() < 80:                       # full-length futures only
            return False
        fut = gt[PAST_END:, XY][v]
        return np.linalg.norm(fut[-1] - fut[0]) > 3.0   # the agent actually moves

    pool = [k for k in keys if usable(k)]
    pool.sort(key=lambda k: ade["no pose"][k])
    pcts = np.linspace(60, 92, args.n)
    chosen, seen = [], set()
    for q in pcts:
        idx = int(round(q / 100 * (len(pool) - 1)))
        for off in range(len(pool)):           # walk out if scenario repeats
            for cand in (pool[min(idx + off, len(pool) - 1)],
                         pool[max(idx - off, 0)]):
                if cand not in chosen and cand[0] not in seen:
                    chosen.append(cand); seen.add(cand[0]); break
            else:
                continue
            break
    print(f"pool after filtering: {len(pool)} agents; percentiles {list(pcts.round(0))}")

    nrow, ncol = args.n, len(ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.62 * ncol, 1.62 * nrow))
    axes = np.atleast_2d(axes)
    for i, k in enumerate(chosen):
        xlim, ylim = scene_bounds([arms[a][k] for a in ORDER])
        for j, a in enumerate(ORDER):
            panel(axes[i, j], arms[a][k], a, show_title=(i == 0),
                  row_label=(f"scene {i+1}" if j == 0 else None),
                  xlim=xlim, ylim=ylim)

    handles = [Line2D([], [], color="0.45", lw=1.6, label="observed"),
               Line2D([], [], color="black", ls="--", lw=1.4, label="ground truth"),
               Line2D([], [], color="0.3", lw=1.9, label="best of 6 modes"),
               Line2D([], [], color="0.3", lw=0.9, alpha=0.4, label="other modes")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=220, bbox_inches="tight")
    print("wrote", args.out + ".pdf")

    print("\nchosen agents (selected on no-pose minADE only):")
    for i, k in enumerate(chosen):
        vals = "  ".join(f"{a} {ade[a][k]:.2f}" for a in ORDER)
        print(f"  scene {i+1}  {k[0][:14]}/{k[1][:8]}  {vals}")


if __name__ == "__main__":
    main()
