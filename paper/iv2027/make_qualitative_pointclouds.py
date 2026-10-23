"""
Point cloud vs fitted SMPL mesh, as a function of range.

Shows what the pose estimator actually has to work with.  Each panel overlays
the raw per-pedestrian LiDAR returns on the mesh the pipeline fitted to them.
Near the sensor the points outline a body; by 40-60 m there are a few dozen
returns, and the articulated limb configuration drawn over them is supplied by
the AMASS motion prior rather than measured.  That is the physical reason the
body-facing estimate sits far below the detection floor of Sec. IV.

Run with an env that has smplx, e.g.
  /home/erik/anaconda3/envs/lidar-hmr/bin/python make_qualitative_pointclouds.py
"""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
import smplx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

WORK = "/home/erik/big1/mpjpe_eval"
PRED = f"{WORK}/amass/raw/validation"
EXTR = f"{WORK}/extracted/validation"
SMPL_DIR = "/home/erik/ssd2/Waymo-3DSkelMo/LiDAR-HMR/smplx_models"
BANDS = [(6, 14), (18, 28), (33, 45), (50, 70)]   # metres
# Median returns per pedestrian measured over the validation segments, used so
# each panel shows a TYPICAL frame for its band rather than the densest one.
TARGET = {(6, 14): 300, (18, 28): 100, (33, 45): 38, (50, 70): 12}


def mesh_for(model, d, j):
    with torch.no_grad():
        o = model(betas=torch.tensor(d["betas"][j:j + 1], dtype=torch.float32),
                  body_pose=torch.tensor(d["pose_body"][j:j + 1], dtype=torch.float32),
                  global_orient=torch.tensor(d["root_orient"][j:j + 1], dtype=torch.float32),
                  transl=torch.tensor(d["trans"][j:j + 1], dtype=torch.float32))
    return o.vertices[0].numpy()


def collect(model, max_per_band=6):
    """One good example per range band: most points available in that band."""
    best = {b: None for b in BANDS}
    for f in sorted(glob.glob(f"{PRED}/*/*.npz")):
        ctx, oid = f.split("/")[-2], os.path.basename(f)[:-4]
        seg = f"{EXTR}/{ctx}.pkl"
        if not os.path.exists(seg):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            if "smpl_timestamps" not in d:
                continue
            recs = pickle.load(open(seg, "rb"))
        except Exception:
            continue
        mine = [r for r in recs if r["basic_info"]["object_id"] == oid]
        if not mine:
            continue
        smt = d["smpl_timestamps"].astype(np.int64)
        for r in mine:
            P = r["world_frame"]["pedestrian_pcd"]
            n = len(P)
            if n < 8:
                continue
            rng = float(np.linalg.norm(r["vehicle_frame"]["center"][:2]))
            band = next((b for b in BANDS if b[0] <= rng < b[1]), None)
            if band is None:
                continue
            ts = int(r["basic_info"]["time_stamp"])
            j = int(np.argmin(np.abs(smt - ts)))
            if abs(int(smt[j]) - ts) > 60_000:
                continue
            # reject frames where the returns cover only part of the body
            # (heavy occlusion); they say nothing about what range costs
            if float(P[:, 2].ptp()) < 1.2:
                continue
            h = np.asarray(r["world_frame"]["heading"], float).ravel()
            yaw = float(np.arctan2(h[1], h[0])) if h.size >= 2 else 0.0
            cost = abs(n - TARGET[band])
            cur = best[band]
            if cur is None or cost < cur[0]:
                best[band] = (cost, f, j, P, rng, n, yaw)
    return best


def draw(ax, V, faces, P, rng, n, yaw):
    """Body-centred frame, widest horizontal axis across the view, so every
    panel shows the same aspect regardless of which way the person walked."""
    c = V.mean(0)
    Vc, Pc = V - c, P - c

    # principal horizontal axis of the mesh = shoulder span; rotate it onto y
    H = Vc[:, :2]
    w, Q = np.linalg.eigh(H.T @ H)
    ax_major = Q[:, int(np.argmax(w))]
    ang = np.arctan2(ax_major[1], ax_major[0]) - np.pi / 2
    ca, sa = np.cos(-ang), np.sin(-ang)
    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Vc, Pc = Vc @ R.T, Pc @ R.T

    tri = Poly3DCollection(Vc[faces], alpha=0.55, linewidths=0)
    tri.set_facecolor("#A8C0DB")
    tri.set_edgecolor("none")
    ax.add_collection3d(tri)
    ax.scatter(Pc[:, 0], Pc[:, 1], Pc[:, 2], s=6.5, c="#C2410C",
               depthshade=False, edgecolors="none", zorder=10)

    # fixed metric scale in every panel: the bodies are the same size, only the
    # number of returns changes, which is the whole point of the figure
    rh, zh = 0.60, 1.05
    zc = (Vc[:, 2].max() + Vc[:, 2].min()) / 2
    ax.set_xlim(-rh, rh); ax.set_ylim(-rh, rh)
    ax.set_zlim(zc - zh, zc + zh)
    ax.set_box_aspect((1, 1, zh / rh))
    ax.view_init(elev=6, azim=-88)
    ax.set_axis_off()
    ax.set_title(f"{rng:.0f} m\n{n} points", fontsize=8.5, pad=-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "fig_pointcloud_mesh"))
    args = ap.parse_args()

    model = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral")
    faces = model.faces.astype(np.int64)

    best = collect(model)
    chosen = [(b, v) for b, v in best.items() if v is not None]
    print(f"bands filled: {len(chosen)}/{len(BANDS)}")

    fig = plt.figure(figsize=(1.35 * len(chosen), 2.9))
    for i, (band, (_, f, j, P, rng, n, yaw)) in enumerate(chosen):
        d = np.load(f, allow_pickle=True)
        V = mesh_for(model, d, j)
        ax = fig.add_subplot(1, len(chosen), i + 1, projection="3d")
        draw(ax, V, faces, P, rng, n, yaw)
        print(f"  band {band}: range {rng:.1f} m, {n} points, {os.path.basename(f)}")

    fig.subplots_adjust(left=0, right=1, top=1.02, bottom=0, wspace=-0.08)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=240, bbox_inches="tight")
    print("wrote", args.out + ".pdf")


if __name__ == "__main__":
    main()
