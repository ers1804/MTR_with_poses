#!/usr/bin/env python
"""Does SMPL body facing beat the free box heading WHERE THE TRACKER DEGRADES?

On average it does not (25.2 vs 44.2 deg median error at H=3s). This asks whether the box
heading's advantage is uniform, or whether it collapses in hard perception conditions while
pose holds up -- which would be a real operating regime for pose, and a positive result.

Stratifies the same paired comparison by range to ego, history validity (occlusion proxy),
and box-heading jitter, and reports the paired pose-minus-box gap in every stratum. The
quantity that matters is the GAP: negative means pose wins there.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from scipy import stats

POSE = "/home/erik/big1/waymo_motion_poses/final_10fps"


def aa_to_R(aa):
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def angdiff(a, b):
    return np.degrees(np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="experiments/onset_events_validation.pkl")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--horizon", type=float, default=3.0)
    a = ap.parse_args()

    rows = pickle.load(open(a.events, "rb"))
    pos = [r for r in rows if r[f"onset_{a.horizon:g}s"] and np.isfinite(r["onset_dir"])
           and r.get("has_pose") and "range_ego" in r]
    print(f"positive onset events with pose + covariates: {len(pos)}")

    R, keep = [], []
    for r in pos:
        try:
            ro = np.load(os.path.join(POSE, a.split, r["scene"], f"{r['object_id']}.npz"))["root_orient"]
        except Exception:
            continue
        if ro.ndim != 2 or not len(ro):
            continue
        R.append(aa_to_R(np.asarray(ro[-1], dtype=np.float64)))
        keep.append(r)
    R = np.stack(R)
    truth = np.array([r["onset_dir"] for r in keep])
    box = np.array([r["heading"] for r in keep])
    print(f"pose loaded for {len(keep)}")

    best = None
    for ax in range(3):
        for sg in (1, -1):
            v = sg * R[:, :, ax]
            yaw = np.arctan2(v[:, 1], v[:, 0])
            for off in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                m = float(np.median(angdiff(yaw + off, box)))
                if best is None or m < best[0]:
                    best = (m, yaw + off)
    pose_yaw = best[1]
    e_pose, e_box = angdiff(pose_yaw, truth), angdiff(box, truth)

    print("\nOVERALL   pose %.1f deg | box %.1f deg | gap %+.1f deg"
          % (np.median(e_pose), np.median(e_box), np.median(e_pose) - np.median(e_box)))

    strata = {
        "range to ego (m)":        np.array([r["range_ego"] for r in keep]),
        "history valid fraction":  np.array([r["hist_valid_frac"] for r in keep]),
        "box heading jitter (deg)": np.array([r["heading_std"] for r in keep]),
    }
    for name, v in strata.items():
        ok = np.isfinite(v)
        qs = np.nanquantile(v[ok], [0, .25, .5, .75, 1.0])
        print("\n" + "=" * 78)
        print(f"stratified by {name}   (quartiles {np.round(qs,2)})")
        print("=" * 78)
        print(f"  {'stratum':<26}{'n':>6}{'pose':>8}{'box':>8}{'gap':>9}{'p':>10}  pose wins")
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            m = ok & (v >= lo) & ((v <= hi) if i == 3 else (v < hi))
            if m.sum() < 25:
                print(f"  Q{i+1} [{lo:.2f},{hi:.2f})  n={int(m.sum())} too few"); continue
            ep, eb = e_pose[m], e_box[m]
            gap = np.median(ep) - np.median(eb)
            p = stats.wilcoxon(ep, eb).pvalue if m.sum() > 10 else np.nan
            tag = "  <-- POSE WINS" if gap < 0 and p < 0.05 else ""
            print(f"  Q{i+1} [{lo:7.2f},{hi:7.2f}){int(m.sum()):>6}{np.median(ep):>8.1f}"
                  f"{np.median(eb):>8.1f}{gap:>+9.1f}{p:>10.2g}  {100*np.mean(ep<eb):>4.0f}%{tag}")
    print("\nGap is pose-minus-box median error: NEGATIVE means SMPL beats the free heading there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
