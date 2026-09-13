#!/usr/bin/env python
"""Does SMPL body facing beat the FREE Waymo box heading at predicting onset direction?

This is the decisive test for the onset-direction task. The Waymo box heading is already a
standard WOMD channel and MTR already consumes it as sin/cos (waymo_pose_dataset.py:1104).
So SMPL pose is only worth anything here if it beats that, not if it beats chance.

SMPL root_orient is stored as axis-angle in WORLD frame. The mapping from the rotation to a
ground-plane "facing" yaw depends on the export's axis convention, so the convention is
CALIBRATED against the box heading (choosing the forward axis and sign that best aligns),
then the comparison is made. Calibrating on the box heading is deliberately generous to the
pose side: it hands SMPL the best possible frame before the contest starts.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

POSE = "/home/erik/big1/waymo_motion_poses/final_10fps"


def aa_to_R(aa: np.ndarray) -> np.ndarray:
    """Axis-angle -> rotation matrix (Rodrigues)."""
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
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()

    rows = pickle.load(open(a.events, "rb"))
    seen, uniq = set(), []
    for r in rows:
        k = (r["scene"], r["object_id"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    pos = [r for r in rows
           if r[f"onset_{a.horizon:g}s"] and np.isfinite(r["onset_dir"]) and r["has_pose"]]
    if a.limit:
        pos = pos[:a.limit]
    print(f"positive onset events (H={a.horizon:g}s) with pose: {len(pos)}")

    # load SMPL root orientation at the last observed frame
    R_last, keep = [], []
    for r in pos:
        p = os.path.join(POSE, a.split, r["scene"], f"{r['object_id']}.npz")
        try:
            d = np.load(p)
            ro = d["root_orient"]
        except Exception:
            continue
        if ro.ndim != 2 or len(ro) == 0:
            continue
        R_last.append(aa_to_R(np.asarray(ro[-1], dtype=np.float64)))
        keep.append(r)
    print(f"loaded pose for {len(keep)}")
    if len(keep) < 50:
        print("too few — abort"); return 1

    truth = np.array([r["onset_dir"] for r in keep])
    box = np.array([r["heading"] for r in keep])
    R = np.stack(R_last)

    # calibrate the body-forward convention against the box heading (generous to pose)
    best = None
    for axis in range(3):
        for sign in (+1, -1):
            v = sign * R[:, :, axis]                    # world-frame image of a body axis
            yaw = np.arctan2(v[:, 1], v[:, 0])
            for off in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                err = angdiff(yaw + off, box)
                med = float(np.median(err))
                if best is None or med < best[0]:
                    best = (med, axis, sign, off, yaw + off)
    med_vs_box, axis, sign, off, pose_yaw = best
    print(f"calibrated body-forward = {'+' if sign>0 else '-'}axis{axis} offset {np.degrees(off):.0f} deg"
          f"  (median |pose_yaw - box_heading| = {med_vs_box:.1f} deg)")

    e_pose = angdiff(pose_yaw, truth)
    e_box = angdiff(box, truth)
    print("\n" + "=" * 70)
    print(f"ONSET DIRECTION, H={a.horizon:g}s, n={len(keep)}")
    print("=" * 70)
    print(f"  {'predictor':<22}{'median |err|':>13}{'<45 deg':>9}{'<90 deg':>9}")
    print(f"  {'chance (uniform)':<22}{90.0:>12.1f}{25.0:>8.1f}%{50.0:>8.1f}%")
    print(f"  {'SMPL body facing':<22}{np.median(e_pose):>12.1f}{100*np.mean(e_pose<45):>8.1f}%{100*np.mean(e_pose<90):>8.1f}%")
    print(f"  {'WAYMO box heading':<22}{np.median(e_box):>12.1f}{100*np.mean(e_box<45):>8.1f}%{100*np.mean(e_box<90):>8.1f}%")

    # paired comparison on the same events
    d = e_pose - e_box
    from scipy import stats
    w = stats.wilcoxon(e_pose, e_box)
    print(f"\n  paired: SMPL - box = {np.mean(d):+.1f} deg mean, {np.median(d):+.1f} deg median")
    print(f"          Wilcoxon p = {w.pvalue:.3g}   (SMPL better on {100*np.mean(d<0):.1f}% of events)")
    print("\n  VERDICT: " + ("SMPL beats the free box heading."
                             if np.median(e_pose) < np.median(e_box) and w.pvalue < 0.05
                             else "SMPL does NOT beat the free box heading."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
