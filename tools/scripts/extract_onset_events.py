#!/usr/bin/env python
"""Extract motion-onset events for stationary pedestrians.

Motivation
----------
Body pose is redundant with velocity for long-horizon position forecasting, which is why
every pose result in this project is null. The one regime where that redundancy CANNOT
hold is a pedestrian who is standing still: velocity then carries zero directional
information. This script builds the dataset for that task.

Task definition (fixed here before any model is fit)
----------------------------------------------------
Candidate  : TYPE_PEDESTRIAN, valid at the current step (index 10), with speed at that
             step < STATIONARY_SPEED, and at least MIN_VALID_FUTURE valid future steps.
Onset      : the first future step t > 10 at which ||p[t] - p[10]|| > ONSET_RADIUS.
Label      : onset occurs within H seconds, for each H in HORIZONS.
Direction  : unit vector from p[10] to p[onset] (defined only for positive events).
Features   : derived from the observable history (indices 0..10) ONLY. Anything computed
             from index > 10 is a label, never a feature.

Trajectory columns are [x, y, z, len, wid, hgt, heading, vel_x, vel_y, valid] at 10 Hz,
91 steps, current_time_index = 10 (so 80 future steps = 8 s).

Usage:
  python extract_onset_events.py --split validation [--limit N] [--workers 24]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCEN = {
    "validation": "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation",
    "training": "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_training",
}
POSE = "/home/erik/big1/waymo_motion_poses/final_10fps"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")

CUR = 10                     # index of "now"
HZ = 10.0
STATIONARY_SPEED = 0.3       # m/s at the current step
ONSET_RADIUS = 1.0           # m displacement from p[10] that counts as having moved
MIN_VALID_FUTURE = 30        # >= 3 s of valid future, else the label is undefined
HORIZONS = (1.0, 2.0, 3.0, 5.0)


def _scene_events(args) -> list:
    path, split = args
    try:
        info = pickle.load(open(path, "rb"))
    except Exception:
        return []
    ti = info["track_infos"]
    trajs = np.asarray(ti["trajs"])
    ids = list(ti["object_id"])
    types = list(ti["object_type"])
    scene = info["scenario_id"]
    pose_dir = os.path.join(POSE, split, str(scene))
    have_pose = os.path.isdir(pose_dir)

    out = []
    for oi, (oid, otype) in enumerate(zip(ids, types)):
        if otype != "TYPE_PEDESTRIAN":
            continue
        tr = trajs[oi]
        if tr[CUR, -1] <= 0:
            continue
        speed0 = float(np.hypot(tr[CUR, 7], tr[CUR, 8]))
        if speed0 >= STATIONARY_SPEED:
            continue
        fut = tr[CUR + 1:]
        fv = fut[:, -1] > 0
        if int(fv.sum()) < MIN_VALID_FUTURE:
            continue

        p0 = tr[CUR, :2].astype(np.float64)
        # onset = first valid future step beyond ONSET_RADIUS of p0
        d = np.full(len(fut), np.nan)
        d[fv] = np.linalg.norm(fut[fv, :2] - p0, axis=1)
        moved = np.where(np.nan_to_num(d, nan=-1.0) > ONSET_RADIUS)[0]
        onset_idx = int(moved[0]) if len(moved) else -1          # 0-based into future
        onset_t = (onset_idx + 1) / HZ if onset_idx >= 0 else np.inf
        if onset_idx >= 0:
            v = fut[onset_idx, :2].astype(np.float64) - p0
            onset_dir = float(np.arctan2(v[1], v[0]))
        else:
            onset_dir = np.nan

        # ---- history-only features (indices 0..CUR) -------------------------------
        hist = tr[:CUR + 1]
        hv = hist[:, -1] > 0
        hp = hist[hv, :2].astype(np.float64)
        hs = np.hypot(hist[hv, 7], hist[hv, 8])
        seg = np.linalg.norm(np.diff(hp, axis=0), axis=1) if len(hp) > 1 else np.array([0.0])
        rec = {
            "scene": str(scene), "object_id": str(oid),
            "speed0": speed0,
            "speed_hist_mean": float(hs.mean()) if len(hs) else 0.0,
            "speed_hist_max": float(hs.max()) if len(hs) else 0.0,
            "speed_hist_last3": float(hs[-3:].mean()) if len(hs) >= 3 else float(hs.mean() if len(hs) else 0.0),
            "hist_pathlen": float(seg.sum()),
            "hist_netdisp": float(np.linalg.norm(hp[-1] - hp[0])) if len(hp) > 1 else 0.0,
            "hist_n_valid": int(hv.sum()),
            "heading": float(tr[CUR, 6]),
            "accel_proxy": float(hs[-1] - hs[0]) if len(hs) > 1 else 0.0,
            "n_valid_future": int(fv.sum()),
            "onset_idx": onset_idx, "onset_time_s": onset_t, "onset_dir": onset_dir,
            "has_pose": bool(have_pose and os.path.exists(os.path.join(pose_dir, f"{oid}.npz"))),
        }
        for H in HORIZONS:
            rec[f"onset_{H:g}s"] = bool(onset_t <= H)
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=list(SCEN))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SCEN[a.split], "sample_*.pkl")))
    if a.limit:
        files = files[:a.limit]
    print(f"{a.split}: {len(files)} scenario files, {a.workers} workers", flush=True)

    rows, done = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(_scene_events, [(f, a.split) for f in files], chunksize=16):
            rows.extend(part)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(files)} scenes, {len(rows)} stationary peds", flush=True)

    # Waymo preprocessing writes several sample_<scene>_<k>.pkl per scene carrying the SAME
    # tracks, so one pedestrian can appear many times (5.4x on validation). Deduplicate by
    # (scene, object_id) or every count and every split is inflated.
    seen, uniq = set(), []
    for r in rows:
        k = (r["scene"], r["object_id"])
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    print(f"deduplicated {len(rows)} -> {len(uniq)} unique (scene, object_id)")
    rows = uniq

    out = a.out or os.path.join(OUT, f"onset_events_{a.split}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(rows, open(out, "wb"))
    print(f"\nwrote {len(rows)} candidate events -> {out}")
    if rows:
        for H in HORIZONS:
            k = f"onset_{H:g}s"
            n = sum(r[k] for r in rows)
            print(f"  onset within {H:g}s: {n} ({100*n/len(rows):.1f}%)")
        print(f"  with pose available: {sum(r['has_pose'] for r in rows)} "
              f"({100*sum(r['has_pose'] for r in rows)/len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
