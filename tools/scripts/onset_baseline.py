#!/usr/bin/env python
"""Kinematics-only baselines for the motion-onset task, plus the confounds that matter.

Two sub-tasks, scored separately because they have different baselines:

  (A) DETECTION  — will this stationary pedestrian move (>1 m) within H seconds?
      Baseline: history kinematics (residual micro-motion, path length, accel proxy).
      Metric: ROC-AUC and average precision, against the base rate.

  (B) DIRECTION  — for pedestrians that DO move, which way do they go?
      This is the sub-task where velocity is structurally uninformative (they are
      standing still), and it is where pose is hypothesised to help.
      Baselines, in increasing order of threat to the pose hypothesis:
        chance          — uniform over the circle; mean |error| = 90 deg
        history heading — direction of whatever micro-motion happened in 0..10
        WAYMO BOX HEADING — the tracker's own orientation estimate for the box
      The third is the one that matters. Waymo already ships a per-object heading. If
      it predicts onset direction, then SMPL pose must beat IT, not chance, to be worth
      anything. Reporting pose against chance while a free box heading does the same job
      would repeat exactly the mistake this project keeps catching.

Splits are by SCENE, never by agent — pedestrians in one scene share context and would
leak across a random agent split.

Usage: python onset_baseline.py [--events experiments/onset_events_validation.pkl]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle

import numpy as np

HORIZONS = (1.0, 2.0, 3.0, 5.0)
FEATURES = ["speed0", "speed_hist_mean", "speed_hist_max", "speed_hist_last3",
            "hist_pathlen", "hist_netdisp", "accel_proxy", "hist_n_valid"]


def scene_split(scene: str, frac_test: float = 0.3) -> str:
    """Deterministic scene-level split by hash — stable across runs and machines."""
    h = int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < frac_test else "train"


def angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute angle between two headings, in degrees."""
    d = np.abs(np.angle(np.exp(1j * (a - b))))
    return np.degrees(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="experiments/onset_events_validation.pkl")
    a = ap.parse_args()
    rows = pickle.load(open(a.events, "rb"))
    n_raw = len(rows)
    seen, uniq = set(), []
    for r in rows:
        k = (r["scene"], r["object_id"])
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    rows = uniq
    print(f"loaded {n_raw} rows from {a.events}; {len(rows)} unique (scene, object_id)")

    split = np.array([scene_split(r["scene"]) for r in rows])
    print(f"  scene-level split: train {int((split=='train').sum())}  test {int((split=='test').sum())}")
    print(f"  distinct scenes: {len({r['scene'] for r in rows})}")
    print(f"  pose available : {sum(r['has_pose'] for r in rows)} "
          f"({100*np.mean([r['has_pose'] for r in rows]):.1f}%)")

    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    tr, te = split == "train", split == "test"

    # ---------------- (A) detection ------------------------------------------------
    print("\n" + "=" * 74)
    print("(A) DETECTION: will a stationary pedestrian move >1 m within H seconds?")
    print("=" * 74)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn unavailable — skipping"); return 1

    print(f"  {'H':>4} {'base rate':>10} {'n_pos test':>11} {'AUC logreg':>11} {'AUC GBT':>9} {'AP GBT':>8}")
    for H in HORIZONS:
        y = np.array([r[f"onset_{H:g}s"] for r in rows], dtype=int)
        if y[tr].sum() < 20 or y[te].sum() < 20:
            print(f"  {H:>4g} too few positives — skipped"); continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        auc_lr = roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1])
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], y[tr])
        p = gb.predict_proba(X[te])[:, 1]
        print(f"  {H:>4g} {y.mean():>10.3f} {int(y[te].sum()):>11d} {auc_lr:>11.3f} "
              f"{roc_auc_score(y[te], p):>9.3f} {average_precision_score(y[te], p):>8.3f}")

    # ---------------- (B) direction ------------------------------------------------
    print("\n" + "=" * 74)
    print("(B) DIRECTION: which way do they go? (positive events only)")
    print("=" * 74)
    for H in HORIZONS:
        m = np.array([r[f"onset_{H:g}s"] and np.isfinite(r["onset_dir"]) for r in rows])
        mt = m & te
        if mt.sum() < 30:
            print(f"  H={H:g}s: only {int(mt.sum())} test events — skipped"); continue
        truth = np.array([r["onset_dir"] for r in rows])[mt]
        box = np.array([r["heading"] for r in rows])[mt]
        # history micro-motion direction, where any exists
        err_box = angdiff(truth, box)
        # chance: expected |error| for a uniform guess is 90 deg
        print(f"\n  H={H:g}s  ({int(mt.sum())} test events)")
        print(f"    chance (uniform guess)      median |err| 90.0 deg   within 45 deg: 25.0%")
        print(f"    WAYMO BOX HEADING           median |err| {np.median(err_box):4.1f} deg   "
              f"within 45 deg: {100*np.mean(err_box < 45):4.1f}%   "
              f"within 90 deg: {100*np.mean(err_box < 90):4.1f}%")
        # box heading has a 180-deg ambiguity for a standing person; report the folded version too
        err_box180 = np.minimum(err_box, 180 - err_box)
        print(f"    box heading, axis only      median |err| {np.median(err_box180):4.1f} deg   "
              f"(180-deg-folded: does it at least give the AXIS of motion?)")
    print("\nNOTE: SMPL pose must beat the BOX HEADING row, not the chance row, to be useful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
