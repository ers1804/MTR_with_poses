#!/usr/bin/env python
"""Per-agent manoeuvre covariates for the AVO eval scenes.

Motivated directly by the horizon-sweep principle: an auxiliary modality can only pay where the
baseline is not already saturated. The sharpest operational definition of "not saturated" is
where CONSTANT-VELOCITY extrapolation fails, so that is the primary covariate.

Computed per pedestrian, over the 80 future steps (10 Hz), from the trajectory only:

  cv_residual   mean || p(t) - (p0 + v0 * t) ||  -- how badly a constant-velocity model does.
                This is the headroom: agents with a large value are the ones whose future is NOT
                determined by their current velocity.
  heading_change total |unwrapped heading change| over the future (turning)
  speed_range   max future speed - min future speed (accelerate / decelerate / stop-start)
  max_accel     max |d speed / dt| over the future
  speed0        speed at the current step

Writes {(scene, object_id): {...}} for every pedestrian in the dev+final eval scenes.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCEN = "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation"
REPO = os.path.join(os.path.dirname(__file__), "..", "..")
CUR, HZ = 10, 10.0


def _scene(path: str) -> list:
    try:
        info = pickle.load(open(path, "rb"))
    except Exception:
        return []
    ti = info["track_infos"]
    trajs = np.asarray(ti["trajs"])
    scene = str(info["scenario_id"])
    out = []
    for oi, (oid, ot) in enumerate(zip(ti["object_id"], ti["object_type"])):
        if ot != "TYPE_PEDESTRIAN":
            continue
        tr = trajs[oi]
        if tr[CUR, -1] <= 0:
            continue
        fut = tr[CUR + 1:]
        fv = fut[:, -1] > 0
        if fv.sum() < 10:
            continue
        p0 = tr[CUR, :2].astype(np.float64)
        v0 = tr[CUR, 7:9].astype(np.float64)
        t = (np.arange(len(fut)) + 1) / HZ
        pred = p0[None, :] + v0[None, :] * t[:, None]
        res = np.linalg.norm(fut[fv, :2] - pred[fv], axis=1)
        sp = np.hypot(fut[fv, 7], fut[fv, 8])
        hd = np.unwrap(fut[fv, 6].astype(np.float64))
        out.append((scene, str(oid), {
            "cv_residual": float(res.mean()),
            "cv_residual_final": float(res[-1]),
            "heading_change": float(np.abs(hd[-1] - hd[0])) if len(hd) > 1 else 0.0,
            "heading_total_var": float(np.abs(np.diff(hd)).sum()) if len(hd) > 1 else 0.0,
            "speed_range": float(sp.max() - sp.min()) if len(sp) else 0.0,
            "max_accel": float(np.abs(np.diff(sp) * HZ).max()) if len(sp) > 1 else 0.0,
            "speed0": float(np.hypot(tr[CUR, 7], tr[CUR, 8])),
            "n_valid_future": int(fv.sum()),
        }))
    return out


def main() -> int:
    want = set()
    for f in ("avo_eval_dev_scenes.txt", "avo_eval_final_scenes.txt"):
        want |= set(open(os.path.join(REPO, "data/waymo", f)).read().split())
    print(f"eval scenes: {len(want)}")

    by_scene: dict[str, str] = {}
    for f in glob.glob(os.path.join(SCEN, "sample_*.pkl")):
        m = re.match(r"sample_(.+)_\d+\.pkl$", os.path.basename(f))
        if m and m.group(1) in want and m.group(1) not in by_scene:
            by_scene[m.group(1)] = f
    print(f"reading {len(by_scene)} scene files (one per scene)")

    cov, done = {}, 0
    with ProcessPoolExecutor(24) as ex:
        for part in ex.map(_scene, list(by_scene.values()), chunksize=16):
            for s, o, d in part:
                cov[(s, o)] = d
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(by_scene)}", flush=True)

    out = os.path.join(REPO, "experiments", "eval_agent_manoeuvre.pkl")
    pickle.dump(cov, open(out, "wb"))
    print(f"wrote {len(cov)} pedestrians -> {out}")
    v = np.array([c["cv_residual"] for c in cov.values()])
    print(f"  cv_residual quartiles: {np.round(np.percentile(v,[25,50,75,90,99]),2)}  max {v.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
