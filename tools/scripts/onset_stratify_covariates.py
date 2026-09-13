#!/usr/bin/env python
"""Pull tracker-quality covariates for the onset events, to test WHERE pose could still win.

The box heading beats SMPL facing on average (25.2 vs 44.2 deg median error at H=3s). The one
remaining live hypothesis is that it does so only where the tracker is healthy, and degrades
faster than pose in hard conditions. This adds the covariates needed to test that.

WOMD motion data has no LiDAR point counts (those live in the Perception set), so we use the
proxies that are available and that genuinely track perception quality:

  range_ego      distance from the self-driving car at t=10 -- fewer returns with range
  hist_valid_frac fraction of the 11 history steps the tracker actually saw the agent
                  (gaps = occlusion / dropped detections)
  heading_std    circular std of the box heading over valid history steps. This is the
                 direct one: a jittery heading estimate IS an unreliable heading estimate,
                 and it is measurable at inference time without labels.
  box_area       length*width of the box; degenerate boxes signal poor fits

Only scenes that actually contain events are read (one sample file per scene), which is ~9k
files for validation instead of 192k.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCEN = {
    "validation": "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation",
    "training": "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_training",
}
CUR = 10


def _scene_cov(args) -> list:
    path, wanted = args
    try:
        info = pickle.load(open(path, "rb"))
    except Exception:
        return []
    ti = info["track_infos"]
    trajs = np.asarray(ti["trajs"])
    ids = [str(x) for x in ti["object_id"]]
    scene = str(info["scenario_id"])
    sdc = info.get("sdc_track_index", None)
    ego = trajs[sdc, CUR, :2].astype(np.float64) if sdc is not None else None

    out = []
    idx = {o: i for i, o in enumerate(ids)}
    for oid in wanted:
        i = idx.get(oid)
        if i is None:
            continue
        tr = trajs[i]
        hist = tr[:CUR + 1]
        hv = hist[:, -1] > 0
        head = hist[hv, 6].astype(np.float64)
        if len(head) >= 2:
            # circular std over the observed headings
            z = np.exp(1j * head)
            R = np.abs(z.mean())
            hstd = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))))
        else:
            hstd = np.nan
        out.append({
            "scene": scene, "object_id": oid,
            "range_ego": float(np.linalg.norm(tr[CUR, :2] - ego)) if ego is not None else np.nan,
            "hist_valid_frac": float(hv.mean()),
            "heading_std": hstd,
            "box_area": float(tr[CUR, 3] * tr[CUR, 4]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--events", default=None)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    ev_path = a.events or f"experiments/onset_events_{a.split}.pkl"
    rows = pickle.load(open(ev_path, "rb"))
    want = defaultdict(set)
    for r in rows:
        want[r["scene"]].add(r["object_id"])
    print(f"{len(rows)} events across {len(want)} scenes")

    # one sample file per scene that we actually need
    by_scene: dict[str, str] = {}
    for f in glob.glob(os.path.join(SCEN[a.split], "sample_*.pkl")):
        m = re.match(r"sample_(.+)_\d+\.pkl$", os.path.basename(f))
        if m and m.group(1) in want and m.group(1) not in by_scene:
            by_scene[m.group(1)] = f
    print(f"reading {len(by_scene)} scene files (one per scene)")

    tasks = [(p, want[s]) for s, p in by_scene.items()]
    cov, done = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(_scene_cov, tasks, chunksize=16):
            for c in part:
                cov[(c["scene"], c["object_id"])] = c
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    n = 0
    for r in rows:
        c = cov.get((r["scene"], r["object_id"]))
        if c:
            r.update({k: v for k, v in c.items() if k not in ("scene", "object_id")})
            n += 1
    pickle.dump(rows, open(ev_path, "wb"))
    print(f"attached covariates to {n}/{len(rows)} events -> {ev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
