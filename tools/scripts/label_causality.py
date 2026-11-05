#!/usr/bin/env python
"""Does the WOMD state label at time t carry information from t+1?

Reviewer objection (Flohr, 2026-11-04): WOMD boxes come from an offboard 3D
auto-labelling system whose refinement network consumes every frame in a track,
so the states our baseline consumes are not what an onboard tracker could
produce. If position, velocity and heading at time t are smoothed with frames
after t, the baseline starts with a future-informed state and there is little
left for body pose to add.

That objection is testable on the labels alone, with no model and no training.
If the labelled velocity at t were causal it would agree with the BACKWARD
difference of positions, (p[t] - p[t-1]) * hz. If the labeller used t+1 it
agrees better with the CENTERED difference, (p[t+1] - p[t-1]) * hz / 2. The
stencil the label matches is therefore a direct read of which frames went into
it.

RESULT (2026-11-05): the velocity field is the BACKWARD difference exactly, to
float32, at every pedestrian timestep measured. So WOMD velocity is an algebraic
function of the position labels and carries nothing beyond them: there is one
state channel to degrade, not three independent ones.

What this does NOT settle. For heading the test is confounded and we say so. The
centered stencil beats the best causal one (2.2 deg against 3.4 deg), which is
consistent with smoothing across t+1 but equally with the centered difference
simply being a lower-variance estimator of the true direction at t, since it
averages two steps of position noise. A causal second-order estimator at t is
worse still (5.5 deg) because extrapolation amplifies that noise. These data
therefore cannot separate a leak from an estimator-variance effect in the
heading channel. The evidence that the labels are future-informed is documentary
rather than statistical: WOMD boxes come from per-frame detection, then
tracking, then an object-centric refinement network that consumes every frame in
the track.

Writes experiments/label_causality.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCEN = "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation"
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
HZ = 10.0
MOVING_MS = 0.5  # below this, heading is not defined by motion and is excluded


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _scene(path: str):
    """Per-timestep stencil residuals for every pedestrian in one scene."""
    try:
        d = pickle.load(open(path, "rb"))
    except Exception:
        return None
    ti = d["track_infos"]
    types = np.asarray(ti["object_type"])
    ped = types == "TYPE_PEDESTRAIN"  # the dataset's own spelling
    if not ped.any():
        ped = types == "TYPE_PEDESTRIAN"
        if not ped.any():
            return None

    tr = np.asarray(ti["trajs"], dtype=np.float64)[ped]
    pos, head, vel, val = tr[:, :, 0:2], tr[:, :, 6], tr[:, :, 7:9], tr[:, :, 9] > 0

    # a timestep is usable when t-1, t and t+1 are all valid
    ok = val[:, 1:-1] & val[:, 0:-2] & val[:, 2:]
    if not ok.any():
        return None
    p_prev, p_cur, p_next = pos[:, 0:-2], pos[:, 1:-1], pos[:, 2:]

    # backward is causal but centred on t-0.5; causal_extrap is causal AND centred
    # on t, so "centered beats causal" cannot be dismissed as half-step bias alone.
    p_prev2 = np.concatenate([pos[:, 0:1], pos[:, 0:-3]], axis=1)
    stencils = {
        "backward": (p_cur - p_prev) * HZ,
        "causal_extrap": (1.5 * (p_cur - p_prev) - 0.5 * (p_prev - p_prev2)) * HZ,
        "forward": (p_next - p_cur) * HZ,
        "centered": (p_next - p_prev) * HZ / 2.0,
    }
    v_lab = vel[:, 1:-1]
    speed = np.linalg.norm(v_lab, axis=-1)

    out = {"n": int(ok.sum()), "speed": speed[ok]}
    for name, v in stencils.items():
        out[f"vel_{name}"] = np.linalg.norm(v_lab - v, axis=-1)[ok]
        # heading is only defined by motion where the agent actually moves
        m = ok & (speed > MOVING_MS) & (np.linalg.norm(v, axis=-1) > MOVING_MS)
        out[f"head_{name}"] = np.degrees(
            np.abs(_wrap(head[:, 1:-1][m] - np.arctan2(v[..., 1], v[..., 0])[m]))
        )
    out["n_moving"] = int((ok & (speed > MOVING_MS)).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "label_causality.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(SCEN, "*.pkl")))[: args.scenes]
    print(f"reading {len(paths)} validation scenes with {args.workers} workers ...")

    STENCILS = ("backward", "causal_extrap", "forward", "centered")
    keys = [f"{q}_{s}" for q in ("vel", "head") for s in STENCILS]
    acc = {k: [] for k in keys}
    acc["speed"] = []
    n_steps = n_moving = n_scenes = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_scene, paths, chunksize=16):
            if r is None:
                continue
            n_scenes += 1
            n_steps += r["n"]
            n_moving += r["n_moving"]
            for k in keys:
                acc[k].append(r[k])
            acc["speed"].append(r["speed"])

    res = {
        "n_scenes_with_pedestrians": n_scenes,
        "n_timesteps": n_steps,
        "n_timesteps_moving": n_moving,
        "moving_threshold_ms": MOVING_MS,
        "velocity_residual_ms": {},
        "heading_residual_deg": {},
    }
    for s in STENCILS:
        v = np.concatenate(acc[f"vel_{s}"])
        h = np.concatenate(acc[f"head_{s}"])
        res["velocity_residual_ms"][s] = dict(
            median=float(np.median(v)), p25=float(np.percentile(v, 25)),
            p75=float(np.percentile(v, 75)), mean=float(v.mean()), n=int(v.size))
        res["heading_residual_deg"][s] = dict(
            median=float(np.median(h)), p25=float(np.percentile(h, 25)),
            p75=float(np.percentile(h, 75)), mean=float(h.mean()), n=int(h.size))

    vm = {s: res["velocity_residual_ms"][s]["median"] for s in STENCILS}
    hm = {s: res["heading_residual_deg"][s]["median"] for s in STENCILS}
    res["best_stencil_velocity"] = min(vm, key=vm.get)
    res["best_stencil_heading"] = min(hm, key=hm.get)
    # the quantity the paper needs: how much better than a causal estimate the
    # label is, in the two channels the baseline consumes
    best_causal = min(hm["backward"], hm["causal_extrap"])
    res["velocity_is_exact_backward_difference"] = vm["backward"] == 0.0
    res["heading_future_vs_best_causal_deg"] = dict(
        best_causal=best_causal, centered=hm["centered"],
        advantage=best_causal - hm["centered"],
        confounded=True,
        note="centered also averages two steps of position noise, so this gap is "
             "an upper bound on any future-information effect, not a measurement "
             "of one",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    print(f"\npedestrian timesteps: {n_steps:,} ({n_moving:,} moving) "
          f"over {n_scenes:,} scenes\n")
    print(f"{'stencil':<14} {'|v_label - v_stencil| (m/s)':>30} {'|heading - atan2(v)| (deg)':>28}")
    for s in STENCILS:
        print(f"{s:<14} {vm[s]:>30.6f} {hm[s]:>28.3f}")
    print(f"\nbest match: velocity -> {res['best_stencil_velocity']}, "
          f"heading -> {res['best_stencil_heading']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
