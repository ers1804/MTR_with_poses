#!/usr/bin/env python
"""Do HEADING_NOISE_DEG and POSITION_NOISE_M corrupt the frame CONSISTENTLY?

The flag perturbs center_objects[:, 6] once, where the array is assembled, on the
assumption that every consumer reads it from there: the past-trajectory
transform, the future-target transform, the map transform, the pose-root
rotation, and the back-rotation of predictions to world. If any one of them
instead reached for a clean heading, predictions would be rotated out of the
frame their targets live in, and every arm would carry a silent per-agent
rotation error. That is the same class of bug as the export id collision, so it
gets a test before any job is submitted.

The invariant: rotating the agent-frame ground-truth future back by
center_objects_world[:, 6] and adding the centre position must recover the raw
world trajectory, and must recover the SAME world trajectory whether the noise
is on or off. A clean heading surviving anywhere downstream breaks this.

Also checks the flag is not a no-op: the agent-frame inputs and the map
polylines must differ between the two conditions.

POSITION_NOISE_M moves the frame ORIGIN as well, so the cross-condition check is
made in absolute world coordinates (float32 at ~1e4 m resolves ~1e-3 m, far
below the decimetre noise it guards against). It must also leave the heading and
the future targets bit-identical, which is what keeps the two ladders separate,
and it reports the error the simulated tracker delivers at prediction time.

Usage: python tools/scripts/test_heading_noise_roundtrip.py [--scenes 40] [--deg 30]
       python tools/scripts/test_heading_noise_roundtrip.py --pos 0.3 [--out f.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from mtr.config import cfg, cfg_from_yaml_file  # noqa: E402
from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset  # noqa: E402
from mtr.utils import common_utils  # noqa: E402

CFG = os.path.join(REPO, "tools", "cfgs", "waymo",
                   "mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5.yaml")


def build(deg: float = 0.0, pos: float = 0.0):
    cfg_from_yaml_file(CFG, cfg)
    cfg.DATA_CONFIG.HEADING_NOISE_DEG = deg
    cfg.DATA_CONFIG.POSITION_NOISE_M = pos
    logger = common_utils.create_logger(log_file=None, rank=0)
    return WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG, training=False, logger=logger)


def world_from_agent_frame(d: dict, centre_relative: bool = True) -> np.ndarray:
    """Rotate the agent-frame GT future back to world, exactly as the eval path does.

    Returned centre-relative by default. World coordinates run to ~1e4 m, where
    float32 spacing is ~1e-3 m, so an absolute comparison is blind below a
    millimetre; subtracting the centre keeps the full ~1e-7 resolution and still
    catches the bug this test is for, which is a per-agent ROTATION error.
    """
    gt = torch.from_numpy(d["center_gt_trajs"]).float()           # (C, T_f, >=2)
    cw = torch.from_numpy(np.asarray(d["center_objects_world"])).float()
    n, t = gt.shape[0], gt.shape[1]
    xy = common_utils.rotate_points_along_z(
        points=gt[:, :, 0:2].reshape(n, t, 2), angle=cw[:, 6].reshape(n)
    ).reshape(n, t, 2)
    out = xy if centre_relative else xy + cw[:, None, 0:2]
    return out.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--deg", type=float, default=30.0)
    ap.add_argument("--pos", type=float, default=0.0,
                    help="test POSITION_NOISE_M at this std instead of the heading flag")
    ap.add_argument("--out", help="write the tracker-error report here (position mode)")
    args = ap.parse_args()
    if args.pos > 0:
        return main_position(args)

    clean, noisy = build(0.0), build(args.deg)
    assert noisy.heading_noise_deg == args.deg and clean.heading_noise_deg == 0.0

    n_agents = n_checked = 0
    worst_roundtrip = worst_cross = 0.0
    heading_shift, input_delta, map_delta, displacement = [], [], [], []

    for i in range(args.scenes):
        a, b = clean[i], noisy[i]
        assert (a["center_objects_id"] == b["center_objects_id"]).all(), \
            "the two conditions must select the same pedestrians"
        n_agents += a["center_objects_world"].shape[0]

        # 1. the frame must be internally consistent in EACH condition
        for d in (a, b):
            rec = world_from_agent_frame(d)
            cw = np.asarray(d["center_objects_world"])
            src = (np.asarray(d["center_gt_trajs_src"])[:, -rec.shape[1]:, 0:2]
                   - cw[:, None, 0:2])
            m = np.asarray(d["center_gt_trajs_mask"]).astype(bool)
            if m.any():
                err = np.abs(rec - src).max(axis=-1)[m]
                worst_roundtrip = max(worst_roundtrip, float(err.max()))
                n_checked += int(m.sum())

        # 2. and both must recover the SAME world trajectory
        ra, rb = world_from_agent_frame(a), world_from_agent_frame(b)
        m = np.asarray(a["center_gt_trajs_mask"]).astype(bool)
        if m.any():
            worst_cross = max(worst_cross, float(np.abs(ra - rb).max(axis=-1)[m].max()))
            displacement.append(np.linalg.norm(ra, axis=-1)[m])

        # 3. the noise must actually bite
        dh = np.abs(np.asarray(b["center_objects_world"])[:, 6]
                    - np.asarray(a["center_objects_world"])[:, 6])
        heading_shift.append(np.degrees((dh + np.pi) % (2 * np.pi) - np.pi))
        input_delta.append(np.abs(a["obj_trajs"] - b["obj_trajs"]).max())
        if "map_polylines" in a:
            map_delta.append(np.abs(a["map_polylines"] - b["map_polylines"]).max())

    shift = np.abs(np.concatenate(heading_shift))
    print(f"\nscenes {args.scenes}, centre agents {n_agents}, "
          f"future steps checked {n_checked:,}\n")
    print(f"  worst agent-frame -> world round-trip error   {worst_roundtrip:.3e} m")
    print(f"  worst clean-vs-noisy world disagreement       {worst_cross:.3e} m")
    print(f"  heading shift applied: median {np.median(shift):.2f} deg, "
          f"max {shift.max():.2f} deg")
    print(f"  agent input max |delta|                       {max(input_delta):.4f}")
    if map_delta:
        print(f"  map polyline max |delta|                      {max(map_delta):.4f}")

    disp = np.concatenate(displacement)
    print(f"  future displacement compared: median {np.median(disp):.2f} m, "
          f"max {disp.max():.2f} m")

    tol = 1e-4  # float32 through two rotations, centre-relative
    ok = True
    if worst_roundtrip > tol:
        print(f"\nFAIL: the frame is not internally consistent ({worst_roundtrip:.3e} m)")
        ok = False
    if worst_cross > tol:
        print(f"\nFAIL: a clean heading survives downstream -- the two conditions "
              f"recover different world trajectories ({worst_cross:.3e} m). "
              f"Predictions would be rotated out of their targets' frame.")
        ok = False
    if np.median(disp) < 0.1:
        print(f"\nFAIL: vacuous -- the agents compared barely move "
              f"(median {np.median(disp):.3f} m), so a rotation error would not show")
        ok = False
    if max(input_delta) < 1e-6:
        print("\nFAIL: the flag is a no-op, the agent inputs are unchanged")
        ok = False
    if map_delta and max(map_delta) < 1e-6:
        print("\nFAIL: the map was not rotated into the corrupted frame")
        ok = False
    if abs(np.median(shift) - 0.6745 * args.deg) > 0.35 * args.deg:
        print(f"\nWARN: median |shift| {np.median(shift):.2f} deg is far from the "
              f"{0.6745 * args.deg:.2f} deg a N(0,{args.deg}) draw implies")

    print("\nPASS: heading noise reaches the frame, the anchors' frame, the map and "
          "the pose root, and the eval path undoes exactly what the data path did."
          if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


def main_position(args) -> None:
    clean, noisy = build(), build(pos=args.pos)
    assert noisy.position_noise_m == args.pos and clean.position_noise_m == 0.0

    worst_roundtrip = worst_cross = 0.0
    n_agents = 0
    pos_err, vel_err, speed, input_delta, map_delta = [], [], [], [], []
    for i in range(args.scenes):
        a, b = clean[i], noisy[i]
        assert (a["center_objects_id"] == b["center_objects_id"]).all()
        cwa = np.asarray(a["center_objects_world"], dtype=np.float64)
        cwb = np.asarray(b["center_objects_world"], dtype=np.float64)
        n_agents += cwa.shape[0]

        # the two ladders must not mix, and the targets must stay the labels
        assert (cwa[:, 6] == cwb[:, 6]).all(), "position noise touched the heading"
        T = noisy.num_past
        assert (np.asarray(a["center_gt_trajs_src"])[:, T:]
                == np.asarray(b["center_gt_trajs_src"])[:, T:]).all(), \
            "position noise touched the future targets"

        for d in (a, b):
            rec = world_from_agent_frame(d)
            cw = np.asarray(d["center_objects_world"])
            src = (np.asarray(d["center_gt_trajs_src"])[:, -rec.shape[1]:, 0:2]
                   - cw[:, None, 0:2])
            m = np.asarray(d["center_gt_trajs_mask"]).astype(bool)
            if m.any():
                worst_roundtrip = max(worst_roundtrip,
                                      float(np.abs(rec - src).max(axis=-1)[m].max()))

        # absolute world: both conditions must predict into the same place
        ra = world_from_agent_frame(a).astype(np.float64) + cwa[:, None, 0:2]
        rb = world_from_agent_frame(b).astype(np.float64) + cwb[:, None, 0:2]
        m = np.asarray(a["center_gt_trajs_mask"]).astype(bool)
        if m.any():
            worst_cross = max(worst_cross, float(np.abs(ra - rb).max(axis=-1)[m].max()))

        pos_err.append(np.linalg.norm(cwb[:, 0:2] - cwa[:, 0:2], axis=-1))
        vel_err.append(np.linalg.norm(cwb[:, 7:9] - cwa[:, 7:9], axis=-1))
        speed.append(np.linalg.norm(cwa[:, 7:9], axis=-1))
        input_delta.append(np.abs(a["obj_trajs"] - b["obj_trajs"]).max())
        if "map_polylines" in a:
            map_delta.append(np.abs(a["map_polylines"] - b["map_polylines"]).max())

    pe, ve, sp = map(np.concatenate, (pos_err, vel_err, speed))
    rep = dict(position_noise_m=args.pos, scenes=args.scenes, agents=int(len(pe)),
               pos_err_median_m=float(np.median(pe)), pos_err_p90_m=float(np.percentile(pe, 90)),
               vel_err_median_ms=float(np.median(ve)), vel_err_p90_ms=float(np.percentile(ve, 90)),
               speed_median_ms=float(np.median(sp)))
    print(f"\nscenes {args.scenes}, centre agents {n_agents}\n")
    print(f"  worst agent-frame -> world round-trip error   {worst_roundtrip:.3e} m")
    print(f"  worst clean-vs-noisy absolute disagreement    {worst_cross:.3e} m")
    print(f"  tracker error at prediction time: position median {rep['pos_err_median_m']:.3f} m "
          f"(p90 {rep['pos_err_p90_m']:.3f}), velocity median {rep['vel_err_median_ms']:.3f} m/s "
          f"(p90 {rep['vel_err_p90_ms']:.3f}); label speed median {rep['speed_median_ms']:.2f} m/s")
    print(f"  agent input max |delta| {max(input_delta):.4f}, map max |delta| "
          f"{max(map_delta) if map_delta else float('nan'):.4f}")
    ok = worst_roundtrip <= 1e-4 and worst_cross <= 5e-3 and max(input_delta) > 1e-6 \
        and (not map_delta or max(map_delta) > 1e-6) and np.median(pe) > 0.1 * args.pos
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
    print("\nPASS: tracked history reaches inputs, frame origin and map; heading and "
          "targets untouched; eval undoes exactly the data path." if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
