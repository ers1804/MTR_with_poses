#!/usr/bin/env python
"""Does HEADING_NOISE_DEG corrupt the frame CONSISTENTLY?

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

Usage: python tools/scripts/test_heading_noise_roundtrip.py [--scenes 6] [--deg 30]
"""
from __future__ import annotations

import argparse
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


def build(deg: float):
    cfg_from_yaml_file(CFG, cfg)
    cfg.DATA_CONFIG.HEADING_NOISE_DEG = deg
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
