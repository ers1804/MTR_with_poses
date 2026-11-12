#!/usr/bin/env python
"""Is the pre-fit pose export a drop-in for the final one?

The pre-fit rerun of the onboard-track ladder is only readable if NOTHING but the pose
tensor differs from the final-pose runs it is compared with. This builds the dataset
from both exports, with the same config otherwise, and checks per scene:

  1. every non-pose key is bit-identical (tracks, map, targets, agent set, the tracker
     noise, which is keyed on scene and object id);
  2. the shuffled arm draws the same donor for every pedestrian in both exports;
  3. the pose validity masks agree except on the rows only the motion prior filled;
  4. the pose tensors do differ (the flag is not a no-op).

Usage: python tools/scripts/test_prefit_dropin.py [--scenes 300] [--train] [--noise 30|10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from mtr.config import cfg, cfg_from_yaml_file  # noqa: E402
from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset  # noqa: E402
from mtr.utils import common_utils  # noqa: E402

CFGS = os.path.join(REPO, "tools", "cfgs", "waymo")
FINAL = "mtr+pose_data_{arm}_posnoise{noise}_scale25k.yaml"
PREFIT = "mtr+pose_data_{arm}_prefit_posnoise{noise}_scale25k.yaml"
POSE_KEYS = {"obj_poses", "obj_poses_mask", "center_gt_poses", "center_gt_poses_mask",
             "obj_poses_future", "obj_poses_future_mask", "obj_betas", "center_betas"}


def build(name: str, training: bool):
    cfg_from_yaml_file(os.path.join(CFGS, name), cfg)
    logger = common_utils.create_logger(log_file=None, rank=0)
    return WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG, training=training, logger=logger)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--noise", default="30", help="ladder rung in cm: 30 or 10")
    a = ap.parse_args()
    bad = 0
    for arm in ("cross_attn_pe_with_map", "cross_attn_pe_with_map_shufflepose"):
        fin = build(FINAL.format(arm=arm, noise=a.noise), a.train)
        pre = build(PREFIT.format(arm=arm, noise=a.noise), a.train)
        assert fin.dataset_cfg.POSITION_NOISE_M == pre.dataset_cfg.POSITION_NOISE_M == int(a.noise) / 100
        assert len(fin) == len(pre), (len(fin), len(pre))
        if "shufflepose" in arm:
            rel = lambda p: os.sep.join(p.split(os.sep)[-3:])
            df = {rel(k): rel(v) for k, v in fin._pose_donor.items()}
            dp = {rel(k): rel(v) for k, v in pre._pose_donor.items()}
            same = df == dp
            print(f"[{arm}] donor map identical: {same} ({len(df)} pedestrians)")
            bad += not same
        idx = np.linspace(0, len(fin) - 1, a.scenes).astype(int)
        nonpose_diff, pose_keys_seen, other_keys = {}, set(), set()
        m_and = m_fin_only = m_pre_only = 0
        pose_changed = pose_valid_rows = 0
        for i in idx:
            # training mode samples centres at random when a scene is capped, so seed both
            np.random.seed(int(i)); df_ = fin.create_scene_level_data(int(i))
            np.random.seed(int(i)); dp_ = pre.create_scene_level_data(int(i))
            assert df_.keys() == dp_.keys()
            for k in df_:
                x, y = df_[k], dp_[k]
                if "pose" in k or "beta" in k:
                    pose_keys_seen.add(k)
                    continue
                other_keys.add(k)
                eq = np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
                if not eq:
                    nonpose_diff[k] = nonpose_diff.get(k, 0) + 1
            mf, mp = df_["obj_poses_mask"] > 0, dp_["obj_poses_mask"] > 0
            m_and += int((mf & mp).sum()); m_fin_only += int((mf & ~mp).sum()); m_pre_only += int((~mf & mp).sum())
            both = mf & mp
            pose_valid_rows += int(both.sum())
            pose_changed += int((np.abs(df_["obj_poses"][both] - dp_["obj_poses"][both]).max(-1) > 1e-6).sum())
        print(f"[{arm}] {len(idx)} scenes, non-pose keys checked: {len(other_keys)}; differing: {nonpose_diff or 'none'}")
        print(f"[{arm}] pose keys: {sorted(pose_keys_seen)}")
        print(f"[{arm}] pose mask: valid in both {m_and}, final only {m_fin_only} "
              f"({m_fin_only / max(m_and + m_fin_only, 1):.2%}), pre-fit only {m_pre_only}")
        print(f"[{arm}] valid pose rows that differ: {pose_changed}/{pose_valid_rows} "
              f"({pose_changed / max(pose_valid_rows, 1):.1%})")
        bad += bool(nonpose_diff) + (m_pre_only > 0) + (pose_changed == 0)
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
