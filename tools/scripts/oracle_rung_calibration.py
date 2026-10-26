#!/usr/bin/env python
"""What median bearing error does each oracle rung actually deliver?

The ladder's rungs are LABELLED by the measured median absolute error of the channel the
branch receives, not by the nominal sigma, and the paper's floor bracket is quoted in those
labels. The labels in detection_floor.py (5.5, 10.2, 14.6, 27.2, 37.9, 45.0, 57.4 deg) were
measured once, ad hoc, and never re-derived. Two things make that worth checking before a
second ladder inherits them:

  * The configured sigmas are exactly rung_key / 0.6745 -- 7.4, 14.8, 22.2, 37.1, ... -- i.e.
    chosen so that the asymptotic median of |N(0, sigma)| equals the rung key. So the labels
    are known by construction to be 5, 10, 15, 25, 35, 45, 60 deg, and the recorded values
    scatter around that by +0.5/-0.4 at the low rungs and +3.0/-2.3 at the high ones. The
    scatter grows with sigma exactly as a finite-sample median error does, which puts the
    original measurement at roughly 130 draws.
  * A rung's corruption is a fixed per-agent rotation drawn from (scene_id, sigma, slot), so
    the delivered error distribution does not depend on the data. It should read the same at
    579 scenes and at 25k -- but "should" is why this script exists.

This measures the delivered channel directly: for every scored agent it compares the unit
vector in obj_poses[..., 0:2] (and 3:5, the +8 s pair) against the clean future displacement
bearing recomputed from obj_trajs_future_state, on the validation split, filtered to agents
that actually move (|displacement| >= 0.5 m, as in facing_error_ladder_units.py -- a bearing
is undefined for a still agent and the oracle encodes a zero vector for one).

Usage: python tools/scripts/oracle_rung_calibration.py [--scenes N] CFG [CFG ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from mtr.config import cfg, cfg_from_yaml_file                      # noqa: E402
from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset  # noqa: E402

MIN_DISP = 0.5      # m
IDX3, IDX8 = 29, 79  # +3 s and +8 s on the 80-step future grid at 10 Hz


def _last_valid_disp(fut, fmask, upto):
    """Replicates the oracle's own choice of displacement: last valid step at or before upto."""
    m = fmask[:, :upto + 1]
    any_valid = m.any(-1)
    last = (m.shape[-1] - 1) - m[:, ::-1].argmax(-1)
    d = np.take_along_axis(fut[:, :upto + 1, :2], last[:, None, None], axis=1)[:, 0, :]
    return d * any_valid[:, None], any_valid


def measure(cfg_file: str, n_scenes: int) -> dict:
    cfg_from_yaml_file(cfg_file, cfg)
    dc = cfg.DATA_CONFIG
    sigma = float(dc.get('ORACLE_POSE_NOISE_DEG', 0.0))
    # the dataset logs unconditionally through self.logger, so pass a real one
    log = logging.getLogger('rung_calibration')
    log.addHandler(logging.NullHandler())
    ds = WaymoPoseDataset(dataset_cfg=dc, training=False, logger=log)
    step = max(len(ds) // n_scenes, 1)
    err3, err8 = [], []
    for i in range(0, len(ds), step):
        d = ds[i]
        if d is None:
            continue
        poses, fut = d['obj_poses'], d['obj_trajs_future_state']
        fmask = d['obj_trajs_future_mask'] > 0
        tip = d['track_index_to_predict']
        C = poses.shape[0]
        rows = np.arange(C)
        # the agent's OWN channel, as the decoder reads it for that centre object
        own = poses[rows, tip]                     # (C, T, >=6)
        # the channel is constant over observed frames; take the last observed one
        own = own[:, -1, :6]
        f, fm = fut[rows, tip], fmask[rows, tip]   # (C, H, >=2), (C, H)
        for idx, bucket, ch in ((IDX3, err3, 0), (IDX8, err8, 3)):
            d_xy, ok = _last_valid_disp(f, fm, idx)
            r = np.linalg.norm(d_xy, axis=-1)
            keep = ok & (r >= MIN_DISP) & (np.linalg.norm(own[:, ch:ch + 2], axis=-1) > 1e-3)
            if not keep.any():
                continue
            clean = np.arctan2(d_xy[keep, 1], d_xy[keep, 0])
            got = np.arctan2(own[keep, ch + 1], own[keep, ch])
            bucket.extend(np.degrees(np.abs(np.arctan2(np.sin(got - clean),
                                                       np.cos(got - clean)))))
    out = dict(cfg=os.path.basename(cfg_file), sigma_deg=sigma,
               asymptotic_deg=round(0.6745 * sigma, 2))
    for lab, v in (('+3s', err3), ('+8s', err8)):
        a = np.asarray(v)
        out[lab] = (dict(n=int(a.size), median=float(np.median(a)),
                         p90=float(np.percentile(a, 90)))
                    if a.size else dict(n=0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cfgs', nargs='+')
    ap.add_argument('--scenes', type=int, default=400)
    a = ap.parse_args()
    print(f'{"config":58}{"sigma":>7}{"0.6745s":>9}{"n":>7}{"med+3s":>8}{"med+8s":>8}')
    print('-' * 97)
    for c in a.cfgs:
        p = c if os.path.isabs(c) else os.path.join(REPO, 'tools', 'cfgs', 'waymo', c)
        if not p.endswith('.yaml'):
            p += '.yaml'
        r = measure(p, a.scenes)
        m3 = r['+3s'].get('median'); m8 = r['+8s'].get('median')
        print(f'{r["cfg"][:58]:58}{r["sigma_deg"]:>7.1f}{r["asymptotic_deg"]:>9.1f}'
              f'{r["+3s"]["n"]:>7}'
              f'{(f"{m3:.2f}" if m3 is not None else "-"):>8}'
              f'{(f"{m8:.2f}" if m8 is not None else "-"):>8}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
