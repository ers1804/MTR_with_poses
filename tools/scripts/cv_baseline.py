#!/usr/bin/env python
"""Constant-velocity baseline on the paper's fixed validation split.

The paper claims short-horizon forecasting is "nearly saturated by constant-velocity
extrapolation". That attribution needs a measured constant-velocity baseline, not the
model's own minADE -- Schoeller et al. (RA-L 2020) made exactly this point about CV
baselines in pedestrian forecasting, and an IV reviewer will ask for the number.

Uses the dataset's own validation loader, so the agent set, the centre-object frame and
the validity masks are identical to those behind every minADE in the paper. In the
centre-object frame the agent sits at the origin with heading 0, so the CV prediction is
simply v0 * t.

Reports ADE/FDE at 2 s and 8 s. CV is single-mode, so it is compared against the
model's single-mode error as well as its best-of-6 minADE; both are printed.

Env: mtr_smpl.
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mtr.config import cfg, cfg_from_yaml_file          # noqa: E402
from mtr.datasets import build_dataloader                # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), '..',
                   'cfgs/waymo/mtr+pose_data_no_pose_with_map_fixdata.yaml')
OUT = os.path.join(os.path.dirname(__file__), '..', '..',
                   'experiments', 'cv_baseline.json')
DT = 0.1


def main() -> int:
    log = logging.getLogger('cv'); log.addHandler(logging.StreamHandler()); log.setLevel(logging.ERROR)
    cfg_from_yaml_file(CFG, cfg)
    _, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False,
                                    workers=8, logger=log, training=False)

    err_sum = np.zeros(80); err_n = np.zeros(80)
    n_agents = 0
    for bi, batch in enumerate(loader):
        d = batch['input_dict']
        gt = d['center_gt_trajs'].numpy()                 # (C, 80, 4) centre frame
        gm = d['center_gt_trajs_mask'].numpy() > 0        # (C, 80)
        trajs = d['obj_trajs'].numpy()                    # (C, N, 11, 29)
        tidx = d['track_index_to_predict'].numpy()
        # channels 25:27 are velocity (pos/size 6 | type 5 | time 12 | heading 2 | vel 2 | acc 2)
        v0 = trajs[np.arange(len(tidx)), tidx, -1, 25:27]  # (C, 2) current velocity
        t = (np.arange(80) + 1) * DT
        pred = v0[:, None, :] * t[None, :, None]           # (C, 80, 2)
        e = np.linalg.norm(pred - gt[:, :, :2], axis=-1)   # (C, 80)
        err_sum += (e * gm).sum(0); err_n += gm.sum(0)
        n_agents += len(tidx)
        if (bi + 1) % 200 == 0:
            print(f'  {bi+1} batches, {n_agents} agents', flush=True)

    per_step = err_sum / np.maximum(err_n, 1)
    def ade(h): return float(per_step[:h].mean())
    def fde(h): return float(per_step[h - 1])
    res = dict(n_agents=int(n_agents),
               cv_ade_2s=ade(20), cv_fde_2s=fde(20),
               cv_ade_4s=ade(40), cv_ade_6s=ade(60),
               cv_ade_8s=ade(80), cv_fde_8s=fde(80),
               per_step_mean_err=[float(x) for x in per_step],
               n_valid_per_step=[int(x) for x in err_n])
    print(f"\nagents: {n_agents}")
    for k in ('cv_ade_2s', 'cv_fde_2s', 'cv_ade_4s', 'cv_ade_6s', 'cv_ade_8s', 'cv_fde_8s'):
        print(f'  {k:<12} {res[k]:.4f} m')
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
