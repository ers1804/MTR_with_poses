#!/usr/bin/env python
"""Per-pedestrian error of the simulated causal tracker at prediction time.

Every result.pkl of a track-noise run stores the history the model was given: the past of
gt_trajs is the tracker's output (_track_past_causally in waymo_pose_dataset.py), the
future is the label. The detection noise is drawn once per (scene, object), so the tracked
history is the same in every arm and seed of a noise level, and one file gives it.

Compares the tracked state at the last observed frame with the label's, from the scenario
files, for every predicted pedestrian valid there in both:

  pos_err  || tracked position - label position ||   (m)
  vel_err  || tracked velocity - label velocity ||   (m/s)
  n_obs    valid observed frames out of 11

Usage: python tools/scripts/tracker_state_error.py [noise_cm]
Writes experiments/val_tracker_state_posnoise<noise_cm>.pkl.
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = os.environ.get('METRICS', '/home/erik/big1/heading_noise_metrics')
SCEN = os.environ.get(
    'SCEN', '/home/erik/big1/waymo_agent_batch_25k/processed_scenarios_validation')
CUR = 10


def main() -> None:
    rung = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = f'cross_attn_pe_with_map_prefit_posnoise{rung:02d}_scale25k'
    path = os.path.join(METRICS, f'mtr+pose_data_{cfg}', f'PF_25k_{cfg}_lr25_s101', 'eval',
                        'eval_with_train', 'epoch_30', 'result.pkl')
    tracked = {}
    for item in pickle.load(open(path, 'rb')):
        for r in (item if isinstance(item, list) else [item]):
            tracked.setdefault(str(r['scenario_id']), {})[str(r['object_id'])] = r['gt_trajs']

    out, future_differs = {}, 0
    for scene, objs in tracked.items():
        info = pickle.load(open(glob.glob(os.path.join(SCEN, f'sample_{scene}_*.pkl'))[0], 'rb'))
        ids = [str(x) for x in info['track_infos']['object_id']]
        trajs = np.asarray(info['track_infos']['trajs'])
        for oid, t in objs.items():
            if oid not in ids:
                continue
            c = trajs[ids.index(oid)]
            if t[CUR, 9] <= 0 or c[CUR, 9] <= 0:
                continue
            fv = t[CUR + 1:, 9] > 0
            future_differs += not np.allclose(t[CUR + 1:, :2][fv], c[CUR + 1:, :2][fv], atol=1e-3)
            out[(scene, oid)] = dict(
                pos_err=float(np.linalg.norm(t[CUR, :2] - c[CUR, :2])),
                vel_err=float(np.linalg.norm(t[CUR, 7:9] - c[CUR, 7:9])),
                n_obs=int((t[:CUR + 1, 9] > 0).sum()))
    assert future_differs == 0, 'the stored future is not the label'

    dst = os.path.join(REPO, 'experiments', f'val_tracker_state_posnoise{rung:02d}.pkl')
    pickle.dump(out, open(dst, 'wb'))
    pe = np.array([v['pos_err'] for v in out.values()])
    ve = np.array([v['vel_err'] for v in out.values()])
    full = np.array([v['n_obs'] == CUR + 1 for v in out.values()])
    print(f'{len(out)} pedestrians valid at prediction time -> {dst}')
    print(f'  position error median {np.median(pe):.3f} m, velocity error median '
          f'{np.median(ve):.3f} m/s')
    print(f'  full 11-frame history {full.mean():.1%}; velocity error median '
          f'{np.median(ve[full]):.3f} on those, {np.median(ve[~full]):.3f} on shorter tracks')


if __name__ == '__main__':
    main()
