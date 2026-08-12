#!/usr/bin/env python
"""Re-cluster MTR intention (goal) points for a 2 s prediction horizon.

MTR's decoder classifies over 64 static "intention points" that are k-means centres of
future ENDPOINTS in the agent-centric frame. The shipped pedestrian set was clustered
for the 8 s horizon and spans 1.7-49.2 m, but a pedestrian covers only ~3 m in 2 s, so
reusing it would leave nearly every goal unreachable and the goal-classification head
mis-specified -- confounding any 8 s vs 2 s comparison.

This regenerates the pedestrian intention points from 2 s endpoints of the training
split (vehicle/cyclist entries are copied through unchanged; this project is
pedestrian-only).

Usage: python make_intention_points_2s.py [--steps 20] [--k 64]
"""
import argparse
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20, help='future steps (20 = 2.0 s @10Hz)')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--cfg', default=os.path.join(
        REPO, 'tools', 'cfgs', 'waymo', 'mtr+pose_data_cross_attn_pe_with_map.yaml'))
    ap.add_argument('--src', default=os.path.join(REPO, 'data', 'waymo', 'cluster_64_center_dict.pkl'))
    ap.add_argument('--out', default=os.path.join(REPO, 'data', 'waymo', 'cluster_64_center_dict_2s.pkl'))
    args = ap.parse_args()

    import logging
    from sklearn.cluster import KMeans
    from mtr.config import cfg, cfg_from_yaml_file
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset

    log = logging.getLogger('ip'); log.addHandler(logging.NullHandler())
    cfg_from_yaml_file(args.cfg, cfg)
    ds = WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG, training=True, logger=log)
    print(f'training scenes: {len(ds)}')

    idx = args.steps - 1          # 0-indexed endpoint of the shortened horizon
    endpoints = []
    for i in range(len(ds)):
        try:
            d = ds[i]
        except Exception:
            continue
        gt, mask = d['center_gt_trajs'], d['center_gt_trajs_mask']
        for k in range(gt.shape[0]):
            if mask[k, idx] > 0:                      # endpoint must be observed
                endpoints.append(gt[k, idx, :2])
        if (i + 1) % 200 == 0:
            print(f'  ...{i+1}/{len(ds)} scenes, {len(endpoints)} endpoints', flush=True)

    endpoints = np.asarray(endpoints, dtype=np.float64)
    r = np.linalg.norm(endpoints, axis=-1)
    print(f'\n{len(endpoints)} valid {args.steps/10:.1f}s endpoints | '
          f'radius min/median/max = {r.min():.2f}/{np.median(r):.2f}/{r.max():.2f} m')

    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(endpoints)
    centres = km.cluster_centers_.astype(np.float32)
    rc = np.linalg.norm(centres, axis=-1)
    print(f'{args.k} centres | radius min/max = {rc.min():.2f}/{rc.max():.2f} m')

    with open(args.src, 'rb') as fh:
        d = pickle.load(fh)
    d['TYPE_PEDESTRIAN'] = centres                    # other types untouched (unused here)
    with open(args.out, 'wb') as fh:
        pickle.dump(d, fh)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
