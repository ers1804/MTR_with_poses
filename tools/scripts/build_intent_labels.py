#!/usr/bin/env python
"""Build pre-registered will-enter-roadway labels (R2) from WORLD-frame GT.

Label (fixed in to_human/project_review_2026-07-20.md Part 6 BEFORE any results):
  population: pedestrians off-road at t0 (min dist to any LANE polyline > 3.0 m)
              with >= 10 valid future steps
  positive:   any valid GT future step within 1.5 m of a lane polyline
  horizons:   primary 8 s (80 steps), secondary 3 s (30 steps)

Reads the same NPZ subset + processed_scenarios pkls the forecasting dataset uses,
so labels are keyed by (scenario_id, object_id) and join everything else.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NPZ_ROOT = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/generated_motion/final_10fps'
SCEN_ROOT = '/home/erik/NAS/personal/waymo_agent_batch'
D_OFF, D_ON = 3.0, 1.5
NUM_PAST, CUR = 11, 10

def lane_points(map_infos):
    pl = map_infos['all_polylines']
    pts = []
    for lane in map_infos.get('lane', []):
        s, e = lane['polyline_index']
        pts.append(pl[s:e, :2])
    return np.concatenate(pts, axis=0) if pts else None

def min_dist(points, lanes):
    """points (N,2) x lanes (M,2) -> (N,) min distance, chunked."""
    out = np.empty(len(points))
    for i in range(0, len(points), 256):
        chunk = points[i:i+256]
        d = np.linalg.norm(chunk[:, None, :] - lanes[None, :, :], axis=-1)
        out[i:i+256] = d.min(axis=1)
    return out

def main(split):
    labels = {}
    stats = {'scenes': 0, 'no_lanes': 0, 'peds': 0, 'onroad_t0': 0, 'short_future': 0, 'labeled': 0, 'pos8': 0, 'pos3': 0}
    scene_dirs = sorted(glob.glob(os.path.join(NPZ_ROOT, split, '*')))
    scen_split = 'validation' if split == 'validation' else 'training'
    for sd in scene_dirs:
        scene = os.path.basename(sd)
        pk = os.path.join(SCEN_ROOT, f'processed_scenarios_{scen_split}', f'sample_{scene}_0.pkl')
        if not os.path.exists(pk):
            continue
        d = pickle.load(open(pk, 'rb'))
        stats['scenes'] += 1
        lanes = lane_points(d['map_infos'])
        if lanes is None:
            stats['no_lanes'] += 1
            continue
        ti = d['track_infos']
        ids = ti['object_id']
        trajs = ti['trajs']  # (num_objects, 91, 10)
        npz_ids = {int(os.path.splitext(os.path.basename(f))[0]) for f in glob.glob(os.path.join(sd, '*.npz'))}
        for oi, oid in enumerate(ids):
            if int(oid) not in npz_ids:
                continue
            stats['peds'] += 1
            tr = trajs[oi]
            if tr[CUR, -1] <= 0:
                continue
            fut = tr[NUM_PAST:]
            fv = fut[:, -1] > 0
            if fv.sum() < 10:
                stats['short_future'] += 1
                continue
            d0 = min_dist(tr[CUR:CUR+1, :2], lanes)[0]
            if d0 <= D_OFF:
                stats['onroad_t0'] += 1
                continue
            dists = min_dist(fut[fv][:, :2], lanes)
            steps = np.where(fv)[0]
            enter8 = bool((dists < D_ON).any())
            m3 = steps < 30                      # dists is aligned with the valid steps
            enter3 = bool((dists[m3] < D_ON).any()) if m3.any() else False
            labels[f'{scene}|{oid}'] = {'enter_8s': enter8, 'enter_3s': enter3, 'd0': float(d0)}
            stats['labeled'] += 1
            stats['pos8'] += int(enter8); stats['pos3'] += int(enter3)
    print(split, stats)
    return labels

if __name__ == '__main__':
    out = {}
    for split in ['training', 'validation']:
        out[split] = main(split)
    path = os.path.join(REPO, 'experiments', 'intent_labels.json')
    json.dump(out, open(path, 'w'))
    for s in out:
        n = len(out[s]); p8 = sum(v['enter_8s'] for v in out[s].values())
        print(f'{s}: {n} labeled, positives 8s: {p8} ({100*p8/max(n,1):.1f}%)')
    print('wrote', path)
