#!/usr/bin/env python
"""Per-pedestrian LiDAR return counts for the 25k validation agents.

The pose pipeline fitted SMPL to per-pedestrian point clouds, and
scripts/extract_waymo_motion.py kept them: one pickle per scenario, each record
a (object, frame) with `pedestrian_pcd`. Point count is the quantity that
actually governs mesh-fit quality -- it absorbs occlusion, which range does not.

Writes experiments/val_agent_pointcount_25k.pkl keyed (scenario_id, object_id):
  {'n_last': returns at the prediction frame, 'n_mean': mean over observed
   frames, 'n_max', 'n_frames'}

Usage: python tools/scripts/make_pointcounts_25k.py
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXTRACTED = ('/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/'
             'waymo_motion_extracted/validation')
METRICS = ('output/waymo/mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5/'
           'MS_scale25k_xattn_pe_s101/eval/eval_with_train/metrics_epoch_29.pkl')
OUT = os.path.join(REPO, 'experiments', 'val_agent_pointcount_25k.pkl')


def _one_scene(args):
    """Open a scenario pickle once and summarise every wanted pedestrian."""
    sid, wanted = args
    path = os.path.join(EXTRACTED, f'{sid}.pkl')
    if not os.path.exists(path):
        return sid, None
    try:
        recs = pickle.load(open(path, 'rb'))
    except Exception:
        return sid, None
    per = defaultdict(list)
    for r in recs:
        oid = str(r['basic_info']['object_id'])
        if oid not in wanted:
            continue
        pcd = r['vehicle_frame'].get('pedestrian_pcd')
        per[oid].append((int(r['basic_info']['time_stamp']),
                         0 if pcd is None else int(len(pcd))))
    out = {}
    for oid, v in per.items():
        v.sort()
        n = np.array([x[1] for x in v], dtype=float)
        out[oid] = {'n_last': float(n[-1]), 'n_mean': float(n.mean()),
                    'n_max': float(n.max()), 'n_frames': int(len(n))}
    return sid, out


def main() -> int:
    recs = pickle.load(open(os.path.join(REPO, METRICS), 'rb'))
    agents = {(str(r['scenario_id']), str(r['object_id'])) for r in recs}
    by_scene = defaultdict(set)
    for sid, oid in agents:
        by_scene[sid].add(oid)
    print(f'agents {len(agents)} across {len(by_scene)} scenarios')

    tasks = [(sid, oids) for sid, oids in by_scene.items()]
    counts, missing = {}, []
    done = 0
    # modest parallelism: these reads are over NFS
    with ProcessPoolExecutor(max_workers=8) as ex:
        for sid, res in ex.map(_one_scene, tasks, chunksize=8):
            done += 1
            if res is None:
                missing.append(sid)
            else:
                for oid, v in res.items():
                    counts[(sid, oid)] = v
            if done % 500 == 0:
                print(f'  {done}/{len(tasks)} scenarios, {len(counts)} agents',
                      flush=True)

    print(f'\nagents with point counts: {len(counts)} / {len(agents)}')
    print(f'scenario files missing: {len(missing)}')
    if counts:
        nl = np.array([v['n_last'] for v in counts.values()])
        print('returns at prediction frame: '
              + '  '.join(f'p{q}={np.percentile(nl, q):.0f}'
                          for q in (10, 25, 50, 75, 90)))
        print(f'  zero-return agents: {int((nl == 0).sum())}')
    pickle.dump(counts, open(OUT, 'wb'))
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
