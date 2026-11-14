#!/usr/bin/env python
"""Is every exported pose sequence attached to its own track?

The first export keyed pose clips by an object identifier that is unique only within a
scene, so most sequences carried another pedestrian's body. The per-scene re-export fixed
the key. This checks the result directly: for every pose file <scene>/<id>.npz, the
horizontal distance between the body's root and the box centre of track <id> in the same
scene, on every frame the file covers. A body that belongs to another pedestrian sits in
another place, usually another scene.

A distance threshold cannot tell a pedestrian from a neighbour in a crowd, so the check
also asks the sharper question: on each posed frame, is the body's root nearer to its own
track's box centre than to the box centre of every other agent in the scene?

Root positions are the export's `trans`. Reads the 25k export and its processed scenarios.
Writes experiments/pose_track_association.json.

Usage: python tools/scripts/pose_track_association.py [--scenarios DIR] [validation] [training]
--scenarios names another copy of the processed scenarios (the same files on a faster
disk). The default is validation alone. The training split is 163,000 files on the disk the
training runs read from, so do not run it beside a training job.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POSES = '/home/erik/big1/waymo_motion_poses/scale_25k'
SCEN = '/home/erik/big1/waymo_agent_batch_25k'
OUT = os.path.join(REPO, 'experiments', 'pose_track_association.json')
DT = 0.1
THRESHOLDS_M = (0.25, 0.5, 1.0)


def one_scene(args: tuple) -> list:
    """Per pose file: (worst, n_frames, n_nearest).
    worst is the largest root-to-box-centre distance over its frames, in metres: nan if the
    track is missing from the scenario, inf if a posed frame has no valid box. n_nearest
    counts the frames on which no other valid agent's box centre is nearer to the root."""
    split, scene = args
    info = pickle.load(open(os.path.join(
        SCEN, f'processed_scenarios_{split}', f'sample_{scene}_0.pkl'), 'rb'))
    ids = [str(i) for i in info['track_infos']['object_id']]
    trajs = np.asarray(info['track_infos']['trajs'])
    out = []
    d = os.path.join(POSES, split, scene)
    for name in sorted(os.listdir(d)):
        if not name.endswith('.npz'):
            continue
        z = np.load(os.path.join(d, name))
        if name[:-4] not in ids:
            out.append((float('nan'), 0, 0))
            continue
        own = ids.index(name[:-4])
        step = np.rint(z['waymo_timestamps'].astype(np.float64) / 1e6 / DT).astype(int)
        root = np.asarray(z['trans'][:, :2], np.float64)
        if (trajs[own, step, 9] <= 0).any():
            out.append((float('inf'), 0, 0))
            continue
        dist = np.linalg.norm(trajs[:, step, :2] - root[None], axis=2)      # (agents, frames)
        dist[trajs[:, step, 9] <= 0] = np.inf
        out.append((float(dist[own].max()), len(step), int((dist.argmin(0) == own).sum())))
    return out


def main() -> None:
    global SCEN
    res = dict(_meta=dict(poses=POSES, scenarios=SCEN, note=(
        'worst_m is, per pose file, the largest horizontal distance between the root and the '
        'box centre of the same scene and object id over the frames the file covers. '
        'own_track_nearest counts posed frames on which no other valid agent of the scene has '
        'its box centre nearer to the root.')))
    argv = sys.argv[1:]
    if '--scenarios' in argv:
        SCEN = argv.pop(argv.index('--scenarios') + 1)
        argv.remove('--scenarios')
        res['_meta']['scenarios'] = SCEN
    for split in (argv or ['validation']):
        scenes = sorted(os.listdir(os.path.join(POSES, split)))
        with Pool(4) as p:
            rows = np.array([x for r in p.imap_unordered(
                one_scene, [(split, s) for s in scenes], chunksize=32) for x in r])
        worst, n_frames, n_nearest = rows[:, 0], rows[:, 1], rows[:, 2]
        ok = np.isfinite(worst)
        res[split] = dict(
            n_scenes=len(scenes), n_files=int(len(worst)),
            n_track_missing=int(np.isnan(worst).sum()),
            n_posed_frame_without_box=int(np.isinf(worst).sum()),
            median_worst_m=float(np.median(worst[ok])), p99_worst_m=float(np.percentile(worst[ok], 99)),
            max_worst_m=float(worst[ok].max()),
            share_within_pct={f'{t:g}': float(100 * (ok & (worst <= t)).sum() / len(worst))
                              for t in THRESHOLDS_M},
            n_frames=int(n_frames.sum()),
            own_track_nearest_pct_of_frames=float(100 * n_nearest.sum() / n_frames.sum()),
            own_track_nearest_on_every_frame_pct_of_files=float(
                100 * (ok & (n_nearest == n_frames)).sum() / len(worst)))
        print(split, json.dumps(res[split]))
    json.dump(res, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
