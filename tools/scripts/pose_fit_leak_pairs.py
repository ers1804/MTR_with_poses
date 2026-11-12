#!/usr/bin/env python
"""Pair the pre-fit (per-frame LiDAR-HMR) and final (NeMF-fitted) pose of each pedestrian.

Input to pose_fit_leak_speed.py. Read only. Covers the validation scenes the model is
scored on (5,171) and keeps a pedestrian when both poses exist on all 11 observed frames
with identical timestamps (11,588). Also stores the box track and the mean LiDAR returns.

Usage: python tools/scripts/pose_fit_leak_pairs.py [/home/erik/big1/pose_fit_leak/pairs_val.npz]
"""
import os, sys, glob, pickle
import numpy as np
from multiprocessing import Pool

R = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed'
FIN = '/home/erik/big1/waymo_motion_poses/final_10fps_subset579/validation'
RAW = f'{R}/amass_format_motion/raw/validation'
EXT = f'{R}/waymo_motion_extracted/validation'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/erik/big1/pose_fit_leak/pairs_val.npz'


def one_scene(scene):
    rows = []
    stats = dict(final=0, no_raw=0, short_raw=0, ts_mismatch=0, kept=0)
    npts, rng = {}, {}
    try:
        for rec in pickle.load(open(f'{EXT}/{scene}.pkl', 'rb')):
            oid = str(rec['basic_info']['object_id'])
            pcd = rec['vehicle_frame']['pedestrian_pcd']
            npts.setdefault(oid, []).append(0 if pcd is None else len(pcd))
            rng.setdefault(oid, []).append(float(np.linalg.norm(rec['vehicle_frame']['center'][:2])))
    except Exception:
        pass
    for f in sorted(glob.glob(f'{FIN}/{scene}/*.npz')):
        oid = os.path.basename(f)[:-4]
        stats['final'] += 1
        rp = f'{RAW}/{scene}/{oid}.npz'
        if not os.path.exists(rp):
            stats['no_raw'] += 1
            continue
        fin = np.load(f, allow_pickle=True)
        raw = np.load(rp, allow_pickle=True)
        if raw['pose_body'].shape[0] != 11 or fin['pose_body'].shape[0] != 11:
            stats['short_raw'] += 1
            continue
        if not np.array_equal(raw['waymo_timestamps'], fin['waymo_timestamps']) or \
                not np.array_equal(raw['smpl_timestamps'], raw['waymo_timestamps']):
            stats['ts_mismatch'] += 1
            continue
        stats['kept'] += 1
        rows.append(dict(
            scene=scene, oid=oid,
            fin_body=fin['pose_body'].astype(np.float32), fin_root=fin['root_orient'].astype(np.float32),
            raw_body=raw['pose_body'].astype(np.float32), raw_root=raw['root_orient'].astype(np.float32),
            center=raw['world_center'].astype(np.float64), heading=raw['heading_world'].astype(np.float32),
            fin_trans=fin['trans'].astype(np.float64), raw_trans=raw['trans'].astype(np.float64),
            npts=float(np.mean(npts.get(oid, [np.nan]))), rng=float(np.mean(rng.get(oid, [np.nan]))),
        ))
    return rows, stats


if __name__ == '__main__':
    scenes = sorted(os.listdir(FIN))
    print(len(scenes), 'scenes', flush=True)
    rows, tot = [], {}
    with Pool(24) as p:
        for i, (r, s) in enumerate(p.imap_unordered(one_scene, scenes, chunksize=8)):
            rows += r
            for k, v in s.items():
                tot[k] = tot.get(k, 0) + v
            if i % 500 == 0:
                print(i, tot, flush=True)
    print('TOTAL', tot, flush=True)
    keys = ['fin_body', 'fin_root', 'raw_body', 'raw_root', 'center', 'heading', 'fin_trans', 'raw_trans']
    out = {k: np.stack([r[k] for r in rows]) for k in keys}
    out['scene'] = np.array([r['scene'] for r in rows])
    out['oid'] = np.array([r['oid'] for r in rows])
    out['npts'] = np.array([r['npts'] for r in rows])
    out['rng'] = np.array([r['rng'] for r in rows])
    np.savez_compressed(OUT, **out)
    print('saved', OUT, {k: v.shape for k, v in out.items()}, flush=True)
