#!/usr/bin/env python
"""No-GT pose-quality proxies for WOMD final pose data: corrupt vs fixed export.

Two checks, both anchored on the INDEPENDENT Waymo GT trajectory (processed_scenarios):
  1. heading alignment: angle between SMPL root forward direction and GT walking
     direction (walking frames only, speed > 0.7 m/s)
  2. articulation-speed coupling: across pedestrians, Pearson r between leg-joint
     angular velocity magnitude and GT walking speed
Correct per-pedestrian poses should beat cross-scene donor content on both.

Writes experiments/pose_proxy_quality.json
"""
import json
import os
import pickle

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CORRUPT = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/generated_motion/final_10fps/training'
FIXED = '/home/erik/big1/waymo_motion_poses/final_10fps/training'
SCEN = '/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_training'
LEG_IDX = [0, 1, 3, 4, 6, 7]        # L/R hip, knee, ankle in SMPL body-joint order


def aa_to_forward(aa):
    """Root axis-angle (T,3) -> forward (facing) unit vector in XY plane (T,2)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.where(theta > 1e-8, aa / np.maximum(theta, 1e-8), 0.0)
    ct, st = np.cos(theta), np.sin(theta)
    # rotate SMPL forward axis. SMPL neutral faces +z is 'up' y? Use column of R:
    # R @ [0,0,1]: forward for SMPL (z axis of body frame)
    v = np.array([0.0, 0.0, 1.0])
    kv = np.cross(k, np.broadcast_to(v, k.shape))
    kdv = (k * v).sum(-1, keepdims=True)
    rv = v * ct + kv * st + k * kdv * (1 - ct)
    f = rv[:, :2]
    n = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.maximum(n, 1e-8)


def evaluate(root_dir, max_scenes=250):
    head_errs, artic, speeds = [], [], []
    scenes = sorted(os.listdir(CORRUPT))[:max_scenes]      # scenes present in BOTH
    for scene in scenes:
        pk = os.path.join(SCEN, f'sample_{scene}_0.pkl')
        d = os.path.join(root_dir, scene)
        if not (os.path.isdir(d) and os.path.exists(pk)):
            continue
        info = pickle.load(open(pk, 'rb'))
        ti = info['track_infos']
        ids = list(ti['object_id'])
        for f in os.listdir(d):
            if not f.endswith('.npz'):
                continue
            try:
                oi = ids.index(int(f[:-4]))
            except ValueError:
                continue
            z = np.load(os.path.join(d, f), allow_pickle=True)
            n = len(z['trans'])
            tr = ti['trajs'][oi]
            valid = tr[:, -1] > 0
            if n < 6 or valid.sum() < 6:
                continue
            gt = tr[valid][:, :2]
            m = min(n, len(gt))
            vel = np.diff(gt[:m], axis=0) * 10.0           # m/s at 10 Hz
            speed = np.linalg.norm(vel, axis=-1)
            walking = speed > 0.7
            if walking.sum() >= 3:
                fwd = aa_to_forward(z['root_orient'][:m - 1])
                hd = vel / np.maximum(np.linalg.norm(vel, axis=-1, keepdims=True), 1e-8)
                cosang = np.clip((fwd * hd).sum(-1), -1, 1)
                errs = np.degrees(np.arccos(cosang))[walking]
                # allow global forward-axis convention flip
                head_errs.append(min(errs.mean(), 180 - errs.mean()))
            pb = z['pose_body'][:m].reshape(m, -1, 3)
            legs = pb[:, LEG_IDX]
            w = np.linalg.norm(np.diff(legs, axis=0), axis=-1).mean() * 10.0
            artic.append(w)
            speeds.append(speed.mean())
    artic, speeds = np.array(artic), np.array(speeds)
    r = float(np.corrcoef(artic, speeds)[0, 1]) if len(artic) > 10 else float('nan')
    return {
        'n_heading': len(head_errs), 'heading_err_deg_mean': float(np.mean(head_errs)),
        'heading_err_deg_median': float(np.median(head_errs)),
        'n_coupling': len(artic), 'articulation_speed_pearson_r': r,
    }


if __name__ == '__main__':
    out = {'corrupt': evaluate(CORRUPT), 'fixed': evaluate(FIXED)}
    print(json.dumps(out, indent=2))
    path = os.path.join(REPO, 'experiments', 'pose_proxy_quality.json')
    json.dump(out, open(path, 'w'), indent=2)
    print('wrote', path)
