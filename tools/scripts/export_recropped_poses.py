#!/usr/bin/env python
"""Export poses estimated from re-cropped points in the layout WaymoPoseDataset reads.

The counterpart of export_prefit_poses.py for the poses that recrop_noisy_boxes.py and
LiDAR-HMR/gen_lidar_hmr_waymo.py produce: per-frame estimates, no motion prior, no fit to
the track. It reads the estimator's result files directly,

    <hmr>/<split>/<scene>/<id>/<timestamp>.npz        (key SMPL_lidarhmr)

which is what Waymo-3DSkelMo scripts/convert_amass.py stacks into amass_format_motion/raw
(sorted by timestamp, nothing else), and mirrors the final export file for file:

    <out>/<split>/<scene>/<id>.npz
        root_orient (N, 3)   LiDAR-HMR on the re-cropped points, world frame
        pose_body   (N, 69)
        betas       (10,)    of the first exported frame, as in the original pipeline
        trans       (N, 3)   the FINAL export's rows at the same timestamps (only read
                             when a scene has no processed WOMD track)
        waymo_timestamps (N,)

Unlike export_prefit_poses.py it cannot demand that the result equals the first export:
  * a displaced box can miss the pedestrian on a frame, so rows are fewer. A pedestrian
    without a single estimate gets no file and leaves the dataset for every arm trained on
    this export (counted as files_without_estimate).
  * a displaced box can catch points on a frame where the label's box was empty. If that
    frame is not a row of the final export it is dropped (rows_outside_final).
  * the shape is estimated again, so betas differ.

With --compare <label-crop pre-fit export> the manifest also says how far the new poses
are from those estimated in the label's box, frame by frame: the angle between the two
root orientations and the mean angle over the 23 body joints. Run on a re-crop with no
noise this gives the floor of the comparison, since the estimator samples its input
points at random.

    python tools/scripts/export_recropped_poses.py \
        --final /home/erik/big1/waymo_motion_poses/scale_25k \
        --hmr   /home/erik/ssd2/recrop/pos30_head25_hmr \
        --out   /home/erik/ssd2/recrop/scale_25k_prefit_crop30 \
        --compare /home/erik/big1/waymo_motion_poses/scale_25k_prefit
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np


def angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between axis-angle rotations, row by row."""
    from scipy.spatial.transform import Rotation
    return np.degrees((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())


def read_estimates(obj_dir: str):
    """Stack one pedestrian's per-frame results, sorted by timestamp."""
    if not os.path.isdir(obj_dir):
        return None
    rows = []
    for name in os.listdir(obj_dir):
        if not name.endswith('.npz'):
            continue
        with np.load(os.path.join(obj_dir, name), allow_pickle=True) as z:
            smpl, info = z['SMPL_lidarhmr'].item(), z['basic_info'].item()
        ts = int(info['time_stamp'])
        assert ts == int(name[:-4]), (obj_dir, name, ts)
        rows.append((ts, smpl['root_orient'], smpl['pose_body'], smpl['betas']))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    assert len(np.unique(ts)) == len(ts), (obj_dir, 'duplicate timestamps')
    return (ts, np.stack([r[1] for r in rows]).astype(np.float32),
            np.stack([r[2] for r in rows]).astype(np.float32),
            np.stack([r[3] for r in rows]).astype(np.float32))


def one_scene(args):
    final_root, out_root, hmr_root, compare_root, split, scene = args
    c, root_deg, body_deg = Counter(), [], []
    src = os.path.join(final_root, split, scene)
    dst = os.path.join(out_root, split, scene)
    for name in sorted(os.listdir(src)):
        if not name.endswith('.npz') or not name[:-4].isdigit():
            continue
        c['files_final'] += 1
        fin = np.load(os.path.join(src, name))
        fts = fin['waymo_timestamps'].astype(np.int64)
        assert np.all(np.diff(fts) > 0), (scene, name, 'unsorted timestamps')
        c['rows_final'] += len(fts)
        est = read_estimates(os.path.join(hmr_root, split, scene, name[:-4]))
        target = os.path.join(dst, name)
        if est is None:
            c['files_without_estimate'] += 1
            if os.path.exists(target):          # left by an earlier export into this directory
                os.remove(target)
            continue
        ts, root_orient, pose_body, betas = est
        assert root_orient.shape[1:] == (3,) and pose_body.shape[1:] == (69,) and betas.shape[1:] == (10,)
        keep = np.isin(ts, fts)
        c['rows_outside_final'] += int((~keep).sum())
        if not keep.any():
            c['files_without_estimate'] += 1
            if os.path.exists(target):
                os.remove(target)
            continue
        ts, root_orient, pose_body, betas = ts[keep], root_orient[keep], pose_body[keep], betas[keep]
        rows = np.searchsorted(fts, ts)
        assert np.array_equal(fts[rows], ts), (scene, name)
        os.makedirs(dst, exist_ok=True)
        tmp = os.path.join(dst, '.' + name + '.tmp')    # not *.npz: the dataset globs for those
        with open(tmp, 'wb') as f:
            np.savez(f, trans=fin['trans'][rows].astype(np.float64), root_orient=root_orient,
                     pose_body=pose_body, betas=betas[0], waymo_timestamps=ts)
        os.replace(tmp, target)
        c['files'] += 1
        c['rows'] += len(ts)
        if compare_root:
            p = os.path.join(compare_root, split, scene, name)
            if not os.path.exists(p):
                c['files_not_in_comparison'] += 1
                continue
            ref = np.load(p)
            rts = ref['waymo_timestamps'].astype(np.int64)
            c['rows_comparison'] += len(rts)
            both, i, j = np.intersect1d(ts, rts, return_indices=True)
            if len(both):
                root_deg.append(angles_deg(root_orient[i], ref['root_orient'][j]))
                body_deg.append(angles_deg(pose_body[i].reshape(-1, 3),
                                           ref['pose_body'][j].reshape(-1, 3)).reshape(-1, 23).mean(1))
    return c, (np.concatenate(root_deg) if root_deg else np.empty(0)), \
        (np.concatenate(body_deg) if body_deg else np.empty(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--final', required=True, help='final pose export, names the files and gives trans')
    ap.add_argument('--hmr', required=True, help='LiDAR-HMR results on the re-cropped points')
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare', help='pre-fit export from label boxes, for the pose difference')
    ap.add_argument('--splits', nargs='+', default=['validation', 'training'])
    ap.add_argument('--scene-list', help='file of scene ids, to restrict the run')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    only = set(open(a.scene_list).read().split()) if a.scene_list else None

    manifest = dict(final=a.final, hmr=a.hmr, compare=a.compare, scene_list=a.scene_list, splits={})
    for split in a.splits:
        scenes = sorted(d for d in os.listdir(os.path.join(a.final, split))
                        if os.path.isdir(os.path.join(a.final, split, d))
                        and (only is None or d in only))
        tot, root_deg, body_deg = Counter(), [], []
        with Pool(a.workers) as p:
            for c, r, b in p.imap_unordered(
                    one_scene, [(a.final, a.out, a.hmr, a.compare, split, s) for s in scenes],
                    chunksize=16):
                tot += c
                root_deg.append(r)
                body_deg.append(b)
        cell = dict(tot, scenes=len(scenes))
        if a.compare:
            r = np.concatenate(root_deg or [np.empty(0)])
            b = np.concatenate(body_deg or [np.empty(0)])
            cell['against_comparison'] = dict(
                frames=int(len(r)),
                root_angle_deg=dict(median=float(np.median(r)), mean=float(r.mean()),
                                    p90=float(np.percentile(r, 90)),
                                    share_over_90=float((r > 90).mean())) if len(r) else None,
                body_joint_angle_deg=dict(median=float(np.median(b)), mean=float(b.mean()),
                                          p90=float(np.percentile(b, 90))) if len(b) else None)
        manifest['splits'][split] = cell
        print(split, json.dumps(cell, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


if __name__ == '__main__':
    main()
