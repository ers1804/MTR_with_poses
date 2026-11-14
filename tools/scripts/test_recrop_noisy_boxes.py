#!/usr/bin/env python
"""Tests for recrop_noisy_boxes.py. Two modes, because the two halves need two environments.

  noise  (mtr_smpl)      the offsets the crop uses are the ones the dataset gives its
                         tracker: recovered from WaymoPoseDataset._track_past_causally
                         frame by frame and from _degrade_past_heading, and compared.
  crop   (waymo_tf_gpu)  on a few scenes: with no noise the stage reproduces the original
                         extraction on the NAS entry by entry, and with noise every crop
                         is the label's point set cut by the displaced box, checked with
                         a second implementation in world coordinates.

    python tools/scripts/test_recrop_noisy_boxes.py noise [--mtr <checkout>]
    python tools/scripts/test_recrop_noisy_boxes.py crop [scene ...]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import types

import numpy as np

import recrop_noisy_boxes as R

STAGE = '/home/erik/ssd2/heldout_stage'
ORIGINAL = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/waymo_motion_extracted'


def test_noise(mtr_root: str) -> None:
    sys.path.insert(0, mtr_root)
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset as D
    src = sys.modules[D.__module__].__file__
    assert os.path.realpath(src).startswith(os.path.realpath(mtr_root) + os.sep), src
    R.check_noise_stream()
    T = R.NUM_PAST
    n = 0
    for scene, oid in (('1001ebb6d3905d92', '197'), ('abcdef0123456789', '4'), ('selfcheck', '7')):
        for s in (0.3, 0.1):
            fake = types.SimpleNamespace(position_noise_m=s, dt=0.1, num_past=T)
            # eleven copies of the track, copy i seen at frame i alone: the filter is born
            # there on the detection itself, so the output is label + noise of frame i
            trajs = np.zeros((T, 91, 10), dtype=np.float32)
            trajs[np.arange(T), np.arange(T), 9] = 1
            D._track_past_causally(fake, trajs, np.array([oid] * T), scene)
            got = trajs[np.arange(T), np.arange(T), 0:2]
            want = R.detection_noise(scene, oid, s)
            assert np.abs(got).min() > 0 and np.allclose(got, want, atol=1e-7), (scene, oid, s)
            n += 1
        if hasattr(D, '_degrade_past_heading'):
            fake = types.SimpleNamespace(track_heading_noise_deg=25.0, num_past=T)
            trajs = np.zeros((1, 91, 10), dtype=np.float32)
            D._degrade_past_heading(fake, trajs, np.array([oid]), scene)
            assert np.allclose(trajs[0, :T, 6], R.heading_error(scene, oid, 25.0), atol=1e-6)
            assert np.all(trajs[0, T:, 6] == 0)
            n += 1
    assert hasattr(D, '_degrade_past_heading'), 'this checkout has no per-track heading error'
    print(f'noise: {n} draws equal the dataset\'s ({mtr_root})')


def inside(points_world, center, size, heading, margin):
    """Box test in world coordinates, with the faces moved by margin (negative: shrunk)."""
    c, s = np.cos(heading), np.sin(heading)
    d = points_world.astype(np.float64) - center.astype(np.float64)
    local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    return np.all(np.abs(local) <= size / 2.0 + margin, axis=1)


def test_crop(scenes: list) -> None:
    final = os.path.join(STAGE, 'poses')
    scen = os.path.join(STAGE, 'scenarios')
    if not scenes:
        have = {n[7:-6] for n in os.listdir(os.path.join(scen, 'processed_scenarios_validation'))}
        scenes = sorted(have & set(os.listdir(os.path.join(final, 'validation'))))[:3]
    R.check_noise_stream()
    for scene in scenes:
        ids = sorted((n[:-4] for n in os.listdir(os.path.join(final, 'validation', scene))
                      if n.endswith('.npz')), key=int)
        SET = ((0.0, 0.0, 0.0, 0), (0.3, 25.0, 0.0, 0), (0.1, 25.0, 0.0, 0), (0.3, 0.0, 0.0, 0),
               (0.3, 25.0, 0.3, 0), (0.3, 25.0, 0.45, 1))
        crops, why = R.crop_scene('validation', scene, ids, scen, R.LIDAR, list(SET))
        assert crops, (scene, why)
        clean = crops[0][0]
        assert clean, scene
        with open(os.path.join(scen, 'processed_scenarios_validation', f'sample_{scene}_0.pkl'), 'rb') as f:
            stamps = [int(t * 1e6) for t in pickle.load(f)['timestamps_seconds']]
        # how far the vehicle's z axis leans from the world's, per frame
        lidar = R._stage()._load_scenario_from_tfrecord(os.path.join(R.LIDAR, 'validation', f'{scene}.tfrecord'))
        lean = [float(np.sin(np.arccos(np.clip(np.array(fr.pose.transform).reshape(4, 4)[2, 2], -1, 1))))
                for fr in lidar.compressed_frame_laser_data]

        # 1. no noise: the original extraction, entry by entry
        with open(os.path.join(ORIGINAL, 'validation', f'{scene}.pkl'), 'rb') as f:
            orig = {(e['basic_info']['object_id'], e['basic_info']['time_stamp']): e
                    for e in pickle.load(f) if e['basic_info']['object_id'] in set(ids)}
        mine = {(e['basic_info']['object_id'], e['basic_info']['time_stamp']): e for e in clean}
        assert set(mine) == set(orig), (scene, len(mine), len(orig))
        n_pts = n_diff = 0
        for k, e in mine.items():
            o = orig[k]
            assert e['basic_info'] == o['basic_info'], k
            for fr in ('vehicle_frame', 'world_frame'):
                for key in ('center', 'size', 'heading', 'bbox_vertices'):
                    assert np.allclose(e[fr][key], o[fr][key], atol=2e-5), (k, fr, key)
            a, b = e['world_frame']['pedestrian_pcd'], o['world_frame']['pedestrian_pcd']
            assert e['crop_box']['n_points_label_box'] == len(a)
            n_pts += len(b)
            if len(a) == len(b):
                assert np.allclose(a, b, atol=1e-4), k
                assert np.allclose(e['vehicle_frame']['pedestrian_pcd'],
                                   o['vehicle_frame']['pedestrian_pcd'], atol=1e-4), k
            else:                                   # a point on a box face, float32 heading
                n_diff += abs(len(a) - len(b))
        assert n_diff <= max(2, 1e-4 * n_pts), (scene, n_diff, n_pts)
        # the frames the estimator ran on must be rows of the final export
        for oid in ids:
            fts = set(np.load(os.path.join(final, 'validation', scene, f'{oid}.npz'))['waymo_timestamps'].tolist())
            for (o, ts), e in mine.items():
                if o == oid and len(e['world_frame']['pedestrian_pcd']):
                    assert ts in fts, (scene, oid, ts)

        # 2. with noise: the label's points cut by the displaced box
        line = f'{scene}: {len(ids)} pedestrians, {len(mine)} crops, {n_pts} points, ' \
               f'{n_diff} differ from the original extraction'
        for (s, deg, margin, refine), (noisy, c) in zip(SET[1:], crops[1:]):
            assert len(noisy) == len(clean)
            shifts, bands, errs, n_deep, n_out = [], [], [], 0, 0
            for e, l in zip(noisy, clean):
                oid, ts = e['basic_info']['object_id'], e['basic_info']['time_stamp']
                assert (oid, ts) == (l['basic_info']['object_id'], l['basic_info']['time_stamp'])
                t = stamps.index(ts)
                box = e['crop_box']
                for fr in ('vehicle_frame', 'world_frame'):     # the stored box is the label's
                    for key in ('center', 'size', 'heading', 'bbox_vertices'):
                        assert np.array_equal(e[fr][key], l[fr][key]), (oid, ts, fr, key)
                # world coordinates are float32 at up to tens of kilometres: about 1 mm steps
                off = box['detection_center_world'].astype(np.float64) - e['world_frame']['center']
                step = 2 * float(np.spacing(np.abs(e['world_frame']['center'][:2]).max()))
                assert np.allclose(off[:2], R.detection_noise(scene, oid, s)[t], atol=step) and off[2] == 0, \
                    (oid, ts, off, step)
                h0 = float(np.arctan2(e['world_frame']['heading'][1], e['world_frame']['heading'][0]))
                want = R.heading_error(scene, oid, deg) if deg else 0.0
                turn = (box['heading_world'] - h0 - want + np.pi) % (2 * np.pi) - np.pi
                assert abs(turn) < 1e-5, (oid, ts, turn)
                shifts.append(np.hypot(*off[:2]))
                # second implementation, in world coordinates. The stage cuts in the vehicle
                # frame, whose z axis leans with the road, so the faces get a band as wide
                # as the lean can move a corner of the box, plus 5 mm for float32.
                size = e['world_frame']['size'].astype(np.float64) + (
                    0 if refine else np.array([2 * margin, 2 * margin, 0]))
                if not refine:                      # the crop sits on the detection
                    assert np.array_equal(box['center_world'], box['detection_center_world'])
                else:                               # it moved, and no further than the larger box reaches
                    moved = np.abs(box['center_world'].astype(np.float64) - box['detection_center_world'])
                    assert moved[2] < 0.05 and np.hypot(*moved[:2]) <= np.hypot(*(size[:2] / 2 + margin)) + 0.01
                    errs.append(np.hypot(*(box['center_world'].astype(np.float64)
                                           - e['world_frame']['center'])[:2]))
                assert np.allclose(box['size'], size, atol=1e-6), (oid, ts)
                band = 0.005 + lean[t] * float(np.linalg.norm(size / 2))
                bands.append(band)
                P, L = e['world_frame']['pedestrian_pcd'], l['world_frame']['pedestrian_pcd']
                if len(P):
                    assert inside(P, box['center_world'], size, box['heading_world'], band).all(), (oid, ts)
                if len(L):
                    deep = L[inside(L, box['center_world'], size, box['heading_world'], -band)]
                    kept = {tuple(np.round(p, 3)) for p in P}
                    assert all(tuple(np.round(p, 3)) in kept for p in deep), (oid, ts)
                    out = L[~inside(L, box['center_world'], size, box['heading_world'], band)]
                    assert not any(tuple(np.round(p, 3)) in kept for p in out), (oid, ts)
                    n_deep += len(deep); n_out += len(out)
                # what the crop shares with the label's box, counted again in world coordinates
                # (a displaced box also holds points from outside the label's)
                nb, lsize = box['n_points_also_in_label_box'], e['world_frame']['size'].astype(np.float64)
                assert box['n_points_label_box'] == len(L) and nb <= min(len(P), len(L)), (oid, ts)
                if len(P):
                    lo = int(inside(P, e['world_frame']['center'], lsize, h0, -band).sum())
                    hi = int(inside(P, e['world_frame']['center'], lsize, h0, band).sum())
                    assert lo <= nb <= hi, (oid, ts, lo, nb, hi)
            share = c['points_also_in_label_box'] / max(c['points_label_box'], 1)
            line += (f'\n    {s * 100:.0f} cm, {deg:g} deg, margin {margin * 100:.0f} cm'
                     + (f', refined (centre error {np.mean(errs):.3f} m)' if refine else '') + ': mean shift '
                     f'{np.mean(shifts):.3f} m, {c["points"]} points, {share:.3f} of the label '
                     f'boxes\' points kept, '
                     f'{c["entries_empty"]} empty crops ({c["entries_empty_label_box"]} with label boxes), '
                     f'second implementation agrees on {n_deep} points in and {n_out} out '
                     f'(band up to {max(bands) * 100:.1f} cm)')
            # the check must have had something to decide, or it proves nothing
            assert (n_deep > 0 or refine) and (s < 0.3 or margin > 0 or n_out > 0), (scene, s, n_deep, n_out)
        print(line, flush=True)
    print('crop: all passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('noise', 'crop'))
    ap.add_argument('scenes', nargs='*')
    ap.add_argument('--mtr', help='checkout whose dataset code the noise is compared with',
                    default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    test_noise(a.mtr) if a.mode == 'noise' else test_crop(a.scenes)
