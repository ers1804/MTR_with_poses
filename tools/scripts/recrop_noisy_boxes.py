#!/usr/bin/env python
"""Crop the pedestrians' LiDAR points again, with the box an onboard detector would give.

Why. Every pose in this project was estimated from the points inside WOMD's own box, an
offboard label. The tracker experiments degrade the track the model sees but leave the
pose input as clean as the label: under a 30 cm tracker the body is still cropped to the
centimetre. This repeats the first stage of the pose pipeline (Waymo-3DSkelMo
scripts/extract_waymo_motion.py) with the detection the simulated tracker receives:

  centre   label centre + N(0, s^2) per frame and horizontal axis. It is the draw that
           WaymoPoseDataset._track_past_causally makes (seed crc32 of
           '<scene>:pos<s>:<object id>', one (11, 2) array per track), so the crop of
           frame t sits exactly where the tracker's detection of frame t is.
  heading  label heading + one error per track, the draw of _degrade_past_heading
           (seed '<scene>:trkhead<deg>:<object id>'), if --heading-noise-deg is given.
  size     the label's, plus a margin on each horizontal side if one is given. A box
           displaced independently of the points cuts bodies in half (at 30 cm it keeps
           72% of the label box's points and the estimated root flips on a fifth of the
           frames), which a detector that follows the points does not do. A pipeline
           that knows its detector's error would crop with room for it.

Only the crop moves. The box written beside the points stays the label's: the pose
estimator never reads it (it centres the points on their own extent) and later stages
read it as the track. Each entry gets one extra key, 'crop_box', with the box that was
used and the number of points the label's box holds, so the loss of points can be audited.

What it does not model. A detector's centre error is not independent of the points (the
box follows the visible surface), and its box size is not the label's. Independent
Gaussian noise on the centre is the tracker experiments' convention and is kept here so
that the crop and the track agree.

Differences from extract_waymo_motion.py, which is not modified:
  * boxes and timestamps come from the processed scenarios the predictor trains on
    (sample_<scene>_0.pkl) and not from the raw scenario shards. Those hold all 487k
    training scenarios, and only the per-scene LiDAR file is needed besides the track.
    Values are the same float32 centre and size. The heading is float32 here and float64
    there, which can move a point on the box face in or out (test_recrop_noisy_boxes.py
    compares the two extractions on label boxes).
  * only scenes and pedestrians that have a file in the final pose export are cropped,
    because nothing else is ever read.

Decoding the LiDAR is nearly all of the cost, so several noise settings are cropped in
one pass, each into its own directory.

Output: <out>/<name>/<split>/<scene>.pkl per setting, a list of entries in the layout of
the original stage, ready for LiDAR-HMR/gen_lidar_hmr_waymo.py (--dataset-path
<out>/<name>). <out>/<name>/recrop_<split>_<n>.json has the settings and the totals of
run n. Existing scene files are kept, so an interrupted run resumes.

A margin alone does not help the estimator, which centres its input on the extent of
whatever points it is given: ground and neighbours inside the larger box move that centre
and the pose gets worse with every centimetre of margin (20-scene comparison, research
log of 2026-11-13). With `refine` the larger box only finds the body: the centre moves
from the detection to the points it points at (median of the points above the feet within
half a box width, iterated), and the crop is the label-sized box there. This is what a
point-based pipeline does with a coarse detection, and it uses nothing but the detection
and the points. The track the predictor sees keeps the detection's error.

Usage (environment waymo_tf_gpu, CPU only). A setting is
name:metres:degrees[:margin[:refine]], the margin in metres per horizontal side
(default 0), refine 1 or 0 (default 0).
    python tools/scripts/recrop_noisy_boxes.py \
        --final /home/erik/big1/waymo_motion_poses/scale_25k \
        --scenarios /home/erik/big1/waymo_agent_batch_25k \
        --out /home/erik/ssd2/recrop \
        --setting pos30_head25_m30:0.3:25:0.3 --setting pos10_head25_m10:0.1:25:0.1 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import zlib
from collections import Counter

import numpy as np

SKELMO = os.environ.get('SKELMO', '/home/erik/ssd2/Waymo-3DSkelMo')
LIDAR = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/lidar_and_camera'
NUM_PAST = 11          # observed frames, the only ones WOMD has LiDAR for
FEET = 0.25            # m above the box floor below which points do not vote for the centre


def detection_noise(scene: str, oid, s: float) -> np.ndarray:
    """(11, 2) offsets of the detections from the label, as _track_past_causally draws them."""
    seed = zlib.crc32(f'{scene}:pos{s:g}:{oid}'.encode()) & 0xffffffff
    return np.random.default_rng(seed).normal(0.0, s, (NUM_PAST, 2))


def heading_error(scene: str, oid, deg: float) -> float:
    """One yaw error per track in radians, as _degrade_past_heading draws it."""
    seed = zlib.crc32(f'{scene}:trkhead{deg:g}:{oid}'.encode()) & 0xffffffff
    return float(np.random.default_rng(seed).normal(0.0, np.deg2rad(deg)))


# The draws under the numpy the predictor trains with (1.x, mtr_smpl). A numpy that
# produced another stream would move every crop away from its detection without any error.
_REFERENCE = {'pos': (0.3, [[0.06145511, 0.05546023], [-0.11182009, 0.30178728]]),
              'head': (25.0, 0.71195964)}


def refine_centre(local: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Where the body is, in the detection box's own frame (x, y offset from its centre).

    local: the points of the enlarged crop in that frame. The centre starts at the detection
    and moves to the median of the points within half a box width, until it stays. Points
    near the box floor do not vote: the ground under a larger footprint would outvote the
    body. With nothing in reach it starts from the nearest point instead.
    """
    cand = local[local[:, 2] > -size[2] / 2 + FEET]
    if len(cand) < 5:
        cand = local
    if len(cand) == 0:
        return np.zeros(2)
    radius = float(max(size[0], size[1])) / 2
    c = np.zeros(2)
    if not (np.linalg.norm(cand[:, :2], axis=1) < radius).any():
        c = cand[np.argmin(np.linalg.norm(cand[:, :2], axis=1)), :2].astype(np.float64)
    for _ in range(20):
        near = np.linalg.norm(cand[:, :2] - c, axis=1) < radius
        new = np.median(cand[near, :2], axis=0).astype(np.float64)
        moved = float(np.linalg.norm(new - c))
        c = new
        if moved < 1e-3:
            break
    return c


def check_noise_stream() -> None:
    s, want = _REFERENCE['pos']
    got = detection_noise('selfcheck', '7', s)[[0, 10]]
    assert np.allclose(got, want, atol=1e-7), ('numpy draws another noise stream', got)
    deg, want = _REFERENCE['head']
    got = heading_error('selfcheck', '7', deg)
    assert abs(got - want) < 1e-7, ('numpy draws another noise stream', got)


_E = None


def _stage():
    """The original stage's module. Imported late: it pulls in TensorFlow."""
    global _E
    if _E is None:
        sys.path.insert(0, os.path.join(SKELMO, 'scripts'))
        import extract_waymo_motion as E
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 2)))
        tf.config.threading.set_inter_op_parallelism_threads(1)
        _E = E
    return _E


def crop_scene(split: str, scene: str, ids: list, scenarios: str, lidar_dir: str, settings: list):
    """One (entries, counts) pair per setting, or None with a reason.

    ids: pedestrians to crop (strings). settings: (position noise in m, heading noise in
    deg, margin in m added to each horizontal side of the crop, refine the centre or not).
    """
    E = _stage()
    with open(os.path.join(scenarios, f'processed_scenarios_{split}', f'sample_{scene}_0.pkl'), 'rb') as f:
        sc = pickle.load(f)
    assert sc['scenario_id'] == scene, (sc['scenario_id'], scene)
    ti = sc['track_infos']
    row = {int(o): i for i, o in enumerate(ti['object_id'])}
    for oid in ids:
        assert ti['object_type'][row[int(oid)]] == 'TYPE_PEDESTRIAN', (scene, oid)
    lidar_path = os.path.join(lidar_dir, split, f'{scene}.tfrecord')
    if not os.path.exists(lidar_path):
        return None, 'no_lidar_file'
    frames = E._load_scenario_from_tfrecord(lidar_path).compressed_frame_laser_data
    timestamps = sc['timestamps_seconds']
    n_frames = min(len(timestamps), len(frames))
    assert n_frames <= NUM_PAST, (scene, n_frames)
    noise = [{oid: (detection_noise(scene, oid, s) if s > 0 else np.zeros((NUM_PAST, 2)))
              for oid in ids} for s, *_ in settings]
    herr = [{oid: (heading_error(scene, oid, deg) if deg > 0 else 0.0) for oid in ids}
            for _, deg, *_ in settings]
    grow = [np.array([2 * m, 2 * m, 0.0], dtype=np.float32) for _, _, m, _ in settings]
    refine = [bool(r) for *_, r in settings]
    out = [([], Counter()) for _ in settings]
    for t in range(n_frames):
        ts_micros = int(timestamps[t] * 1e6)
        try:
            pc_vehicle, vehicle_pose = E._extract_frame_point_cloud(frames[t])
        except Exception as exc:                        # as the original: skip the frame
            print(f'[WARN] skipping frame {t} of {scene}: {exc}', flush=True)
            for _, c in out:
                c['frames_undecodable'] += 1
            continue
        pose_inv = np.linalg.inv(vehicle_pose.astype(np.float64)).astype(np.float32)

        def to_vehicle(center_world, heading_world):
            center = E._transform_points(center_world[None, :], pose_inv)[0]
            d = pose_inv[:3, :3] @ np.array([np.cos(heading_world), np.sin(heading_world), 0],
                                            dtype=np.float32)
            return center, float(np.arctan2(d[1], d[0]))

        for oid in ids:
            st = ti['trajs'][row[int(oid)], t]
            if st[9] <= 0:
                continue
            # the label's box, written out as the original stage writes it
            center_world = st[0:3].astype(np.float32)
            size = st[3:6].astype(np.float32)
            heading_world = float(st[6])
            center_vehicle, heading_vehicle = to_vehicle(center_world, heading_world)
            direction = np.array([np.cos(heading_world), np.sin(heading_world), 0], dtype=np.float32)
            direction_unit = direction / (np.linalg.norm(direction) + 1e-8)
            n_label = len(E._crop_points_by_bbox(pc_vehicle, center_vehicle, size, heading_vehicle))

            label_vertices_vehicle = E._bbox_vertices(center_vehicle, size, heading_vehicle)
            label_vertices_world = E._bbox_vertices(center_world, size, heading_world)

            for k, (entries, c) in enumerate(out):
                # the detection's box, used for the crop alone
                crop_center_world = center_world.copy()
                crop_center_world[0:2] += noise[k][oid][t].astype(np.float32)
                crop_heading_world = heading_world + herr[k][oid]
                crop_center_vehicle, crop_heading_vehicle = to_vehicle(crop_center_world, crop_heading_world)
                crop_size = size + grow[k]
                pts_vehicle = E._crop_points_by_bbox(pc_vehicle, crop_center_vehicle, crop_size,
                                                     crop_heading_vehicle)
                detection_center_world = crop_center_world
                if refine[k]:
                    # the larger box only finds the body, the crop is the label-sized box there
                    rot = E._rotation_matrix_z(crop_heading_vehicle)
                    off = refine_centre((pts_vehicle - crop_center_vehicle) @ rot, crop_size)
                    crop_center_vehicle = (crop_center_vehicle
                                           + rot @ np.array([off[0], off[1], 0.0], dtype=np.float32))
                    crop_center_world = E._transform_points(crop_center_vehicle[None, :], vehicle_pose)[0]
                    crop_size = size
                    pts_vehicle = E._crop_points_by_bbox(pc_vehicle, crop_center_vehicle, crop_size,
                                                         crop_heading_vehicle)
                pts_world = E._transform_points(pts_vehicle, vehicle_pose)
                # how many of them the label's box holds too: what is left of the body
                n_body = len(E._crop_points_by_bbox(pts_vehicle, center_vehicle, size, heading_vehicle))

                entries.append({
                    'basic_info': {'object_id': str(oid), 'time_stamp': ts_micros,
                                   'context_name': scene, 'split': split},
                    'vehicle_frame': {'keypoints': None, 'keypoint_type': None,
                                      'keypoint_is_occluded': None, 'center': center_vehicle,
                                      'size': size, 'heading': heading_vehicle,
                                      'pedestrian_pcd': pts_vehicle,
                                      'bbox_vertices': label_vertices_vehicle},
                    'world_frame': {'keypoints': None, 'keypoint_type': None,
                                    'keypoint_is_occluded': None, 'center': center_world,
                                    'size': size, 'heading': direction_unit,
                                    'pedestrian_pcd': pts_world,
                                    'bbox_vertices': label_vertices_world},
                    'crop_box': {'center_world': crop_center_world,
                                 'detection_center_world': detection_center_world,
                                 'heading_world': crop_heading_world, 'size': crop_size,
                                 'n_points_label_box': n_label,
                                 'n_points_also_in_label_box': n_body},
                })
                c['entries'] += 1
                c['points'] += len(pts_vehicle)
                c['points_also_in_label_box'] += n_body
                c['points_label_box'] += n_label
                c['centre_error_mm'] += int(round(1000 * float(np.linalg.norm(
                    (crop_center_world - center_world)[:2]))))
                c['entries_empty'] += len(pts_vehicle) == 0
                c['entries_empty_label_box'] += n_label == 0
    return out, None


def _work(job):
    split, scene, a = job
    names = [n for n, *_ in a['settings']]
    paths = [os.path.join(a['out'], n, split, f'{scene}.pkl') for n in names]
    todo = [k for k, p in enumerate(paths) if not os.path.exists(p)]
    res = {n: Counter() for n in names}
    for k in set(range(len(names))) - set(todo):
        res[names[k]]['scenes_already_there'] += 1
    if not todo:
        return res
    ids = sorted((n[:-4] for n in os.listdir(os.path.join(a['final'], split, scene))
                  if n.endswith('.npz') and n[:-4].isdigit()), key=int)
    if not ids:
        for k in todo:
            res[names[k]]['scenes_without_pose_files'] += 1
        return res
    try:
        out, why = crop_scene(split, scene, ids, a['scenarios'], a['lidar_dir'],
                              [a['settings'][k][1:] for k in todo])
    except Exception as exc:        # one scene must not end a run of hours. main() reports it.
        print(f'[ERROR] {split} {scene}: {type(exc).__name__}: {exc}', flush=True)
        out, why = None, f'failed_{type(exc).__name__}'
    for j, k in enumerate(todo):
        if out is None:
            res[names[k]][f'scenes_{why}'] += 1
            continue
        entries, c = out[j]
        c['scenes'] += 1
        c['pedestrians'] += len(ids)
        res[names[k]] = c
        if not entries:             # an empty file, so that a resumed run does not decode it again
            c['scenes_without_entries'] += 1
        tmp = paths[k] + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(entries, f)
        os.replace(tmp, paths[k])
    return res


def parse_setting(text: str):
    name, s, deg, *more = text.split(':')
    assert name and '/' not in name and len(more) <= 2, text
    margin = float(more[0]) if more else 0.0
    refine = int(more[1]) if len(more) > 1 else 0
    assert refine in (0, 1) and (margin > 0 or not refine), f'{text}: refine needs a margin'
    return name, float(s), float(deg), margin, refine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--final', required=True,
                    help='final pose export. Its <split>/<scene>/<id>.npz files name what is cropped')
    ap.add_argument('--scenarios', required=True,
                    help='directory holding processed_scenarios_<split>/sample_<scene>_0.pkl')
    ap.add_argument('--lidar-dir', default=LIDAR)
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=['training', 'validation'])
    ap.add_argument('--setting', action='append', required=True, type=parse_setting,
                    metavar='NAME:METRES:DEGREES[:MARGIN]',
                    help='POSITION_NOISE_M and TRACK_HEADING_NOISE_DEG of the tracker runs the '
                         'poses are for, the margin in metres added to each horizontal side of '
                         'the crop, and the directory name. 0 keeps the label. Repeatable.')
    ap.add_argument('--scene-list', help='file of scene ids, to restrict the run')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    check_noise_stream()
    names = [n for n, *_ in a.setting]
    assert len(set(names)) == len(names), names
    only = set(open(a.scene_list).read().split()) if a.scene_list else None
    opts = dict(final=a.final, out=a.out, scenarios=a.scenarios, lidar_dir=a.lidar_dir,
                settings=a.setting)
    import multiprocessing as mp
    status = 0
    for split in a.splits:
        scenes = sorted(s for s in os.listdir(os.path.join(a.final, split))
                        if only is None or s in only)
        for n in names:
            os.makedirs(os.path.join(a.out, n, split), exist_ok=True)
        print(f'{split}: {len(scenes)} scenes, {a.workers} workers, settings {a.setting}', flush=True)
        total = {n: Counter() for n in names}
        jobs = [(split, s, opts) for s in scenes]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
                for n in names:
                    total[n].update(res[n])
                if i % 500 == 0 or i == len(jobs):
                    kept = ', '.join(
                        f'{n} {total[n]["points_also_in_label_box"] / max(total[n]["points_label_box"], 1):.3f}'
                        for n in names)
                    print(f"  {split} {i}/{len(jobs)} scenes. Share of the label boxes' points "
                          f'kept: {kept}', flush=True)
        for name, s, deg, margin, refine in a.setting:
            t = total[name]
            summary = dict(settings=dict(final=a.final, scenarios=a.scenarios, lidar_dir=a.lidar_dir,
                                         split=split, scene_list=a.scene_list, name=name,
                                         pos_noise=s, heading_noise_deg=deg, margin_m=margin,
                                         refine=bool(refine)),
                           totals={k: int(v) for k, v in sorted(t.items())})
            if t['points_label_box']:
                summary['share_of_label_box_points_kept'] = (t['points_also_in_label_box']
                                                             / t['points_label_box'])
                # what the crop holds beyond the label's box: ground, neighbours, clutter
                summary['share_of_crop_outside_label_box'] = (
                    1 - t['points_also_in_label_box'] / max(t['points'], 1))
                # how far the crop's centre is from the label's, on average
                summary['mean_centre_error_m'] = t['centre_error_mm'] / 1000 / max(t['entries'], 1)
            # a resumed run counts only what it did itself, so each run writes its own file
            n = 0
            while os.path.exists(os.path.join(a.out, name, f'recrop_{split}_{n}.json')):
                n += 1
            with open(os.path.join(a.out, name, f'recrop_{split}_{n}.json'), 'w') as f:
                json.dump(summary, f, indent=1)
            print(name, json.dumps(summary['totals']), flush=True)
        failed = sorted(k for n in names for k in total[n] if k.startswith('scenes_failed_'))
        if failed:
            print(f'{split}: SCENES FAILED ({failed}), see the [ERROR] lines. They have no file '
                  'and are tried again by the next run.', flush=True)
            status = 1
    sys.exit(status)


if __name__ == '__main__':
    main()
