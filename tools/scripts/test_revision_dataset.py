#!/usr/bin/env python
"""Tests for the three dataset options added for the revision runs.

  TRACK_WARM_START, TRACK_HEADING_NOISE_DEG, CONTEXT_AGENTS   (waymo_pose_dataset.py)

Each test reads validation scenes through the dataset on the CPU. What is checked:

  defaults   with none of the keys set, every array of a collated scene is identical to
             what another checkout of the repository produces (--reference <repo root>)
  context    CONTEXT_AGENTS all leaves the centres, their labels, the map and the rows of
             the posed pedestrians as they were, adds the scenario's other tracks without a
             pose, and marks the ego vehicle as the SDC
  warm       the warm-started filter has converged before the window opens: its velocity
             error on a constant-velocity walker is the steady-state error at every
             observed frame, where the cold start needs seconds to get there
  heading    the yaw error is the same on every observed frame of a track, has the
             configured spread, leaves the centre's own heading channel at zero, and with
             AGENT_CENTRIC_POSE_ROT the body's root turns against it by the same angle
  roundtrip  with all three options on, a prediction equal to the label in the centre frame
             comes back from generate_prediction_dicts as the label in world coordinates

Usage, from tools/:
  python scripts/test_revision_dataset.py <pose root> <scenario root> [--reference <repo root>]
The roots are laid out as DATA_ROOT and PROCESSED_SCENARIOS_PATH (validation split).
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

BASE = 'cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map_prefit_posnoise30_scale25k.yaml'
N_SCENES = 120


def build(pose_root: str, scen_root: str, **keys):
    from easydict import EasyDict
    from mtr.config import cfg_from_yaml_file
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    c = EasyDict()
    cfg_from_yaml_file(BASE, c)
    d = c.DATA_CONFIG
    d.DATA_ROOT, d.PROCESSED_SCENARIOS_PATH = pose_root, scen_root
    for k, v in keys.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    log = logging.getLogger('test')
    log.addHandler(logging.NullHandler())
    return WaymoPoseDataset(dataset_cfg=d, training=False, logger=log)


def subset_root(pose_root: str, scen_root: str, n: int) -> str:
    """A pose root holding links to n scenes that have a scenario file, so that building
    the dataset does not index the whole split."""
    src = os.path.join(pose_root, 'validation')
    have = set(os.listdir(os.path.join(scen_root, 'processed_scenarios_validation')))
    tmp = tempfile.mkdtemp(prefix='revtest_')
    os.makedirs(os.path.join(tmp, 'validation'))
    k = 0
    for s in sorted(os.listdir(src)):
        if f'sample_{s}_0.pkl' in have and os.listdir(os.path.join(src, s)):
            os.symlink(os.path.join(src, s), os.path.join(tmp, 'validation', s))
            k += 1
            if k == n:
                break
    assert k == n, f'only {k} scenes with a scenario file'
    return tmp


def digest(ds) -> str:
    h = hashlib.sha256()
    for i in range(len(ds)):
        item = ds[i]
        for k in sorted(item):
            v = item[k]
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()


def yaw_of_root6d(r6: np.ndarray) -> np.ndarray:
    """Heading of the rotation whose first two columns are r6[..., :3] and r6[..., 3:6]."""
    return np.arctan2(r6[..., 1], r6[..., 0])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def test_defaults(pose_root, scen_root, reference):
    for keys in (dict(), dict(POSITION_NOISE_M=None), dict(SHUFFLE_POSES_ACROSS_AGENTS=True)):
        mine = digest(build(pose_root, scen_root, **keys))
        code = ('import sys; sys.path.insert(0, ".."); sys.path.insert(0, "scripts"); '
                'import test_revision_dataset as t; '
                f'print(t.digest(t.build({pose_root!r}, {scen_root!r}, **{keys!r})))')
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
        ref = subprocess.run([sys.executable, '-c', code.replace('sys.path.insert(0, "scripts"); ', '')],
                             cwd=os.path.join(reference, 'tools'), env=env, capture_output=True,
                             text=True)
        assert ref.returncode == 0, ref.stderr[-2000:]
        assert mine == ref.stdout.strip().splitlines()[-1], f'defaults changed with {keys}'
        print(f'  defaults {keys or "{}"}: identical to {reference} ({mine[:12]})')


def test_context(pose_root, scen_root):
    import pickle
    a = build(pose_root, scen_root, POSITION_NOISE_M=None)
    b = build(pose_root, scen_root, POSITION_NOISE_M=None, CONTEXT_AGENTS='all')
    n_extra = n_scenes_with_ego = 0
    for i in range(len(a)):
        x, y = a[i], b[i]
        sid = str(x['scenario_id'][0])
        for k in ('center_objects_world', 'center_objects_id', 'center_gt_trajs',
                  'center_gt_trajs_mask', 'center_gt_trajs_src', 'map_polylines',
                  'map_polylines_mask', 'center_gt_poses', 'center_shape_params'):
            assert np.array_equal(x[k], y[k]), (sid, k)
        n = x['obj_trajs'].shape[1]                       # posed pedestrians with a valid past
        assert list(y['obj_ids'][:n]) == list(x['obj_ids']), sid
        assert np.array_equal(x['track_index_to_predict'], y['track_index_to_predict']), sid
        same = np.isclose(x['obj_trajs'], y['obj_trajs'][:, :n]).all(axis=(0, 1, 2))
        # one channel may differ: the SDC flag, which moves from a pedestrian to the ego
        assert (~same).sum() <= 1, (sid, np.flatnonzero(~same))
        assert np.array_equal(x['obj_poses'], y['obj_poses'][:, :n]), sid
        assert np.array_equal(x['obj_poses_mask'], y['obj_poses_mask'][:, :n]), sid
        assert not y['obj_poses_mask'][:, n:].any() and not y['obj_poses'][:, n:].any(), sid
        sc = pickle.load(open(os.path.join(scen_root, 'processed_scenarios_validation',
                                           f'sample_{sid}_0.pkl'), 'rb'))
        ti = sc['track_infos']
        posed = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(pose_root, 'validation', sid)))
        past_ok = (np.asarray(ti['trajs'])[:, :11, 9] > 0).any(1)
        want = [(str(o), str(t)) for o, t, ok in zip(ti['object_id'], ti['object_type'], past_ok)
                if ok and str(o) not in posed]
        got = list(zip(map(str, y['obj_ids'][n:]), map(str, y['obj_types'][n:])))
        assert got == want, (sid, len(got), len(want))
        n_extra += len(want)
        if (~same).any():
            ch = int(np.flatnonzero(~same)[0])
            ego = str(ti['object_id'][sc['sdc_track_index']])
            flagged = np.flatnonzero(y['obj_trajs'][0, :, -1, ch] > 0)
            assert len(flagged) == 1 and str(y['obj_ids'][flagged[0]]) == ego, (sid, flagged)
            n_scenes_with_ego += 1
    assert n_extra > 10 * len(a) and n_scenes_with_ego > 0.9 * len(a)
    print(f'  context: {len(a)} scenes, {n_extra} context tracks added, ego flagged as SDC in '
          f'{n_scenes_with_ego}, centres, labels, map and posed rows unchanged')


def test_warm():
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    rng = np.random.default_rng(0)
    n, T, dt, s = 20000, 11, 0.1, 0.3
    speed, ang = rng.uniform(0.8, 1.6, n), rng.uniform(0, 2 * np.pi, n)
    v = np.stack([speed * np.cos(ang), speed * np.sin(ang)], 1)
    truth = np.zeros((n, T, 10))
    truth[:, :, 0:2] = v[:, None] * (np.arange(T) * dt)[None, :, None]
    truth[:, :, 7:9] = v[:, None]
    truth[:, :, 9] = 1
    ids = np.arange(n).astype(str)
    med = {}
    for warm in (False, True):
        me = SimpleNamespace(position_noise_m=s, dt=dt, num_past=T, track_warm_start=warm)
        tr = truth.copy()
        WaymoPoseDataset._track_past_causally(me, tr, ids, 'warmtest')
        med[warm] = np.median(np.linalg.norm(tr[:, :, 7:9] - truth[:, :, 7:9], axis=2), axis=0)
    # steady state on the same model, from a filter run for 20 s
    long_T = 200
    lt = np.zeros((n, long_T, 10))
    lt[:, :, 0:2] = v[:, None] * (np.arange(long_T) * dt)[None, :, None]
    lt[:, :, 7:9] = v[:, None]
    lt[:, :, 9] = 1
    ref = lt.copy()
    WaymoPoseDataset._track_past_causally(
        SimpleNamespace(position_noise_m=s, dt=dt, num_past=long_T), ref, ids, 'warmtest')
    steady = float(np.median(np.linalg.norm(ref[:, -1, 7:9] - lt[:, -1, 7:9], axis=1)))
    assert abs(med[True] - steady).max() < 0.1 * steady, (med[True], steady)
    assert med[False][-1] > 1.5 * steady
    print(f'  warm: velocity error {med[True].min():.3f} to {med[True].max():.3f} m/s over the '
          f'eleven frames (steady state {steady:.3f}), cold start {med[False][-1]:.3f} at the last')


def test_heading(pose_root, scen_root, sigma=25.0):
    keys = dict(TRACK_WARM_START=True, AGENT_CENTRIC_POSE_ROT=True)
    a = build(pose_root, scen_root, **keys)
    b = build(pose_root, scen_root, TRACK_HEADING_NOISE_DEG=sigma, **keys)
    errs, root_turn = [], []
    for i in range(len(a)):
        x, y = a[i], b[i]
        e = wrap(y['center_gt_trajs_src'][:, :11, 6] - x['center_gt_trajs_src'][:, :11, 6])
        valid = x['center_gt_trajs_src'][:, :11, 9] > 0
        for c in range(len(e)):
            ev = e[c][valid[c]]
            assert np.allclose(ev, ev[0], atol=1e-4), 'the error changes within a track'
        e = wrap(y['center_objects_world'][:, 6] - x['center_objects_world'][:, 6])
        errs.append(e)
        assert np.array_equal(x['center_gt_trajs_src'][:, 11:], y['center_gt_trajs_src'][:, 11:])
        assert np.allclose(x['center_objects_world'][:, :2], y['center_objects_world'][:, :2])
        # the centre in its own frame: every channel of its current frame is what it was
        t = y['track_index_to_predict']
        cur_a = x['obj_trajs'][np.arange(len(t)), x['track_index_to_predict'], -1]
        cur_b = y['obj_trajs'][np.arange(len(t)), t, -1]
        vel = slice(-4, -2)                                    # velocity, turns with the frame
        keep = np.ones(cur_a.shape[1], bool)
        keep[vel] = False
        keep[-2:] = False                                      # acceleration, from velocity
        assert np.allclose(cur_a[:, keep], cur_b[:, keep], atol=1e-4), 'own heading channel moved'
        # the body's root, in the frame, turns against the error
        m = x['obj_poses_mask'][np.arange(len(t)), x['track_index_to_predict'], -1]
        ra = yaw_of_root6d(x['obj_poses'][np.arange(len(t)), x['track_index_to_predict'], -1, :6])
        rb = yaw_of_root6d(y['obj_poses'][np.arange(len(t)), t, -1, :6])
        root_turn.append((wrap(rb - ra) + e)[m])
    errs, root_turn = np.concatenate(errs), np.concatenate(root_turn)
    sd = np.rad2deg(errs.std())
    assert abs(sd - sigma) < 0.15 * sigma, sd
    assert np.abs(wrap(root_turn)).max() < 1e-3, np.abs(wrap(root_turn)).max()
    print(f'  heading: {len(errs)} centres, error s.d. {sd:.1f} deg for {sigma:g} configured, '
          f'constant within a track, own channel at zero, root turns by minus the error '
          f'({len(root_turn)} posed frames, worst deviation {np.abs(wrap(root_turn)).max():.1e} rad)')


def test_roundtrip(pose_root, scen_root):
    import torch
    ds = build(pose_root, scen_root, TRACK_WARM_START=True, TRACK_HEADING_NOISE_DEG=25.0,
               CONTEXT_AGENTS='all', AGENT_CENTRIC_POSE_ROT=True)
    worst, n = 0.0, 0
    for i in range(0, len(ds), 3):
        b = ds.collate_batch([ds[i]])
        inp = b['input_dict']
        gt = inp['center_gt_trajs']                                  # (C, 80, 4) centre frame
        pred = torch.zeros(gt.shape[0], 6, gt.shape[1], 7)
        pred[..., :2] = gt[:, None, :, :2]
        b['pred_trajs'], b['pred_scores'] = pred, torch.full((gt.shape[0], 6), 1 / 6)
        out = ds.generate_prediction_dicts(b)
        for scene in out:
            for r in scene:
                fut, ok = r['gt_trajs'][11:, :2], r['gt_trajs'][11:, 9] > 0
                if ok.any():
                    worst = max(worst, float(np.abs(r['pred_trajs'][0][ok] - fut[ok]).max()))
                    n += 1
    assert n > 50 and worst < 2e-3, (n, worst)
    # the check must be able to fail: turn the frame by 0.2 rad and it has to show
    inp = dict(inp)
    inp['center_objects_world'] = inp['center_objects_world'].clone()
    inp['center_objects_world'][:, 6] += 0.2
    off = 0.0
    for scene in ds.generate_prediction_dicts(dict(b, input_dict=inp)):
        for r in scene:
            ok = r['gt_trajs'][11:, 9] > 0
            if ok.any():
                off = max(off, float(np.abs(r['pred_trajs'][0][ok] - r['gt_trajs'][11:, :2][ok]).max()))
    assert off > 1e-2, off
    print(f'  roundtrip: {n} pedestrians, label fed back as the prediction returns the world '
          f'label to {worst:.1e} m with all three options on (a frame turned by 0.2 rad is '
          f'caught, {off:.2f} m)')


def main() -> None:
    args = sys.argv[1:]
    reference = None
    if '--reference' in args:
        reference = args.pop(args.index('--reference') + 1)
        args.remove('--reference')
    pose_root, scen_root = args
    sys.path.insert(0, '..')
    small = subset_root(pose_root, scen_root, N_SCENES)
    if reference:
        test_defaults(small, scen_root, reference)
    test_warm()
    test_context(small, scen_root)
    test_heading(small, scen_root)
    test_roundtrip(small, scen_root)
    print('all passed')


if __name__ == '__main__':
    main()
