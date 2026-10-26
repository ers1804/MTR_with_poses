#!/usr/bin/env python
"""Body-facing accuracy expressed in the detection ladder's own units.

The floor is calibrated in the error on FUTURE DISPLACEMENT BEARING at +3 s and
+8 s, over every predicted pedestrian -- that is what the direction-only oracle
encodes and what each rung's angular corruption is applied to.

The paper's 44.2 deg is a different quantity: onset direction for the stationary
subset. Both numbers are defensible, but comparing one against the other is not
apples-to-apples, so this measures SMPL facing against the ladder's target on
the ladder's population.

The body-forward convention is calibrated against the tracker box heading, as in
onset_pose_vs_heading.py -- a choice that is generous to pose.

Usage: python tools/scripts/facing_error_ladder_units.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# the ladder ran at the 579 rung; this dump defines its validation population
RESULT = ('output/waymo/mtr+pose_data_no_pose_with_map_fixdata/'
          'MS_map_nopose_fix_s101/eval/eval_with_train/epoch_30/result.pkl')
POSE = '/home/erik/big1/waymo_motion_poses/final_10fps'   # corrected re-export
OUT = os.path.join(REPO, 'experiments', 'facing_error_ladder_units.json')

# The bracket edges come from the ladder itself, so a relabelled rung cannot leave this
# script measuring the share above a number the ladder no longer reports. The keys are
# named for their role, not their value, for the same reason: 'smpl_frac_above_14_6'
# would still have looked right after 14.6 became 15.1.
_FLOOR = json.load(open(os.path.join(REPO, 'experiments', 'detection_floor.json')))
LO, HI = _FLOOR['floor_bracket_deg']

CUR, VALID, HEAD = 10, 9, 6
H3, H8 = CUR + 30, CUR + 80          # +3 s and +8 s at 10 Hz
MIN_DISP = 0.5                        # m; bearing is undefined for a still agent


def aa_to_R(aa):
    th = np.linalg.norm(aa)
    if th < 1e-8:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def angdiff(a, b):
    return np.degrees(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def main() -> int:
    data = pickle.load(open(os.path.join(REPO, RESULT), 'rb'))
    recs = []
    for item in data:
        for r in (item if isinstance(item, list) else [item]):
            recs.append(r)
    print(f'ladder validation population: {len(recs)} agents')

    rows = []
    for r in recs:
        gt = r['gt_trajs']
        if not (gt[CUR, VALID] and gt[H3, VALID] and gt[H8, VALID]):
            continue
        p0, p3, p8 = gt[CUR, :2], gt[H3, :2], gt[H8, :2]
        d3, d8 = p3 - p0, p8 - p0
        rows.append({
            'scene': str(r['scenario_id']), 'oid': str(r['object_id']),
            'box': float(gt[CUR, HEAD]),
            'b3': float(np.arctan2(d3[1], d3[0])), 'n3': float(np.linalg.norm(d3)),
            'b8': float(np.arctan2(d8[1], d8[0])), 'n8': float(np.linalg.norm(d8)),
            'speed0': float(np.linalg.norm(gt[CUR, 7:9])),
        })
    print(f'agents with valid current, +3 s and +8 s states: {len(rows)}')

    R_last, keep = [], []
    for r in rows:
        p = os.path.join(POSE, 'validation', r['scene'], f"{r['oid']}.npz")
        try:
            ro = np.load(p)['root_orient']
        except Exception:
            continue
        if ro.ndim != 2 or len(ro) == 0:
            continue
        R_last.append(aa_to_R(np.asarray(ro[-1], float)))
        keep.append(r)
    print(f'with a pose sequence: {len(keep)}')
    if len(keep) < 100:
        print('too few -- abort')
        return 1

    R = np.stack(R_last)
    box = np.array([r['box'] for r in keep])
    b3 = np.array([r['b3'] for r in keep]); n3 = np.array([r['n3'] for r in keep])
    b8 = np.array([r['b8'] for r in keep]); n8 = np.array([r['n8'] for r in keep])
    sp = np.array([r['speed0'] for r in keep])

    best = None
    for axis in range(3):
        for sign in (+1, -1):
            v = sign * R[:, :, axis]
            yaw = np.arctan2(v[:, 1], v[:, 0])
            for off in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                med = float(np.median(angdiff(yaw + off, box)))
                if best is None or med < best[0]:
                    best = (med, axis, sign, off, yaw + off)
    med_box, axis, sign, off, pose_yaw = best
    print(f"calibrated body-forward = {'+' if sign > 0 else '-'}axis{axis}, "
          f'offset {np.degrees(off):.0f} deg (median vs box {med_box:.1f} deg)')

    res = {'n_population': len(recs), 'n_scored': len(keep),
           'calibration_vs_box_deg': med_box, 'targets': {}}

    print(f'\n{"target":<34}{"n":>7}{"SMPL":>9}{"box":>9}{"<45":>8}')
    print('-' * 68)
    for lab, bearing, disp in (('+3 s displacement bearing', b3, n3),
                               ('+8 s displacement bearing', b8, n8)):
        m = disp >= MIN_DISP
        e_p = angdiff(pose_yaw[m], bearing[m])
        e_b = angdiff(box[m], bearing[m])
        # The median badly understates the problem. Each ladder rung corrupts
        # EVERY agent by about theta, so a channel is only comparable to a rung
        # if its whole distribution sits there. What matters is the share of
        # agents outside the bracket, not the centre of the distribution.
        res['targets'][lab] = {
            'n': int(m.sum()),
            'smpl_med': float(np.median(e_p)),
            'box_med': float(np.median(e_b)),
            'smpl_under45': float(np.mean(e_p < 45)),
            'smpl_pct': {str(q): float(np.percentile(e_p, q))
                         for q in (25, 50, 75, 90, 95)},
            'box_pct': {str(q): float(np.percentile(e_b, q))
                        for q in (25, 50, 75, 90, 95)},
            'smpl_frac_above_lo': float(np.mean(e_p > LO)),
            'box_frac_above_lo': float(np.mean(e_b > LO)),
            'smpl_frac_above_hi': float(np.mean(e_p > HI)),
            'box_frac_above_hi': float(np.mean(e_b > HI)),
            'smpl_vs_box_med': float(np.median(angdiff(pose_yaw[m], box[m])))}
        print(f'{lab:<34}{int(m.sum()):>7}{np.median(e_p):>8.1f}°'
              f'{np.median(e_b):>8.1f}°{100*np.mean(e_p < 45):>7.1f}%')
        print('    percentiles  ' + '  '.join(
            f'p{q}: SMPL {np.percentile(e_p, q):.1f}° box {np.percentile(e_b, q):.1f}°'
            for q in (50, 75, 90)))
        print(f'    above {LO}°: SMPL {100*np.mean(e_p > LO):.1f}%'
              f'  box {100*np.mean(e_b > LO):.1f}%')
        print(f'    above {HI}°: SMPL {100*np.mean(e_p > HI):.1f}%'
              f'  box {100*np.mean(e_b > HI):.1f}%')

    # the moving/stationary split matters: for movers the pipeline snaps root
    # orientation to the direction of travel, which flatters pose
    print('\nsplit by motion at prediction time:')
    for lab, m in (('moving (>=0.3 m/s)', (sp >= 0.3) & (n3 >= MIN_DISP)),
                   ('stationary (<0.3 m/s)', (sp < 0.3) & (n3 >= MIN_DISP))):
        if m.sum() < 50:
            continue
        e_p = angdiff(pose_yaw[m], b3[m]); e_b = angdiff(box[m], b3[m])
        res['targets'][f'+3 s / {lab}'] = {
            'n': int(m.sum()), 'smpl_med': float(np.median(e_p)),
            'box_med': float(np.median(e_b))}
        print(f'  {lab:<24} n={int(m.sum()):>6}  SMPL {np.median(e_p):>5.1f}°  '
              f'box {np.median(e_b):>5.1f}°')

    # What the floor model predicts at the measured accuracy. The per-seed fits used to
    # sit here as three hardcoded pairs; they are now read from the ladder that produced
    # them, so a relabelled rung refits the law instead of leaving this prediction
    # describing rungs that no longer exist.
    pars = [(f['amplitude_pct'], f['tau_deg'])
            for f in _FLOOR.get('decay_fit', {}).get('per_seed', {}).values()]
    if not pars:
        raise SystemExit('detection_floor.json has no decay_fit; re-run detection_floor.py')
    th = res['targets']['+3 s displacement bearing']['smpl_med']
    pred = [A * np.exp(-th / t) for A, t in pars]
    print(f'\nat {th:.1f} deg the per-seed ladder fits predict a gain of '
          f'{min(pred):.3f}-{max(pred):.3f}%')
    print('  against a branch-capacity cost of ~1.6%')
    res['floor_prediction_pct'] = [float(min(pred)), float(max(pred))]

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
