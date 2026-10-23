#!/usr/bin/env python
"""Would a longer observation window make body facing accurate enough to pay?

The detection floor is about the ANGULAR ACCURACY of the facing cue: the gain
survives 5.5 deg and is gone by ~10 deg, while SMPL facing sits at 44.2 deg.
A longer pose history can only help by making that estimate better.

WOMD cannot answer this by extending the window -- its LiDAR covers exactly the
1 s history (11 frames at 10 Hz) and nothing else exists to extend into. But the
window is not uniformly filled: coverage varies per pedestrian from 1 to 11
frames. That variation is a natural experiment for the same question. If facing
accuracy improves with the number of observed pose frames, a longer window
would plausibly help; if it does not, extending it cannot rescue the cue.

Reuses the calibration of onset_pose_vs_heading.py: the body-forward convention
is chosen to best match the box heading, which is generous to pose.

Usage: python tools/scripts/facing_vs_observation_length.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# the CORRECTED per-scene re-export (the NAS copy under generated_motion is
# the pre-fix one; see the 2026-08 export-bug entry in research-log.md)
POSE = '/home/erik/big1/waymo_motion_poses/final_10fps'
EVENTS = os.path.join(REPO, 'experiments', 'onset_events_validation.pkl')
OUT = os.path.join(REPO, 'experiments', 'facing_vs_observation_length.json')


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
    rows = pickle.load(open(EVENTS, 'rb'))
    seen, uniq = set(), []
    for r in rows:
        k = (r['scene'], r['object_id'])
        if k not in seen:
            seen.add(k); uniq.append(r)
    pos = [r for r in uniq if r['onset_3s'] and np.isfinite(r['onset_dir'])
           and r['has_pose']]
    print(f'positive onset events (H=3s) with pose: {len(pos)}')

    R_last, keep, nfr = [], [], []
    for r in pos:
        p = os.path.join(POSE, 'validation', r['scene'], f"{r['object_id']}.npz")
        try:
            ro = np.load(p)['root_orient']
        except Exception:
            continue
        if ro.ndim != 2 or len(ro) == 0:
            continue
        R_last.append(aa_to_R(np.asarray(ro[-1], dtype=np.float64)))
        keep.append(r); nfr.append(len(ro))
    print(f'loaded pose for {len(keep)}')
    if len(keep) < 100:
        print('too few -- abort'); return 1

    truth = np.array([r['onset_dir'] for r in keep])
    box = np.array([r['heading'] for r in keep])
    R = np.stack(R_last)
    nfr = np.array(nfr)

    best = None
    for axis in range(3):
        for sign in (+1, -1):
            v = sign * R[:, :, axis]
            yaw = np.arctan2(v[:, 1], v[:, 0])
            for off in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                med = float(np.median(angdiff(yaw + off, box)))
                if best is None or med < best[0]:
                    best = (med, axis, sign, off, yaw + off)
    med_vs_box, axis, sign, off, pose_yaw = best
    print(f"calibrated body-forward = {'+' if sign > 0 else '-'}axis{axis} "
          f'offset {np.degrees(off):.0f} deg (median vs box {med_vs_box:.1f} deg)')

    e_pose = angdiff(pose_yaw, truth)
    e_box = angdiff(box, truth)
    print(f'\noverall: SMPL facing {np.median(e_pose):.1f} deg, '
          f'box heading {np.median(e_box):.1f} deg, n={len(keep)}')

    print('\nfacing error by NUMBER OF OBSERVED POSE FRAMES')
    print(f'  {"frames":<10}{"n":>7}{"SMPL med":>11}{"box med":>10}{"<45 deg":>10}')
    res = {'overall': {'n': len(keep), 'smpl_med': float(np.median(e_pose)),
                       'box_med': float(np.median(e_box))}, 'by_frames': []}
    groups = [(1, 3), (4, 6), (7, 9), (10, 10), (11, 11)]
    for lo, hi in groups:
        m = (nfr >= lo) & (nfr <= hi)
        if m.sum() < 25:
            continue
        lab = f'{lo}' if lo == hi else f'{lo}-{hi}'
        res['by_frames'].append({'lo': lo, 'hi': hi, 'n': int(m.sum()),
                                 'smpl_med': float(np.median(e_pose[m])),
                                 'box_med': float(np.median(e_box[m])),
                                 'frac_under45': float(np.mean(e_pose[m] < 45))})
        print(f'  {lab:<10}{int(m.sum()):>7}{np.median(e_pose[m]):>10.1f}°'
              f'{np.median(e_box[m]):>9.1f}°{100*np.mean(e_pose[m] < 45):>9.1f}%')

    r_s, p_s = stats.spearmanr(nfr, e_pose)
    print(f'\n  trend: Spearman(frames, SMPL facing error) = {r_s:+.3f} '
          f'(p={p_s:.3f})')
    print('  a longer window would help only if this is clearly NEGATIVE')
    res['trend'] = {'spearman_r': float(r_s), 'spearman_p': float(p_s)}

    # what accuracy would even the best-observed group need to reach?
    full = e_pose[nfr == 11]
    if len(full) > 25:
        med_full = float(np.median(full))
        print(f'\n  best case here (11/11 frames): {med_full:.1f} deg, '
              f'against a floor bracket of 5.5-10.2 deg')
        need = med_full / 10.2
        print(f'  would need a further {need:.1f}x improvement in bearing error '
              f'to reach the top of the bracket')
        res['full_window'] = {'n': int(len(full)), 'median_deg': med_full,
                              'factor_to_bracket': float(need)}

    # How long a window would the cue need? Fit error = A * frames^b and
    # extrapolate to the floor bracket. This is an UPPER BOUND on the benefit:
    # pedestrians with few frames are also the occluded and distant ones, so
    # part of the improvement with frame count is selection, not observation.
    g = res['by_frames']
    if len(g) >= 3:
        x = np.array([(b['lo'] + b['hi']) / 2 for b in g], float)
        y = np.array([b['smpl_med'] for b in g], float)
        w = np.sqrt([b['n'] for b in g])
        bb, loga = np.polyfit(np.log(x), np.log(y), 1, w=w)
        A = float(np.exp(loga))
        print(f'\n  scaling: error = {A:.1f} * frames^({bb:+.3f})')
        print(f'  predicted at 11 frames: {A * 11 ** bb:.1f} deg '
              f'(measured {np.median(e_pose[nfr == 11]):.1f})')
        need = {}
        for target, lab in ((10.2, 'top of bracket'), (5.5, 'bottom')):
            f_need = float((target / A) ** (1.0 / bb))
            need[str(target)] = f_need
            print(f'  to reach {target:4.1f} deg ({lab}): {f_need:,.0f} frames '
                  f'= {f_need / 10:,.0f} s of continuous observation')
        for mult in (3, 10):
            print(f'  a {mult}x window ({mult * 11} frames, {mult * 1.1:.0f} s) '
                  f'-> {A * (mult * 11) ** bb:.1f} deg')
        print('  WOMD provides 11 frames = 1.0 s, and its LiDAR covers only '
              'that window.')
        res['scaling'] = {'A': A, 'exponent': float(bb),
                          'frames_needed': need,
                          'note': 'upper bound; frame count is confounded with '
                                  'occlusion and range'}

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
