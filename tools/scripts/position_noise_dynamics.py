#!/usr/bin/env python
"""Under a causal track, on well-resolved bodies: where does pose pay?

position_noise_subpopulation.py leaves one content effect standing once the label leak is
removed: pre-fit poses at 30 cm of detection noise, on pedestrians with at least 200 LiDAR
returns. This scores that effect along two kinds of axis.

What the pedestrian goes on to do: the four manoeuvre covariates of highdyn_covariates.py
(the ones the offboard-track null was stratified by), constant-velocity residual, heading
variation, speed range and peak acceleration over the 8 s future, from the clean labels.
Speed at prediction time is added in the bands of position_noise_subpopulation.py.

How wrong the tracker is: the error of the tracked position and velocity at prediction
time (tracker_state_error.py, which keeps pedestrians valid at that frame in both the
tracked history and the label). The detection noise is drawn independently of the
pedestrian, which is the argument for reading this axis as a dose. The code does not test
that. It does score the axis on full 11-frame histories alone, which removes the cold
start.

Quartiles and the top decile are cut inside the scored population, the design of the
offboard test. The manoeuvre covariates need 10 valid future steps, so tracks that end
within a second drop out, and every axis is scored on the pedestrians that have them.

Reads experiments/eval_agent_manoeuvre.pkl, val_agent_pointcount_25k.pkl,
val_tracker_state_posnoise30.pkl and the per-agent metrics pickles of the 30 cm runs (see
position_noise_subpopulation.py). Writes experiments/position_noise_dynamics.json.

Usage: python tools/scripts/position_noise_dynamics.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from position_noise_subpopulation import LADDER, PC, REPO, ci, load, load_prefit

MAN = os.path.join(REPO, 'experiments', 'eval_agent_manoeuvre.pkl')
OUT = os.path.join(REPO, 'experiments', 'position_noise_dynamics.json')
RUNG = 30
TRK = os.path.join(REPO, 'experiments', f'val_tracker_state_posnoise{RUNG:02d}.pkl')
MIN_RETURNS = 200
FULL_HISTORY = 11
AXES = {'tracker_vel_err': 'tracker velocity error at prediction time (m/s)',
        'tracker_pos_err': 'tracker position error at prediction time (m)',
        'cv_residual': 'constant-velocity residual (m)',
        'heading_total_var': 'heading variation (rad)',
        'speed_range': 'speed range (m/s)',
        'max_accel': 'peak acceleration (m/s^2)'}
SPEED_BANDS = {'stationary': (0.0, 0.3), 'slow': (0.3, 0.8), 'walking': (0.8, 1.6),
               'fast': (1.6, np.inf)}


def strata(v: np.ndarray) -> tuple[dict, list]:
    """Quartiles and the top decile of v, cut on v itself."""
    q = np.percentile(v, [25, 50, 75, 90])
    return ({'Q1': v <= q[0], 'Q2': (v > q[0]) & (v <= q[1]), 'Q3': (v > q[1]) & (v <= q[2]),
             'Q4': v > q[2], 'top10': v > q[3]}, [float(x) for x in q])


def main() -> None:
    prow = next(p for p in json.load(open(LADDER))['prefit'] if p['noise_cm'] == RUNG)
    seeds = prow['seeds']
    man = pickle.load(open(MAN, 'rb'))
    pc = pickle.load(open(PC, 'rb'))
    trk = pickle.load(open(TRK, 'rb'))

    runs = {('nopose', s): load('nopose', RUNG, s) for s in seeds}
    runs.update({('fit_pose', s): load('pose', RUNG, s) for s in seeds})
    runs.update({('fit_shuf', s): load('shuffled', RUNG, s) for s in seeds})
    runs.update({('pre_pose', s): load_prefit('pose', RUNG, s) for s in seeds})
    runs.update({('pre_shuf', s): load_prefit('shuffled', RUNG, s) for s in seeds})
    scored = set(pc)
    for x in runs.values():
        scored &= set(x)
    common = sorted(k for k in scored if k in man and k in trk)
    X = {k: np.array([v[a] for a in common]) for k, v in runs.items()}
    npts = np.array([pc[k]['n_mean'] for k in common])
    cov = {a: np.array([man[k][a] for k in common]) for a in AXES if not a.startswith('tracker')}
    cov['tracker_vel_err'] = np.array([trk[k]['vel_err'] for k in common])
    cov['tracker_pos_err'] = np.array([trk[k]['pos_err'] for k in common])
    full_hist = np.array([trk[k]['n_obs'] == FULL_HISTORY for k in common])
    sp = np.array([man[k]['speed0'] for k in common])

    def rel(a: str, b: str, mask: np.ndarray) -> np.ndarray:
        return np.array([(X[a, s][mask].mean() / X[b, s][mask].mean() - 1) * 100
                         for s in seeds])

    def row(mask: np.ndarray) -> dict:
        return dict(n_agents=int(mask.sum()),
                    nopose_minADE=float(np.mean([X['nopose', s][mask].mean() for s in seeds])),
                    content=ci(rel('pre_pose', 'pre_shuf', mask)),
                    pose_vs_nopose=ci(rel('pre_pose', 'nopose', mask)),
                    content_fitted=ci(rel('fit_pose', 'fit_shuf', mask)))

    def show(name: str, r: dict) -> None:
        c, pn, f = r['content'], r['pose_vs_nopose'], r['content_fitted']
        print(f'    {name:<10} n={r["n_agents"]:>5} base {r["nopose_minADE"]:.3f} m   '
              f'content {c["rel"]:+6.2f}% p={c["p"]:.3f} [{c["lo"]:+.1f},{c["hi"]:+.1f}]   '
              f'vs nopose {pn["rel"]:+6.2f}% p={pn["p"]:.3f}   fitted {f["rel"]:+6.2f}% '
              f'p={f["p"]:.3f}')

    def axis(v: np.ndarray, keep: np.ndarray, label: str) -> dict:
        """Content effect in the quartiles of v inside keep, and its trend."""
        masks, cuts = strata(v[keep])
        cell = dict(cuts_p25_p50_p75_p90=cuts)
        print(f'\n  {label}   cuts {np.round(cuts, 2)}')
        for name, m in masks.items():
            full = np.zeros(len(common), bool)
            full[np.flatnonzero(keep)[m]] = True
            cell[name] = row(full)
            show(name, cell[name])
        for key, out in (('content', 'Q4_minus_Q1'), ('content_fitted', 'Q4_minus_Q1_fitted'),
                         ('pose_vs_nopose', 'Q4_minus_Q1_vs_nopose')):
            cell[out] = ci(np.array(cell['Q4'][key]['per_seed'])
                           - np.array(cell['Q1'][key]['per_seed']))
        d, f = cell['Q4_minus_Q1'], cell['Q4_minus_Q1_fitted']
        print(f'    Q4 minus Q1: pre-fit {d["rel"]:+.2f} pp p={d["p"]:.3f}   '
              f'fitted {f["rel"]:+.2f} pp p={f["p"]:.3f}')
        return cell

    res = dict(_meta=dict(
        noise_cm=RUNG, seeds=seeds, min_returns=MIN_RETURNS, axes=AXES,
        n_scored=len(scored), n_with_covariates=len(common),
        note='pre-fit poses unless a key says fitted; best-epoch per-agent minADE; paired t '
             'over seeds; negative favours own pose. Manoeuvre covariates describe the 8 s '
             'future on the clean labels and need 10 valid future steps. Tracker errors are '
             'at the last observed frame. Cuts are made inside each population.'))

    for pop, keep in ((f'returns>={MIN_RETURNS}', npts >= MIN_RETURNS),
                      ('all_returns', np.ones(len(common), bool))):
        print(f'\n##### {pop}: n={int(keep.sum())} of {len(common)} with covariates #####')
        cell = dict(all=row(keep))
        show('everyone', cell['all'])
        for a, label in AXES.items():
            cell[a] = axis(cov[a], keep, label)

        cell['speed0'] = {}
        print('\n  speed at prediction time (m/s)')
        for name, (lo, hi) in SPEED_BANDS.items():
            cell['speed0'][name] = row(keep & (sp >= lo) & (sp < hi))
            show(name, cell['speed0'][name])

        # A track younger than a second starts the filter at rest, so its error is larger
        # for a reason that belongs to the pedestrian's visibility. On full histories the
        # error is the noise draw and the pedestrian's own acceleration.
        print('\n  observed history')
        cell['history'] = dict(full=row(keep & full_hist), short=row(keep & ~full_hist))
        show('full', cell['history']['full'])
        show('short', cell['history']['short'])
        cell['history']['short_minus_full'] = ci(
            np.array(cell['history']['short']['content']['per_seed'])
            - np.array(cell['history']['full']['content']['per_seed']))
        cell['full_history_only'] = {
            a: axis(cov[a], keep & full_hist, f'full histories only: {AXES[a]}')
            for a in ('tracker_vel_err', 'tracker_pos_err')}
        res[pop] = cell

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
