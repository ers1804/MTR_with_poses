#!/usr/bin/env python
"""Where could a pose benefit hide on offboard tracks: by horizon and by manoeuvre.

The content test (own pose against a stranger's, same architecture) on the 25k arms the
paper reports, at each run's best validation epoch, from the per-agent metrics already on
disk. No GPU.

  horizons    minADE over the first 2 s, 4 s and 8 s
  manoeuvre   quartiles and the top decile of four measures of what the pedestrian goes on
              to do (constant-velocity residual, heading change, speed range, peak
              acceleration, from highdyn_covariates.py), twenty tests

A stratum's effect is the per-seed relative difference of the arms' mean minADE over the
stratum's agents, tested against zero with a one-sample t over the five seeds, as in
pointcount_stratified_content.py. Each stratum also gets own pose against no pose, which
the summary of the twenty content tests does not count.

Usage: python tools/scripts/offboard_content_by_horizon_and_manoeuvre.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')
MAN = os.path.join(REPO, 'experiments', 'eval_agent_manoeuvre.pkl')
OUT_JSON = os.path.join(REPO, 'experiments', 'offboard_content_by_horizon_manoeuvre.json')
SEEDS = (101, 202, 303, 404, 505)
MS = json.load(open(os.path.join(REPO, 'experiments', 'multiseed_analysis.json')))['cells']
CELLS = {
    'pose': ('mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5',
             'MS_scale25k_xattn_pe_s{}', 'scale25k_xattn_pe'),
    'shuf': ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k_lr5',
             'MS_scale25k_shufflepose_s{}', 'scale25k_shufflepose'),
    'nopose': ('mtr+pose_data_no_pose_with_map_scale5k_lr5',
               'MS_scale25k_nopose_s{}', 'scale25k_nopose'),
}
HORIZONS = ('minADE_2s', 'minADE_4s', 'minADE_8s', 'minADE')
MEASURES = ('cv_residual', 'heading_change', 'speed_range', 'max_accel')
FULL_FUTURE = 80


def load_run(cfg: str, tag: str, ms_cell: str, seed: int) -> dict:
    ep = MS[ms_cell]['seeds'][str(seed)]['best_epoch']
    p = os.path.join(OUT_ROOT, cfg, tag.format(seed), 'eval', 'eval_with_train',
                     f'metrics_epoch_{ep}.pkl')
    return {(str(r['scenario_id']), str(r['object_id'])): r for r in pickle.load(open(p, 'rb'))}


def test(pose: np.ndarray, shuf: np.ndarray, mask: np.ndarray) -> dict:
    """pose, shuf: (seeds, agents). Relative difference of stratum means per seed."""
    a, b = pose[:, mask].mean(1), shuf[:, mask].mean(1)
    rel = (a - b) / b * 100
    n = len(rel)
    t, p = stats.ttest_1samp(rel, 0.0)
    h = stats.t.ppf(0.975, n - 1) * rel.std(ddof=1) / np.sqrt(n)
    return dict(n_agents=int(mask.sum()), n_seeds=n, rel=float(rel.mean()), p=float(p),
                lo=float(rel.mean() - h), hi=float(rel.mean() + h),
                per_seed=[float(x) for x in rel])


def main() -> int:
    runs = {c: [load_run(cfg, tpl, cell, s) for s in SEEDS] for c, (cfg, tpl, cell) in CELLS.items()}
    agents = sorted(set.intersection(*(set(r) for rs in runs.values() for r in rs)))
    assert all(len(r) == len(agents) for rs in runs.values() for r in rs), 'runs score different agents'
    # A horizon column is absent for the few agents with no valid step inside its window.
    col = {c: {k: np.array([[r[a].get(k, np.nan) for a in agents] for r in rs], float) for k in HORIZONS}
           for c, rs in runs.items()}
    steps = np.array([runs['pose'][0][a]['num_valid_future_steps'] for a in agents])
    everyone = np.all([np.isfinite(col[c][k]).all(0) for c in col for k in HORIZONS], axis=0)
    full = everyone & (steps >= FULL_FUTURE)
    out = dict(arms={c: v[2] for c, v in CELLS.items()}, seeds=list(SEEDS), epoch='best validation epoch',
               n_agents=len(agents), n_all_horizons=int(everyone.sum()),
               n_full_future=int(full.sum()), horizons={}, manoeuvre={})
    for k in HORIZONS:
        out['horizons'][k] = dict(all=test(col['pose'][k], col['shuf'][k], everyone),
                                  full_future=test(col['pose'][k], col['shuf'][k], full))

    man = pickle.load(open(MAN, 'rb'))
    have = np.array([a in man and all(np.isfinite(man[a].get(m, np.nan)) for m in MEASURES)
                     for a in agents])
    out['n_with_manoeuvre'] = int(have.sum())
    pose, shuf, nopose = (col[c]['minADE'] for c in ('pose', 'shuf', 'nopose'))
    tests = []
    for m in MEASURES:
        v = np.array([man[a][m] if h else np.nan for a, h in zip(agents, have)])
        q = np.nanpercentile(v, [25, 50, 75, 90])
        strata = {'Q1': have & (v <= q[0]), 'Q2': have & (v > q[0]) & (v <= q[1]),
                  'Q3': have & (v > q[1]) & (v <= q[2]), 'Q4': have & (v > q[2]),
                  'top_decile': have & (v > q[3])}
        out['manoeuvre'][m] = {s: dict(test(pose, shuf, mk), vs_nopose=test(pose, nopose, mk),
                                       shuffled_vs_nopose=test(shuf, nopose, mk))
                               for s, mk in strata.items()}
        tests += [(m, s, c['rel'], c['p']) for s, c in out['manoeuvre'][m].items()]
    out['manoeuvre_summary'] = dict(
        n_tests=len(tests), n_below_005=sum(p < 0.05 for *_, p in tests),
        min_p=min(p for *_, p in tests), rel_min=min(r for _, _, r, _ in tests),
        rel_max=max(r for _, _, r, _ in tests),
        below_005=[dict(measure=m, stratum=s, rel=r, p=p) for m, s, r, p in tests if p < 0.05])
    json.dump(out, open(OUT_JSON, 'w'), indent=1)

    print(f"{len(agents)} agents, {int(everyone.sum())} with every horizon, {int(full.sum())} with a full future, "
          f"{int(have.sum())} with all four measures")
    for k in HORIZONS:
        a, f = out['horizons'][k]['all'], out['horizons'][k]['full_future']
        print(f"  {k:10s} all {a['rel']:+.2f}% p={a['p']:.2f} [{a['lo']:+.2f},{a['hi']:+.2f}]   "
              f"full future {f['rel']:+.2f}% p={f['p']:.2f}")
    for m in MEASURES:
        print('  ' + m + ': ' + '  '.join(f"{s} {c['rel']:+.2f}% (p={c['p']:.2f}, n={c['n_agents']})"
                                         for s, c in out['manoeuvre'][m].items()))
    print('  summary:', {k: v for k, v in out['manoeuvre_summary'].items() if k != 'below_005'})
    for b in out['manoeuvre_summary']['below_005']:
        c = out['manoeuvre'][b['measure']][b['stratum']]
        print(f"  p<0.05: {b['measure']} {b['stratum']} content {c['rel']:+.2f}% p={c['p']:.4f} "
              f"Bonferroni {min(1.0, c['p'] * len(tests)):.2f} per seed {np.round(c['per_seed'], 2)}")
        for k in ('vs_nopose', 'shuffled_vs_nopose'):
            print(f"      {k} {c[k]['rel']:+.2f}% p={c[k]['p']:.3f} per seed {np.round(c[k]['per_seed'], 2)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
