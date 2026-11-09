#!/usr/bin/env python
"""Where does pose pay once the observed track is a causal tracker's?

position_noise_ladder.py finds the project's first content effect: under simulated onboard
tracking, own pose beats another pedestrian's. The tracker leaves the box heading clean and
spoils speed and path, so if the body is supplying gait, the gain belongs to walkers, not
to pedestrians standing at prediction time. Scores content (pose vs shuffled) and pose vs
no-pose per speed band, on the seeds position_noise_ladder.json counts as complete, from
the per-agent metrics_epoch pickles of each run's best epoch. Speed bands use the label
speed at prediction time, the same speed0 as heading_noise_subpopulation.py.

Pickles: METRICS (default big1/heading_noise_metrics), laid out as output/waymo; best
epochs from experiments/heading_noise_records.
Writes experiments/position_noise_subpopulation.json.

Usage: python tools/scripts/position_noise_subpopulation.py
"""
from __future__ import annotations

import json
import os
import pickle
import re

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = os.environ.get('METRICS', '/home/erik/big1/heading_noise_metrics')
RECORDS = os.path.join(REPO, 'experiments', 'heading_noise_records')
DYN = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
LADDER = os.path.join(REPO, 'experiments', 'position_noise_ladder.json')
OUT = os.path.join(REPO, 'experiments', 'position_noise_subpopulation.json')

ARMS = {'nopose': 'no_pose_with_map',
        'pose': 'cross_attn_pe_with_map',
        'shuffled': 'cross_attn_pe_with_map_shufflepose'}
RUNGS = (0, 10, 30)
STATIONARY = 0.3


def load(arm: str, rung: int, seed: int) -> dict:
    if rung == 0:
        cfg = f'{ARMS[arm]}_headnoise00_scale25k'
        tag = f'HN_25k_{cfg}_lr25_s{seed}'
    else:
        cfg = f'{ARMS[arm]}_posnoise{rung:02d}_scale25k'
        tag = f'PN_25k_{cfg}_lr25_s{seed}'
    sub = os.path.join(f'mtr+pose_data_{cfg}', tag, 'eval', 'eval_with_train')
    ep = re.findall(r'best_epoch_(\d+)',
                    open(os.path.join(RECORDS, sub, 'best_eval_record.txt')).read())[-1]
    recs = pickle.load(open(os.path.join(METRICS, sub, f'metrics_epoch_{ep}.pkl'), 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE']) for r in recs}


def ci(x: np.ndarray) -> dict:
    t, p = stats.ttest_1samp(x, 0.0)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(n=len(x), rel=float(x.mean()), p=float(p), lo=float(x.mean() - h),
                hi=float(x.mean() + h), per_seed=[float(v) for v in x])


def main() -> None:
    seeds = {r: json.load(open(LADDER))['seeds'][f'pn{r:02d}'] for r in RUNGS}
    dyn = {(str(k[0]), str(k[1])): v for k, v in pickle.load(open(DYN, 'rb')).items()}
    runs = {(a, r, s): load(a, r, s) for r in RUNGS for s in seeds[r] for a in ARMS}
    common = set(dyn)
    for x in runs.values():
        common &= set(x)
    common = sorted(common)
    sp = np.array([dyn[k]['speed0'] for k in common])
    subs = {'ALL': np.ones(len(common), bool),
            'stationary': sp < STATIONARY,
            'slow': (sp >= STATIONARY) & (sp < 0.8),
            'walking': (sp >= 0.8) & (sp < 1.6),
            'fast': sp >= 1.6}
    X = {k: np.array([v[a] for a in common]) for k, v in runs.items()}

    def rel(a, b, r, mask):
        return np.array([(X[a, r, s][mask].mean() / X[b, r, s][mask].mean() - 1) * 100
                         for s in seeds[r]])

    res = dict(_meta=dict(n_common_agents=len(common), seeds=seeds,
                          bands_ms={'stationary': [0, STATIONARY], 'slow': [STATIONARY, 0.8],
                                    'walking': [0.8, 1.6], 'fast': [1.6, None]},
                          sizes={k: int(v.sum()) for k, v in subs.items()},
                          note='best-epoch per-agent minADE; paired t over seeds; negative '
                               'favours the first-named arm'))
    for sub, mask in subs.items():
        print(f'\n=== {sub} (n={int(mask.sum())}) ===')
        res[sub] = {}
        for r in RUNGS:
            row = dict(content=ci(rel('pose', 'shuffled', r, mask)),
                       pose_vs_nopose=ci(rel('pose', 'nopose', r, mask)))
            res[sub][f'pn{r:02d}'] = row
            c, pn = row['content'], row['pose_vs_nopose']
            print(f'  {r:>2} cm n={c["n"]}  content {c["rel"]:+6.2f}% p={c["p"]:.3f} '
                  f'[{c["lo"]:+.1f},{c["hi"]:+.1f}]   vs nopose {pn["rel"]:+6.2f}% p={pn["p"]:.3f}')
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
