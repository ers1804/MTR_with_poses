#!/usr/bin/env python
"""Does pose pay under a degraded heading where the mechanism says it should: when standing?

The heading ladder (heading_noise_ladder.py) is null on overall minADE. But for a walking
pedestrian the observed track states the direction of travel whatever the heading says,
so overall minADE is dominated by agents for whom a bad heading is partly recoverable from
the history. A pedestrian standing still at prediction time has no such fallback: the
heading is the only direction the state carries, and body facing is the cue that could
replace it. If pose substitutes for a degraded heading anywhere, it is there.

Scores the content test (own pose vs shuffled) and its change with sigma inside the
stationary, standing-then-moves and moving subpopulations, from the per-agent
metrics_epoch pickles of the best epoch of each ladder run. No training. Subpopulation
definitions match nohead_subpopulation.py.

The pickles live on big1 (ssd2 is full); fetch them with the tar pipe in the commit that
added this script, or point METRICS at any copy laid out as output/waymo.

Usage: python tools/scripts/heading_noise_subpopulation.py
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
DYN = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
OUT = os.path.join(REPO, 'experiments', 'heading_noise_subpopulation.json')

SIGMAS = (0, 15, 30)
SEEDS = (101, 202, 303)
ARMS = {'nopose': 'no_pose_with_map',
        'pose': 'cross_attn_pe_with_map',
        'shuffled': 'cross_attn_pe_with_map_shufflepose'}
STATIONARY = 0.3     # m/s at prediction time; matches the onset-task definition
ONSET_DISP = 1.0     # m travelled within 3 s


def load_run(arm: str, sigma: int, seed: int) -> dict:
    cfg = f'{ARMS[arm]}_headnoise{sigma:02d}_scale25k'
    d = os.path.join(METRICS, f'mtr+pose_data_{cfg}', f'HN_25k_{cfg}_lr25_s{seed}',
                     'eval', 'eval_with_train')
    ep = re.findall(r'best_epoch_(\d+)', open(os.path.join(d, 'best_eval_record.txt')).read())[-1]
    recs = pickle.load(open(os.path.join(d, f'metrics_epoch_{ep}.pkl'), 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE']) for r in recs}


def ci(x: np.ndarray) -> dict:
    n = len(x)
    t, p = stats.ttest_1samp(x, 0.0)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(n=n, rel=float(x.mean()), t=float(t), p=float(p),
                lo=float(x.mean() - h), hi=float(x.mean() + h),
                per_seed=[float(v) for v in x])


def main() -> None:
    dyn = {(str(k[0]), str(k[1])): v for k, v in pickle.load(open(DYN, 'rb')).items()}
    runs = {(a, s, seed): load_run(a, s, seed) for a in ARMS for s in SIGMAS for seed in SEEDS}

    # one fixed population: agents every run scored and whose dynamics are known
    common = set(dyn)
    for r in runs.values():
        common &= set(r)
    common = sorted(common)
    sp = np.array([dyn[a]['speed0'] for a in common])
    dp = np.array([dyn[a]['disp3s'] for a in common])
    subs = {'ALL': np.ones(len(common), bool),
            'stationary': sp < STATIONARY,
            'standing_then_moves': (sp < STATIONARY) & (dp > ONSET_DISP),
            'movers': sp >= STATIONARY}
    X = {k: np.array([r[a] for a in common]) for k, r in runs.items()}

    def m(a, s, seed, mask):
        return X[a, s, seed][mask].mean()

    def rel(a, sa, b, sb, mask):
        """Per-seed relative difference of (a, sa) against (b, sb), percent."""
        return np.array([(m(a, sa, x, mask) / m(b, sb, x, mask) - 1) * 100 for x in SEEDS])

    def content(s, mask):
        return rel('pose', s, 'shuffled', s, mask)

    res = dict(_meta=dict(n_common_agents=len(common), seeds=list(SEEDS),
                          stationary_speed=STATIONARY, onset_disp=ONSET_DISP,
                          sizes={k: int(v.sum()) for k, v in subs.items()},
                          note='best-epoch per-agent minADE; paired t over seeds; '
                               'did = content(sigma) - content(0) in pp, the hypothesis '
                               'itself; did_agentwise averages seeds per agent first and '
                               'is a sensitivity analysis, not a retraining claim'))
    for sub, mask in subs.items():
        print(f'\n=== {sub} (n={int(mask.sum())}) ===')
        out = {}
        for s in SIGMAS:
            r = dict(mean={a: float(np.mean([m(a, s, x, mask) for x in SEEDS])) for a in ARMS},
                     content=ci(content(s, mask)),
                     pose_vs_nopose=ci(rel('pose', s, 'nopose', s, mask)))
            if s:
                r['degradation'] = ci(rel('nopose', s, 'nopose', 0, mask))
                r['did'] = ci(content(s, mask) - content(0, mask))
                # agent-level: seeds averaged per agent, content difference in metres
                A = {(a, t): np.mean([X[a, t, x][mask] for x in SEEDS], axis=0)
                     for a in ('pose', 'shuffled') for t in (0, s)}
                d = (A['pose', s] - A['shuffled', s]) - (A['pose', 0] - A['shuffled', 0])
                denom = A['shuffled', 0].mean()
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                r['did_agentwise'] = dict(n_agents=int(len(d)), did_pp=float(100 * d.mean() / denom),
                                          p=float(stats.ttest_1samp(d, 0.0)[1]),
                                          lo_pp=float(100 * (d.mean() - h) / denom),
                                          hi_pp=float(100 * (d.mean() + h) / denom))
            out[f'sigma{s:02d}'] = r
            line = (f'  sigma={s:>2}  nopose {r["mean"]["nopose"]:.4f}  pose {r["mean"]["pose"]:.4f}'
                    f'  shuf {r["mean"]["shuffled"]:.4f}  content {r["content"]["rel"]:+6.2f}% '
                    f'p={r["content"]["p"]:.2f} [{r["content"]["lo"]:+.1f},{r["content"]["hi"]:+.1f}]')
            if s:
                g, dd, da = r['degradation'], r['did'], r['did_agentwise']
                line += (f'\n             degradation {g["rel"]:+6.2f}% p={g["p"]:.3f}'
                         f'   DiD {dd["rel"]:+6.2f}pp p={dd["p"]:.2f} [{dd["lo"]:+.1f},{dd["hi"]:+.1f}]'
                         f'   agentwise {da["did_pp"]:+.2f}pp [{da["lo_pp"]:+.2f},{da["hi_pp"]:+.2f}]')
            print(line)
        res[sub] = out
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
