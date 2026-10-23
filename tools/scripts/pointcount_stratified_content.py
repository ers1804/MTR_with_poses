#!/usr/bin/env python
"""Does pose pay for pedestrians the sensor actually resolves?

Range was only a proxy. Return count is the quantity that governs mesh-fit
quality directly, and it absorbs occlusion, which range does not: a close but
half-occluded pedestrian and a distant clear one can share a range and differ
by an order of magnitude in returns.

Scores the content test (own pose vs another pedestrian's) in point-count
strata at the 25k rung from per-agent metrics already on disk. No GPU.

Counts come from the pose pipeline's own extraction (make_pointcounts_25k.py),
summarised as the mean returns over the 11 observed frames, since the pose
input is the whole 11-frame sequence.

Usage: python tools/scripts/pointcount_stratified_content.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')
PC = os.path.join(REPO, 'experiments', 'val_agent_pointcount_25k.pkl')
RNG = os.path.join(REPO, 'experiments', 'val_agent_range_25k.pkl')
OUT_JSON = os.path.join(REPO, 'experiments', 'pointcount_stratified_content.json')
SEEDS = (101, 202, 303, 404, 505)
MS = json.load(open(os.path.join(REPO, 'experiments',
                                 'multiseed_analysis.json')))['cells']
CELLS = {
    'nopose': ('mtr+pose_data_no_pose_with_map_scale5k_lr5',
               'MS_scale25k_nopose_s{}', 'scale25k_nopose'),
    'pose':   ('mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5',
               'MS_scale25k_xattn_pe_s{}', 'scale25k_xattn_pe'),
    'shuf':   ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k_lr5',
               'MS_scale25k_shufflepose_s{}', 'scale25k_shufflepose'),
}


def load_run(cfg, tag, ms_cell, seed):
    ep = MS[ms_cell]['seeds'][str(seed)]['best_epoch']
    p = os.path.join(OUT_ROOT, cfg, tag.format(seed),
                     'eval', 'eval_with_train', f'metrics_epoch_{ep}.pkl')
    if not os.path.exists(p):
        return None
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE'])
            for r in pickle.load(open(p, 'rb'))}


def paired(runs, a, b, agents, mask):
    seeds = sorted(set(runs[a]) & set(runs[b]))
    av = np.array([np.array([runs[a][s][x] for x in agents])[mask].mean()
                   for s in seeds])
    bv = np.array([np.array([runs[b][s][x] for x in agents])[mask].mean()
                   for s in seeds])
    rel = (av - bv) / bv * 100
    n = len(seeds)
    t, p = stats.ttest_1samp(rel, 0.0)
    se = rel.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    mde = se * (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))
    return dict(n=n, a=float(av.mean()), b=float(bv.mean()),
                rel=float(rel.mean()), t=float(t), p=float(p),
                lo=float(rel.mean() - h), hi=float(rel.mean() + h),
                mde80=float(mde), per_seed=[float(x) for x in rel])


def main() -> int:
    runs = {}
    for cell, (cfg, tpl, ms_cell) in CELLS.items():
        runs[cell] = {s: r for s in SEEDS
                      if (r := load_run(cfg, tpl, ms_cell, s)) is not None}
        print(f'{cell:<8} seeds {sorted(runs[cell])}')

    common = set.intersection(*(set(runs[c][s]) for c in runs for s in runs[c]))
    pc = pickle.load(open(PC, 'rb'))
    rng = pickle.load(open(RNG, 'rb')) if os.path.exists(RNG) else {}
    common = sorted(common & set(pc))
    N = np.array([pc[a]['n_mean'] for a in common])
    print(f'\nagents with point counts: {len(common)}')
    print('mean returns over the 11 observed frames: '
          + '  '.join(f'p{q}={np.percentile(N, q):.0f}'
                      for q in (10, 25, 50, 75, 90)))

    if rng:
        R = np.array([rng.get(a, np.nan) for a in common])
        ok = np.isfinite(R) & (R < 200)
        r_s, p_s = stats.spearmanr(R[ok], N[ok])
        print(f'range vs returns: Spearman {r_s:+.3f} (p={p_s:.1e}) '
              f'-- they are related but not the same cut')

    res = {'n_agents': len(common), 'strata': {}}
    strata = {'ALL': np.ones(len(N), bool)}
    for thr in (30, 50, 100, 200):
        strata[f'>={thr} pts'] = N >= thr
    q = np.percentile(N, [25, 50, 75])
    strata[f'Q1 (<{q[0]:.0f})'] = N < q[0]
    strata[f'Q2 ({q[0]:.0f}-{q[1]:.0f})'] = (N >= q[0]) & (N < q[1])
    strata[f'Q3 ({q[1]:.0f}-{q[2]:.0f})'] = (N >= q[1]) & (N < q[2])
    strata[f'Q4 (>={q[2]:.0f})'] = N >= q[2]

    print('\nCONTENT TEST (own pose vs stranger\'s), minADE. Negative favours pose.')
    print(f'{"stratum":<18}{"n":>7}{"pose":>9}{"shuf":>9}{"rel":>9}'
          f'{"p":>8}{"MDE":>8}   95% CI')
    print('-' * 88)
    for name, m in strata.items():
        if m.sum() < 50:
            continue
        c = paired(runs, 'pose', 'shuf', common, m)
        v = paired(runs, 'pose', 'nopose', common, m)
        res['strata'][name] = {'n': int(m.sum()), 'content': c,
                               'pose_vs_nopose': v}
        print(f'{name:<18}{int(m.sum()):>7}{c["a"]:>9.4f}{c["b"]:>9.4f}'
              f'{c["rel"]:>+8.2f}%{c["p"]:>8.3f}{c["mde80"]:>7.2f}%   '
              f'[{c["lo"]:+.2f},{c["hi"]:+.2f}]')

    # the gradient is the real test: a genuine benefit must grow with returns
    print('\ncontent effect in 8 equal-count bins of return count:')
    edges = np.percentile(N, np.linspace(0, 100, 9))
    mids, rels, bins = [], [], []
    for i in range(8):
        m = (N >= edges[i]) & (N < edges[i + 1]) if i < 7 else (N >= edges[i])
        if m.sum() < 50:
            continue
        c = paired(runs, 'pose', 'shuf', common, m)
        mids.append(float(np.median(N[m]))); rels.append(c['rel'])
        bins.append({'lo': float(edges[i]), 'hi': float(edges[min(i + 1, 7)]),
                     'n': int(m.sum()), **c})
        print(f'  [{edges[i]:6.0f},{edges[i+1]:6.0f}) n={int(m.sum()):>5} '
              f'{c["rel"]:>+7.2f}%  p={c["p"]:.3f}  MDE={c["mde80"]:.2f}%')
    r_p, p_p = stats.pearsonr(mids, rels)
    r_sp, p_sp = stats.spearmanr(mids, rels)
    print(f'  trend with return count: Pearson r={r_p:+.3f} (p={p_p:.3f}), '
          f'Spearman={r_sp:+.3f} (p={p_sp:.3f})')
    print('  (a real benefit from better-resolved bodies would make this '
          'strongly NEGATIVE)')
    res['range_bins'] = bins
    res['trend'] = {'pearson_r': float(r_p), 'pearson_p': float(p_p),
                    'spearman_r': float(r_sp), 'spearman_p': float(p_sp)}

    print('\npose vs no-pose (includes the branch-capacity tax):')
    for name in ('ALL', '>=100 pts', '>=200 pts'):
        if name in res['strata']:
            v = res['strata'][name]['pose_vs_nopose']
            print(f'  {name:<12} {v["rel"]:+6.2f}%  p={v["p"]:.4f}  '
                  f'CI=[{v["lo"]:+.2f},{v["hi"]:+.2f}]')

    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'\nwrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
