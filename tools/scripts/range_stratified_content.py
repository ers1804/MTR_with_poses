#!/usr/bin/env python
"""Does pose pay for pedestrians close to the sensor, where the mesh fit is best?

The detection floor says pose fails because body facing is imprecise (44.2 deg
pooled). Facing accuracy improves with proximity -- 32.9 deg in the nearest
onset quartile -- because a close pedestrian returns far more LiDAR points
(median 329 under 15 m against 11 beyond 45 m). So the one regime left where
pose could clear the floor is the near field.

This scores the content test (own pose vs another pedestrian's) inside range
strata at the 25k rung, from the per-agent metrics already on disk. No GPU.

Range is distance from the self-driving car at the prediction timestep, taken
from the processed scenario infos exactly as tools/scripts/onset_stratify_
covariates.py defines it.

Usage: python tools/scripts/range_stratified_content.py
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')
SCEN = "/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation"
OUT_JSON = os.path.join(REPO, 'experiments', 'range_stratified_content.json')
CACHE = os.path.join(REPO, 'experiments', 'val_agent_range_25k.pkl')
CUR = 10
SEEDS = tuple(int(x) for x in os.environ.get('SEEDS', '101,202,303,404,505').split(','))

CELLS = {   # cell -> (config dir, run tag, multiseed_analysis.json cell)
    # canonical multiseed runs at the 25k rung (n=5), the same cells
    # analyze_multiseed.py reports as scale25k_* in Table I
    'nopose': ('mtr+pose_data_no_pose_with_map_scale5k_lr5',
               'MS_scale25k_nopose_s{}', 'scale25k_nopose'),
    'pose':   ('mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5',
               'MS_scale25k_xattn_pe_s{}', 'scale25k_xattn_pe'),
    'shuf':   ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k_lr5',
               'MS_scale25k_shufflepose_s{}', 'scale25k_shufflepose'),
}
# Two 25k seeds never got a best_eval_record.txt written. The canonical best
# epoch for every seed is the one the paper's Table I uses, so read it from
# the analysis record rather than re-deriving it here.
MS = json.load(open(os.path.join(REPO, 'experiments', 'multiseed_analysis.json')))['cells']
METRICS = ('minADE', 'minADE_2s', 'minADE_4s', 'minADE_8s', 'minFDE')


def best_epoch(d):
    f = os.path.join(d, 'eval', 'eval_with_train', 'best_eval_record.txt')
    if not os.path.exists(f):
        return None
    ep = None
    for line in open(f):
        m = re.search(r'best_epoch_(\d+)', line)
        if m:
            ep = int(m.group(1))
    return ep


def load_run(cfg, tag, ms_cell=None, seed=None):
    d = os.path.join(OUT_ROOT, cfg, tag)
    ep = None
    if ms_cell is not None:
        ep = MS.get(ms_cell, {}).get('seeds', {}).get(str(seed), {}).get('best_epoch')
    if ep is None:
        ep = best_epoch(d)
    if ep is None:
        return None, None
    p = os.path.join(d, 'eval', 'eval_with_train', f'metrics_epoch_{ep}.pkl')
    if not os.path.exists(p):
        return None, None
    recs = pickle.load(open(p, 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])):
            {m: float(r[m]) for m in METRICS if m in r} for r in recs}, ep


def _scene_range(args):
    path, wanted = args
    try:
        info = pickle.load(open(path, 'rb'))
    except Exception:
        return []
    ti = info['track_infos']
    trajs = np.asarray(ti['trajs'])
    ids = [str(x) for x in ti['object_id']]
    sdc = info.get('sdc_track_index', None)
    if sdc is None:
        return []
    ego = trajs[sdc, CUR, :2].astype(np.float64)
    idx = {o: i for i, o in enumerate(ids)}
    out = []
    for oid in wanted:
        i = idx.get(oid)
        if i is None:
            continue
        out.append((oid, float(np.linalg.norm(trajs[i, CUR, :2] - ego))))
    return out


def build_ranges(agents):
    """One info file per scene is enough: track_infos covers the whole scene."""
    if os.path.exists(CACHE):
        r = pickle.load(open(CACHE, 'rb'))
        if set(agents) <= set(r):
            print(f'range cache hit ({len(r)} agents)')
            return r
    by_scene = defaultdict(list)
    for sc, oid in agents:
        by_scene[sc].append(oid)
    one = {}
    for f in glob.glob(os.path.join(SCEN, 'sample_*.pkl')):
        m = re.match(r'sample_(.+)_\d+\.pkl$', os.path.basename(f))
        if m and m.group(1) in by_scene and m.group(1) not in one:
            one[m.group(1)] = f
    print(f'scenes needed {len(by_scene)}, info files found {len(one)}')
    tasks = [(one[sc], by_scene[sc]) for sc in one]
    rng = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for sc, res in zip([s for s in one], ex.map(_scene_range, tasks, chunksize=16)):
            for oid, d in res:
                rng[(sc, oid)] = d
    pickle.dump(rng, open(CACHE, 'wb'))
    print(f'computed range for {len(rng)} agents -> {CACHE}')
    return rng


def paired(runs, a, b, agents, mask, metric):
    seeds = sorted(set(runs[a]) & set(runs[b]))
    av, bv = [], []
    for s in seeds:
        av.append(np.array([runs[a][s][x][metric] for x in agents])[mask].mean())
        bv.append(np.array([runs[b][s][x][metric] for x in agents])[mask].mean())
    av, bv = np.array(av), np.array(bv)
    rel = (av - bv) / bv * 100
    n = len(seeds)
    t, p = stats.ttest_1samp(rel, 0.0)
    h = stats.t.ppf(0.975, n - 1) * rel.std(ddof=1) / np.sqrt(n)
    return dict(n=n, a=float(av.mean()), b=float(bv.mean()),
                rel=float(rel.mean()), t=float(t), p=float(p),
                lo=float(rel.mean() - h), hi=float(rel.mean() + h),
                per_seed=[float(x) for x in rel])


def main():
    runs, eps = {}, {}
    for cell, (cfg, tpl, ms_cell) in CELLS.items():
        runs[cell] = {}
        for s in SEEDS:
            r, ep = load_run(cfg, tpl.format(s), ms_cell, s)
            if r is not None:
                runs[cell][s] = r
                eps[(cell, s)] = ep
        print(f'{cell:<8} seeds {sorted(runs[cell])}  epochs '
              f'{[eps[(cell, s)] for s in sorted(runs[cell])]}')
    usable = sorted(set.intersection(*(set(runs[c]) for c in runs)))
    for c in runs:
        runs[c] = {s: v for s, v in runs[c].items() if s in usable}
    print(f'seeds usable in every cell: {usable}')

    common = None
    for c in runs:
        for s, r in runs[c].items():
            common = set(r) if common is None else (common & set(r))
    common = sorted(common)
    rng = build_ranges(common)
    common = [a for a in common
              if a in rng and np.isfinite(rng[a]) and rng[a] < 200.0]
    R = np.array([rng[a] for a in common])
    print(f'\nagents with range: {len(common)}')
    print('range percentiles: ' + '  '.join(
        f'p{q}={np.percentile(R, q):.1f}m' for q in (10, 25, 50, 75, 90, 99)))

    q = np.percentile(R, [25, 50, 75])
    strata = {
        'ALL': np.ones(len(R), bool),
        '<19m': R < 19.0,
        '<15m': R < 15.0,
        '<10m': R < 10.0,
        f'Q1 [{R.min():.0f},{q[0]:.0f})': R < q[0],
        f'Q2 [{q[0]:.0f},{q[1]:.0f})': (R >= q[0]) & (R < q[1]),
        f'Q3 [{q[1]:.0f},{q[2]:.0f})': (R >= q[1]) & (R < q[2]),
        f'Q4 [{q[2]:.0f},{R.max():.0f}]': R >= q[2],
    }

    res = {'seeds': usable, 'n_agents': len(common),
           'epochs': {f'{c}_s{s}': eps[(c, s)] for (c, s) in eps if s in usable},
           'strata': {}}
    print('\nCONTENT TEST (own pose vs stranger\'s), minADE. Negative favours pose.')
    print(f'{"stratum":<18}{"n":>7}{"pose":>9}{"shuf":>9}{"rel":>9}{"t":>8}{"p":>9}   95% CI')
    print('-' * 86)
    for name, m in strata.items():
        if m.sum() < 50:
            continue
        c = paired(runs, 'pose', 'shuf', common, m, 'minADE')
        v = paired(runs, 'pose', 'nopose', common, m, 'minADE')
        res['strata'][name] = {'n': int(m.sum()), 'content': c, 'pose_vs_nopose': v}
        print(f'{name:<18}{int(m.sum()):>7}{c["a"]:>9.4f}{c["b"]:>9.4f}'
              f'{c["rel"]:>+8.2f}%{c["t"]:>8.2f}{c["p"]:>9.4f}   '
              f'[{c["lo"]:+.2f},{c["hi"]:+.2f}]')

    # only horizons stored by every run in every cell
    have = set(METRICS)
    for c in runs:
        for sd in runs[c]:
            for a_ in common:
                have &= set(runs[c][sd][a_])
                if not have:
                    break
    extra = [m for m in ('minADE_2s', 'minADE_4s') if m in have]
    if extra:
        print('\nsame strata, shorter horizons (content test):')
        for metric in extra:
            row = []
            for name in ('ALL', '<19m', '<10m'):
                if name in res['strata']:
                    c = paired(runs, 'pose', 'shuf', common, strata[name], metric)
                    res['strata'][name][f'content_{metric}'] = c
                    row.append(f'{name} {c["rel"]:+.2f}% (p={c["p"]:.2f})')
            print(f'  {metric:<10} ' + '   '.join(row))
    else:
        print('\nper-horizon metrics not stored by every 25k run; skipped')

    print('\npose vs no-pose (includes the branch-capacity tax):')
    for name in ('ALL', '<19m', '<10m'):
        if name in res['strata']:
            v = res['strata'][name]['pose_vs_nopose']
            print(f'  {name:<8} {v["rel"]:+6.2f}%  p={v["p"]:.4f}  '
                  f'CI=[{v["lo"]:+.2f},{v["hi"]:+.2f}]')

    # A real proximity benefit would grow monotonically as range falls.  Coarse
    # strata can manufacture one by straddling bin edges, so test the gradient
    # directly in equal-count bins.
    print('\ncontent effect in 8 equal-count range bins (the gradient test):')
    print(f'  {"bin":<16}{"n":>7}{"rel":>9}{"p":>9}{"MDE@80%":>10}')
    edges = np.percentile(R, np.linspace(0, 100, 9))
    mids, rels, bins = [], [], []
    for i in range(8):
        m = (R >= edges[i]) & (R < edges[i + 1]) if i < 7 else (R >= edges[i])
        c = paired(runs, 'pose', 'shuf', common, m, 'minADE')
        se = np.std(c['per_seed'], ddof=1) / np.sqrt(c['n'])
        mde = se * (stats.t.ppf(.975, c['n'] - 1) + stats.t.ppf(.80, c['n'] - 1))
        mids.append(float(R[m].mean())); rels.append(c['rel'])
        bins.append({'lo': float(edges[i]), 'hi': float(edges[min(i + 1, 7)]),
                     'n': int(m.sum()), **c, 'mde80': float(mde)})
        print(f'  [{edges[i]:5.1f},{edges[i+1]:5.1f})'.ljust(18)
              + f'{int(m.sum()):>7}{c["rel"]:>+8.2f}%{c["p"]:>9.3f}{mde:>9.2f}%')
    r_p, p_p = stats.pearsonr(mids, rels)
    r_s, p_s = stats.spearmanr(mids, rels)
    print(f'  trend with range: Pearson r={r_p:+.3f} (p={p_p:.3f}), '
          f'Spearman={r_s:+.3f} (p={p_s:.3f})')
    print('  (a genuine proximity benefit would make this strongly positive)')
    res['range_bins'] = bins
    res['trend'] = {'pearson_r': float(r_p), 'pearson_p': float(p_p),
                    'spearman_r': float(r_s), 'spearman_p': float(p_s)}

    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'\nwrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
