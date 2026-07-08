#!/usr/bin/env python
"""Analyze the multi-seed replication matrix.

For every run (cell x seed):
  - parse log_train_*.txt for per-epoch validation minADE/minFDE
  - protocols: best-checkpoint (min over evaluated epochs), epoch-30, mean of last-5 evals
Per cell: mean +/- std over seeds (each protocol).
Pairwise: paired bootstrap CIs over validation pedestrians using per-agent metrics
(metrics_epoch_N.pkl) at each run's best epoch, seed-averaged per agent.

Usage: python analyze_multiseed.py [--out results.json]
"""
import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(ROOT, 'output', 'waymo')

CELLS = {
    'baseline':   'mtr+pose_data_no_pose',
    'wta01':      'mtr+pose_data_geo_only',
    'geo_pure':   'mtr+pose_data_geo_pure',
    'map_nopose': 'mtr+pose_data_no_pose_with_map',
    'map_wta01':  'mtr+pose_data_geo_only_with_map',
    'ft_nopose':  'mtr+full_ped_finetune_no_pose',
    'ft_wta01':   'mtr+full_ped_finetune_geo',
    'xattn':      'mtr+pose_data_cross_attn',
    'xattn_pe':   'mtr+pose_data_cross_attn_pe',
    'gmm_only':   'mtr+pose_data_gmm_only',
    'mpjpe':      'mtr+pose_data_mpjpe_only',
    'full':       'mtr+pose_data',
}
SEEDS = [101, 202, 303, 404, 505]  # 404/505 only exist for headline cells (phase 2)

# pairs for paired bootstrap (cellA = reference, cellB = treatment)
PAIRS = [
    ('baseline', 'wta01'),
    ('baseline', 'geo_pure'),
    ('baseline', 'gmm_only'),
    ('wta01', 'geo_pure'),
    ('wta01', 'gmm_only'),
    ('wta01', 'mpjpe'),
    ('wta01', 'full'),
    ('baseline', 'xattn'),
    ('baseline', 'xattn_pe'),
    ('baseline', 'mpjpe'),
    ('baseline', 'full'),
    ('xattn', 'xattn_pe'),
    ('xattn_pe', 'wta01'),
    ('map_nopose', 'map_wta01'),
    ('baseline', 'map_nopose'),
    ('ft_nopose', 'ft_wta01'),
    ('map_nopose', 'ft_nopose'),
    ('map_wta01', 'ft_wta01'),
]

EPOCH_RE = re.compile(r'Performance of EPOCH (\d+)')
ADE_RE = re.compile(r'^minADE:\s*([\d.]+)')
FDE_RE = re.compile(r'^minFDE:\s*([\d.]+)')


def parse_log(run_dir):
    """Return {epoch: (minADE, minFDE)} from the run's training log(s)."""
    logs = sorted(glob.glob(os.path.join(run_dir, 'log_train_*.txt')))
    per_epoch = {}
    for log in logs:
        cur_epoch = None
        cur_ade = None
        with open(log) as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    cur_epoch = int(m.group(1))
                    cur_ade = None
                    continue
                m = ADE_RE.match(line)
                if m and cur_epoch is not None:
                    cur_ade = float(m.group(1))
                    continue
                m = FDE_RE.match(line)
                if m and cur_epoch is not None and cur_ade is not None:
                    per_epoch[cur_epoch] = (cur_ade, float(m.group(1)))
                    cur_epoch, cur_ade = None, None
    return per_epoch


def run_summary(run_dir):
    per_epoch = parse_log(run_dir)
    if not per_epoch:
        return None
    epochs = sorted(per_epoch)
    ades = {e: per_epoch[e][0] for e in epochs}
    best_epoch = min(ades, key=ades.get)
    last5 = [e for e in epochs if e > max(epochs) - 5]
    return {
        'per_epoch': {str(e): per_epoch[e] for e in epochs},
        'best_epoch': best_epoch,
        'best_minADE': per_epoch[best_epoch][0],
        'best_minFDE': per_epoch[best_epoch][1],
        'epoch30_minADE': per_epoch[max(epochs)][0],
        'epoch30_minFDE': per_epoch[max(epochs)][1],
        'last5_mean_minADE': float(np.mean([per_epoch[e][0] for e in last5])),
        'last5_mean_minFDE': float(np.mean([per_epoch[e][1] for e in last5])),
    }


def load_agent_metrics(run_dir, epoch):
    path = os.path.join(run_dir, 'eval', 'eval_with_train', f'metrics_epoch_{epoch}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        recs = pickle.load(f)
    return {(r['scenario_id'], str(r['object_id'])): (r['minADE'], r['minFDE']) for r in recs}


def cell_agent_means(cell):
    """Seed-averaged per-agent minADE at each seed's best epoch. {agent_key: mean ADE}."""
    per_seed = []
    for seed in SEEDS:
        run_dir = os.path.join(OUT_ROOT, CELLS[cell], f'MS_{cell}_s{seed}')
        s = run_summary(run_dir)
        if s is None:
            continue
        m = load_agent_metrics(run_dir, s['best_epoch'])
        if m is not None:
            per_seed.append(m)
    if not per_seed:
        return None
    keys = set(per_seed[0])
    for m in per_seed[1:]:
        keys &= set(m)
    return {k: float(np.mean([m[k][0] for m in per_seed])) for k in keys}


def paired_bootstrap(a_means, b_means, n_boot=10000, seed=0):
    keys = sorted(set(a_means) & set(b_means))
    a = np.array([a_means[k] for k in keys])
    b = np.array([b_means[k] for k in keys])
    diff = b - a
    rng = np.random.RandomState(seed)
    n = len(keys)
    idx = rng.randint(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return {
        'n_agents': n,
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'mean_diff': float(diff.mean()),
        'rel_diff_pct': float(100 * diff.mean() / a.mean()),
        'ci95_lo': float(np.percentile(boots, 2.5)),
        'ci95_hi': float(np.percentile(boots, 97.5)),
        'p_boot_two_sided': float(2 * min((boots >= 0).mean(), (boots <= 0).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',
                    default=os.path.join(ROOT, 'experiments', 'multiseed_analysis.json'))
    args = ap.parse_args()

    results = {'cells': {}, 'pairs': {}}

    for cell, cfg in CELLS.items():
        seeds = {}
        for seed in SEEDS:
            run_dir = os.path.join(OUT_ROOT, cfg, f'MS_{cell}_s{seed}')
            s = run_summary(run_dir)
            if s is not None:
                seeds[str(seed)] = {k: v for k, v in s.items() if k != 'per_epoch'}
        agg = {}
        for proto in ['best_minADE', 'best_minFDE', 'epoch30_minADE',
                      'last5_mean_minADE', 'last5_mean_minFDE']:
            vals = [seeds[s][proto] for s in seeds]
            if vals:
                agg[proto] = {'mean': float(np.mean(vals)),
                              'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              'values': vals}
        results['cells'][cell] = {'seeds': seeds, 'agg': agg}

    means_cache = {}
    for a, b in PAIRS:
        for c in (a, b):
            if c not in means_cache:
                means_cache[c] = cell_agent_means(c)
        if means_cache[a] is None or means_cache[b] is None:
            results['pairs'][f'{a}->{b}'] = 'missing data'
            continue
        results['pairs'][f'{a}->{b}'] = paired_bootstrap(means_cache[a], means_cache[b])

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    # human-readable summary
    print(f"{'cell':<12} {'best minADE (mean±std over seeds)':<38} {'last5-mean':<22} seeds")
    for cell in CELLS:
        agg = results['cells'][cell]['agg']
        if 'best_minADE' in agg:
            m = agg['best_minADE']
            l5 = agg['last5_mean_minADE']
            sd = f"±{m['std']:.4f}" if m['std'] is not None else ""
            sd5 = f"±{l5['std']:.4f}" if l5['std'] is not None else ""
            print(f"{cell:<12} {m['mean']:.4f}{sd:<10} {str([f'{v:.4f}' for v in m['values']]):<28} "
                  f"{l5['mean']:.4f}{sd5:<10} n={len(m['values'])}")
    print()
    print(f"{'pair':<24} {'Δ minADE':<12} {'rel %':<9} {'95% CI':<22} p")
    for k, v in results['pairs'].items():
        if isinstance(v, str):
            print(f"{k:<24} {v}")
            continue
        print(f"{k:<24} {v['mean_diff']:+.4f}    {v['rel_diff_pct']:+.2f}%   "
              f"[{v['ci95_lo']:+.4f}, {v['ci95_hi']:+.4f}]   p={v['p_boot_two_sided']:.4f}")
    print(f"\nfull JSON: {args.out}")


if __name__ == '__main__':
    main()
