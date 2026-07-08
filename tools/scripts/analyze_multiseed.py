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
        # Last EVALUATED epoch — not necessarily 30 (a run may die early); the old
        # 'epoch30_minADE' label was a lie for short runs.
        'last_eval_epoch': int(max(epochs)),
        'last_eval_minADE': per_epoch[max(epochs)][0],
        'last_eval_minFDE': per_epoch[max(epochs)][1],
        'n_evals': len(epochs),
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


def _two_sided_p(boots, n_boot):
    """Two-sided bootstrap p, floored at 1/n_boot and capped at 1.0.
    A literal 0.0 (no bootstrap crossed zero) is not evidence of p==0."""
    p = 2.0 * min((boots >= 0).mean(), (boots <= 0).mean())
    return float(min(max(p, 1.0 / n_boot), 1.0))


def paired_bootstrap(a_means, b_means, n_boot=10000, seed=0):
    """Paired bootstrap over pedestrians on SEED-AVERAGED per-agent minADE.

    This marginalizes over seeds (each agent's value is its mean across seeds), so the
    CI reflects pedestrian sampling variance only, NOT seed-to-seed variance. Read
    alongside hierarchical_bootstrap, which resamples seeds as well.
    """
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
        'p_boot_two_sided': _two_sided_p(boots, n_boot),
    }


def cell_per_seed_agent_metrics(cell):
    """{seed: {agent_key: minADE}} at each seed's own best epoch (NOT seed-averaged)."""
    per_seed = {}
    for seed in SEEDS:
        run_dir = os.path.join(OUT_ROOT, CELLS[cell], f'MS_{cell}_s{seed}')
        s = run_summary(run_dir)
        if s is None:
            continue
        m = load_agent_metrics(run_dir, s['best_epoch'])
        if m is not None:
            per_seed[seed] = {k: v[0] for k, v in m.items()}
    return per_seed


def hierarchical_bootstrap(a_per_seed, b_per_seed, n_boot=10000, seed=0):
    """Two-level paired bootstrap: resample SEEDS (with replacement), then PEDESTRIANS.

    Captures both seed-to-seed and pedestrian-to-pedestrian variance, which is what the
    paper's seed-level significance claims actually require. Returns None if fewer than
    2 common seeds (seed resampling is meaningless with 1 seed).
    """
    seeds = sorted(set(a_per_seed) & set(b_per_seed))
    if len(seeds) < 2:
        return None
    keys = sorted(set.intersection(*[set(a_per_seed[s]) for s in seeds],
                                   *[set(b_per_seed[s]) for s in seeds]))
    if not keys:
        return None
    A = np.array([[a_per_seed[s][k] for k in keys] for s in seeds])  # (S, N)
    B = np.array([[b_per_seed[s][k] for k in keys] for s in seeds])  # (S, N)
    D = B - A  # (S, N)
    S, N = D.shape
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for bi in range(n_boot):
        s_idx = rng.randint(0, S, size=S)   # resample seeds (clusters)
        a_idx = rng.randint(0, N, size=N)   # resample pedestrians
        boots[bi] = D[np.ix_(s_idx, a_idx)].mean()
    a_mean = float(A.mean())
    mean_diff = float(D.mean())
    return {
        'n_seeds': S,
        'n_agents': N,
        'mean_diff': mean_diff,
        'rel_diff_pct': float(100 * mean_diff / a_mean),
        'ci95_lo': float(np.percentile(boots, 2.5)),
        'ci95_hi': float(np.percentile(boots, 97.5)),
        'p_boot_two_sided': _two_sided_p(boots, n_boot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',
                    default=os.path.join(ROOT, 'experiments', 'multiseed_analysis.json'))
    args = ap.parse_args()

    results = {'cells': {}, 'pairs': {}, 'sanity': {}}

    for cell, cfg in CELLS.items():
        seeds = {}
        for seed in SEEDS:
            run_dir = os.path.join(OUT_ROOT, cfg, f'MS_{cell}_s{seed}')
            s = run_summary(run_dir)
            if s is not None:
                seeds[str(seed)] = {k: v for k, v in s.items() if k != 'per_epoch'}
        agg = {}
        for proto in ['best_minADE', 'best_minFDE', 'last_eval_minADE',
                      'last5_mean_minADE', 'last5_mean_minFDE']:
            vals = [seeds[s][proto] for s in seeds]
            if vals:
                agg[proto] = {'mean': float(np.mean(vals)),
                              'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              'values': vals}
        results['cells'][cell] = {'seeds': seeds, 'agg': agg}

    # Epoch-count sanity check: a run that died early has fewer evaluated epochs and a
    # higher-than-real "best" minADE (fewer chances to improve) — it must not silently
    # pollute a cell mean. Flag any seed whose n_evals is below the modal count.
    all_nevals = [seeds_d['n_evals']
                  for c in results['cells'].values()
                  for seeds_d in c['seeds'].values()]
    if all_nevals:
        modal = int(np.bincount(all_nevals).argmax())
        results['sanity']['modal_n_evals'] = modal
        short = []
        for cell, cdata in results['cells'].items():
            for sd, sdata in cdata['seeds'].items():
                if sdata['n_evals'] < modal:
                    short.append({'cell': cell, 'seed': sd,
                                  'n_evals': sdata['n_evals'],
                                  'last_eval_epoch': sdata['last_eval_epoch']})
        results['sanity']['short_runs'] = short

    means_cache = {}
    per_seed_cache = {}
    for a, b in PAIRS:
        for c in (a, b):
            if c not in means_cache:
                means_cache[c] = cell_agent_means(c)
                per_seed_cache[c] = cell_per_seed_agent_metrics(c)
        if means_cache[a] is None or means_cache[b] is None:
            results['pairs'][f'{a}->{b}'] = 'missing data'
            continue
        entry = {'paired': paired_bootstrap(means_cache[a], means_cache[b])}
        hier = hierarchical_bootstrap(per_seed_cache[a], per_seed_cache[b])
        entry['hierarchical'] = hier if hier is not None else 'insufficient seeds (<2)'
        results['pairs'][f'{a}->{b}'] = entry

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
    if results['sanity'].get('short_runs'):
        print(f"\n[sanity] modal n_evals={results['sanity']['modal_n_evals']}; "
              f"SHORT runs (excluded-quality warning): "
              f"{[(s['cell'], s['seed'], s['n_evals']) for s in results['sanity']['short_runs']]}")

    print()
    hdr = f"{'pair':<24} {'method':<13} {'rel %':<9} {'95% CI (abs)':<24} p"
    print(hdr)
    for k, v in results['pairs'].items():
        if isinstance(v, str):
            print(f"{k:<24} {v}")
            continue
        for method in ('paired', 'hierarchical'):
            m = v[method]
            if isinstance(m, str):
                print(f"{k:<24} {method:<13} {m}")
                continue
            print(f"{k:<24} {method:<13} {m['rel_diff_pct']:+.2f}%   "
                  f"[{m['ci95_lo']:+.4f}, {m['ci95_hi']:+.4f}]   p={m['p_boot_two_sided']:.4f}")
    print(f"\nfull JSON: {args.out}")


if __name__ == '__main__':
    main()
