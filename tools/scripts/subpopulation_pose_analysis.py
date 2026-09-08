#!/usr/bin/env python
"""Event-conditioned re-analysis of the 25k rung: does pose help where its cue exists?

The aggregate minADE averages over ~16.5k validation pedestrians. The one behavioural cue we
measured as real — body facing predicting the direction of motion onset — applies only to
pedestrians who are STANDING and then walk off (~1.3% of agents). A genuine effect there is
invisible in the mean. This script asks the decisive question (correct pose vs. its own
shuffled control) separately within each subpopulation, reusing the per-agent metrics that
already exist on disk. No training, no cluster.

Stage 1 builds per-agent dynamics from the Waymo validation trajectories (independent of the
model): speed at t=0 and displacement over the next 3 s.
Stage 2 loads per-agent minADE at each run's best epoch, seed-averages, and runs the same
hierarchical bootstrap used for the headline numbers, restricted to each subpopulation.
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCEN = '/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_validation'
DYN_CACHE = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
OUT_JSON = os.path.join(REPO, 'experiments', 'subpopulation_pose_analysis.json')

CELLS = {
    'nopose': 'mtr+pose_data_no_pose_with_map_scale5k_lr5/HELMA_25k_no_pose_with_map_lr25_s{}',
    'pose': 'mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5/HELMA_25k_cross_attn_pe_with_map_lr25_s{}',
    'shuffle': ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k_lr5/'
                'HELMA_25k_cross_attn_pe_with_map_shufflepose_lr25_s{}'),
}
SEEDS = (101, 202, 303)
CUR = 10                       # index of "now" on the 91-step grid
N_BOOT = 10000


# ---------------------------------------------------------------- stage 1: dynamics
def _scene_dynamics(scene: str) -> dict:
    path = os.path.join(SCEN, f'sample_{scene}_0.pkl')
    if not os.path.exists(path):
        return {}
    try:
        info = pickle.load(open(path, 'rb'))
    except Exception:
        return {}
    ti = info['track_infos']
    ids, trajs = list(ti['object_id']), ti['trajs']
    out = {}
    for oi, oid in enumerate(ids):
        tr = trajs[oi]
        if tr[CUR, -1] <= 0:
            continue
        speed = float(np.linalg.norm(tr[CUR, 7:9]))
        fut = tr[CUR + 1:CUR + 31]                       # next 3 s
        fv = fut[:, -1] > 0
        disp3 = float(np.linalg.norm(fut[fv][-1, :2] - tr[CUR, :2])) if fv.sum() >= 10 else np.nan
        out[(scene, str(oid))] = {'speed0': speed, 'disp3s': disp3}
    return out


def build_dynamics(scenes: list[str]) -> dict:
    if os.path.exists(DYN_CACHE):
        dyn = pickle.load(open(DYN_CACHE, 'rb'))
        if len(dyn) >= 0.95 * len(scenes):
            print(f'dynamics cache: {len(dyn)} agents')
            return dyn
    dyn: dict = {}
    with ProcessPoolExecutor(16) as ex:
        for i, part in enumerate(ex.map(_scene_dynamics, scenes, chunksize=8)):
            dyn.update(part)
            if (i + 1) % 1000 == 0:
                print(f'  ...{i + 1}/{len(scenes)} scenes, {len(dyn)} agents', flush=True)
    pickle.dump(dyn, open(DYN_CACHE, 'wb'))
    print(f'dynamics built: {len(dyn)} agents -> {DYN_CACHE}')
    return dyn


# ---------------------------------------------------------------- stage 2: metrics
def per_agent_at_best_epoch(run_dir: str) -> dict:
    """{(scenario, object): minADE} at the epoch with the lowest overall mean minADE."""
    best, best_m = None, np.inf
    for p in glob.glob(os.path.join(run_dir, 'eval/eval_with_train/metrics_epoch_*.pkl')):
        try:
            recs = pickle.load(open(p, 'rb'))
        except Exception:
            continue
        vals = np.array([r['minADE'] for r in recs], dtype=float)
        if not len(vals) or not np.isfinite(vals).all():
            continue
        m = vals.mean()
        if m < best_m:
            best_m, best = m, {(r['scenario_id'], str(r['object_id'])): float(r['minADE'])
                               for r in recs}
    return best or {}


def hierarchical_bootstrap(a_seeds: list[dict], b_seeds: list[dict], keys: list, rng) -> tuple:
    """Resample seeds, then agents (the protocol used for every headline number)."""
    A = np.array([[s[k] for k in keys] for s in a_seeds])
    B = np.array([[s[k] for k in keys] for s in b_seeds])
    obs = A.mean() - B.mean()
    n_seed, n_ag = A.shape
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        si = rng.integers(0, n_seed, n_seed)
        ai = rng.integers(0, n_ag, n_ag)
        boots[i] = A[np.ix_(si, ai)].mean() - B[np.ix_(si, ai)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots >= 0).mean(), (boots <= 0).mean())
    return obs, lo, hi, min(max(p, 1.0 / N_BOOT), 1.0)


def main() -> int:
    runs = {c: [per_agent_at_best_epoch(os.path.join(REPO, 'output/waymo', tpl.format(s)))
                for s in SEEDS] for c, tpl in CELLS.items()}
    for c, seeds in runs.items():
        print(f'{c}: agents per seed {[len(s) for s in seeds]}')
        if any(not s for s in seeds):
            print(f'  missing metrics for {c} — aborting'); return 1

    common = set(runs['nopose'][0])
    for seeds in runs.values():
        for s in seeds:
            common &= set(s)
    print(f'agents common to all 9 runs: {len(common)}')

    scenes = sorted({k[0] for k in common})
    dyn = build_dynamics(scenes)

    groups = {
        'ALL': lambda d: True,
        'standing_then_moves': lambda d: d['speed0'] < 0.3 and np.isfinite(d['disp3s']) and d['disp3s'] > 1.0,
        'standing_stays': lambda d: d['speed0'] < 0.3 and np.isfinite(d['disp3s']) and d['disp3s'] <= 1.0,
        'walking': lambda d: d['speed0'] >= 0.5,
    }

    rng = np.random.default_rng(0)
    results = {}
    for gname, pred in groups.items():
        keys = sorted(k for k in common if k in dyn and pred(dyn[k]))
        if len(keys) < 30:
            print(f'\n[{gname}] only {len(keys)} agents — skipped'); continue
        means = {c: float(np.mean([[s[k] for k in keys] for s in seeds])) for c, seeds in runs.items()}
        print(f'\n[{gname}] n={len(keys)} agents ({100*len(keys)/len(common):.1f}% of eval)')
        for c, m in means.items():
            print(f'    {c:<8} {m:.4f}')
        entry = {'n_agents': len(keys), 'means': means, 'tests': {}}
        for a, b in (('pose', 'shuffle'), ('nopose', 'pose'), ('nopose', 'shuffle')):
            obs, lo, hi, p = hierarchical_bootstrap(runs[a], runs[b], keys, rng)
            rel = 100 * obs / means[b]
            entry['tests'][f'{a}_vs_{b}'] = {'abs': obs, 'rel_pct': rel, 'ci': [lo, hi], 'p': p}
            star = '  <-- DECISIVE' if (a, b) == ('pose', 'shuffle') else ''
            print(f'    {a} vs {b}: {rel:+.2f}%  CI [{lo:+.4f},{hi:+.4f}]  p={p:.4f}{star}')
        results[gname] = entry

    import json
    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    print(f'\nwrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
