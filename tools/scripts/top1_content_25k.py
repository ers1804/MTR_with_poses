"""Top-1 and mode-ranking content test at 25k scenes, on the corrected export.

minADE takes the best of six modes regardless of rank, so it cannot see a cue
that only improves which mode the forecaster puts first. Sec. V-A argues the
oracle's surviving term is future-side mode selection, which makes the ranking
columns the place a pose effect would hide. This scores them.

There is an older `top1_confirmation.json`, but its arms are
`mtr+pose_data_cross_attn_pe_with_map{,_shufflepose}` -- the pre-fix export,
paired over 6,688 agents. Those numbers are not quotable for the same reason the
joints3d and norootorient cells are not: before the id-collision fix every pose
arm was already a shuffled arm. This script reads the 25k arms instead, the same
ones the content test and the stratified nulls use, over 16,486 agents at each
run's own best epoch.
"""
import json
import os
import pickle

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')
OUT_JSON = os.path.join(REPO, 'experiments', 'top1_content_25k.json')
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
# minADE is the published yardstick and is carried here only as a sanity check:
# it must reproduce the content test's own +0.04% at this scale.
COLS = ('top1_ADE', 'best_mode_rank', 'mode_spread_endpoint', 'minADE')


def load_run(cfg, tag, ms_cell, seed):
    ep = MS[ms_cell]['seeds'][str(seed)]['best_epoch']
    p = os.path.join(OUT_ROOT, cfg, tag.format(seed),
                     'eval', 'eval_with_train', f'metrics_epoch_{ep}.pkl')
    if not os.path.exists(p):
        return None
    out = {}
    for r in pickle.load(open(p, 'rb')):
        key = (str(r['scenario_id']), str(r['object_id']))
        out[key] = {c: float(r[c]) for c in COLS if c in r}
    return ep, out


def main():
    runs, epochs = {}, {}
    for name, (cfg, tag, cell) in CELLS.items():
        runs[name] = {}
        for sd in SEEDS:
            got = load_run(cfg, tag, cell, sd)
            if got is None:
                continue
            epochs[f'{name}_s{sd}'], runs[name][sd] = got

    res = {'n_seeds': {k: len(v) for k, v in runs.items()}, 'epochs': epochs,
           'metrics': {}}

    for col in COLS:
        # Agents the column exists for in every run of every arm, so the three
        # arms are compared over one population rather than three.
        agents = None
        for name in runs:
            for sd, rec in runs[name].items():
                have = {k for k, v in rec.items() if col in v}
                agents = have if agents is None else (agents & have)
        agents = sorted(agents)
        if not agents:
            continue

        means = {}
        for name in runs:
            means[name] = {sd: float(np.mean([rec[x][col] for x in agents]))
                           for sd, rec in runs[name].items()}

        entry = {'n_agents': len(agents), 'arm_means': means, 'pairs': {}}
        for ref, tr in (('shuf', 'pose'), ('nopose', 'pose'),
                        ('nopose', 'shuf')):
            sds = sorted(set(means[ref]) & set(means[tr]))
            a = np.array([means[ref][s] for s in sds])
            b = np.array([means[tr][s] for s in sds])
            d = b - a
            n = len(sds)
            t, p = stats.ttest_rel(b, a)
            se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            rel = 100.0 * d.mean() / a.mean()
            entry['pairs'][f'{ref}->{tr}'] = {
                'n': n, 'seeds': sds,
                'ref_mean': float(a.mean()), 'treat_mean': float(b.mean()),
                'rel_pct': float(rel),
                't': float(t), 'p': float(p),
                'lo_pct': float(100.0 * (d.mean() - tcrit * se) / a.mean()),
                'hi_pct': float(100.0 * (d.mean() + tcrit * se) / a.mean()),
                'mde80_pct': float(100.0 * se *
                                   (tcrit + stats.t.ppf(0.80, n - 1)) / a.mean()),
            }
        res['metrics'][col] = entry

    json.dump(res, open(OUT_JSON, 'w'), indent=2)

    for col, e in res['metrics'].items():
        print(f'\n{col}  (n={e["n_agents"]} agents)')
        for k, v in e['pairs'].items():
            print(f'  {k:16s} {v["rel_pct"]:+7.2f}%  '
                  f'[{v["lo_pct"]:+.2f},{v["hi_pct"]:+.2f}]  '
                  f'p={v["p"]:.4f}  MDE80={v["mde80_pct"]:.2f}pp  n={v["n"]}')
    print(f'\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
