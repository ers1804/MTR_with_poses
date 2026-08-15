#!/usr/bin/env python
"""Best-epoch analysis of the top-1 confirmation runs (t1_* cells).

Best epoch is selected per run by minADE (the paper's protocol); top1_ADE /
best_mode_rank / mode_spread are then read AT THAT EPOCH from metrics_epoch_N.pkl.
This tests whether the +13.4% top-1 degradation seen on last-epoch checkpoints
survives under the real protocol.
"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_multiseed import OUT_ROOT, hierarchical_bootstrap, run_summary

CELLS = {
    't1_nopose':  'mtr+pose_data_no_pose_with_map',
    't1_xattnpe': 'mtr+pose_data_cross_attn_pe_with_map',
    't1_shuffle': 'mtr+pose_data_cross_attn_pe_with_map_shufflepose',
    't1_const':   'mtr+pose_data_cross_attn_pe_with_map_constpose',
}
SEEDS = [101, 202, 303, 404, 505]
METRICS = ['minADE', 'top1_ADE', 'best_mode_rank', 'mode_spread_endpoint', 'minADE_2s']

def cell_data(cell):
    out = {}
    for s in SEEDS:
        rd = os.path.join(OUT_ROOT, CELLS[cell], f'MS_{cell}_s{s}')
        summ = run_summary(rd)
        if summ is None: continue
        p = os.path.join(rd, 'eval', 'eval_with_train', f"metrics_epoch_{summ['best_epoch']}.pkl")
        if not os.path.exists(p): continue
        recs = pickle.load(open(p, 'rb'))
        out[s] = {(r['scenario_id'], str(r['object_id'])): r for r in recs}
    return out

data = {c: cell_data(c) for c in CELLS}
print("seeds per cell:", {c: len(v) for c, v in data.items()})
res = {}
for m in METRICS:
    keys = sorted(set.intersection(*[{k for k, v in x.items() if m in v}
                                     for c in data.values() for x in c.values()]))
    print(f"\n=== {m} (best-epoch, n={len(keys)}) ===")
    res[m] = {'n': len(keys), 'cells': {}, 'pairs': {}}
    for c in CELLS:
        vals = [np.mean([x[k][m] for k in keys]) for x in data[c].values()]
        res[m]['cells'][c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1))}
        print(f"   {c:12} {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
    for ref, tr in [('t1_nopose','t1_xattnpe'), ('t1_const','t1_xattnpe'), ('t1_shuffle','t1_xattnpe'), ('t1_nopose','t1_const')]:
        a = {s: {k: x[k][m] for k in keys} for s, x in data[ref].items()}
        b = {s: {k: x[k][m] for k in keys} for s, x in data[tr].items()}
        r = hierarchical_bootstrap(a, b, n_boot=6000)
        res[m]['pairs'][f'{ref}->{tr}'] = {'rel_pct': r['rel_diff_pct'], 'p': r['p_boot_two_sided']}
        print(f"     {ref:11} -> {tr}: {r['rel_diff_pct']:+7.2f}%  p={r['p_boot_two_sided']:.4f}")

with open(os.path.join(OUT_ROOT, '..', '..', 'experiments', 'top1_confirmation.json'), 'w') as f:
    json.dump(res, f, indent=2)
print("\nwrote experiments/top1_confirmation.json")
