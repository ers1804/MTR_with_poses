#!/usr/bin/env python
"""Does pose pay for standing pedestrians once the model is trained on them alone?

The stationary split of the heading ladder (heading_noise_subpopulation.py) is null, and
the obvious reply is that full-data training lets the walking majority set the gradient,
so a cue that matters only for standing pedestrians never gets learned. This tests that
directly: each arm's sigma=0 model (the heading ladder's checkpoint_epoch_30) is fine-tuned
for 5 epochs at LR 6.25e-6, no decay, on centres below TRAIN_CENTER_SPEED_MAX = 0.3 m/s
only (slurm/60_heading_noise.sh standft). Eval still scores every agent, so the standing
subset is read from the per-agent pickles with the same speed0 definition.

Scored at the FIXED final epoch against the init model's own epoch-30 pickle, so no epoch
is chosen on the outcome. Epochs 1-4 are printed as a sensitivity check only.

Pickles: METRICS (default big1/heading_noise_metrics), laid out as output/waymo.
Writes experiments/standing_finetune.json.

Usage: python tools/scripts/standing_finetune.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = os.environ.get('METRICS', '/home/erik/big1/heading_noise_metrics')
DYN = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
OUT = os.path.join(REPO, 'experiments', 'standing_finetune.json')

SEEDS = (101, 202, 303)
ARMS = {'nopose': 'no_pose_with_map',
        'pose': 'cross_attn_pe_with_map',
        'shuffled': 'cross_attn_pe_with_map_shufflepose'}
STATIONARY = 0.3
ONSET_DISP = 1.0
FINAL = 5


def load(arm: str, seed: int, epoch: int | None) -> dict:
    """epoch=None is the init model (heading ladder sigma=0, epoch 30)."""
    if epoch is None:
        cfg, tag, ep = f'{ARMS[arm]}_headnoise00_scale25k', 'HN_25k_{cfg}_lr25_s{seed}', 30
    else:
        cfg, tag, ep = f'{ARMS[arm]}_standft_scale25k', 'FT_25k_{cfg}_lr625_s{seed}', epoch
    d = os.path.join(METRICS, f'mtr+pose_data_{cfg}', tag.format(cfg=cfg, seed=seed),
                     'eval', 'eval_with_train')
    recs = pickle.load(open(os.path.join(d, f'metrics_epoch_{ep}.pkl'), 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE']) for r in recs}


def ci(x: np.ndarray) -> dict:
    t, p = stats.ttest_1samp(x, 0.0)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(rel=float(x.mean()), p=float(p), lo=float(x.mean() - h),
                hi=float(x.mean() + h), per_seed=[float(v) for v in x])


def main() -> None:
    dyn = {(str(k[0]), str(k[1])): v for k, v in pickle.load(open(DYN, 'rb')).items()}
    stages = [None] + list(range(1, FINAL + 1))
    runs = {(a, seed, e): load(a, seed, e) for a in ARMS for seed in SEEDS for e in stages}
    common = set(dyn)
    for r in runs.values():
        common &= set(r)
    common = sorted(common)
    sp = np.array([dyn[a]['speed0'] for a in common])
    dp = np.array([dyn[a]['disp3s'] for a in common])
    subs = {'stationary': sp < STATIONARY,
            'standing_then_moves': (sp < STATIONARY) & (dp > ONSET_DISP),
            'stays_put': (sp < STATIONARY) & (dp <= ONSET_DISP),
            'ALL': np.ones(len(common), bool)}
    X = {k: np.array([r[a] for a in common]) for k, r in runs.items()}

    def m(a, seed, e, mask):
        return X[a, seed, e][mask].mean()

    def rel(a, ea, b, eb, mask):
        return np.array([(m(a, s, ea, mask) / m(b, s, eb, mask) - 1) * 100 for s in SEEDS])

    res = dict(_meta=dict(n_common_agents=len(common), seeds=list(SEEDS), final_epoch=FINAL,
                          stationary_speed=STATIONARY, onset_disp=ONSET_DISP,
                          sizes={k: int(v.sum()) for k, v in subs.items()},
                          note='init = sigma=0 heading-ladder model at epoch 30; ft = its '
                               'standing-only fine-tune; paired t over seeds; negative '
                               'favours the first-named; did = content(ft) - content(init) pp'))
    for sub, mask in subs.items():
        out = {}
        for e in stages:
            k = 'init' if e is None else f'ft_epoch{e}'
            r = dict(mean={a: float(np.mean([m(a, s, e, mask) for s in SEEDS])) for a in ARMS},
                     content=ci(rel('pose', e, 'shuffled', e, mask)),
                     pose_vs_nopose=ci(rel('pose', e, 'nopose', e, mask)))
            if e is not None:
                r['ft_vs_init'] = {a: ci(rel(a, e, a, None, mask)) for a in ARMS}
                r['did'] = ci(rel('pose', e, 'shuffled', e, mask)
                              - rel('pose', None, 'shuffled', None, mask))
            out[k] = r
        res[sub] = out
        print(f'\n=== {sub} (n={int(mask.sum())}) ===')
        for k, r in out.items():
            c, pn = r['content'], r['pose_vs_nopose']
            line = (f'  {k:<10} nopose {r["mean"]["nopose"]:.4f}  pose {r["mean"]["pose"]:.4f}  '
                    f'shuf {r["mean"]["shuffled"]:.4f}   content {c["rel"]:+6.2f}% p={c["p"]:.2f}'
                    f'   vs nopose {pn["rel"]:+6.2f}% p={pn["p"]:.2f}')
            if 'did' in r:
                f = r['ft_vs_init']
                line += (f'   DiD {r["did"]["rel"]:+5.2f}pp p={r["did"]["p"]:.2f}   ft vs init '
                         + ' '.join(f'{a} {f[a]["rel"]:+.1f}%' for a in ARMS))
            print(line)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
