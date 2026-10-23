#!/usr/bin/env python
"""How accurate must a side channel be before the pose branch converts it into a gain?

The content test bounds what SMPL pose is worth (+/-1.7%), and the oracle control shows the
branch itself is live (-34.8%). Neither says how much signal a channel must carry before THIS
pathway turns it into accuracy. That threshold is what makes the null quantitative rather than
merely bounded: if the floor sits between the box heading's accuracy and SMPL's, the null is
explained by measurement quality rather than by pose being uninformative in principle.

Each rung feeds the branch a direction-only oracle (future displacement bearing at +3 s and
+8 s) corrupted by a fixed per-agent angular error. Rungs are labelled by the MEASURED median
absolute bearing error of the delivered channel, not the nominal sigma.

Reference points must be in THIS ladder's units -- median error on future displacement bearing,
pooled over every predicted pedestrian -- not the onset-direction numbers. Pooled: box heading
3.8 deg, SMPL facing 12.3 deg. Stationary subset, where facing is the cue velocity cannot supply:
box 24.7 deg, SMPL 41.2 deg. Mixing the two is what made Sec. IV-B compare 44.2 against a floor
calibrated on a different quantity (see research-log 2026-10-23).

Usage: python tools/scripts/detection_floor.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(REPO, 'experiments', 'detection_floor.json')
SEEDS = (101, 202, 303, 404, 505)
# rung -> measured median |bearing error| of the delivered channel (smoke-tested 2026-10-19)
RUNGS = {'00': 0.0, '05': 5.5, '10': 10.2, '15': 14.6, '25': 27.2,
         '35': 37.9, '45': 45.0, '60': 57.4}
# ladder units (pooled over all predicted pedestrians), from facing_error_ladder_units.py
BOX_HEADING_DEG, SMPL_FACING_DEG = 3.8, 12.3
# the same two channels on the stationary subset, the population pose was meant to serve
BOX_STATIONARY_DEG, SMPL_STATIONARY_DEG = 24.7, 41.2
# map_nopose_fix_s404 is a ~2 SD outlier for its cell (0.4954 vs 0.443-0.453) although seed 404
# is better than average across 37 other cells. Kept: a baseline landing badly is training noise,
# and it biases the oracle's apparent gain UPWARD, which is the conservative direction for the
# claim that the gain is gone by 10.2 deg. The leave-one-out column makes it visible.
WATCH_SEED = 404


CFG = 'mtr+pose_data_cross_attn_pe_with_map_oracledir{}_fixdata'
FINAL_EPOCH = 30


def best_minade(key: str, seed: int) -> float | None:
    """Best-epoch minADE, from the run's own best_eval_record.txt.

    That file is the canonical artifact the rest of the project scores against; the
    /tmp training log is only a fallback for runs whose output dir has been pruned.

    A RUNNING job must never be scored. The log fallback takes min() over every
    minADE printed so far, so an in-progress run contributes its epoch-2 value and
    reads as a catastrophic regression -- rung 10.2 showed +25.0% from a seed that
    was 2/30 epochs in. The fallback therefore applies only to runs that finished:
    a DONE marker (written by the queue scripts), or a best_eval_record.txt that has
    reached FINAL_EPOCH -- that file is appended after every eval epoch, so its mere
    existence means the run started, not that it ended.
    """
    tag = f'MS_oracledir{key}_s{seed}'
    run_dir = os.path.join(REPO, 'output', 'waymo', CFG.format(key), tag)
    rec = os.path.join(run_dir, 'eval', 'eval_with_train', 'best_eval_record.txt')
    done = os.path.exists(os.path.join(run_dir, 'DONE'))
    if not done:
        seen = (max((int(m) for m in re.findall(r'^epoch_(\d+)\s', open(rec).read(),
                                                re.M)), default=0)
                if os.path.exists(rec) else 0)
        if seen < FINAL_EPOCH:
            if os.path.exists(run_dir):
                print(f'  (skipping {tag}: at epoch {seen}/{FINAL_EPOCH}, still running)')
            return None
    if os.path.exists(rec):
        best = None
        for line in open(rec):
            m = re.search(r'best_epoch_\d+\s+minADE\s+([0-9.]+)', line)
            if m:
                best = float(m.group(1))
        if best is not None:
            return best
    log = f'/tmp/{tag}.log'
    if os.path.exists(log):
        v = [float(x) for x in re.findall(r'minADE: ([0-9.]+)', open(log).read())]
        if v:
            return min(v)
    return None


def main() -> int:
    base = json.load(open(os.path.join(REPO, 'experiments', 'multiseed_analysis.json')))
    B = {int(k): v['best_minADE'] for k, v in base['cells']['map_nopose_fix']['seeds'].items()
         if v.get('best_minADE') is not None}

    rows = []
    print(f'{"median err":>11}{"n":>4}{"oracle":>9}{"no-pose":>9}{"rel":>9}'
          f'{"t":>8}{"p":>9}{"MDE":>7}   95% CI')
    print('-' * 86)
    for key, med in RUNGS.items():
        vals = {s: best_minade(key, s) for s in SEEDS}
        s_ok = [s for s in SEEDS if vals[s] is not None and s in B]
        if len(s_ok) < 2:
            print(f'{med:>9.1f}d  insufficient runs'); continue
        av = np.array([vals[s] for s in s_ok]); bv = np.array([B[s] for s in s_ok])
        rel = (av - bv) / bv * 100
        n = len(s_ok); t, p = stats.ttest_1samp(rel, 0.0); sd = rel.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        lo, hi = rel.mean() - h, rel.mean() + h
        # minimum effect this rung could resolve at 80% power given its realised spread
        mde = sd / np.sqrt(n) * (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))
        # leave-one-out: a rung whose verdict rests on one seed is not a measurement
        loo = [float(np.delete(rel, i).mean()) for i in range(n)] if n >= 4 else []
        loo_s = f'  LOO [{min(loo):+.2f},{max(loo):+.2f}]' if loo else ''
        print(f'{med:>9.1f}d{n:>4}{av.mean():>9.4f}{bv.mean():>9.4f}'
              f'{(av.mean()-bv.mean())/bv.mean()*100:>+9.2f}%{t:>8.2f}{p:>9.4f}'
              f'{mde:>7.2f}   [{lo:+.2f},{hi:+.2f}]{loo_s}')
        if WATCH_SEED in s_ok and n >= 4:
            i = s_ok.index(WATCH_SEED)
            print(f'{"":>11}  s{WATCH_SEED} alone {rel[i]:+.2f}%, '
                  f'without it {np.delete(rel, i).mean():+.2f}%')
        rows.append(dict(rung=key, median_err_deg=med, n=n, oracle=float(av.mean()),
                         nopose=float(bv.mean()), rel=float(rel.mean()), t=float(t),
                         p=float(p), lo=float(lo), hi=float(hi), mde80=float(mde),
                         seeds=list(s_ok), loo_rel=loo,
                         per_seed_rel=[float(x) for x in rel],
                         per_seed=[float(x) for x in av]))

    sig = [r for r in rows if r['hi'] < 0]
    ns = [r for r in rows if r['hi'] >= 0]
    print()
    if not rows:
        print('no rungs could be read -- nothing to conclude')
        return 1
    if sig and ns:
        last, first = max(r['median_err_deg'] for r in sig), min(r['median_err_deg'] for r in ns)
        print(f'gain is significant up to {last:.1f} deg and indistinguishable from zero '
              f'from {first:.1f} deg onward')
        print(f'  => detection floor lies between {last:.1f} and {first:.1f} deg')
        def verdict(th):
            return ('inside the bracket (pays)' if th < last else
                    'outside (does not pay)' if th > first else 'in the gap')
        for lab, th in (('box heading  (pooled)    ', BOX_HEADING_DEG),
                        ('SMPL facing  (pooled)    ', SMPL_FACING_DEG),
                        ('box heading  (stationary)', BOX_STATIONARY_DEG),
                        ('SMPL facing  (stationary)', SMPL_STATIONARY_DEG)):
            print(f'  {lab} {th:>5.1f} deg : {verdict(th)}')
        print('  NB a median is not a rung: a rung corrupts every agent uniformly, while SMPL')
        print('     is a heavy tail with 56.8% of agents outside the 10.2 deg bracket.')
        floor = [last, first]
    elif not ns:
        print('every rung still shows a significant gain: the floor is beyond the ladder')
        floor = [max(RUNGS.values()), None]
    else:
        print('no rung shows a significant gain: the floor is below the finest rung tested')
        floor = [None, min(r['median_err_deg'] for r in ns)]

    json.dump(dict(rungs=rows, floor_bracket_deg=floor, seeds=list(SEEDS),
                   box_heading_deg=BOX_HEADING_DEG, smpl_facing_deg=SMPL_FACING_DEG,
                   box_stationary_deg=BOX_STATIONARY_DEG,
                   smpl_stationary_deg=SMPL_STATIONARY_DEG,
                   note='direction-only oracle; rungs labelled by measured median bearing error; '
                        'paired t over seeds against map_nopose_fix'),
              open(OUT, 'w'), indent=2)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
