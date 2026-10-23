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

Two ladders deliver the SAME corrupted channel through different pathways, so both are scored
here: 'late' through the cross-attention pose branch (the published ladder) and 'early' through
extra per-timestep agent feature channels. The delivered tensors are bit-identical at a given
rung -- early obj_trajs[..., 29:35] equals late obj_poses[..., :6] -- so the rungs carry the
same measured-error labels.

Usage: python tools/scripts/detection_floor.py [late|early]
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SEEDS = (101, 202, 303, 404, 505)
# rung -> measured median |bearing error| of the delivered channel (smoke-tested 2026-10-19)
_RUNGS = {'00': 0.0, '05': 5.5, '10': 10.2, '15': 14.6, '25': 27.2,
          '35': 37.9, '45': 45.0, '60': 57.4}
LADDERS = {
    'late': dict(cfg='mtr+pose_data_cross_attn_pe_with_map_oracledir{}_fixdata',
                 tag='MS_oracledir{key}_s{seed}', rungs=_RUNGS,
                 out='detection_floor.json',
                 label='cross-attention pose branch (published ladder)'),
    'early': dict(cfg='mtr+pose_data_early_oracledir{}_fixdata',
                  tag='MS_earlyoracle{key}_s{seed}',
                  rungs={k: _RUNGS[k] for k in ('00', '05', '10')},
                  out='detection_floor_early.json',
                  label='early fusion, per-timestep agent channels'),
}
# ladder units (pooled over all predicted pedestrians), from facing_error_ladder_units.py
BOX_HEADING_DEG, SMPL_FACING_DEG = 3.8, 12.3
# the same two channels on the stationary subset, the population pose was meant to serve
BOX_STATIONARY_DEG, SMPL_STATIONARY_DEG = 24.7, 41.2
FINAL_EPOCH = 30

# map_nopose_fix_s404 = 0.4954 is a ~2 SD outlier for its cell (others 0.4428-0.4584) although
# seed 404 averages BETTER than its cell mean across 37 other cells, and the sibling cells
# map_xattn_pe_fix / map_shufflepose_fix put the same seed at 0.4570 / 0.4547. It is the paired
# denominator for the two rungs Part A sharpens, and it dominates them: at rung 10.2 its pair
# reads -15.45% against -2.38/-2.75/-6.00. Every rung is therefore reported under each available
# denominator, and an edge that moves between them is reported as unresolved rather than picked.
WATCH_SEED = 404
BASELINE_CFG = 'mtr+pose_data_no_pose_with_map_fixdata'
RESAMPLE_TAG = 'MS_map_nopose_fix_s404_resample'


def _read_best(run_dir: str, tag: str) -> float | None:
    """Best-epoch minADE from a run's own best_eval_record.txt, or None if unfinished.

    A RUNNING job must never be scored. best_eval_record.txt is appended after EVERY eval
    epoch, so its existence means the run started, not that it ended -- reading it mid-run
    once made rung 10.2 report +25.0% from a seed that was 2/30 epochs in. A run counts as
    finished only with a DONE marker (written by the queue scripts) or a record that has
    reached FINAL_EPOCH.
    """
    rec = os.path.join(run_dir, 'eval', 'eval_with_train', 'best_eval_record.txt')
    if not os.path.exists(os.path.join(run_dir, 'DONE')):
        seen = (max((int(m) for m in re.findall(r'^epoch_(\d+)\s', open(rec).read(), re.M)),
                    default=0) if os.path.exists(rec) else 0)
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
    # only for runs whose output dir has been pruned; gated above, so never a live job
    log = f'/tmp/{tag}.log'
    if os.path.exists(log):
        v = [float(x) for x in re.findall(r'minADE: ([0-9.]+)', open(log).read())]
        if v:
            return min(v)
    return None


def best_minade(key: str, seed: int, L: dict) -> float | None:
    tag = L['tag'].format(key=key, seed=seed)
    return _read_best(os.path.join(REPO, 'output', 'waymo', L['cfg'].format(key), tag), tag)


def stats_for(rel: np.ndarray) -> dict:
    """Paired statistics on per-seed relative differences (negative favours the oracle)."""
    n = len(rel)
    t, p = stats.ttest_1samp(rel, 0.0)
    se = rel.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(n=n, rel=float(rel.mean()), t=float(t), p=float(p),
                lo=float(rel.mean() - h), hi=float(rel.mean() + h),
                # minimum effect this rung could resolve at 80% power given its spread
                mde80=float(se * (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))),
                # a rung whose verdict rests on one seed is not a measurement
                loo_rel=([float(np.delete(rel, i).mean()) for i in range(n)]
                         if n >= 4 else []),
                per_seed_rel=[float(x) for x in rel])


def infer_bracket(rows, sig_of=lambda r: r['hi'] < 0) -> dict:
    """Floor bracket from a MONOTONE prefix of significant rungs.

    "significant up to max(sig), gone from min(n.s.)" is only meaningful if significance
    falls monotonically in theta. It need not: rungs carry different n (the bracket rungs
    go to 5 seeds, the rest stay at 3) and MDE scales as 1/sqrt(n), so a higher rung can
    read significant purely because it is better powered. max/min over that returns an
    INVERTED bracket, which the reference-point test then reads as "pays". So walk the
    rungs in delivered-error order, stop at the FIRST transition, and report any later
    significant rung as an anomaly instead of folding it into an edge.
    """
    order = sorted(rows, key=lambda r: r['median_err_deg'])
    flags = [sig_of(r) for r in order]
    i = next((j for j, f in enumerate(flags) if not f), None)
    if i is None:
        return dict(lo=order[-1]['median_err_deg'], hi=None, anomalies=[],
                    note='every rung significant: the floor is beyond the ladder')
    if i == 0:
        return dict(lo=None, hi=order[0]['median_err_deg'], anomalies=[],
                    note='no rung significant: the floor is below the finest rung tested')
    return dict(lo=order[i - 1]['median_err_deg'], hi=order[i]['median_err_deg'],
                anomalies=[order[j]['median_err_deg']
                           for j in range(i + 1, len(order)) if flags[j]],
                note='bracketed')


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else 'late'
    if which not in LADDERS:
        print(f'usage: detection_floor.py [{"|".join(LADDERS)}]')
        return 2
    L = LADDERS[which]
    RUNGS, OUT = L['rungs'], os.path.join(REPO, 'experiments', L['out'])
    print(f'ladder: {which} -- {L["label"]}\n')

    base = json.load(open(os.path.join(REPO, 'experiments', 'multiseed_analysis.json')))
    B = {int(k): v['best_minADE'] for k, v in base['cells']['map_nopose_fix']['seeds'].items()
         if v.get('best_minADE') is not None}

    # denominators to score under: the published one, it minus the outlier seed, and --
    # once it exists -- a second sample of that seed's baseline
    resample = _read_best(os.path.join(REPO, 'output', 'waymo', BASELINE_CFG, RESAMPLE_TAG),
                          RESAMPLE_TAG)
    variants = [('all seeds', B, SEEDS)]
    if WATCH_SEED in B:
        variants.append((f'without s{WATCH_SEED}', B,
                         tuple(s for s in SEEDS if s != WATCH_SEED)))
    if resample is not None:
        print(f's{WATCH_SEED} baseline resample: {resample:.4f} '
              f'(original {B[WATCH_SEED]:.4f})\n')
        variants.append((f's{WATCH_SEED} resampled', {**B, WATCH_SEED: resample}, SEEDS))

    raw = {key: {s: best_minade(key, s, L) for s in SEEDS} for key in RUNGS}

    def build(Bmap, seeds):
        rows = []
        for key, med in RUNGS.items():
            s_ok = [s for s in seeds if raw[key][s] is not None and s in Bmap]
            if len(s_ok) < 2:
                continue
            av = np.array([raw[key][s] for s in s_ok])
            bv = np.array([Bmap[s] for s in s_ok])
            rows.append(dict(rung=key, median_err_deg=med, seeds=list(s_ok),
                             oracle=float(av.mean()), nopose=float(bv.mean()),
                             per_seed=[float(x) for x in av],
                             **stats_for((av - bv) / bv * 100)))
        return rows

    rows = build(B, SEEDS)
    if not rows:
        print('no rungs could be read -- nothing to conclude')
        return 1

    print(f'{"median err":>11}{"n":>4}{"oracle":>9}{"no-pose":>9}{"rel":>9}'
          f'{"t":>8}{"p":>9}{"MDE":>7}   95% CI')
    print('-' * 86)
    for r in sorted(rows, key=lambda r: r['median_err_deg']):
        loo = (f'  LOO [{min(r["loo_rel"]):+.2f},{max(r["loo_rel"]):+.2f}]'
               if r['loo_rel'] else '')
        print(f'{r["median_err_deg"]:>9.1f}d{r["n"]:>4}{r["oracle"]:>9.4f}'
              f'{r["nopose"]:>9.4f}{r["rel"]:>+9.2f}%{r["t"]:>8.2f}{r["p"]:>9.4f}'
              f'{r["mde80"]:>7.2f}   [{r["lo"]:+.2f},{r["hi"]:+.2f}]{loo}')
        if WATCH_SEED in r['seeds'] and r['n'] >= 4:
            i = r['seeds'].index(WATCH_SEED)
            print(f'{"":>11}  s{WATCH_SEED} alone {r["per_seed_rel"][i]:+.2f}%, without it '
                  f'{np.delete(np.array(r["per_seed_rel"]), i).mean():+.2f}%')
    missing = [RUNGS[k] for k in RUNGS if k not in {r['rung'] for r in rows}]
    if missing:
        print(f'  (no usable runs yet at {", ".join(f"{m:.1f}" for m in missing)} deg)')

    print('\ndetection floor (gain significant below, indistinguishable above):')
    brackets = {}
    for lab, Bmap, seeds in variants:
        rws = build(Bmap, seeds)
        b = (infer_bracket(rws) if rws else
             dict(lo=None, hi=None, anomalies=[], note='no usable rungs'))
        brackets[lab] = b
        if b['note'] != 'bracketed':
            print(f'  {lab:<18} {b["note"]}')
            continue
        power = ', '.join(f'{r["median_err_deg"]:.1f}d n={r["n"]} MDE={r["mde80"]:.1f}'
                          for r in sorted(rws, key=lambda r: r['median_err_deg'])
                          if r['median_err_deg'] in (b['lo'], b['hi']))
        print(f'  {lab:<18} [{b["lo"]:.1f}, {b["hi"]:.1f}] deg   ({power})')
        if b['anomalies']:
            print(f'  {"":<18} WARNING non-monotone: significant again at '
                  f'{", ".join(f"{a:.1f}" for a in b["anomalies"])} deg -- a rung above the '
                  f'bracket reading significant is a power or noise artifact, not a floor')

    edges = {(b['lo'], b['hi']) for b in brackets.values() if b['note'] == 'bracketed'}
    if len(edges) > 1:
        shown = ', '.join(f'{lab} [{b["lo"]:.1f},{b["hi"]:.1f}]'
                          for lab, b in brackets.items() if b['note'] == 'bracketed')
        print(f'\n  => THE BRACKET EDGE MOVES between denominators: {shown}.')
        print(f"     s{WATCH_SEED}'s no-pose baseline is a ~2 SD within-cell outlier, so the")
        print('     edge is NOT safely reportable. Quote it both ways, or add seeds, before')
        print('     putting a number in the paper.')

    primary = brackets['all seeds']
    if primary['note'] == 'bracketed':
        last, first = primary['lo'], primary['hi']

        def verdict(th):
            return ('inside the bracket (pays)' if th < last else
                    'outside (does not pay)' if th > first else 'in the gap')
        print("\nwhere the real channels sit, in this ladder's units:")
        for lab, th in (('box heading  (pooled)    ', BOX_HEADING_DEG),
                        ('SMPL facing  (pooled)    ', SMPL_FACING_DEG),
                        ('box heading  (stationary)', BOX_STATIONARY_DEG),
                        ('SMPL facing  (stationary)', SMPL_STATIONARY_DEG)):
            print(f'  {lab} {th:>5.1f} deg : {verdict(th)}')
        print('  NB a median is not a rung: a rung corrupts every agent uniformly, while SMPL')
        print('     is a heavy tail with 56.8% of agents outside the 10.2 deg bracket.')

    json.dump(dict(ladder=which, pathway=L['label'], rungs=rows,
                   floor_bracket_deg=[primary['lo'], primary['hi']],
                   floor_brackets=brackets, bracket_edge_stable=(len(edges) <= 1),
                   watch_seed=WATCH_SEED, watch_seed_baseline=B.get(WATCH_SEED),
                   watch_seed_baseline_resample=resample, seeds=list(SEEDS),
                   box_heading_deg=BOX_HEADING_DEG, smpl_facing_deg=SMPL_FACING_DEG,
                   box_stationary_deg=BOX_STATIONARY_DEG,
                   smpl_stationary_deg=SMPL_STATIONARY_DEG,
                   note='direction-only oracle; rungs labelled by measured median bearing '
                        'error; paired t over seeds against map_nopose_fix; bracket inferred '
                        'from the first monotone significant->n.s. transition, not max/min'),
              open(OUT, 'w'), indent=2)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
