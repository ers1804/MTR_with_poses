#!/usr/bin/env python
"""Onboard-position ladder: does pose pay once the observed track is a causal tracker's?

The heading ladder (heading_noise_ladder.py) answers the offboard-label objection for the
box heading only. WOMD positions are offboard-refined too, and velocity is their exact
backward difference (label_causality.py), so the whole observed history is better than
an onboard tracker would report. POSITION_NOISE_M replaces it with what a causal tracker
would: per-frame detection noise of that std on x and y (deterministic per scene and
object), then a forward-only constant-velocity Kalman filter that writes back positions
and velocities. Heading and the future targets are untouched, so this ladder is separate
from the heading one (test_heading_noise_roundtrip.py --pos checks both, and writes the
tracker error at prediction time to experiments/position_noise_tracker_error_*.json).

The clean rung is the heading ladder's sigma=0 cell: same arms, seeds and recipe. The
noisy rungs add seeds 404/505 (n=3 put the project's first content effect at 30 cm, and an
earlier 3-seed win did not survive n=5), so each rung is scored on the seeds for which all
three arms are done, and the degradation on the seeds it shares with the clean rung.

Writes experiments/position_noise_ladder.json.

Usage: python tools/scripts/position_noise_ladder.py [--pull]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_floor import _read_best  # noqa: E402
from heading_noise_ladder import ARMS, RECORDS, REPO, paired, pull  # noqa: E402
from heading_noise_ladder import best as heading_best  # noqa: E402

OUT = os.path.join(REPO, 'experiments', 'position_noise_ladder.json')
RUNGS = (0, 10, 30)   # cm of per-frame detection noise
FIXED_EPOCHS = (26, 28, 30)   # the best-epoch rule, checked against fixed late epochs
SEEDS = {0: (101, 202, 303), 10: (101, 202, 303, 404, 505), 30: (101, 202, 303, 404, 505)}


def best(arm: str, rung: int, seed: int) -> float | None:
    if rung == 0:
        return heading_best(arm, 0, seed)
    cfg = f'{ARMS[arm]}_posnoise{rung:02d}_scale25k'
    tag = f'PN_25k_{cfg}_lr25_s{seed}'
    return _read_best(os.path.join(RECORDS, f'mtr+pose_data_{cfg}', tag), tag)


def at_epoch(arm: str, rung: int, seed: int, epoch: int) -> float:
    cfg = (f'{ARMS[arm]}_headnoise00_scale25k' if rung == 0
           else f'{ARMS[arm]}_posnoise{rung:02d}_scale25k')
    tag = (f'HN_25k_{cfg}_lr25_s{seed}' if rung == 0 else f'PN_25k_{cfg}_lr25_s{seed}')
    rec = open(os.path.join(RECORDS, f'mtr+pose_data_{cfg}', tag, 'eval', 'eval_with_train',
                            'best_eval_record.txt')).read()
    return float(re.search(rf'^epoch_{epoch} minADE ([\d.]+)', rec, re.M).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true', help='refresh the records from Helma first')
    args = ap.parse_args()
    if args.pull:
        pull('mtr+pose_data_*posnoise*/PN_25k_*/DONE '
             'mtr+pose_data_*posnoise*/PN_25k_*/eval/eval_with_train/best_eval_record.txt')

    raw = {(a, r, s): best(a, r, s) for r in RUNGS for s in SEEDS[r] for a in ARMS}
    # a seed counts at a rung once all three arms have finished it, so every test is paired
    done = {r: [s for s in SEEDS[r] if all(raw[a, r, s] is not None for a in ARMS)]
            for r in RUNGS}
    v = {(a, r): [raw[a, r, s] for s in done[r]] for a in ARMS for r in RUNGS}
    for r in RUNGS:
        if len(done[r]) < len(SEEDS[r]):
            print(f'{r} cm: {len(done[r])}/{len(SEEDS[r])} seeds complete, scoring those')

    res = dict(
        lr=2.5e-5, seeds={f'pn{r:02d}': done[r] for r in RUNGS},
        note='best-epoch minADE; paired t over the seeds complete in all three arms; '
             'negative favours the first-named arm; rung 0 is the heading ladder sigma=0 cell',
        cells={f'{a}_pn{r:02d}': dict(seeds=done[r], per_seed=v[a, r],
                                      mean=float(np.mean(v[a, r])) if v[a, r] else None)
               for a in ARMS for r in RUNGS},
        rungs=[],
    )
    for r in RUNGS:
        row = dict(noise_cm=r, n=len(done[r]),
                   content=paired(v['pose', r], v['shuffled', r]),
                   pose_vs_nopose=paired(v['pose', r], v['nopose', r]),
                   shuffled_vs_nopose=paired(v['shuffled', r], v['nopose', r]))
        if r:
            common = [s for s in done[r] if s in done[0]]
            row['degradation_seeds'] = common
            err = json.load(open(os.path.join(
                REPO, 'experiments', f'position_noise_tracker_error_{r / 100:g}.json')))
            row['tracker_error'] = err
            row['degradation'] = paired([raw['nopose', r, s] for s in common],
                                        [raw['nopose', 0, s] for s in common])
            if row['content'] and row['degradation']:
                row['max_recovered_share'] = (max(0.0, -row['content']['lo'])
                                              / row['degradation']['rel'])
        row['fixed_epoch'] = {
            str(e): dict(content=paired([at_epoch('pose', r, s, e) for s in done[r]],
                                        [at_epoch('shuffled', r, s, e) for s in done[r]]),
                         pose_vs_nopose=paired([at_epoch('pose', r, s, e) for s in done[r]],
                                               [at_epoch('nopose', r, s, e) for s in done[r]]))
            for e in FIXED_EPOCHS}
        res['rungs'].append(row)
    json.dump(res, open(OUT, 'w'), indent=1)

    fmt = lambda c: (f'{c["rel"]:+6.2f}%  p={c["p"]:.2f}  [{c["lo"]:+.1f},{c["hi"]:+.1f}]'
                     if c else 'n/a')
    print(f'\n{"noise":>5} {"nopose":>7} {"pose":>7} {"shuf":>7}   '
          f'{"content (pose vs shuf)":<32} {"pose vs nopose":<32} degradation')
    for row in res['rungs']:
        r = row['noise_cm']
        m = [res['cells'][f'{a}_pn{r:02d}']['mean'] for a in ARMS]
        print(f'{r:>3}cm n={row["n"]} ' + ' '.join(f'{x:7.4f}' if x else '    n/a' for x in m)
              + f'   {fmt(row["content"]):<32} {fmt(row["pose_vs_nopose"]):<32} '
              + fmt(row.get('degradation'))
              + (f'   cap {row["max_recovered_share"]:.0%}' if 'max_recovered_share' in row else ''))
    for row in res['rungs']:
        print(f'{row["noise_cm"]:>3}cm fixed epochs: ' + '   '.join(
            f'{e}: content {x["content"]["rel"]:+.2f}% p={x["content"]["p"]:.3f}'
            for e, x in row['fixed_epoch'].items()))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
