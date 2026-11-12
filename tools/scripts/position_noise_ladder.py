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
from detection_floor import _read_best, stats_for  # noqa: E402
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


# --- pre-fit control ------------------------------------------------------------------
# The poses above come out of a motion-prior fit whose loss matches the gait to the WOMD
# box track (Waymo-3DSkelMo, NeMF latent_optimization, lambda_trans). That track is the
# offboard label the simulated tracker replaces, so the fitted body carries the clean
# speed the tracker lost and the content effect cannot be read as body information.
# The PF runs repeat the pose and shuffled arms on the per-frame LiDAR-HMR output, which
# never saw the track (export_prefit_poses.py; test_prefit_dropin.py shows nothing but
# the pose tensor differs). The no-pose arm reads no pose, so it is shared.
# 30 cm ran on Helma (records pulled next to the PN ones). 10 cm runs on the workstation
# (run_prefit_local.sh) and is read from output/waymo once a run has its DONE marker.
PREFIT_RUNGS = (10, 30)
LOCAL_OUT = os.path.join(REPO, 'output', 'waymo')


def _prefit_dir(arm: str, rung: int, seed: int) -> str | None:
    cfg = f'{ARMS[arm]}_prefit_posnoise{rung:02d}_scale25k'
    tag = f'PF_25k_{cfg}_lr25_s{seed}'
    for root in (RECORDS, LOCAL_OUT):
        d = os.path.join(root, f'mtr+pose_data_{cfg}', tag)
        if os.path.exists(os.path.join(d, 'DONE')):   # a running job is never scored
            return d
    return None


def best_prefit(arm: str, rung: int, seed: int) -> float | None:
    d = _prefit_dir(arm, rung, seed)
    return _read_best(d, os.path.basename(d)) if d else None


def at_epoch_prefit(arm: str, rung: int, seed: int, epoch: int) -> float:
    rec = open(os.path.join(_prefit_dir(arm, rung, seed), 'eval', 'eval_with_train',
                            'best_eval_record.txt')).read()
    return float(re.search(rf'^epoch_{epoch} minADE ([\d.]+)', rec, re.M).group(1))


def prefit_rows(raw: dict, done: dict, rungs_out: list) -> list:
    """Score the pre-fit arms against the final-pose arms, seed by seed."""
    rows = []
    for r in PREFIT_RUNGS:
        pf = {(a, s): best_prefit(a, r, s) for a in ('pose', 'shuffled') for s in SEEDS[r]}
        seeds = [s for s in done[r] if pf['pose', s] is not None and pf['shuffled', s] is not None]
        half = [s for s in SEEDS[r] if (pf['pose', s] is None) != (pf['shuffled', s] is None)]
        if not seeds:
            if half:
                print(f'pre-fit {r} cm: no complete pair yet (one arm done for seeds {half})')
            continue
        P, S = [pf['pose', s] for s in seeds], [pf['shuffled', s] for s in seeds]
        fin = {a: [raw[a, r, s] for s in seeds] for a in ARMS}
        row = dict(noise_cm=r, n=len(seeds), seeds=seeds,
                   per_seed=dict(prefit_pose=P, prefit_shuffled=S, final_pose=fin['pose'],
                                 final_shuffled=fin['shuffled'], nopose=fin['nopose']))
        if len(seeds) >= 2:
            c_pre, c_fin = paired(P, S), paired(fin['pose'], fin['shuffled'])
            d = np.array(c_pre['per_seed_rel']) - np.array(c_fin['per_seed_rel'])
            row.update(
                content=c_pre, content_final_same_seeds=c_fin,
                # positive: the fitted pose bought more than the pre-fit pose does
                content_prefit_minus_final=stats_for(d),
                surviving_share=c_pre['rel'] / c_fin['rel'],
                pose_vs_nopose=paired(P, fin['nopose']),
                shuffled_vs_nopose=paired(S, fin['nopose']),
                prefit_pose_vs_final_pose=paired(P, fin['pose']),
                prefit_shuffled_vs_final_shuffled=paired(S, fin['shuffled']),
                fixed_epoch={str(e): dict(
                    content=paired([at_epoch_prefit('pose', r, s, e) for s in seeds],
                                   [at_epoch_prefit('shuffled', r, s, e) for s in seeds]),
                    pose_vs_nopose=paired([at_epoch_prefit('pose', r, s, e) for s in seeds],
                                          [at_epoch('nopose', r, s, e) for s in seeds]))
                    for e in FIXED_EPOCHS})
            deg = next(x for x in rungs_out if x['noise_cm'] == r).get('degradation')
            if deg:
                row['max_recovered_share'] = max(0.0, -c_pre['lo']) / deg['rel']
        else:
            row['content_per_seed_rel'] = [(p / s - 1.0) * 100.0 for p, s in zip(P, S)]
            row['content_final_per_seed_rel'] = [
                (p / s - 1.0) * 100.0 for p, s in zip(fin['pose'], fin['shuffled'])]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true', help='refresh the records from Helma first')
    args = ap.parse_args()
    if args.pull:
        # NOTE pull() opens its own ssh. Automation should fetch through slurm/helma_ssh.sh
        # instead and run this without --pull (same tar command, see heading_noise_ladder).
        pull('mtr+pose_data_*posnoise*/P[NF]_25k_*/DONE '
             'mtr+pose_data_*posnoise*/P[NF]_25k_*/eval/eval_with_train/best_eval_record.txt')

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
    res['prefit'] = prefit_rows(raw, done, res['rungs'])
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

    print('\npre-fit poses (per-frame LiDAR-HMR, never fitted to the track):')
    for row in res['prefit']:
        r, n = row['noise_cm'], row['n']
        if 'content' not in row:
            print(f'{r:>3}cm n={n} seeds {row["seeds"]}: content per seed '
                  f'{["%+.2f" % x for x in row["content_per_seed_rel"]]} '
                  f'(final pose, same seeds: '
                  f'{["%+.2f" % x for x in row["content_final_per_seed_rel"]]})')
            continue
        print(f'{r:>3}cm n={n} seeds {row["seeds"]}')
        for k, label in (('content', 'content, pre-fit (pose vs shuffled)'),
                         ('content_final_same_seeds', 'content, final fit, same seeds'),
                         ('content_prefit_minus_final', 'pre-fit minus final content (pp)'),
                         ('pose_vs_nopose', 'pre-fit pose vs no-pose'),
                         ('shuffled_vs_nopose', 'pre-fit shuffled vs no-pose'),
                         ('prefit_pose_vs_final_pose', 'pre-fit pose vs final pose'),
                         ('prefit_shuffled_vs_final_shuffled', 'pre-fit shuffled vs final shuffled')):
            c = row[k]
            print(f'    {label:<38} {c["rel"]:+6.2f}  p={c["p"]:.4f}  [{c["lo"]:+.2f},{c["hi"]:+.2f}]'
                  f'  per seed {" ".join("%+.2f" % x for x in c["per_seed_rel"])}')
        print(f'    surviving share of the final-fit content effect: {row["surviving_share"]:.0%}'
              + (f'   cap on recovered tracker loss {row["max_recovered_share"]:.0%}'
                 if 'max_recovered_share' in row else ''))
        print('    fixed epochs: ' + '   '.join(
            f'{e}: content {x["content"]["rel"]:+.2f}% p={x["content"]["p"]:.3f}'
            for e, x in row['fixed_epoch'].items()))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
