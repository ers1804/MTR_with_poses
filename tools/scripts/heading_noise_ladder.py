#!/usr/bin/env python
"""Onboard-heading ladder: does pose pay once the box heading is worse than WOMD's?

Reviewer objection (Flohr, 2026-11-04): WOMD boxes come from an offboard auto-labeller
that refines each box with the whole track, future frames included, so the heading the
baseline consumes is better than an onboard tracker would supply. Pose might pay only
where that heading is worse. ZERO_HEADING_CHANNELS cannot test this: it removes the
sin/cos channel but leaves the agent-centric frame, and the intention anchors defined
in it, aligned to the clean heading.

HEADING_NOISE_DEG corrupts the heading once, where center_objects is assembled, with a
fixed per-agent error N(0, sigma) (deterministic per scene and object), so the frame,
the anchors, the map and the pose root all inherit it (test_heading_noise_roundtrip.py
checks that the eval path undoes exactly that rotation). In the pose arm the root is
rotated into the same corrupted frame, so it states the correction the model would need;
in the shuffled arm it is another pedestrian's body and states nothing.

Runs on Helma (slurm/60_heading_noise.sh). --pull copies each run's best_eval_record.txt
and DONE marker into experiments/heading_noise_records/, laid out as output/waymo is, so
the analysis reruns without the cluster and scores only finished runs.

Writes experiments/heading_noise_ladder.json.

Usage: python tools/scripts/heading_noise_ladder.py [--pull]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_floor import _read_best, stats_for  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RECORDS = os.path.join(REPO, 'experiments', 'heading_noise_records')
OUT = os.path.join(REPO, 'experiments', 'heading_noise_ladder.json')
REMOTE = '/hnvme/workspace/v103fe12-mtr_smpl/MTR_with_poses/output/waymo'

SIGMAS = (0, 15, 30)
SEEDS = (101, 202, 303)
ARMS = {'nopose': 'no_pose_with_map',
        'pose': 'cross_attn_pe_with_map',
        'shuffled': 'cross_attn_pe_with_map_shufflepose'}
# the same arms at sigma=0 on the workstation, five seeds; the cluster must reproduce them
WORKSTATION = {'nopose': 'scale25k_nopose', 'pose': 'scale25k_xattn_pe',
               'shuffled': 'scale25k_shufflepose'}
LR_CHECK_SIGMA = 30


def pull(files: str = ('mtr+pose_data_*headnoise*/HN_25k_*/DONE '
                        'mtr+pose_data_*headnoise*/HN_25k_*/eval/eval_with_train/best_eval_record.txt')
         ) -> None:
    os.makedirs(RECORDS, exist_ok=True)
    ssh = subprocess.Popen(['ssh', '-F', os.path.join(REPO, 'slurm', 'ssh_config'), 'helma',
                            f'cd {REMOTE} && tar cf - {files}'], stdout=subprocess.PIPE)
    subprocess.run(['tar', 'xf', '-', '-C', RECORDS], stdin=ssh.stdout, check=True)
    if ssh.wait() != 0:
        sys.exit('pull failed')


def best(arm: str, sigma: int, seed: int, lrtag: str = '25') -> float | None:
    cfg = f'{ARMS[arm]}_headnoise{sigma:02d}_scale25k'
    tag = f'HN_25k_{cfg}_lr{lrtag}_s{seed}'
    return _read_best(os.path.join(RECORDS, f'mtr+pose_data_{cfg}', tag), tag)


def paired(arm: list, base: list) -> dict | None:
    """Per-seed relative difference of arm against base, in percent; negative favours arm."""
    if None in arm or None in base:
        return None
    return stats_for((np.array(arm) / np.array(base) - 1.0) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', action='store_true', help='refresh the records from Helma first')
    args = ap.parse_args()
    if args.pull:
        pull()

    v = {(a, s): [best(a, s, seed) for seed in SEEDS] for a in ARMS for s in SIGMAS}
    lr5 = [best('nopose', LR_CHECK_SIGMA, seed, lrtag='5') for seed in SEEDS]
    missing = [f'{a} sigma={s}' for (a, s), x in v.items() if None in x]
    if missing:
        print('incomplete, not scoring partial cells:', ', '.join(missing))

    res = dict(
        seeds=list(SEEDS), lr=2.5e-5,
        note='best-epoch minADE; paired t over seeds; negative favours the first-named arm',
        cells={f'{a}_s{s:02d}': dict(per_seed=x, mean=(float(np.mean(x)) if None not in x
                                                       else None))
               for (a, s), x in v.items()},
        rungs=[],
    )
    for s in SIGMAS:
        r = dict(sigma_deg=s,
                 # the error ADDED to the label; median |N(0, sigma)| = 0.6745 sigma
                 median_added_err_deg=round(0.6745 * s, 1),
                 content=paired(v['pose', s], v['shuffled', s]),
                 pose_vs_nopose=paired(v['pose', s], v['nopose', s]),
                 shuffled_vs_nopose=paired(v['shuffled', s], v['nopose', s]))
        if s:
            r['degradation'] = paired(v['nopose', s], v['nopose', 0])
            # the most of that degradation the content effect could be hiding: the
            # content CI's favourable edge as a share of what the corruption cost
            if r['content'] and r['degradation']:
                r['max_recovered_share'] = max(0.0, -r['content']['lo']) / r['degradation']['rel']
        res['rungs'].append(r)

    res['lr_check'] = dict(sigma_deg=LR_CHECK_SIGMA, lr=5e-5, per_seed=lr5,
                           vs_default=paired(lr5, v['nopose', LR_CHECK_SIGMA]))

    ms = json.load(open(os.path.join(REPO, 'experiments', 'multiseed_analysis.json')))
    res['reproduction'] = {}
    for a, cell in WORKSTATION.items():
        ref = ms['cells'][cell]['agg']['best_minADE']['mean']
        here = res['cells'][f'{a}_s00']['mean']
        res['reproduction'][a] = dict(workstation=ref, cluster=here,
                                      rel=(here / ref - 1) * 100 if here else None)

    json.dump(res, open(OUT, 'w'), indent=1)

    fmt = lambda c: (f'{c["rel"]:+6.2f}%  p={c["p"]:.2f}  [{c["lo"]:+.1f},{c["hi"]:+.1f}]'
                     if c else 'n/a')
    print(f'\n{"sigma":>5} {"added":>6} {"nopose":>7} {"pose":>7} {"shuf":>7}   '
          f'{"content (pose vs shuf)":<32} {"pose vs nopose":<32} degradation')
    for r in res['rungs']:
        s = r['sigma_deg']
        m = [res['cells'][f'{a}_s{s:02d}']['mean'] for a in ARMS]
        print(f'{s:>5} {r["median_added_err_deg"]:>5.1f}d '
              + ' '.join(f'{x:7.4f}' if x else '    n/a' for x in m)
              + f'   {fmt(r["content"]):<32} {fmt(r["pose_vs_nopose"]):<32} '
              + fmt(r.get('degradation')))
    print(f'\nLR check at sigma={LR_CHECK_SIGMA}: 5e-5 vs 2.5e-5  {fmt(res["lr_check"]["vs_default"])}')
    print('sigma=0 vs workstation: ' + ', '.join(
        f'{a} {x["rel"]:+.2f}%' for a, x in res['reproduction'].items() if x['rel'] is not None))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
