#!/usr/bin/env python
"""Does the well-resolved pre-fit gain replicate on validation scenes nobody has looked at?

The gain of pre-fit pose at 30 cm of tracker noise on pedestrians with at least 200 LiDAR
returns (-3.47% against a stranger's pose, p = 0.010) came out of a stratification chosen
on the 5,171 validation scenes that also selected each run's best epoch. This scores the
same fifteen models (own pose, stranger's pose, no pose, five seeds) on exported validation
scenes that were never used for these models: not for training, not for selecting an epoch
and not for choosing the stratum. (Sec. IV-D scores other, clean-track models on them for
the standing-then-moving test.)

Fixed before any of the evaluations was run:
  population   every scored pedestrian with at least 200 mean LiDAR returns over the
               observed second, in the 21,985 unused scenes. 5,807 of them hold such a
               pedestrian (experiments/heldout_wellresolved_scenes.txt), and only those are
               evaluated, to keep the cost on a shared GPU down. Pedestrians below 200
               returns in these scenes are scored too, but they are not a sample of their
               stratum and are not reported.
  checkpoint   epoch 30, the only one kept, so no selection on any validation set
  statistic    relative difference of arm means per seed, paired t over the five seeds
  primary      own pose against a stranger's pose. Predicted sign: negative.
  secondary    the three claims the paper makes against no pose, each with the split that
               was chosen after the result on the original scenes and is now fixed:
               (a) own pose against no pose on the pedestrians that also have a second of
                   future (2.1% on the original scenes)
               (b) the same on the half with the larger tracker velocity error (4.0%),
                   the median cut inside the held-out population
               (c) the trend of (a) from the first to the fourth quartile of the tracker's
                   velocity error and of its position error (3.7 and 5.1 points)
               The secondary tests need the tracker's error and the manoeuvre covariates
               of the held-out pedestrians (tracker_state_error.py, highdyn_covariates.py)
               and are skipped while those files are missing.

The 1,142 original scenes with a well-resolved pedestrian are evaluated in the same pass
and reported beside it. On them the re-evaluated own-pose and no-pose models must return,
pedestrian by pedestrian, the minADE stored at epoch 30 during training, which checks the
staged poses, the chunked forward pass and the checkpoints together (up to a mode
selection that sits on its threshold, see the check in main). The stranger arm need not
agree: its donor permutation is drawn over the split it is given.

Reads output/waymo/<cfg>/HELDOUT_<tag>/eval/epoch_30/default/metrics_epoch_30.pkl
(run_heldout_eval.sh), experiments/val_agent_pointcount_25k.pkl and
val_agent_pointcount_heldout.pkl. Writes experiments/heldout_replication.json.

Usage: python tools/scripts/heldout_replication.py
"""
from __future__ import annotations

import json
import os
import pickle
import re

import numpy as np

from position_noise_subpopulation import LOCAL_OUT, METRICS, RECORDS, REPO, ci, load_prefit

OUT = os.path.join(REPO, 'experiments', 'heldout_replication.json')
ROOT = os.path.join(REPO, 'output', 'waymo')
EXP = os.path.join(REPO, 'experiments')
ARMS = {'own': ('cross_attn_pe_with_map_prefit_posnoise30_scale25k', 'PF'),
        'stranger': ('cross_attn_pe_with_map_shufflepose_prefit_posnoise30_scale25k', 'PF'),
        'nopose': ('no_pose_with_map_posnoise30_scale25k', 'PN')}
SEEDS = (101, 202, 303, 404, 505)
MIN_RETURNS = 200
USED = [os.path.join(REPO, 'data', 'waymo', f)
        for f in ('avo_eval_dev_scenes.txt', 'avo_eval_final_scenes.txt')]
# held-out covariates, and the files of the original scenes they are compared with
TRK = {'heldout': os.path.join(EXP, 'val_tracker_state_posnoise30_heldout.pkl'),
       'original': os.path.join(EXP, 'val_tracker_state_posnoise30.pkl')}
MAN = {'heldout': os.path.join(EXP, 'eval_agent_manoeuvre_heldout.pkl'),
       'original': os.path.join(EXP, 'eval_agent_manoeuvre.pkl')}


def read(path: str) -> dict:
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE'])
            for r in pickle.load(open(path, 'rb'))}


def load(arm: str, seed: int):
    cfg, pre = ARMS[arm]
    p = os.path.join(ROOT, f'mtr+pose_data_{cfg}', f'HELDOUT_{pre}_25k_{cfg}_lr25_s{seed}',
                     'eval', 'epoch_30', 'default', 'metrics_epoch_30.pkl')
    return read(p) if os.path.exists(p) else None


def stored_epoch30(arm: str, seed: int):
    """Per-pedestrian minADE written at epoch 30 while the run trained, original scenes."""
    if arm == 'own':
        try:
            return load_prefit('pose', 30, seed, epoch=30)
        except FileNotFoundError:
            return None
    cfg, pre = ARMS[arm]                       # no-pose runs kept their best epoch only
    p = os.path.join(METRICS, f'mtr+pose_data_{cfg}', f'{pre}_25k_{cfg}_lr25_s{seed}', 'eval',
                     'eval_with_train', 'metrics_epoch_30.pkl')
    return read(p) if os.path.exists(p) else None


def final_over_best(arm: str, seed: int) -> float:
    """Validation minADE at epoch 30 over the run's best epoch, in percent (all pedestrians,
    from the record train.py writes)."""
    cfg, pre = ARMS[arm]
    sub = os.path.join(f'mtr+pose_data_{cfg}', f'{pre}_25k_{cfg}_lr25_s{seed}', 'eval',
                       'eval_with_train', 'best_eval_record.txt')
    txt = next(open(os.path.join(root, sub)).read() for root in (RECORDS, LOCAL_OUT)
               if os.path.exists(os.path.join(root, sub)))
    ep = {int(a): float(b) for a, b in re.findall(r'^epoch_(\d+) minADE ([\d.]+)', txt, re.M)}
    best = int(re.findall(r'best_epoch_(\d+)', txt)[-1])
    assert ep[best] == min(ep.values()), (arm, seed)
    return (ep[30] / ep[best] - 1) * 100


def main() -> None:
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    seeds = [s for s in SEEDS if all(runs[a, s] is not None for a in ARMS)]
    # a failed evaluation must not turn the five-seed test into a four-seed one unnoticed
    assert len(seeds) == len(SEEDS), f'evaluations missing: complete seeds are {seeds}'
    common = sorted(set.intersection(*[set(runs[a, s]) for a in ARMS for s in seeds]))
    used = set()
    for f in USED:
        used |= set(open(f).read().split())
    pc = pickle.load(open(os.path.join(EXP, 'val_agent_pointcount_25k.pkl'), 'rb'))
    pc.update(pickle.load(open(os.path.join(EXP, 'val_agent_pointcount_heldout.pkl'), 'rb')))
    X = {k: np.array([v[a] for a in common]) for k, v in runs.items() if v is not None}
    npts = np.array([pc[k]['n_mean'] if k in pc else np.nan for k in common])
    old = np.array([k[0] in used for k in common])
    well = npts >= MIN_RETURNS

    def rel(a: str, b: str, mask: np.ndarray) -> np.ndarray:
        return np.array([(X[a, s][mask].mean() / X[b, s][mask].mean() - 1) * 100 for s in seeds])

    def row(mask: np.ndarray) -> dict:
        return dict(n_agents=int(mask.sum()),
                    n_scenes=len({k[0] for k, m in zip(common, mask) if m}),
                    nopose_minADE=float(np.mean([X['nopose', s][mask].mean() for s in seeds])),
                    content=ci(rel('own', 'stranger', mask)),
                    own_vs_nopose=ci(rel('own', 'nopose', mask)),
                    stranger_vs_nopose=ci(rel('stranger', 'nopose', mask)))

    def show(name: str, r: dict) -> None:
        c, n, st = r['content'], r['own_vs_nopose'], r['stranger_vs_nopose']
        print(f'  {name:<26} n={r["n_agents"]:>6}  own vs stranger {c["rel"]:+6.2f}% p={c["p"]:.3f} '
              f'[{c["lo"]:+.1f},{c["hi"]:+.1f}]   own vs no pose {n["rel"]:+6.2f}% p={n["p"]:.3f} '
              f'[{n["lo"]:+.1f},{n["hi"]:+.1f}]   stranger vs no pose {st["rel"]:+6.2f}% '
              f'p={st["p"]:.3f}')

    # The re-evaluation against what training stored, pedestrian by pedestrian. Only the
    # own-pose runs kept per-pedestrian metrics at epoch 30, so this arm must be compared
    # on every seed. The no-pose runs kept their best epoch alone and are compared if found.
    # Values agree to 1e-4 m across GPUs and batch sizes, with one kind of exception: the
    # six modes are picked from 64 by suppressing endpoints closer than 2.5 m, and an
    # endpoint pair that sits on that threshold flips with the last digit. Seed 202 has one
    # such pedestrian (493261795d80d465/7345, pair at 2.5000 m: 0.877 m here, 0.387 m
    # stored, and noise of 0.1 mm on the modes picks either half of the time). Such values
    # are counted and may be at most one in ten thousand.
    check = {}
    for arm in ('own', 'nopose'):
        worst, n, n_seeds, flipped = 0.0, 0, 0, []
        for s in seeds:
            ref = stored_epoch30(arm, s)
            both = [k for k in common if ref is not None and k in ref]
            if not both:
                continue
            d = np.abs(np.array([runs[arm, s][k] for k in both]) - np.array([ref[k] for k in both]))
            assert np.isfinite(d).all(), f'{arm} seed {s}: a minADE is not finite'
            flipped += [(s,) + both[i] + (float(d[i]),) for i in np.nonzero(d >= 1e-3)[0]]
            close = d[d < 1e-3]
            worst = max(worst, float(close.max()) if close.size else 0.0)
            n, n_seeds = n + len(both), n_seeds + 1
        check[arm] = dict(n_seeds=n_seeds, n_compared=n, largest_abs_difference_m=worst,
                          n_other_mode_selected=len(flipped), other_mode_selected=flipped)
        print(f'stored epoch 30 against this pass, {arm}: {n} values on {n_seeds} seeds, '
              f'largest difference {worst:.2e} m, {len(flipped)} with another mode selected')
        assert len(flipped) <= 1e-4 * n, f'{arm}: the re-evaluation does not reproduce training'
    assert check['own']['n_seeds'] == len(seeds) and check['own']['n_compared'] >= 5000 * len(seeds), \
        'the own-pose re-evaluation was not compared with the stored metrics on every seed'

    res = dict(_meta=dict(
        seeds=seeds, epoch=30, n_scored=len(common), min_returns=MIN_RETURNS,
        noise_cm=int(re.search(r'posnoise(\d+)', ARMS['own'][0]).group(1)),
        n_without_point_count=int(np.isnan(npts).sum()), reproduces_stored_epoch30=check,
        note='pre-fit poses, 30 cm tracker noise, epoch-30 checkpoints, pedestrians with at '
             'least 200 mean returns. Relative difference of arm means per seed in percent, '
             'paired t over seeds, negative favours the first arm. heldout: validation '
             'scenes never used before. original: scenes of the paper, same pass. Only '
             'scenes that hold a well-resolved pedestrian were evaluated.'))
    # What the best-epoch rule of the paper's tables is worth to each arm. The held-out
    # pass has the final epoch only, and the no-pose arm keeps more of its best epoch.
    res['final_epoch_over_best_epoch_pct'] = {
        a: ci(np.array([final_over_best(a, s) for s in seeds])) for a in ARMS}
    print('validation minADE, final epoch over best epoch: ' + ', '.join(
        f'{a} {c["rel"]:+.2f}%' for a, c in res['final_epoch_over_best_epoch_pct'].items()))
    for pop, keep in (('heldout', ~old & well), ('original', old & well)):
        cell = dict(primary=row(keep))
        print(f'\n##### {pop}: seeds {seeds}')
        show('at least 200 returns', cell['primary'])
        if os.path.exists(TRK[pop]) and os.path.exists(MAN[pop]):
            trk, man = pickle.load(open(TRK[pop], 'rb')), pickle.load(open(MAN[pop], 'rb'))
            has = np.array([k in trk and k in man for k in common])
            sec = keep & has                            # a second of future, tracker state known
            vel = np.array([trk[k]['vel_err'] if k in trk else np.nan for k in common])
            pos = np.array([trk[k]['pos_err'] if k in trk else np.nan for k in common])
            cell['with_a_second_of_future'] = row(sec)
            show('  with a second of future', cell['with_a_second_of_future'])
            cell['larger_velocity_error_half'] = row(sec & (vel > np.median(vel[sec])))
            show('  larger velocity error', cell['larger_velocity_error_half'])
            for name, v in (('velocity_error', vel), ('position_error', pos)):
                q = np.percentile(v[sec], [25, 50, 75])
                quart = [row(sec & m) for m in (v <= q[0], (v > q[0]) & (v <= q[1]),
                                                (v > q[1]) & (v <= q[2]), v > q[2])]
                trend = {key: ci(np.array(quart[3][key]['per_seed'])
                                 - np.array(quart[0][key]['per_seed']))
                         for key in ('own_vs_nopose', 'content')}
                cell[f'quartiles_of_tracker_{name}'] = dict(
                    cuts=[float(x) for x in q], quartiles=quart, Q4_minus_Q1=trend)
                t = trend['own_vs_nopose']
                print(f'  {name}: own vs no pose by quartile '
                      + ' '.join(f'{r["own_vs_nopose"]["rel"]:+.1f}' for r in quart)
                      + f'   Q4 minus Q1 {t["rel"]:+.2f} points p={t["p"]:.3f}')
        else:
            print('  (secondary tests skipped: tracker state or manoeuvre file missing)')
        res[pop] = cell
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
