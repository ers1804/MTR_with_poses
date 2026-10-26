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
3.9 deg, SMPL facing 12.3 deg. Stationary subset, where facing is the cue velocity cannot supply:
box 24.7 deg, SMPL 41.2 deg. Mixing the two is what made Sec. IV-B compare 44.2 against a floor
calibrated on a different quantity (see research-log 2026-10-23).

Three ladders are scored here. Two deliver the SAME corrupted channel through different
PATHWAYS: 'late' through the cross-attention pose branch (the published ladder) and 'early'
through extra per-timestep agent feature channels. The delivered tensors are bit-identical at
a given rung -- early obj_trajs[..., 29:35] equals late obj_poses[..., :6] -- so the rungs
carry the same measured-error labels.

The third, '25k', is the late pathway again at a different SCALE. It exists because the floor
was measured on 579 scenes while the branch-capacity tax it is weighed against was measured at
25k, and a bracket at one scale compared against a tax at another is the weakest joint in the
argument. 579 is a cheap iteration rung (~52 min a run), not a data limit. The 25k arm also has
a far tighter denominator -- no-pose sd 0.0027 over five seeds against a 0.4428-0.4954 spread
at 579 -- so n=3 there should resolve more than n=5 did here.

Usage: python tools/scripts/detection_floor.py [late|early|25k]
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


def _load_rungs() -> dict:
    """rung -> measured median |bearing error| of the delivered channel.

    Read from the artifact oracle_rung_calibration.py writes, so the labels the paper
    quotes cannot drift from the channel the branch was fed. The values that stood here
    until 2026-10-26 (5.5, 10.2, 14.6, 27.2, 37.9, 45.0, 57.4) came from a single ad hoc
    smoke test and were wrong by up to 2.7 deg: the configured sigmas are exactly
    rung_key / 0.6745, so a rung delivers its key by construction, and measuring the
    channel over the full validation population (n=11,328, against roughly 130 draws
    before) confirms it. The one that mattered was the bracket's upper edge, 14.6 -> 15.1.
    """
    cal = os.path.join(REPO, 'experiments', 'oracle_rung_calibration.json')
    if not os.path.exists(cal):
        raise SystemExit(f'missing {cal}; run tools/scripts/oracle_rung_calibration.py')
    d = json.load(open(cal))
    # +3 s is the paper's reference target; the +8 s channel agrees to within 0.3 deg
    return {k: round(v['+3s']['median'], 1) for k, v in sorted(d.items())
            if v.get('+3s', {}).get('n')}


_RUNGS = _load_rungs()
LADDERS = {
    'late': dict(cfg='mtr+pose_data_cross_attn_pe_with_map_oracledir{}_fixdata',
                 tag='MS_oracledir{key}_s{seed}', rungs=_RUNGS,
                 baseline_cell='map_nopose_fix', watch=True,
                 out='detection_floor.json',
                 label='cross-attention pose branch, 579 scenes (published ladder)'),
    'early': dict(cfg='mtr+pose_data_early_oracledir{}_fixdata',
                  tag='MS_earlyoracle{key}_s{seed}',
                  rungs={k: _RUNGS[k] for k in ('00', '05', '10')},
                  baseline_cell='map_nopose_fix', watch=True,
                  out='detection_floor_early.json',
                  label='early fusion, per-timestep agent channels, 579 scenes'),
    '25k': dict(cfg='mtr+pose_data_cross_attn_pe_with_map_oracledir{}_scale25k',
                tag='MS_oracledir{key}_25k_s{seed}',
                rungs={k: _RUNGS[k] for k in ('00', '05', '10', '15')},
                # no watch seed: the 2 SD baseline outlier is a property of the 579 cell.
                # The 25k no-pose cell spans 0.3683-0.3752 over five seeds, so there is no
                # draw to resample and the single denominator is the only one.
                baseline_cell='scale25k_nopose', watch=False,
                out='detection_floor_25k.json',
                label='cross-attention pose branch, 25k scenes'),
}
def _units() -> dict:
    """Where the real channels sit, read from facing_error_ladder_units.py's output.

    These were four hardcoded literals, and one of them (box heading, 3.884 deg) was
    carried as 3.8 in two places while the paper said 3.9. Reading the artifact removes
    that class of mismatch. The dependency runs one way only: this script's statistics
    do not use these numbers, it only reports where the channels fall relative to the
    bracket it computed, so there is no circularity with that script reading the bracket
    back out of detection_floor.json.
    """
    f = os.path.join(REPO, 'experiments', 'facing_error_ladder_units.json')
    if not os.path.exists(f):
        return {}
    u = json.load(open(f))
    t = u['targets']
    g = lambda k, d: t.get(k, {}).get(d)                                 # noqa: E731
    return dict(box=g('+3 s displacement bearing', 'box_med'),
                smpl=g('+3 s displacement bearing', 'smpl_med'),
                box_stat=g('+3 s / stationary (<0.3 m/s)', 'box_med'),
                smpl_stat=g('+3 s / stationary (<0.3 m/s)', 'smpl_med'),
                smpl_above_hi=g('+3 s displacement bearing', 'smpl_frac_above_hi'))


_U = _units()
# ladder units (pooled over all predicted pedestrians), from facing_error_ladder_units.py
BOX_HEADING_DEG = _U.get('box') or 3.9
SMPL_FACING_DEG = _U.get('smpl') or 12.3
# the same two channels on the stationary subset, the population pose was meant to serve
BOX_STATIONARY_DEG = _U.get('box_stat') or 24.7
SMPL_STATIONARY_DEG = _U.get('smpl_stat') or 41.2
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


def fit_decay(rows) -> dict:
    """Fit Delta_minADE(theta) = -A * exp(-theta / tau) to the ladder.

    The paper's Eq. (1) and its "amplitudes of 12-20%, half-lives of 3.3-8.1 deg" came
    from a fit carried as three hardcoded parameter pairs in another script. Refitting
    here keeps the law tied to the rungs it describes, which matters because the rung
    labels themselves were corrected on 2026-10-26 and a hardcoded fit would have gone
    on describing the old ones.

    A pooled fit over rung means, plus one fit per seed so the paper can report the
    spread rather than a single curve. tau is the e-folding scale; the half-life the
    paper quotes is tau * ln 2.
    """
    from scipy.optimize import curve_fit

    def model(th, A, tau):
        return -A * np.exp(-th / np.maximum(tau, 1e-6))

    def one(th, y):
        if len(th) < 3:
            return None
        try:
            (A, tau), _ = curve_fit(model, np.asarray(th, float), np.asarray(y, float),
                                    p0=(16.0, 8.0), maxfev=20000,
                                    bounds=([0.0, 0.1], [100.0, 200.0]))
        except Exception:
            return None
        return dict(amplitude_pct=float(A), tau_deg=float(tau),
                    half_life_deg=float(tau * np.log(2.0)))

    order = sorted(rows, key=lambda r: r['median_err_deg'])
    pooled = one([r['median_err_deg'] for r in order], [r['rel'] for r in order])
    per_seed = {}
    for sd in SEEDS:
        th = [r['median_err_deg'] for r in order if sd in r['seeds']]
        y = [r['per_seed_rel'][r['seeds'].index(sd)] for r in order if sd in r['seeds']]
        f = one(th, y)
        if f:
            per_seed[str(sd)] = f
    out = dict(pooled=pooled, per_seed=per_seed)
    if per_seed:
        out['amplitude_range_pct'] = [min(f['amplitude_pct'] for f in per_seed.values()),
                                      max(f['amplitude_pct'] for f in per_seed.values())]
        out['half_life_range_deg'] = [min(f['half_life_deg'] for f in per_seed.values()),
                                      max(f['half_life_deg'] for f in per_seed.values())]
    return out


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
    B = {int(k): v['best_minADE']
         for k, v in base['cells'][L['baseline_cell']]['seeds'].items()
         if v.get('best_minADE') is not None}
    print(f'baseline cell: {L["baseline_cell"]}  n={len(B)}  '
          f'mean {np.mean(list(B.values())):.4f}  '
          f'spread {min(B.values()):.4f}-{max(B.values()):.4f}\n')

    # denominators to score under: the published one, it minus the outlier seed, and --
    # once it exists -- a second sample of that seed's baseline
    resample = (_read_best(os.path.join(REPO, 'output', 'waymo', BASELINE_CFG, RESAMPLE_TAG),
                           RESAMPLE_TAG) if L['watch'] else None)
    PRIMARY = ['all seeds']
    variants = [('all seeds', B, SEEDS)]
    if L['watch'] and WATCH_SEED in B:
        variants.append((f'without s{WATCH_SEED}', B,
                         tuple(s for s in SEEDS if s != WATCH_SEED)))
    if resample is not None:
        both = (B[WATCH_SEED] + resample) / 2
        print(f's{WATCH_SEED} baseline: original {B[WATCH_SEED]:.4f}, resample {resample:.4f}, '
              f'mean {both:.4f}')
        print(f'  the two draws differ by {abs(B[WATCH_SEED]-resample):.4f}; the other four '
              f'baselines span {max(o for s,o in B.items() if s!=WATCH_SEED)-min(o for s,o in B.items() if s!=WATCH_SEED):.4f}.')
        print('  Both draws are legitimate samples, so the MEAN is the primary denominator and')
        print('  the single-draw readings below are the sensitivity analysis.\n')
        variants.append((f's{WATCH_SEED} resampled', {**B, WATCH_SEED: resample}, SEEDS))
        variants.append((f's{WATCH_SEED} both draws', {**B, WATCH_SEED: both}, SEEDS))
        PRIMARY[0] = f's{WATCH_SEED} both draws'

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

    Bprim = dict(next(v[1] for v in variants if v[0] == PRIMARY[0]))
    rows = build(Bprim, SEEDS)
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
        if L['watch'] and WATCH_SEED in r['seeds'] and r['n'] >= 4:
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

    primary = brackets[PRIMARY[0]]
    print(f'\n  primary denominator: {PRIMARY[0]}')
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
        if _U.get('smpl_above_hi') is not None:
            print(f"     is a heavy tail: {100 * _U['smpl_above_hi']:.1f}% of agents sit "
                  "above the bracket's upper edge.")

    decay = fit_decay(rows)
    if decay.get('pooled'):
        d = decay['pooled']
        print(f"\ndecay law fitted to this ladder: "
              f"{-d['amplitude_pct']:.1f}% * exp(-theta / {d['tau_deg']:.1f} deg)"
              f"   (half-life {d['half_life_deg']:.1f} deg)")
        if decay.get('amplitude_range_pct'):
            a, b = decay['amplitude_range_pct']; c, e = decay['half_life_range_deg']
            print(f'  per-seed fits: amplitudes {a:.1f}-{b:.1f}%, '
                  f'half-lives {c:.1f}-{e:.1f} deg (n={len(decay["per_seed"])} seeds)')

    json.dump(dict(ladder=which, pathway=L['label'], rungs=rows, decay_fit=decay,
                   floor_bracket_deg=[primary['lo'], primary['hi']],
                   floor_brackets=brackets, bracket_edge_stable=(len(edges) <= 1),
                   primary_denominator=PRIMARY[0],
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
