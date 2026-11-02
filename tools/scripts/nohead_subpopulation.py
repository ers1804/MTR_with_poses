#!/usr/bin/env python
"""Does pose pay once the heading channel is gone -- in the place the mechanism says it should?

The heading ablation measured on OVERALL minADE came back uninformative (content -0.51%,
CI +/-13%): zeroing the heading channels roughly doubles seed variance, and the cue in question
applies to a small subpopulation that the mean averages away. Both problems are addressed by
changing the endpoint rather than the seed budget.

The mechanism localises body facing to pedestrians who are STATIONARY NOW and step off shortly
after. If pose substitutes for the tracker heading at all, it is there. This script scores the
content test (own pose vs another pedestrian's) inside that subpopulation, with and without the
heading channel, from the per-agent metrics already on disk. No training, no GPU.

Each run's per-epoch metrics_epoch_N.pkl holds one record per validation agent, so the best epoch
named in best_eval_record.txt can be re-scored on any subset without re-running inference.

Usage: python tools/scripts/nohead_subpopulation.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')
DYN = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
# per-agent turn angles, written by facing_error_ladder_units.py
TURN = os.path.join(REPO, 'experiments', 'val_agent_turn.pkl')
OUT_JSON = os.path.join(REPO, 'experiments', 'nohead_subpopulation.json')
# Seeds are discovered from disk rather than fixed, so the extension to n=8 needs no edit.
SEEDS = tuple(int(x) for x in os.environ.get(
    'SEEDS', '101,202,303,404,505,606,707,808').split(','))

CELLS = {   # cell -> (config dir, run-tag template)
    'nohead_nopose': ('mtr+pose_data_no_pose_with_map_nohead_fixdata', 'MS_nohead_nopose_s{}'),
    'nohead_pose':   ('mtr+pose_data_cross_attn_pe_with_map_nohead_fixdata', 'MS_nohead_pose_s{}'),
    'nohead_shuf':   ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_nohead_fixdata',
                      'MS_nohead_shuf_s{}'),
    'head_nopose':   ('mtr+pose_data_no_pose_with_map_fixdata', 'MS_map_nopose_fix_s{}'),
    'head_pose':     ('mtr+pose_data_cross_attn_pe_with_map_fixdata', 'MS_map_xattn_pe_fix_s{}'),
    'head_shuf':     ('mtr+pose_data_cross_attn_pe_with_map_shufflepose_fixdata',
                      'MS_map_shufflepose_fix_s{}'),
}

STATIONARY = 0.3     # m/s at prediction time; matches the onset-task definition
ONSET_DISP = 1.0     # m travelled within 3 s


def best_epoch(d: str) -> int | None:
    f = os.path.join(d, 'eval', 'eval_with_train', 'best_eval_record.txt')
    if not os.path.exists(f):
        return None
    ep = None
    for line in open(f):
        m = re.search(r'best_epoch_(\d+)', line)
        if m:
            ep = int(m.group(1))
    return ep


def load_run(cfg: str, tag: str) -> dict | None:
    d = os.path.join(OUT_ROOT, cfg, tag)
    ep = best_epoch(d)
    if ep is None:
        return None
    p = os.path.join(d, 'eval', 'eval_with_train', f'metrics_epoch_{ep}.pkl')
    if not os.path.exists(p):
        return None
    recs = pickle.load(open(p, 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE']) for r in recs}


def main() -> int:
    dyn = pickle.load(open(DYN, 'rb'))
    dyn = {(str(k[0]), str(k[1])): v for k, v in dyn.items()}

    runs: dict[str, dict[int, dict]] = {}
    for cell, (cfg, tpl) in CELLS.items():
        runs[cell] = {}
        for s in SEEDS:
            r = load_run(cfg, tpl.format(s))
            if r is not None:
                runs[cell][s] = r
        print(f'{cell:<14} seeds found: {sorted(runs[cell])}')
    # The difference-in-differences pairs a seed across conditions, so a seed is usable
    # only if it exists in every cell. Dropping the rest keeps every contrast on one set.
    usable = sorted(set.intersection(*(set(runs[c]) for c in runs)))
    for c in runs:
        runs[c] = {s: v for s, v in runs[c].items() if s in usable}
    print(f'\nseeds usable in every cell: {usable} (n={len(usable)})')

    # agents scored by every run, so every comparison is over one fixed population
    common = None
    for cell in runs:
        for s, r in runs[cell].items():
            common = set(r) if common is None else (common & set(r))
    common &= set(dyn)
    common = sorted(common)
    sp = np.array([dyn[a]['speed0'] for a in common])
    dp = np.array([dyn[a]['disp3s'] for a in common])
    subs = {
        'ALL': np.ones(len(common), bool),
        'stationary': sp < STATIONARY,
        'standing_then_moves': (sp < STATIONARY) & (dp > ONSET_DISP),
        'movers': sp >= STATIONARY,
    }

    # Movers stratified by TURN: the angle by which the +3 s displacement bearing departs
    # from the current box heading. This is the stratum where the heading ablation is
    # actually informative. For a mover, deleting the heading channel does not delete the
    # direction of travel -- velocity stays in the input -- so a null there is expected
    # whether or not facing is redundant. Where the agent turns, neither velocity nor
    # heading states the future bearing, and body facing could in principle. Note
    # box_err3s == turn3s by construction: the box heading's 3.9 deg median "accuracy" IS
    # the fact that pedestrians mostly keep going straight.
    if os.path.exists(TURN):
        turn = {(str(k[0]), str(k[1])): v for k, v in pickle.load(open(TURN, 'rb')).items()}
        tv = np.array([turn[a]['turn3s_deg'] if a in turn else np.nan for a in common])
        known = np.isfinite(tv)
        mv = (sp >= STATIONARY) & known
        if mv.sum() > 100:
            q75, q90 = np.percentile(tv[mv], [75, 90])
            subs['movers_turn_q4'] = mv & (tv >= q75)
            subs['movers_turn_top10'] = mv & (tv >= q90)
            subs['movers_turn_q1'] = mv & (tv < np.percentile(tv[mv], 25))
            print(f'\n  turn3s among movers: q75={q75:.1f} deg, q90={q90:.1f} deg '
                  f'(n={int(mv.sum())} with a turn measured)')
    else:
        print(f'\n  NOTE: {TURN} missing -- run facing_error_ladder_units.py for the '
              'turn strata')
    print(f'\nagents common to all runs: {len(common)}')
    for k, m in subs.items():
        print(f'  {k:<22} n={int(m.sum())}')

    def arm_means(cell, mask):
        return {s: float(np.array([runs[cell][s][a] for a in common])[mask].mean())
                for s in sorted(runs[cell])}

    def paired(a, b, mask, label):
        A, B = arm_means(a, mask), arm_means(b, mask)
        s = sorted(set(A) & set(B))
        av = np.array([A[x] for x in s]); bv = np.array([B[x] for x in s])
        rel = (av - bv) / bv * 100
        n = len(s); t, p = stats.ttest_1samp(rel, 0.0); sd = rel.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        print(f'  {label:<42} n={n} {av.mean():.4f} vs {bv.mean():.4f} '
              f'{(av.mean()-bv.mean())/bv.mean()*100:+6.2f}%  t={t:+6.2f} p={p:.4f} '
              f'CI=[{rel.mean()-h:+.2f},{rel.mean()+h:+.2f}]')
        return dict(n=n, a=av.mean(), b=bv.mean(), rel=float(rel.mean()), t=float(t),
                    p=float(p), lo=float(rel.mean()-h), hi=float(rel.mean()+h))

    def did(mask, label):
        """content(heading removed) - content(heading present), paired over seeds.

        This is the hypothesis itself: does removing the heading channel make pose more
        useful? Testing it as one contrast is far more sensitive than comparing two CIs,
        and avoids reading a sign out of four separate underpowered tests.
        """
        per = []
        for s in usable:
            v = lambda c: np.array([runs[c][s][a] for a in common])[mask].mean()
            per.append((v('nohead_pose') - v('nohead_shuf')) / v('nohead_shuf') * 100
                       - (v('head_pose') - v('head_shuf')) / v('head_shuf') * 100)
        per = np.array(per); n = len(per)
        t, p = stats.ttest_1samp(per, 0.0); sd = per.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        print(f'  {label:<42} n={n} DiD={per.mean():+6.2f}pp t={t:+6.2f} p={p:.4f} '
              f'CI=[{per.mean()-h:+.2f},{per.mean()+h:+.2f}]')
        return dict(n=n, did_pp=float(per.mean()), t=float(t), p=float(p),
                    lo=float(per.mean()-h), hi=float(per.mean()+h),
                    per_seed=[float(x) for x in per], sd_pp=float(sd))

    def did_agentwise(mask, label):
        """The same DiD, but with the agent as the unit and seeds averaged first.

        WHY BOTH. The seed-level test above is the one that generalises: a new training
        run is a new draw, so the seed is the replication unit for "would this hold if
        you retrained". But its df is 7, and in a high-turn stratum where minADE is
        ~1.1 m the seed spread swamps a few pp -- the turn strata come back at +/-10 to
        18 pp, which is the same blindness that manufactured a detection floor at 579
        scenes. Averaging the eight runs per agent first cuts run noise by ~sqrt(8) and
        then tests over agents, which answers the narrower question actually at issue:
        on these runs, is there a population of agents for whom pose content pays once
        the heading channel is gone? It does NOT license a claim about retraining, so it
        is reported as a sensitivity analysis beside the seed-level test, never instead
        of it.
        """
        idx = np.flatnonzero(mask)
        if len(idx) < 30:
            return None
        A = {c: np.stack([np.array([runs[c][s][a] for a in common])[mask]
                          for s in usable]).mean(0)
             for c in ('nohead_pose', 'nohead_shuf', 'head_pose', 'head_shuf')}
        # per agent, in metres: content effect without heading minus content effect with
        d = ((A['nohead_pose'] - A['nohead_shuf'])
             - (A['head_pose'] - A['head_shuf']))
        n = len(d)
        t, p = stats.ttest_1samp(d, 0.0)
        sd = d.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        # express as pp of the with-heading shuffled baseline, so it is comparable
        # to the seed-level DiD above
        denom = A['head_shuf'].mean()
        print(f'  {label:<42} n={n} DiD={d.mean():+.4f} m '
              f'({100*d.mean()/denom:+.2f}pp) t={t:+6.2f} p={p:.4f} '
              f'CI=[{100*(d.mean()-h)/denom:+.2f},{100*(d.mean()+h)/denom:+.2f}]pp')
        return dict(n_agents=n, did_m=float(d.mean()),
                    did_pp=float(100 * d.mean() / denom),
                    t=float(t), p=float(p),
                    lo_pp=float(100 * (d.mean() - h) / denom),
                    hi_pp=float(100 * (d.mean() + h) / denom))

    res = {}
    for sub, mask in subs.items():
        print(f'\n=== {sub} (n={int(mask.sum())}) ===')
        res[sub] = {
            'content_noheading': paired('nohead_pose', 'nohead_shuf', mask,
                                        'CONTENT, heading removed'),
            'content_heading':   paired('head_pose', 'head_shuf', mask,
                                        'CONTENT, heading present (reference)'),
            'pose_vs_nopose_noheading': paired('nohead_pose', 'nohead_nopose', mask,
                                               'pose vs no-pose, heading removed'),
            'heading_cost': paired('nohead_nopose', 'head_nopose', mask,
                                   'cost of removing heading (baseline)'),
            'did': did(mask, 'DiD: does removing heading help pose?'),
            'did_agentwise': did_agentwise(
                mask, 'DiD, agent-level (seeds averaged first)'),
        }
    res['_meta'] = dict(n_common_agents=len(common), seeds=list(SEEDS),
                        subpopulation_sizes={k: int(v.sum()) for k, v in subs.items()},
                        stationary_speed=STATIONARY, onset_disp=ONSET_DISP,
                        note='best-epoch per-agent metrics; paired t over seeds, plus an agent-level sensitivity analysis with seeds averaged first')
    json.dump(res, open(OUT_JSON, 'w'), indent=2)
    print(f'\nwrote {OUT_JSON}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
