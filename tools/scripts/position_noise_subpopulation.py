#!/usr/bin/env python
"""Where does pose pay once the observed track is a causal tracker's?

position_noise_ladder.py finds the project's first content effect: under simulated onboard
tracking, own pose beats another pedestrian's. The tracker leaves the box heading clean and
spoils speed and path, so if the body is supplying gait, the gain belongs to walkers, not
to pedestrians standing at prediction time. Scores content (pose vs shuffled) and pose vs
no-pose per speed band, on the seeds position_noise_ladder.json counts as complete, from
the per-agent metrics_epoch pickles of each run's best epoch. Speed bands use the label
speed at prediction time, the same speed0 as heading_noise_subpopulation.py.

Pickles: METRICS (default big1/heading_noise_metrics), laid out as output/waymo; best
epochs from experiments/heading_noise_records.
Writes experiments/position_noise_subpopulation.json.

Usage: python tools/scripts/position_noise_subpopulation.py
"""
from __future__ import annotations

import json
import os
import pickle
import re

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = os.environ.get('METRICS', '/home/erik/big1/heading_noise_metrics')
RECORDS = os.path.join(REPO, 'experiments', 'heading_noise_records')
DYN = os.path.join(REPO, 'experiments', 'val_agent_dynamics_25k.pkl')
LADDER = os.path.join(REPO, 'experiments', 'position_noise_ladder.json')
OUT = os.path.join(REPO, 'experiments', 'position_noise_subpopulation.json')
LOCAL_OUT = os.path.join(REPO, 'output', 'waymo')
PC = os.path.join(REPO, 'experiments', 'val_agent_pointcount_25k.pkl')

ARMS = {'nopose': 'no_pose_with_map',
        'pose': 'cross_attn_pe_with_map',
        'shuffled': 'cross_attn_pe_with_map_shufflepose'}
RUNGS = (0, 10, 30)
STATIONARY = 0.3


def load(arm: str, rung: int, seed: int) -> dict:
    if rung == 0:
        cfg = f'{ARMS[arm]}_headnoise00_scale25k'
        tag = f'HN_25k_{cfg}_lr25_s{seed}'
    else:
        cfg = f'{ARMS[arm]}_posnoise{rung:02d}_scale25k'
        tag = f'PN_25k_{cfg}_lr25_s{seed}'
    sub = os.path.join(f'mtr+pose_data_{cfg}', tag, 'eval', 'eval_with_train')
    ep = re.findall(r'best_epoch_(\d+)',
                    open(os.path.join(RECORDS, sub, 'best_eval_record.txt')).read())[-1]
    recs = pickle.load(open(os.path.join(METRICS, sub, f'metrics_epoch_{ep}.pkl'), 'rb'))
    return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE']) for r in recs}


def load_prefit(arm: str, rung: int, seed: int, epoch: int | None = None) -> dict:
    """A pre-fit run (position_noise_ladder.py explains the control). Helma runs are read
    from RECORDS and METRICS, workstation runs from output/waymo. Best epoch unless one
    is given."""
    cfg = f'{ARMS[arm]}_prefit_posnoise{rung:02d}_scale25k'
    sub = os.path.join(f'mtr+pose_data_{cfg}', f'PF_25k_{cfg}_lr25_s{seed}', 'eval',
                       'eval_with_train')
    for rec_root, met_root in ((RECORDS, METRICS), (LOCAL_OUT, LOCAL_OUT)):
        rec = os.path.join(rec_root, sub, 'best_eval_record.txt')
        if os.path.exists(rec):
            ep = epoch or re.findall(r'best_epoch_(\d+)', open(rec).read())[-1]
            recs = pickle.load(open(os.path.join(met_root, sub, f'metrics_epoch_{ep}.pkl'), 'rb'))
            return {(str(r['scenario_id']), str(r['object_id'])): float(r['minADE'])
                    for r in recs}
    raise FileNotFoundError(sub)


def ci(x: np.ndarray) -> dict:
    t, p = stats.ttest_1samp(x, 0.0)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(n=len(x), rel=float(x.mean()), p=float(p), lo=float(x.mean() - h),
                hi=float(x.mean() + h), per_seed=[float(v) for v in x])


def main() -> None:
    seeds = {r: json.load(open(LADDER))['seeds'][f'pn{r:02d}'] for r in RUNGS}
    dyn = {(str(k[0]), str(k[1])): v for k, v in pickle.load(open(DYN, 'rb')).items()}
    runs = {(a, r, s): load(a, r, s) for r in RUNGS for s in seeds[r] for a in ARMS}
    common = set(dyn)
    for x in runs.values():
        common &= set(x)
    common = sorted(common)
    sp = np.array([dyn[k]['speed0'] for k in common])
    subs = {'ALL': np.ones(len(common), bool),
            'stationary': sp < STATIONARY,
            'slow': (sp >= STATIONARY) & (sp < 0.8),
            'walking': (sp >= 0.8) & (sp < 1.6),
            'fast': sp >= 1.6}
    X = {k: np.array([v[a] for a in common]) for k, v in runs.items()}

    def rel(a, b, r, mask):
        return np.array([(X[a, r, s][mask].mean() / X[b, r, s][mask].mean() - 1) * 100
                         for s in seeds[r]])

    res = dict(_meta=dict(n_common_agents=len(common), seeds=seeds,
                          bands_ms={'stationary': [0, STATIONARY], 'slow': [STATIONARY, 0.8],
                                    'walking': [0.8, 1.6], 'fast': [1.6, None]},
                          sizes={k: int(v.sum()) for k, v in subs.items()},
                          note='best-epoch per-agent minADE; paired t over seeds; negative '
                               'favours the first-named arm'))
    for sub, mask in subs.items():
        print(f'\n=== {sub} (n={int(mask.sum())}) ===')
        res[sub] = {}
        for r in RUNGS:
            row = dict(content=ci(rel('pose', 'shuffled', r, mask)),
                       pose_vs_nopose=ci(rel('pose', 'nopose', r, mask)))
            res[sub][f'pn{r:02d}'] = row
            c, pn = row['content'], row['pose_vs_nopose']
            print(f'  {r:>2} cm n={c["n"]}  content {c["rel"]:+6.2f}% p={c["p"]:.3f} '
                  f'[{c["lo"]:+.1f},{c["hi"]:+.1f}]   vs nopose {pn["rel"]:+6.2f}% p={pn["p"]:.3f}')

    # --- pre-fit control: the same bands with poses that never saw the track ----------
    res['prefit'] = {}
    for prow in json.load(open(LADDER)).get('prefit', []):
        r, ps = prow['noise_cm'], prow['seeds']
        if len(ps) < 2:
            continue
        P = {(a, s): load_prefit(a, r, s) for a in ('pose', 'shuffled') for s in ps}
        assert all(set(common) <= set(x) for x in P.values()), 'pre-fit runs miss agents'
        Y = {k: np.array([v[a] for a in common]) for k, v in P.items()}

        def relp(num, den, mask):
            return np.array([(num[s][mask].mean() / den[s][mask].mean() - 1) * 100 for s in ps])

        # LiDAR returns, mean over the 11 observed frames. The fit can only inject the
        # track's speed where the sensor did not already see the legs, so the gap between
        # fitted and pre-fit poses should be widest on the sparsest bodies.
        pc = pickle.load(open(PC, 'rb'))
        npts = np.array([pc[k]['n_mean'] if k in pc else np.nan for k in common])
        psubs = dict(subs)
        psubs.update({'pts<30': npts < 30, 'pts30-100': (npts >= 30) & (npts < 100),
                      'pts100-200': (npts >= 100) & (npts < 200), 'pts>=200': npts >= 200})

        print(f'\n=== pre-fit poses, {r} cm, seeds {ps} ===')
        res['prefit'][f'pn{r:02d}'] = {}
        for sub, mask in psubs.items():
            pp = {s: Y['pose', s] for s in ps}
            sh = {s: Y['shuffled', s] for s in ps}
            fp = {s: X['pose', r, s] for s in ps}
            fs = {s: X['shuffled', r, s] for s in ps}
            npo = {s: X['nopose', r, s] for s in ps}
            c_pre, c_fin = relp(pp, sh, mask), relp(fp, fs, mask)
            row = dict(n_agents=int(mask.sum()), content=ci(c_pre),
                       content_final_same_seeds=ci(c_fin),
                       content_prefit_minus_final=ci(c_pre - c_fin),
                       pose_vs_nopose=ci(relp(pp, npo, mask)),
                       shuffled_vs_final_shuffled=ci(relp(sh, fs, mask)))
            res['prefit'][f'pn{r:02d}'][sub] = row
            c, f, d, pn = (row[k] for k in ('content', 'content_final_same_seeds',
                                            'content_prefit_minus_final', 'pose_vs_nopose'))
            print(f'  {sub:<10} n={int(mask.sum()):>5}  content {c["rel"]:+6.2f}% p={c["p"]:.3f} '
                  f'[{c["lo"]:+.1f},{c["hi"]:+.1f}]   final {f["rel"]:+6.2f}%   '
                  f'diff {d["rel"]:+5.2f} p={d["p"]:.4f}   vs nopose {pn["rel"]:+6.2f}% p={pn["p"]:.3f}')

        # The gradient is the test. A body cue must grow with the returns that resolve the
        # body, while a leaked label has no reason to. Per seed: the best-resolved stratum
        # against the sparsest, and the slope over the four strata.
        order = ('pts<30', 'pts30-100', 'pts100-200', 'pts>=200')
        cell = res['prefit'][f'pn{r:02d}']
        grad = {}
        for key, name in (('content', 'prefit'), ('content_final_same_seeds', 'final')):
            per = np.array([cell[k][key]['per_seed'] for k in order])        # (4, seeds)
            grad[name] = dict(
                top_minus_bottom=ci(per[-1] - per[0]),
                slope_per_stratum=ci(np.array([np.polyfit(range(4), per[:, i], 1)[0]
                                               for i in range(per.shape[1])])))
        cell['returns_gradient'] = grad
        print(f'  gradient >=200 minus <30: pre-fit {grad["prefit"]["top_minus_bottom"]["rel"]:+.2f} pp '
              f'p={grad["prefit"]["top_minus_bottom"]["p"]:.4f}, final '
              f'{grad["final"]["top_minus_bottom"]["rel"]:+.2f} pp p={grad["final"]["top_minus_bottom"]["p"]:.2f}')
        # The best-epoch rule, checked at fixed late epochs (pre-fit arms only: both arms of
        # the content test are pre-fit runs, so no final-pose pickles are needed).
        cell['fixed_epoch'] = {}
        for e in (26, 28, 30):
            E = {(a, s): load_prefit(a, r, s, e) for a in ('pose', 'shuffled') for s in ps}
            E = {k: np.array([v[a] for a in common]) for k, v in E.items()}
            cell['fixed_epoch'][str(e)] = {
                sub: ci(relp({s: E['pose', s] for s in ps}, {s: E['shuffled', s] for s in ps},
                             psubs[sub])) for sub in ('ALL', 'pts<30', 'pts>=200')}
            x = cell['fixed_epoch'][str(e)]
            print(f'  epoch {e}: ALL {x["ALL"]["rel"]:+.2f}% p={x["ALL"]["p"]:.3f}   '
                  f'>=200 {x["pts>=200"]["rel"]:+.2f}% p={x["pts>=200"]["p"]:.3f}   '
                  f'<30 {x["pts<30"]["rel"]:+.2f}% p={x["pts<30"]["p"]:.3f}')
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
