#!/usr/bin/env python
"""Where in a trajectory the oracle's gain lives, and whether it is mode selection.

WHY. The 25k ladder decomposes into a term that survives any corruption we applied
(-13.1%) and a precision term with a 3.0 deg half-life (-21.4%). The reading is
that the persistent term is coarse future-side SELECTION -- which of the forecast's
modes the agent actually took -- and the precision term is bearing refinement
inside the chosen mode. That reading is testable without training anything:

  * rung 15.1 deg is almost purely the persistent term (the exponential has decayed
    to 0.7 pp of its 21.4 pp amplitude there), so rung 15 ~= selection alone;
  * rung 0 minus rung 15 is almost purely the precision term.

Selection and refinement make opposite predictions. Selection pays at the END of
the trajectory, where the modes have fanned out, and pays most for agents whose
modes disagree; refinement pays along the whole trajectory and does not care how
far apart the modes are.

NO RETRAINING NEEDED. eval/eval_with_train/metrics_epoch_<N>.pkl survives for every
run at every eval epoch, and holds one record PER AGENT with minADE, minFDE,
minADE_{2,4,8}s, top1_ADE, best_mode_rank and mode_spread_endpoint. Agents are keyed
by (scenario_id, object_id), so baseline and oracle are paired per agent, not just
per seed. Inference is still reported over the three SEEDS: per-agent pairing says
where the gain lives, the seed spread says whether it is real.

Each run is read at its OWN best epoch, the operating point the ladder scores, which
is why the baseline's missing minFDE could not be read off the training log -- the
logs retain only the final-epoch eval block and the baselines peaked at 23/27/27.

Usage:  python tools/scripts/horizon_decomposition.py
Writes: experiments/horizon_decomposition.json
"""
import json
import os
import pickle
import re
import statistics as st

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOTS = ('/home/erik/big1/mtr_output/waymo',
         os.path.join(REPO, 'output', 'waymo'))
SEEDS = (101, 202, 303)
FULL_FUTURE = 80                      # 8 s at 10 Hz; the horizon columns need it

BASELINE = ('mtr+pose_data_no_pose_with_map_scale5k_lr5', 'MS_scale25k_nopose_s{seed}')
RUNGS = {                             # rung key -> (median delivered bearing error, cfg)
    '00': (0.0, 'mtr+pose_data_cross_attn_pe_with_map_oracledir00_scale25k'),
    '05': (5.0, 'mtr+pose_data_cross_attn_pe_with_map_oracledir05_scale25k'),
    '10': (10.0, 'mtr+pose_data_cross_attn_pe_with_map_oracledir10_scale25k'),
    '15': (15.1, 'mtr+pose_data_cross_attn_pe_with_map_oracledir15_scale25k'),
}
TAG = 'MS_oracledir{key}_25k_s{seed}'
COLS = ('minADE', 'minFDE', 'minADE_2s', 'minADE_4s', 'minADE_8s', 'top1_ADE',
        'mode_spread_endpoint', 'best_mode_rank')


def run_dir(cfg, tag):
    for r in ROOTS:
        d = os.path.join(r, cfg, tag)
        if os.path.isdir(d):
            return d
    return None


def best_epoch(run_dir_):
    """The epoch the ladder scores: the last best_epoch_N line in the run's own record."""
    rec = os.path.join(run_dir_, 'eval', 'eval_with_train', 'best_eval_record.txt')
    if not os.path.exists(rec):
        return None
    ep = None
    for line in open(rec):
        if (m := re.match(r'best_epoch_(\d+)\s+minADE\s+([0-9.]+)', line.strip())):
            ep = int(m.group(1))
    return ep


def agents(run_dir_):
    """{(scenario_id, object_id): record} at this run's best epoch, or None."""
    ep = best_epoch(run_dir_)
    if ep is None:
        return None
    pkl = os.path.join(run_dir_, 'eval', 'eval_with_train', f'metrics_epoch_{ep}.pkl')
    if not os.path.exists(pkl):
        return None
    recs = pickle.load(open(pkl, 'rb'))
    return ep, {(str(r['scenario_id']), str(r['object_id'])): r for r in recs}


def cell(cfg, tag_fmt, key=None):
    out = {}
    for s in SEEDS:
        tag = tag_fmt.format(seed=s, key=key) if key else tag_fmt.format(seed=s)
        d = run_dir(cfg, tag)
        if d is None:
            print(f'  missing run dir for {tag}')
            continue
        got = agents(d)
        if got is None:
            print(f'  missing best-epoch metrics for {tag}')
            continue
        ep, a = got
        out[s] = (ep, a)
        print(f'  {tag}: epoch {ep}, {len(a)} agents')
    return out


print('baseline')
base = cell(*BASELINE)
arms = {}
for k, (deg, cfg) in RUNGS.items():
    print(f'rung {k} ({deg} deg)')
    arms[k] = cell(cfg, TAG, key=k)

report = {'ladder': '25k', 'seeds': list(SEEDS), 'full_future_steps': FULL_FUTURE,
          'note': 'paired per agent at each run\'s own best epoch; inference over seeds',
          'rungs': []}

for k, (deg, _) in RUNGS.items():
    per_seed = []
    for s in SEEDS:
        if s not in base or s not in arms[k]:
            continue
        (_, b), (_, a) = base[s], arms[k][s]
        keys = sorted(set(b) & set(a))
        full = [x for x in keys if b[x]['num_valid_future_steps'] >= FULL_FUTURE]
        row = {'seed': s, 'n_paired': len(keys), 'n_full_future': len(full)}

        def paired(col, subset):
            """Both sides' values over the agents where BOTH report col.

            A horizon column is absent for an agent whose future is shorter than the
            window, so the subset is per column rather than global -- dropping those
            agents everywhere would silently restrict every metric to long futures.
            """
            ks = [x for x in subset if col in b[x] and col in a[x]]
            return (np.array([b[x][col] for x in ks], dtype=float),
                    np.array([a[x][col] for x in ks], dtype=float))

        for col in COLS:
            bv, av = paired(col, keys)
            row[col] = {'base': float(bv.mean()), 'arm': float(av.mean()), 'n': len(bv),
                        'rel_pct': float((av.mean() / bv.mean() - 1) * 100)}
        # the horizon profile needs a fixed window, so restrict it to full futures
        for col in ('minADE_2s', 'minADE_4s', 'minADE_8s', 'minFDE', 'minADE'):
            bv, av = paired(col, full)
            row[f'{col}_fullfut'] = {'base': float(bv.mean()), 'arm': float(av.mean()),
                                     'n': len(bv),
                                     'rel_pct': float((av.mean() / bv.mean() - 1) * 100)}
        # mode-selection stratification: quartiles of the BASELINE's own mode spread,
        # so the strata are a property of the baseline and never of the treatment
        sp = np.array([b[x]['mode_spread_endpoint'] for x in keys], dtype=float)
        qs = np.quantile(sp, [0.25, 0.5, 0.75])
        strata = []
        for lo, hi in zip([-np.inf, *qs], [*qs, np.inf]):
            sel = [x for x, v in zip(keys, sp) if lo <= v < hi]
            bv = np.array([b[x]['minADE'] for x in sel], dtype=float)
            av = np.array([a[x]['minADE'] for x in sel], dtype=float)
            strata.append({'spread_lo': None if lo == -np.inf else float(lo),
                           'spread_hi': None if hi == np.inf else float(hi),
                           'n': len(sel), 'base': float(bv.mean()),
                           'rel_pct': float((av.mean() / bv.mean() - 1) * 100)})
        row['by_mode_spread'] = strata
        # cost of picking the wrong mode, in metres: top-1 minus best-of-K
        row['selection_gap_m'] = {
            'base': float(np.mean([b[x]['top1_ADE'] - b[x]['minADE'] for x in keys])),
            'arm': float(np.mean([a[x]['top1_ADE'] - a[x]['minADE'] for x in keys]))}
        row['top1_is_best_frac'] = {
            'base': float(np.mean([b[x]['best_mode_rank'] == 0 for x in keys])),
            'arm': float(np.mean([a[x]['best_mode_rank'] == 0 for x in keys]))}
        per_seed.append(row)
    if per_seed:
        report['rungs'].append({'rung': k, 'median_err_deg': deg, 'per_seed': per_seed})


def across(rung, path):
    """Mean and range over seeds of a nested rel_pct, for printing."""
    vals = []
    for row in rung['per_seed']:
        v = row
        for p in path:
            v = v[p]
        vals.append(v)
    return st.mean(vals), min(vals), max(vals)


print('\n' + '=' * 78)
print('HORIZON PROFILE  (agents with a full 8 s future; mean over seeds, relative %)')
print(f"{'rung':>6} {'n':>6} {'ADE@2s':>9} {'ADE@4s':>9} {'ADE@8s':>9} {'minADE':>9} {'minFDE':>9}")
for rung in report['rungs']:
    n = st.mean(r['n_full_future'] for r in rung['per_seed'])
    cells_ = [across(rung, (f'{c}_fullfut', 'rel_pct'))[0]
              for c in ('minADE_2s', 'minADE_4s', 'minADE_8s', 'minADE', 'minFDE')]
    print(f"{rung['median_err_deg']:>6.1f} {n:>6.0f} " + ' '.join(f'{v:>+9.2f}' for v in cells_))

print('\nMODE SELECTION  (mean over seeds)')
print(f"{'rung':>6} {'gap base':>10} {'gap arm':>10} {'gap rel%':>10} "
      f"{'top1=best base':>15} {'arm':>8}")
for rung in report['rungs']:
    gb = st.mean(r['selection_gap_m']['base'] for r in rung['per_seed'])
    ga = st.mean(r['selection_gap_m']['arm'] for r in rung['per_seed'])
    tb = st.mean(r['top1_is_best_frac']['base'] for r in rung['per_seed'])
    ta = st.mean(r['top1_is_best_frac']['arm'] for r in rung['per_seed'])
    print(f"{rung['median_err_deg']:>6.1f} {gb:>10.4f} {ga:>10.4f} "
          f"{100*(ga/gb-1):>+10.2f} {tb:>15.4f} {ta:>8.4f}")

print('\nGAIN BY BASELINE MODE SPREAD AT THE ENDPOINT  (quartiles, relative %)')
print(f"{'rung':>6} {'Q1 (tight)':>12} {'Q2':>12} {'Q3':>12} {'Q4 (wide)':>12}")
for rung in report['rungs']:
    q = [st.mean(r['by_mode_spread'][i]['rel_pct'] for r in rung['per_seed'])
         for i in range(4)]
    print(f"{rung['median_err_deg']:>6.1f} " + ' '.join(f'{v:>+12.2f}' for v in q))

print('\nSEED SPREAD on the two headline columns')
for rung in report['rungs']:
    m, lo, hi = across(rung, ('minADE', 'rel_pct'))
    fm, flo, fhi = across(rung, ('minFDE', 'rel_pct'))
    print(f"  {rung['median_err_deg']:>5.1f} deg  minADE {m:+.2f}% [{lo:+.2f},{hi:+.2f}]   "
          f"minFDE {fm:+.2f}% [{flo:+.2f},{fhi:+.2f}]")

# ---------------------------------------------------------------------------------
# Derived summaries, recorded rather than left to be retyped into the paper. Every
# number Sec. V-A quotes from this analysis is one of these keys; verify_paper_numbers
# checks the tex against them. Hardcoding a fit into a second script is the exact bug
# that let Eq. (1) go on describing superseded rung labels.
from scipy.optimize import curve_fit                                      # noqa: E402

TH = np.array([r['median_err_deg'] for r in report['rungs']], dtype=float)


def two_component(y):
    """-(P + A exp(-theta/tau)) over the four rungs; needs all four to be a fit."""
    if len(TH) < 4:
        return None

    def model(th, P, A, tau):
        return -(P + A * np.exp(-th / np.maximum(tau, 1e-6)))

    (P, A, tau), _ = curve_fit(model, TH, np.asarray(y, float), p0=(10.0, 20.0, 5.0),
                               maxfev=200000, bounds=([0, 0, 0.1], [100, 100, 200]))
    return dict(persistent_pct=float(P), amplitude_pct=float(A),
                half_life_deg=float(tau * np.log(2.0)),
                persistent_share=float(P / (P + A)) if (P + A) else None)


summary = {'n_paired': int(st.mean(r['n_paired'] for r in report['rungs'][0]['per_seed'])),
           'n_full_future': int(st.mean(r['n_full_future']
                                        for r in report['rungs'][0]['per_seed'])),
           'by_horizon': {}}
for col, lab in (('minADE_2s', 'ADE_2s'), ('minADE_4s', 'ADE_4s'),
                 ('minADE_8s', 'ADE_8s'), ('minFDE', 'FDE')):
    y = [st.mean(r[f'{col}_fullfut']['rel_pct'] for r in rung['per_seed'])
         for rung in report['rungs']]
    fit = two_component(y)
    if fit:
        summary['by_horizon'][lab] = {'rel_pct_by_rung': y, **fit}
if summary['by_horizon']:
    shares = [v['persistent_share'] for v in summary['by_horizon'].values()]
    hl = [v['half_life_deg'] for v in summary['by_horizon'].values()]
    summary['persistent_share_range'] = [min(shares), max(shares)]
    summary['half_life_range_deg'] = [min(hl), max(hl)]

# selection: how much of the perfect oracle's effect a corrupted channel keeps
first, last = report['rungs'][0], report['rungs'][-1]


def mean_of(rung, path):
    v = []
    for row in rung['per_seed']:
        x = row
        for p in path:
            x = x[p]
        v.append(x)
    return st.mean(v)


gap_base = mean_of(first, ('selection_gap_m', 'base'))
gap_0 = mean_of(first, ('selection_gap_m', 'arm'))
gap_15 = mean_of(last, ('selection_gap_m', 'arm'))
summary['selection'] = {
    'rung_lo_deg': first['median_err_deg'], 'rung_hi_deg': last['median_err_deg'],
    'gap_m_base': gap_base, 'gap_m_exact_oracle': gap_0, 'gap_m_corrupted': gap_15,
    # fraction of the exact oracle's gap reduction that survives the corruption
    'gap_closed_frac': float((gap_base - gap_15) / (gap_base - gap_0)),
    'minADE_recovered_frac': float(mean_of(last, ('minADE', 'rel_pct'))
                                   / mean_of(first, ('minADE', 'rel_pct'))),
    'top1_is_best_base': mean_of(first, ('top1_is_best_frac', 'base')),
    'top1_is_best_exact_oracle': mean_of(first, ('top1_is_best_frac', 'arm')),
    'top1_is_best_corrupted': mean_of(last, ('top1_is_best_frac', 'arm')),
}

# absolute gain per baseline mode-spread quartile, at the uncorrupted rung
abs_gain = []
for i in range(4):
    base = st.mean(r['by_mode_spread'][i]['base'] for r in first['per_seed'])
    rel = st.mean(r['by_mode_spread'][i]['rel_pct'] for r in first['per_seed'])
    abs_gain.append(float(-base * rel / 100))
summary['abs_gain_m_by_mode_spread_quartile_exact'] = abs_gain
report['summary'] = summary

print('\nSUMMARY recorded for the paper')
for lab, v in summary['by_horizon'].items():
    print(f"  {lab:>7}: persistent {v['persistent_pct']:5.2f}%  precise "
          f"{v['amplitude_pct']:5.2f}%  half-life {v['half_life_deg']:4.2f}d  "
          f"share {v['persistent_share']:.1%}")
sel = summary['selection']
print(f"  selection gap {sel['gap_m_base']:.2f} m -> {sel['gap_m_exact_oracle']:.2f} m "
      f"(exact) / {sel['gap_m_corrupted']:.2f} m (at {sel['rung_hi_deg']} deg)")
print(f"  corrupted channel closes {sel['gap_closed_frac']:.1%} of the selection gap "
      f"while recovering {sel['minADE_recovered_frac']:.1%} of minADE")
print(f"  abs gain by mode-spread quartile (exact oracle, m): "
      + ' '.join(f'{v:.3f}' for v in abs_gain))

out = os.path.join(REPO, 'experiments', 'horizon_decomposition.json')
json.dump(report, open(out, 'w'), indent=1)
print(f'\nwrote {out}')
