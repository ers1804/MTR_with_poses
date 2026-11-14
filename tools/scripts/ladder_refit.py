#!/usr/bin/env python
"""Refit the 25k oracle ladder on all six levels, with a slow term that decays.

detection_floor.py fits a constant plus one exponential to the four levels up to 15.1 deg
and scores 24.9 and 44.4 deg as held-out predictions. Both miss low (-10.4% against a
predicted -13.2%), so the constant is not constant, and the 3.0 deg half-life of the fast
term rests on it. This fits two exponentials to all six levels instead,

    gain(theta) = A_fast exp(-theta / tau_fast) + A_slow exp(-theta / tau_slow)

pooled and per seed, and profiles the fit over the fast half-life: no level lies between 0
and 5 deg, so the data bound that half-life from above more than they measure it.

It also gives the fit-free quantities the paper now quotes: the share of the exact oracle's
gain each level keeps, overall and by horizon (from horizon_decomposition.json).

Reads experiments/detection_floor_25k.json and horizon_decomposition.json.
Writes experiments/ladder_refit_25k.json.

Usage: python tools/scripts/ladder_refit.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import curve_fit

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LADDER = os.path.join(REPO, 'experiments', 'detection_floor_25k.json')
HORIZON = os.path.join(REPO, 'experiments', 'horizon_decomposition.json')
OUT = os.path.join(REPO, 'experiments', 'ladder_refit_25k.json')
LN2 = float(np.log(2.0))
PROFILE_DEG = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
BOUNDS = ([0.0, 0.05, 0.0, 5.0], [60.0, 20.0, 60.0, 1000.0])    # A_fast, tau_fast, A_slow, tau_slow


def two(th, a1, t1, a2, t2):
    return a1 * np.exp(-th / t1) + a2 * np.exp(-th / t2)


def fit(th: np.ndarray, gain: np.ndarray) -> dict:
    (a1, t1, a2, t2), _ = curve_fit(two, th, gain, p0=[18, 3, 17, 50], bounds=BOUNDS, maxfev=40000)
    res = two(th, a1, t1, a2, t2) - gain
    return dict(fast_amplitude_pct=float(a1), fast_half_life_deg=float(t1 * LN2),
                slow_amplitude_pct=float(a2), slow_half_life_deg=float(t2 * LN2),
                max_abs_residual_pp=float(np.abs(res).max()))


def profile(th: np.ndarray, gain: np.ndarray) -> list:
    """Best fit of the other three parameters with the fast half-life held fixed."""
    out = []
    for hl in PROFILE_DEG:
        t1 = hl / LN2
        (a1, a2, t2), _ = curve_fit(lambda t, a1, a2, t2: two(t, a1, t1, a2, t2), th, gain,
                                    p0=[18, 17, 50], bounds=([0, 0, 5], [60, 60, 1000]),
                                    maxfev=40000)
        res = two(th, a1, t1, a2, t2) - gain
        out.append(dict(fast_half_life_deg=hl, max_abs_residual_pp=float(np.abs(res).max()),
                        slow_half_life_deg=float(t2 * LN2)))
    return out


def main() -> None:
    rungs = json.load(open(LADDER))['rungs']
    th = np.array([r['median_err_deg'] for r in rungs], float)
    gain = np.array([-r['rel'] for r in rungs], float)             # percent, positive
    n_seeds = min(len(r['per_seed_rel']) for r in rungs)
    # per-seed ladders are read by position, so every level must list the same seeds
    assert all(r.get('seeds') == rungs[0].get('seeds') for r in rungs), 'levels differ in seeds'
    per_seed = [fit(th, np.array([-r['per_seed_rel'][i] for r in rungs])) for i in range(n_seeds)]

    by_h = json.load(open(HORIZON))['summary']['by_horizon']
    hz = json.load(open(HORIZON))['rungs']
    exact, coarse = 0, len(hz) - 1                                 # 0.0 deg and 15.1 deg
    assert hz[exact]['median_err_deg'] == 0 and abs(hz[coarse]['median_err_deg'] - 15.1) < 0.1
    res = dict(
        _meta=dict(source=LADDER, levels_deg=[float(t) for t in th], n_seeds=n_seeds,
                   note='gain is minus the relative minADE change against no pose, in percent. '
                        'Two exponentials on all six levels. per_seed fits each seed\'s own '
                        'ladder. profile holds the fast half-life fixed and refits the rest.'),
        gain_pct=[float(g) for g in gain],
        share_of_exact_gain_kept_pct=[float(g / gain[0] * 100) for g in gain],
        pooled=fit(th, gain), per_seed=per_seed,
        fast_half_life_range_deg=[min(f['fast_half_life_deg'] for f in per_seed),
                                  max(f['fast_half_life_deg'] for f in per_seed)],
        slow_half_life_range_deg=[min(f['slow_half_life_deg'] for f in per_seed),
                                  max(f['slow_half_life_deg'] for f in per_seed)],
        profile=profile(th, gain),
        horizon=dict(
            coarse_level_deg=float(hz[coarse]['median_err_deg']),
            share_of_exact_gain_kept_pct={
                k: float(v['rel_pct_by_rung'][coarse] / v['rel_pct_by_rung'][exact] * 100)
                for k, v in by_h.items()}))
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != '_meta'}, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
