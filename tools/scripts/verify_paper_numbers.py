#!/usr/bin/env python
"""Does the paper quote the numbers its own experiment artifacts contain?

Sec. IV carries a few dozen numbers derived from JSON under experiments/. Several
near-misses have already happened: 44.2 deg quoted against a floor calibrated on a
different quantity, a box-heading median written 3.8 when the artifact says 3.884
(-> 3.9), and a printed rel column that used a different estimator from the one
tested. Those are all cheap to catch mechanically and expensive to catch in review.

This checks the couplings, not the prose: for each quantity it rounds the artifact
value the way the paper writes it and asserts that string is present. A FAIL means
either the paper or the artifact moved without the other.

Usage: python tools/scripts/verify_paper_numbers.py
Exit 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEX = os.path.join(REPO, 'paper', 'iv2027', 'main.tex')


def load(name):
    p = os.path.join(REPO, 'experiments', name)
    return json.load(open(p)) if os.path.exists(p) else None


def main() -> int:
    tex = open(TEX).read()
    checks = []   # (group, label, artifact value, how the paper writes it)

    F = load('facing_error_ladder_units.json')
    if F:
        t3, t8 = (F['targets']['+3 s displacement bearing'],
                  F['targets']['+8 s displacement bearing'])
        mv = F['targets']['+3 s / moving (>=0.3 m/s)']
        st = F['targets']['+3 s / stationary (<0.3 m/s)']
        g = 'facing (ladder units)'
        checks += [
            (g, 'SMPL median +3 s', t3['smpl_med'], '12.3'),
            (g, 'SMPL median +8 s', t8['smpl_med'], '15.1'),
            (g, 'box median +3 s', t3['box_med'], '3.9'),
            (g, 'box median +8 s', t8['box_med'], '7.1'),
            (g, 'n scored +3 s', t3['n'], '5{,}824'),
            (g, 'n scored +8 s', t8['n'], '6{,}206'),
            (g, 'SMPL p25', t3['smpl_pct']['25'], '5.5'),
            (g, 'SMPL p75', t3['smpl_pct']['75'], '23.9'),
            (g, 'SMPL p90', t3['smpl_pct']['90'], '45.9'),
            # the bracket's UPPER edge moved to 14.6 when its two rungs went to n=5,
            # so the share above 14.6 is the figure the argument now turns on
            (g, 'SMPL above 14.6 +3 s', 100 * t3['smpl_frac_above_14_6'], '43.5'),
            (g, 'SMPL above 14.6 +8 s', 100 * t8['smpl_frac_above_14_6'], '51.1'),
            (g, 'box above 14.6 +3 s', 100 * t3['box_frac_above_14_6'], '14.9'),
            (g, 'SMPL above 10.2 +3 s', 100 * t3['smpl_frac_outside_bracket'], '56.8'),
            (g, 'SMPL vs box median', t3['smpl_vs_box_med'], '10.0'),
            (g, 'movers median', mv['smpl_med'], '11.9'),
            (g, 'stationary median', st['smpl_med'], '41.2'),
            (g, 'stationary n', st['n'], '262'),
            (g, 'movers share of scored', 100 * mv['n'] / (mv['n'] + st['n']), '95.5'),
            (g, 'floor prediction low', F['floor_prediction_pct'][0], '1.5'),
            (g, 'floor prediction high', F['floor_prediction_pct'][1], '4.1'),
        ]

    D = load('detection_floor.json')
    if D:
        g = 'detection floor (Table II)'
        # the rung's n is in the label on purpose: while a queue is in flight the
        # artifact moves ahead of the table, and a mismatch tagged n=4 against a
        # table built at n=3 reads as "the paper needs refreshing", not "wrong"
        for r in sorted(D['rungs'], key=lambda r: r['median_err_deg']):
            th = r['median_err_deg']
            checks.append((g, f'rung {th:.1f} MDE (n={r["n"]})', r['mde80'],
                           f'{r["mde80"]:.1f}'))
        # the bracket the paper quotes is the PRIMARY denominator's, not the
        # literal published-data one -- checking 'all seeds' was a false pass,
        # since 5.5 and 10.2 also appear as rung labels
        prim = D.get('primary_denominator', 'all seeds')
        b = D.get('floor_brackets', {}).get(prim) or {}
        if b.get('note') == 'bracketed':
            checks += [(g, 'bracket low', b['lo'], f'{b["lo"]:.1f}'),
                       (g, 'bracket high', b['hi'], f'{b["hi"]:.1f}')]
        if not D.get('bracket_edge_stable', True):
            print('NOTE: detection_floor.json reports the bracket edge as NOT stable across\n'
                  '      denominators -- the paper should not state a single edge yet.\n')

    bad, group = [], None
    for g, label, val, written in checks:
        if g != group:
            print(f'\n{g}')
            print('-' * 74)
            group = g
        if isinstance(val, int):
            rounded_ok = written.replace('{,}', '') == str(val)
        else:
            rounded_ok = f'{val:.1f}' == written
        present = written in tex
        status = ('OK' if rounded_ok and present else
                  'ROUNDS TO ' + (f'{val:.1f}' if not isinstance(val, int) else str(val))
                  if not rounded_ok else 'NOT IN main.tex')
        if status != 'OK':
            bad.append((g, label, val, written, status))
        shown = str(val) if isinstance(val, int) else f'{val:.4g}'
        print(f'  {label:<30}{shown:>10}  paper: {written:<10} {status}')

    print()
    if bad:
        print(f'{len(bad)} MISMATCH(ES):')
        for g, label, val, written, status in bad:
            print(f'  [{g}] {label}: artifact {val}, paper writes {written} -- {status}')
        return 1
    print(f'all {len(checks)} checks pass')
    return 0


if __name__ == '__main__':
    sys.exit(main())
