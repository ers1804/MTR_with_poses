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

It then greps for RETIRED claims -- see the registry below. Number checks cannot
catch a withdrawn claim restated in prose, which has happened twice: "a factor of
four in bearing" survived the Sec. IV-B units fix and reached the conclusion, and
"gait at the noise floor" re-entered through a figure caption after the per-joint
table had retired it. Append to RETIRED whenever a claim is withdrawn; the list is
manual and goes stale otherwise.

Usage: python tools/scripts/verify_paper_numbers.py
       PAPER_TEX=/path/to/scratch.tex python tools/scripts/verify_paper_numbers.py
Exit 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# PAPER_TEX overrides the target so the retired-claim grep can be negative-tested
# against a scratch copy without touching the real paper
TEX = os.environ.get('PAPER_TEX',
                     os.path.join(REPO, 'paper', 'iv2027', 'main.tex'))

# Claims this project has RETIRED. The number checks below verify quantities that are
# in the artifacts; they cannot catch a withdrawn claim restated in prose, which has now
# happened twice -- "a factor of four in bearing" survived the Sec. IV-B units fix and
# reached the conclusion, and "gait at the noise floor" re-entered through a figure
# caption after the per-joint table had retired it. Both were found by grep, so the grep
# lives here. `allow` is a regex that, matched anywhere in the surrounding window, makes
# an occurrence legitimate -- several of these numbers are fine in their correct scope
# and wrong as headlines.
RETIRED = [
    dict(pat=r'factor of four|four times coarser',
         why='mixed units: 44.2 deg is onset direction on the stationary subset, the floor '
             'is calibrated on displacement bearing over every predicted pedestrian',
         instead='the share of agents above the bracket; the factor is three, and only '
                 'for the stationary subset'),
    dict(pat=r'far below the floor',
         why='same mixed-units comparison; pooled facing is 12.3 deg, inside the bracket',
         instead='facing clears the bracket mainly for agents whose velocity already '
                 'supplies it'),
    dict(pat=r'noise floor',
         why='"gait at the noise floor" was retired: over 534 pedestrians every limb joint '
             'scores +0.46 to +0.64 skill, so limbs are measured',
         instead='the head is the prior-dominated joint (skill 0.05); limbs are recovered'),
    dict(pat=r'gait or facing cue read from it',
         why='asserted prior-dominance for limbs generally, contradicting tab:skill',
         instead='scope prior-dominance to the far-range panels the figure depicts'),
    dict(pat=r'may still pay if fused early|might still pay if fused early',
         why='withdrawn after the early-fusion ladder: that pathway left an uncorrupted '
             'oracle inert on two seeds in three',
         instead='say the cheaper pathway was the less reliable one',
         # the paper states this claim in order to withdraw it, which is legitimate
         allow=r'withdraw'),
    dict(pat=r'160\.6',
         why='MPJPE from the misaligned harness protocol; the fair number is 87.6 mm',
         instead='87.6 mm root-aligned, timestamp-aligned'),
    dict(pat=r'44\.2',
         why='only valid as the stationary-onset measurement, never as "the" facing error',
         instead='12.3 deg pooled in the ladder units, 41.2 deg stationary',
         allow=r'onset'),
    dict(pat=r'\[5\.0\^',
         why='the superseded floor bracket; three of four denominators give [10.0, 15.1]',
         instead='[10.0, 15.1], with [5.0, 10.0] cited only as the single-draw sensitivity',
         allow=r'original draw|single original'),
    dict(pat=r'14\.6\^\\circ|10\.2\^\\circ|27\.2\^\\circ|37\.9\^\\circ|57\.4\^\\circ',
         why='rung labels from the original ad hoc calibration, wrong by up to 2.7 deg; a '
             'rung delivers rung_key/0.6745 * 0.6745 = its key by construction, confirmed '
             'over 11,328 scored agents',
         instead='the measured labels in experiments/oracle_rung_calibration.json: '
                 '5.0, 10.0, 15.1, 24.9, 35.7, 44.4, 60.1'),
]


def check_retired(tex):
    """Grep for claims the project has withdrawn. Returns a list of violations."""
    bad = []
    for r in RETIRED:
        for m in re.finditer(r['pat'], tex):
            window = tex[max(0, m.start() - 250):m.end() + 250]
            if r.get('allow') and re.search(r['allow'], window):
                continue
            line = tex[:m.start()].count(chr(10)) + 1
            bad.append((line, m.group(0), r))
    return bad



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
            # the share above the bracket's UPPER edge is the figure the argument turns
            # on; the keys are role-named because 'smpl_frac_above_14_6' would still have
            # looked right after the edge was corrected to 15.1
            (g, 'SMPL above bracket hi +3 s', 100 * t3['smpl_frac_above_hi'], '42.2'),
            (g, 'SMPL above bracket hi +8 s', 100 * t8['smpl_frac_above_hi'], '50.0'),
            (g, 'box above bracket hi +3 s', 100 * t3['box_frac_above_hi'], '14.2'),
            (g, 'SMPL above bracket lo +3 s', 100 * t3['smpl_frac_above_lo'], '57.5'),
            (g, 'SMPL vs box median', t3['smpl_vs_box_med'], '10.0'),
            (g, 'movers median', mv['smpl_med'], '11.9'),
            (g, 'stationary median', st['smpl_med'], '41.2'),
            (g, 'stationary n', st['n'], '262'),
            (g, 'movers share of scored', 100 * mv['n'] / (mv['n'] + st['n']), '95.5'),
            (g, 'floor prediction low', F['floor_prediction_pct'][0], '1.3'),
            (g, 'floor prediction high', F['floor_prediction_pct'][1], '4.0'),
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
        # Eq. (1) and the per-seed spread, refitted from this ladder rather than carried
        # as hardcoded constants in another script
        fit = D.get('decay_fit', {})
        if fit.get('pooled'):
            checks += [(g, 'decay tau (Eq. 1)', fit['pooled']['tau_deg'], '7.7'),
                       (g, 'decay amplitude', fit['pooled']['amplitude_pct'], '15.7')]
        if fit.get('half_life_range_deg'):
            lo, hi = fit['half_life_range_deg']
            checks += [(g, 'half-life low', lo, '3.1'), (g, 'half-life high', hi, '8.0')]
        if not D.get('bracket_edge_stable', True):
            print('NOTE: detection_floor.json reports the bracket edge as NOT stable across\n'
                  '      denominators -- the paper should not state a single edge yet.\n')

    C = load('oracle_rung_calibration.json')
    if C and D:
        g = 'rung calibration'
        # the labels the table prints must be the delivered medians, not an older
        # estimate of them; both come from JSON, so this is artifact-against-artifact
        meas = {k: round(v['+3s']['median'], 1) for k, v in C.items() if v.get('+3s', {}).get('n')}
        for r in sorted(D['rungs'], key=lambda r: r['median_err_deg']):
            m = meas.get(r['rung'])
            if m is not None:
                checks.append((g, f"rung {r['rung']} label", r['median_err_deg'], f'{m:.1f}'))

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

    retired = check_retired(tex)
    print()
    print('retired claims')
    print('-' * 74)
    if retired:
        for line, hit, r in retired:
            print(f'  main.tex:{line}  "{hit}"')
            print(f'      retired because: {r["why"]}')
            print(f'      say instead:     {r["instead"]}')
    else:
        print(f'  none of the {len(RETIRED)} withdrawn claims appear outside their '
              f'permitted scope')

    print()
    if bad:
        print(f'{len(bad)} MISMATCH(ES):')
        for g, label, val, written, status in bad:
            print(f'  [{g}] {label}: artifact {val}, paper writes {written} -- {status}')
    if retired:
        print(f'{len(retired)} RETIRED CLAIM(S) PRESENT (listed above)')
    if bad or retired:
        return 1
    print(f'all {len(checks)} number checks pass; no retired claims present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
