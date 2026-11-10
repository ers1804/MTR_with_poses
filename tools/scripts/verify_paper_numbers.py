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
         why='mixed units: 44.2 deg is onset direction on the stationary subset, the '
             'ladder is calibrated on displacement bearing over every predicted '
             'pedestrian',
         instead='compare facing against the box heading on one target and one '
                 'population'),
    dict(pat=r'far below the floor',
         why='there is no floor to be below: at 25k every rung converts, including one '
             'worse at every percentile than the box heading the model already has',
         instead='facing is redundant where it is accurate and inaccurate where it '
                 'would be non-redundant'),
    # --- retired 2026-11-02, when the 25k ladder came in at n=3 per rung -----------
    dict(pat=r'detection floor',
         why='the 579 ladder read a floor at [10.0, 15.1] that was its own MDE (6.7 pp '
             'against an expected ~6.2); at 25k every rung converts down to -13.6% at '
             '15.1 deg, so the pathway has no measured accuracy threshold at all',
         instead='say what the ladder does measure: a positive control, an upper bound '
                 'on the pathway, and a precision/selection decomposition',
         # Sec. V-A narrates the retirement, and Sec. V-C cites it as the worked example
         allow=r'reads as a clean|went blind|low power|underpowered|worked example'),
    dict(pat=r'\[10\.0\^\\circ,\s*15\.1\^\\circ\]|bracket',
         why='the floor bracket was three-seed blindness at 579 scenes, not a property '
             'of the task',
         instead='no threshold; quote the two-component decay and the box-heading '
                 'comparison instead'),
    dict(pat=r'42\.2|two in five|50\.0\\%',
         why='tail fractions keyed to the retired bracket edge -- the share of agents '
             'above 15.1 deg means nothing once 15.1 deg is shown to convert',
         instead='the percentile distribution itself (5.5 / 23.9 / 45.9 deg), as '
                 'evidence of a heavy tail rather than of crossing an edge'),
    dict(pat=r'15\.7\\%|exp\(-\\theta/7\.7',
         why='the old Eq. (1): a single exponential fitted to the 579 ladder. One '
             'exponential cannot fit the 25k ladder at any constants -- residuals '
             '-1.5/+3.2/+0.3/-2.5 pp, systematic in theta',
         instead='the two-component form in detection_floor_25k.json'),
    # --- removed 2026-11-04 from the submission ------------------------------------
    # Not wrong, but no longer reported: the directed integration search and the
    # pre-registered eight-seed confirmation that killed its one acceptance are out of
    # the paper, so no claim may lean on them. The null now rests on scale, encoding,
    # integration and the oracle controls.
    dict(pat=r'coding agent|directed architecture search|directed search',
         why='the search is not reported in the submission',
         instead='the controlled null across scale, encoding and integration'),
    dict(pat=r'eight-seed confirmation|acceptance floor|development half',
         why='these all describe the removed search and its confirmation arms',
         instead='nothing -- the evidence they supported is simply not claimed'),
    dict(pat=r'-1\.21',
         why="the removed search's confirmation-arm effect",
         instead='nothing -- the evidence it supported is simply not claimed',
         # the onboard-track ladder's 10 cm pose-vs-no-pose effect is also -1.21% at n=5
         allow=r'against no pose|\$s\{=\}10\$'),
    # --- retired 2026-11-07, when the onboard-heading ladder landed ----------------
    dict(pat=r"tracker's box\s+heading|the tracker\s+heading",
         why='WOMD boxes are offboard auto-labels refined over the whole track, future '
             'frames included (Ettinger et al. 2021, Qi et al. 2021), not tracker output',
         instead="the labelled box heading; Sec. V-B degrades it to test the difference"),
    # --- retired 2026-11-04, when the early-fusion ladder landed at 25k -------------
    # Sec. V-C still NARRATES the 579 reading before overturning it, so these patterns
    # target the conclusions that reading supported, not the reading itself.
    dict(pat=r'does not rescue it: that pathway fails|do not offer early fusion as a rescue',
         why='at 25k the early-fusion pathway converts an uncorrupted direction channel '
             'on every seed (-31.5%, p=0.011); the 579 per-seed signature was a power '
             'artifact, at MDE 25.5 pp',
         instead='both pathways read what they are given, and neither finds anything in '
                 'the body to read'),
    dict(pat=r'partly illusory|partly by\s+not reading the channel',
         why='the free-carriage result means what it appears to mean: early fusion '
             'integrates the channel at no capacity cost, shown by an oracle through '
             'that pathway at 25k',
         instead='the freedom is real, and the oracle through early fusion is what '
                 'establishes it'),
    dict(pat=r'converts it on only one\s+of three seeds',
         why='that count is the 579 ladder at MDE 25.5 pp; at 25k it is every seed',
         instead='converts an uncorrupted oracle on every seed at 25k scenes'),
    # --- retired 2026-11-04, after a full-paper review -----------------------------
    dict(pat=r'six pose\s+encodings|across six pose',
         why='two of the six (3D joint positions, root-orientation-free) predate the '
             'export fix and have no shuffled-pose arm, so the control was never run '
             'against them; before the fix every pose arm was already a shuffled arm',
         instead='the three encodings with a shuffled control on the corrected export, '
                 'with constant-pose named as a capacity control'),
    dict(pat=r'corruption-invariant|survives every\s+corruption|survives arbitrary angular',
         why='the persistent term is slow, not permanent: the 25k rung held out at '
             '24.9 deg reads -10.39% against the -13.2% Eq. 1 extrapolates, and a 60 deg '
             'median is near-uniform on the circle',
         instead='a slowly decaying term, four fifths of it left at 24.9 deg'),
    # --- retired 2026-11-09, when the 25k rung at 24.9 deg landed -------------------
    dict(pat=r'intervals that exclude the|decay somewhere beyond\s+our rungs',
         why='the 579 rungs above 15.1 deg read ~0 because they were blind, not because '
             'the term had gone: scaled from 25k, 24.9 deg should read ~4.7% there, below '
             'its MDE of 6.6 pp, while the 25k rung itself converts -10.39% (p=0.003)',
         instead='the held-out 25k rung: four fifths of the slow term survive 24.9 deg'),
    dict(pat=r'Selection is not available to any observation-time',
         why='Sec. V-B measures observation-time cues that do carry partial mode '
             'information (box heading 25.2 deg, SMPL 44.2 deg against 90 deg chance), '
             'so selection is not categorically unavailable, only never perfect',
         instead='a cue buys selection to the extent it predicts the realised mode '
                 'beyond what the existing inputs already predict'),
    dict(pat=r'about \$3\\%\$ of the displacement, so little error',
         why='the 2 s residual is not relatively saturated: a perfect oracle still cuts '
             'it by 14.7%. The binding limit is absolute, 1.4 cm',
         instead='the absolute form: even an ideal cue is worth 1.4 cm at 2 s'),
    dict(pat=r'measures the cue it is meant to carry more\s+accurately than the inputs',
         why='the marginal-accuracy form of clause 3 is refuted by the paper itself: '
             'the 15.1 deg rung is less accurate at every percentile than an existing '
             'input and still converts',
         instead='the shared-information form: the modality must carry information about '
                 'the cue that the existing inputs do not already carry'),
    dict(pat=r'1\.3\$?--\$?4\.0|0\.002\$?--\$?0\.32',
         why='gains predicted for SMPL facing by reading the retired decay law at its '
             'median; the ladder does not license predicting a real cue from a rung',
         instead='the content test, which measures what facing is worth directly'),
    dict(pat=r'noise floor',
         why='"gait at the noise floor" was retired: over 534 pedestrians every limb joint '
             'scores +0.46 to +0.64 skill, so limbs are measured',
         instead='the head is the prior-dominated joint (skill 0.05); limbs are recovered'),
    dict(pat=r'gait or facing cue read from it',
         why='asserted prior-dominance for limbs generally, contradicting tab:skill',
         instead='scope prior-dominance to the far-range panels the figure depicts'),
    dict(pat=r'may still pay if fused early|might still pay if fused early',
         why='the early-fusion ladder left an uncorrupted oracle inert on two seeds in '
             'three, so we do not offer early fusion as a rescue',
         instead='say we do not offer it as a rescue, and that the stronger claim needs '
                 'that ladder at 25k -- at 579 its MDE is 25.5 pp',
         # the paper states the claim in order to decline it, which is legitimate
         allow=r'withdraw|do not offer'),
    dict(pat=r'the cheaper integration\s+was the less reliable one|cheaper one was not the safer',
         why='an n=3 ladder at 579 scenes with MDE 25.5 pp cannot support a comparative '
             'reliability claim; the defensible evidence is the per-seed signature',
         instead='two of three seeds showed no movement at all from a perfect channel, '
                 'where the branch moved on all three'),
    # --- retired 2026-11-09, when the onboard-track ladder landed ---------------------
    dict(pat=r'neither finds anything in the\s+body to read',
         why='under a simulated causal tracker own pose beats a stranger\'s by 6% on moving '
             'pedestrians; the body has nothing to add to an OFFBOARD track only',
         instead='pose adds nothing to an offboard track and pays on a causal one'),
    dict(pat=r'this ladder does not degrade them',
         why='positions are degraded by the onboard-track ladder (Sec. IV-E)',
         instead='point to Sec. IV-E'),
    dict(pat=r'does not measurably improve pedestrian trajectory accuracy\.',
         why='unscoped: the null holds on offboard tracks, not on a causal tracker\'s',
         instead='scope it to WOMD\'s offboard tracks'),
    dict(pat=r'160\.6',
         why='MPJPE from the misaligned harness protocol; the fair number is 87.6 mm',
         instead='87.6 mm root-aligned, timestamp-aligned'),
    dict(pat=r'44\.2',
         why='only valid as the stationary-onset measurement, never as "the" facing error',
         instead='12.3 deg pooled in the ladder units, 41.2 deg stationary',
         allow=r'onset'),
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
            # the bracket-crossing fractions (42.2 / 50.0 / 14.2 / 57.5) were retired
            # with the bracket on 2026-11-02 and are now in the RETIRED registry;
            # what the argument turns on instead is that box beats SMPL at every
            # percentile, which is checked here and at the quartiles above
            (g, 'SMPL vs box median', t3['smpl_vs_box_med'], '10.0'),
            (g, 'box p75', t3['box_pct']['75'], '9.3'),
            (g, 'box p90', t3['box_pct']['90'], '20.1'),
            (g, 'movers median', mv['smpl_med'], '11.9'),
            (g, 'stationary median', st['smpl_med'], '41.2'),
            (g, 'stationary n', st['n'], '262'),
            (g, 'movers share of scored', 100 * mv['n'] / (mv['n'] + st['n']), '95.5'),
            # floor_prediction_pct is retired: reading the decay law at a real cue's
            # median is the comparison the 25k ladder invalidated
        ]

    D = load('detection_floor.json')
    if D:
        g = 'detection floor (Table II)'
        # the rung's n is in the label on purpose: while a queue is in flight the
        # artifact moves ahead of the table, and a mismatch tagged n=4 against a
        # table built at n=3 reads as "the paper needs refreshing", not "wrong"
        # Table I lists 0.0-15.1 individually and collapses 24.9-60.1 into one row,
        # so the high rungs are checked as the range that row states
        LOW = (0.0, 5.0, 10.0, 15.1)
        hi_rungs = [r for r in D['rungs'] if r['median_err_deg'] not in LOW]
        for r in sorted((x for x in D['rungs'] if x['median_err_deg'] in LOW),
                        key=lambda r: r['median_err_deg']):
            th = r['median_err_deg']
            checks.append((g, f'rung {th:.1f} MDE (n={r["n"]})', r['mde80'],
                           f'{r["mde80"]:.1f}'))
        if hi_rungs:
            checks += [
                (g, 'high rungs MDE low', min(r['mde80'] for r in hi_rungs), '5.5'),
                (g, 'high rungs MDE high', max(r['mde80'] for r in hi_rungs), '9.1'),
                (g, 'high rungs rel most negative',
                 abs(min(r['rel'] for r in hi_rungs)), '0.1'),
                (g, 'high rungs rel most positive',
                 max(r['rel'] for r in hi_rungs), '2.7')]
        # This ladder is no longer the paper's primary one. It appears in Table I as the
        # underpowered predecessor, so its per-rung MDEs are still checked above, but its
        # bracket and its single-exponential Eq. (1) are retired -- see RETIRED. The one
        # claim the paper makes ABOUT it is that it could not have seen the 25k effect:
        # proportionally to its own theta=0 amplitude, rung 15.1 should have read ~6.2%
        # against an MDE of 6.7.
        by_deg = {r['median_err_deg']: r for r in D['rungs']}
        D25 = load('detection_floor_25k.json')
        if D25 and 0.0 in by_deg and 15.1 in by_deg:
            n25 = {r['median_err_deg']: r['rel'] for r in D25['rungs']}
            if 0.0 in n25 and 15.1 in n25:
                expected = by_deg[0.0]['rel'] * (n25[15.1] / n25[0.0])
                checks.append((g, 'rung 15.1 effect the 579 ladder should have expected',
                               -expected, '6.2'))
            # and its flat top: the 25k rung at 24.9 deg, scaled the same way
            if 0.0 in n25 and 24.9 in n25 and 24.9 in by_deg:
                expected = by_deg[0.0]['rel'] * (n25[24.9] / n25[0.0])
                checks += [(g, 'rung 24.9 effect the 579 ladder should have expected',
                            -expected, '4.7'),
                           (g, 'rung 24.9 MDE at 579', by_deg[24.9]['mde80'], '6.6')]

    D25 = load('detection_floor_25k.json')
    if D25:
        g = 'oracle ladder 25k (Table I)'
        for r in sorted(D25['rungs'], key=lambda r: r['median_err_deg']):
            th = r['median_err_deg']
            checks += [(g, f'rung {th:.1f} rel (n={r["n"]})', abs(r['rel']),
                        f'{abs(r["rel"]):.2f}'),
                       (g, f'rung {th:.1f} MDE (n={r["n"]})', r['mde80'],
                        f'{r["mde80"]:.1f}')]
        two = D25.get('two_component_fit', {})
        if two.get('pooled'):
            t = two['pooled']
            checks += [(g, 'Eq. 1 persistent', t['persistent_pct'], '13.1'),
                       (g, 'Eq. 1 amplitude', t['amplitude_pct'], '21.4'),
                       (g, 'Eq. 1 half-life', t['half_life_deg'], '3.0')]
        for key, lo_s, hi_s in (('persistent_range_pct', '12.3', '14.2'),
                                ('amplitude_range_pct', '20.8', '21.8'),
                                ('half_life_range_deg', '2.8', '3.4')):
            if two.get(key):
                lo, hi = two[key]
                checks += [(g, f'{key} low', lo, lo_s), (g, f'{key} high', hi, hi_s)]
        if two.get('residuals_pp'):
            for v in two['residuals_pp']:
                pass    # two-component residuals are quoted only as "to 0.4 pp"
        for h in D25.get('held_out', []):
            if h['median_err_deg'] == 24.9:
                r = next(r for r in D25['rungs'] if r['median_err_deg'] == 24.9)
                checks += [(g, f'held-out 24.9 p (n={r["n"]})', r['p'], '0.003'),
                           (g, f'held-out 24.9 CI low (n={r["n"]})', abs(r['lo']), '12.9'),
                           (g, f'held-out 24.9 CI high (n={r["n"]})', abs(r['hi']), '7.9'),
                           (g, 'held-out 24.9 Eq. 1 extrapolation', abs(h['predicted']),
                            '13.2')]
                # "Four fifths of the term survive" and "below 0.1%"
                assert 0.75 <= h['slow_share'] < 0.85, h['slow_share']
                assert h['precision_term'] < 0.1, h['precision_term']
        fit1 = D25.get('decay_fit', {})
        if fit1.get('residuals_pp'):
            # the paper quotes these as the reason the functional form is wrong
            for v, want in zip(fit1['residuals_pp'], ('1.5', '3.2', '0.3', '2.5')):
                checks.append((g, f'single-exp residual {want}', abs(v), want))

    N = load('nohead_subpopulation.json')
    if N:
        g = 'heading-removal counterfactual (Sec. V-B)'
        mv, stt, hi = N.get('movers'), N.get('stationary'), N.get('movers_turn_top10')
        if mv:
            checks += [(g, 'movers n agents', mv['did']['n'] and
                        N['_meta']['subpopulation_sizes']['movers'], '13{,}006'),
                       (g, 'movers DiD (seeds)', abs(mv['did']['did_pp']), '2.9')]
            if mv.get('did_agentwise'):
                a = mv['did_agentwise']
                checks += [(g, 'movers DiD agent-level low', a['lo_pp'], '2.4'),
                           (g, 'movers DiD agent-level high', a['hi_pp'], '3.3')]
            checks.append((g, 'cost of removing heading on movers',
                           mv['heading_cost']['rel'], '1.5'))
        if stt:
            checks.append((g, 'stationary DiD (seeds)', abs(stt['did']['did_pp']), '0.6'))
            if stt.get('did_agentwise'):
                a = stt['did_agentwise']
                checks += [(g, 'stationary DiD agent-level low', a['lo_pp'], '-1.5'),
                           (g, 'stationary DiD agent-level high', a['hi_pp'], '0.2')]
        if hi and hi.get('did_agentwise'):
            a = hi['did_agentwise']
            checks += [(g, 'top-decile turn DiD agent-level', abs(a['did_pp']), '1.2'),
                       (g, 'top-decile turn agent-level low', a['lo_pp'], '-2.6'),
                       (g, 'top-decile turn agent-level high', a['hi_pp'], '0.2')]

    H = load('horizon_decomposition.json')
    if H and H.get('summary'):
        g = 'horizon decomposition (Sec. V-A)'
        S = H['summary']
        checks.append((g, 'agents paired per arm', S['n_paired'], '16{,}486'))
        for lab, want in (('ADE_2s', '21'), ('ADE_8s', '39'), ('FDE', '53')):
            if lab in S['by_horizon']:
                checks.append((g, f'persistent share {lab}',
                               100 * S['by_horizon'][lab]['persistent_share'], want))
        if S.get('half_life_range_deg'):
            lo, hi = S['half_life_range_deg']
            checks += [(g, 'precision half-life low across horizons', lo, '2.9'),
                       (g, 'precision half-life high across horizons', hi, '3.9')]
        sel = S['selection']
        checks += [(g, 'selection gap baseline', sel['gap_m_base'], '0.67'),
                   (g, 'selection gap exact oracle', sel['gap_m_exact_oracle'], '0.26'),
                   (g, 'selection gap corrupted', sel['gap_m_corrupted'], '0.29'),
                   (g, 'selection gap closed at 15.1 deg',
                    100 * sel['gap_closed_frac'], '92'),
                   (g, 'minADE recovered at 15.1 deg',
                    100 * sel['minADE_recovered_frac'], '39')]
        q = S.get('abs_gain_m_by_mode_spread_quartile_exact')
        if q:
            checks += [(g, 'abs gain widest mode-spread quartile', max(q), '0.150'),
                       (g, 'abs gain other quartiles low', min(q), '0.111'),
                       (g, 'abs gain other quartiles high',
                        max(v for v in q if v != max(q)), '0.130')]

    J = load('per_joint_recoverability_mpjpe_eval113.json')
    if J:
        g = 'per-joint recovery (Table V)'
        by = {r['joint']: r for r in J['per_joint']}
        LEG = ('LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE')
        ARM = ('LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST')
        # Sec. II differentiates this paper from Le et al. by saying the lower-body
        # keypoints they find most useful are the ones our sensor recovers worst, and
        # that the leg motion we do recover is damped. Both claims are quoted as ranges,
        # so check the endpoints rather than a single joint.
        legs_skill = [by[j]['skill'] for j in LEG if j in by]
        legs_sr = [by[j]['spread_ratio'] for j in LEG if j in by]
        arms_sr = [by[j]['spread_ratio'] for j in ARM if j in by]
        elbows = [by[j]['skill'] for j in ('LEFT_ELBOW', 'RIGHT_ELBOW') if j in by]
        if legs_skill:
            checks += [(g, 'knee/ankle skill low', min(legs_skill), '0.46'),
                       (g, 'knee/ankle skill high', max(legs_skill), '0.49'),
                       (g, 'elbow skill low', min(elbows), '0.63'),
                       (g, 'elbow skill high', max(elbows), '0.64'),
                       (g, 'leg spread ratio low', 100 * min(legs_sr), '84'),
                       (g, 'leg spread ratio high', 100 * max(legs_sr), '91'),
                       (g, 'arm spread ratio low', 100 * min(arms_sr), '96'),
                       (g, 'arm spread ratio high', 100 * max(arms_sr), '99')]

    C = load('oracle_rung_calibration.json')
    if C and D:
        g = 'rung calibration'
        # the labels the table prints must be the delivered medians, not an older
        # estimate of them; both come from JSON, so this is artifact-against-artifact
        meas = {k: round(v['+3s']['median'], 1) for k, v in C.items() if v.get('+3s', {}).get('n')}
        for r in sorted(D['rungs'], key=lambda r: r['median_err_deg']):
            m = meas.get(r['rung'])
            if m is not None and f'{m:.1f}' in tex:
                checks.append((g, f"rung {r['rung']} label", r['median_err_deg'], f'{m:.1f}'))

    T1 = load('top1_content_25k.json')
    if T1:
        g = 'ranking columns at 25k (Sec. V-A)'
        # minADE is carried in this artifact purely as a sanity check: it must
        # reproduce the content test's own +0.04%, which is what licenses reading
        # the three ranking columns off the same pairing.
        for col, lab, how in (
                ('top1_ADE', 'top-1 ADE content', '-2.07'),
                ('best_mode_rank', 'best-mode rank content', '-0.93'),
                ('mode_spread_endpoint', 'mode spread content', '0.17'),
                ('minADE', 'minADE (reproduces content test)', '0.04')):
            pr = T1['metrics'][col]['pairs']['shuf->pose']
            checks.append((g, lab, pr['rel_pct'], how))
        checks.append((g, 'top-1 ADE MDE',
                       T1['metrics']['top1_ADE']['pairs']['shuf->pose']['mde80_pct'],
                       '3.7'))
        checks.append((g, 'paired agents', T1['metrics']['minADE']['n_agents'],
                       '16{,}486'))

    PC = load('pointcount_stratified_content.json')
    if PC:
        g = 'stratified nulls (Sec. IV-E)'
        for key, how, hown in (('>=200 pts', '-0.37', '2{,}372'),
                               ('>=30 pts', '-0.08', '10{,}902')):
            st = PC['strata'].get(key)
            if st:
                checks.append((g, f'{key} content', st['content']['rel'], how))
                checks.append((g, f'{key} n', st['n'], hown))

    RG = load('range_stratified_content.json')
    if RG:
        g = 'stratified nulls (Sec. IV-E)'
        st = RG['strata'].get('<10m')
        if st:
            checks.append((g, '<10 m content', st['content']['rel'], '-0.14'))

    HZ = load('horizon_decomposition.json')
    if HZ:
        g = '2 s headroom (Sec. V-F)'
        r0 = HZ['rungs'][0]['per_seed']
        rel2 = sum(x['minADE_2s']['rel_pct'] for x in r0) / len(r0)
        gain = (sum(x['minADE_2s']['base'] - x['minADE_2s']['arm'] for x in r0)
                / len(r0))
        checks.append((g, '2 s oracle relative', abs(rel2), '14.7'))
        checks.append((g, '2 s oracle absolute (cm)', gain * 100.0, '1.4'))

    EF = load('detection_floor_ef25k.json')
    if EF and EF.get('rungs'):
        g = 'early-fusion ladder at 25k (Sec. V-C)'
        r0 = min(EF['rungs'], key=lambda r: r['median_err_deg'])
        # n is in the label on purpose: this queue is in flight, so when the third
        # seed lands these four values move and the mismatch is the signal to
        # refresh Sec. V-C rather than evidence that it is wrong.
        checks += [
            (g, f'theta=0 rel (n={r0["n"]})', abs(r0['rel']), '31.2'),
            (g, f'theta=0 p (n={r0["n"]})', r0['p'], '0.0002'),
            (g, f'theta=0 CI lo (n={r0["n"]})', abs(r0['lo']), '32.9'),
            (g, f'theta=0 CI hi (n={r0["n"]})', abs(r0['hi']), '29.5'),
        ]

    HN = load('heading_noise_ladder.json')
    if HN:
        g = 'onboard-heading ladder (Sec. V-B)'
        R = {r['sigma_deg']: r for r in HN['rungs']}
        checks += [
            (g, 'sigma=15 baseline loss', R[15]['degradation']['rel'], '5.8'),
            (g, 'sigma=30 baseline loss', R[30]['degradation']['rel'], '12.6'),
            (g, 'sigma=30 median added error', R[30]['median_added_err_deg'], '20.2'),
            (g, 'sigma=15 content', abs(R[15]['content']['rel']), '0.14'),
            (g, 'sigma=15 content p', R[15]['content']['p'], '0.90'),
            (g, 'sigma=30 content', abs(R[30]['content']['rel']), '0.11'),
            (g, 'sigma=30 content p', R[30]['content']['p'], '0.93'),
            (g, 'sigma=30 recovery cap (%)', R[30]['max_recovered_share'] * 100, '38'),
            (g, 'clean rung vs workstation (max |rel|)',
             max(abs(x['rel']) for x in HN['reproduction'].values()), '0.4'),
            (g, 'LR check', HN['lr_check']['vs_default']['rel'], '0.7'),
            (g, 'LR check p', HN['lr_check']['vs_default']['p'], '0.48'),
        ]
    HS = load('heading_noise_subpopulation.json')
    if HS:
        g = 'onboard-heading ladder, stationary (Sec. V-B)'
        st = HS['stationary']
        checks += [
            (g, 'stationary n', HS['_meta']['sizes']['stationary'], '3{,}336'),
            (g, 'sigma=15 baseline change', abs(st['sigma15']['degradation']['rel']), '1.1'),
            (g, 'sigma=30 baseline change', st['sigma30']['degradation']['rel'], '4.3'),
            (g, 'sigma=15 content', st['sigma15']['content']['rel'], '1.2'),
            (g, 'sigma=30 content', st['sigma30']['content']['rel'], '1.3'),
            (g, 'sigma=30 content p', st['sigma30']['content']['p'], '0.049'),
        ]

    PN = load('position_noise_ladder.json')
    if PN:
        g = 'onboard-track ladder (Sec. IV-E, Table onboard)'
        R = {r['noise_cm']: r for r in PN['rungs']}
        # n is in every label, so a change in the seed count shows up as a mismatch.
        # The track rungs are at n=5 since 2026-11-10; the paper also quotes the
        # first-three-seed reading, recomputed here from the per-seed values.
        n10, n30 = R[10]['n'], R[30]['n']
        c30, c10 = R[30]['content'], R[10]['content']
        fe = [abs(x['content']['rel']) for x in R[30]['fixed_epoch'].values()]
        e10 = R[10]['tracker_error']
        e30 = R[30]['tracker_error']
        checks += [
            (g, f'30 cm content (n={n30})', abs(c30['rel']), '5.32'),
            (g, f'30 cm content, abstract (n={n30})', abs(c30['rel']), '5.3'),
            (g, f'30 cm content p (n={n30})', c30['p'], '0.002'),
            (g, f'30 cm CI lo (n={n30})', abs(c30['lo']), '7.3'),
            (g, f'30 cm CI hi (n={n30})', abs(c30['hi']), '3.4'),
            (g, f'30 cm per-seed min (n={n30})', min(abs(x) for x in c30['per_seed_rel']), '2.7'),
            (g, f'30 cm per-seed max (n={n30})', max(abs(x) for x in c30['per_seed_rel']), '6.5'),
            (g, '30 cm content, first three seeds',
             abs(sum(c30['per_seed_rel'][:3]) / 3), '6.05'),
            (g, f'30 cm fixed-epoch min (n={n30})', min(fe), '5.2'),
            (g, f'30 cm fixed-epoch max (n={n30})', max(fe), '5.8'),
            (g, f'30 cm vs no-pose (n={n30})', abs(R[30]['pose_vs_nopose']['rel']), '4.06'),
            (g, f'30 cm vs no-pose p (n={n30})', R[30]['pose_vs_nopose']['p'], '0.004'),
            (g, f'10 cm content (n={n10})', abs(c10['rel']), '2.08'),
            (g, f'10 cm content p (n={n10})', c10['p'], '0.005'),
            (g, f'10 cm vs no-pose (n={n10})', abs(R[10]['pose_vs_nopose']['rel']), '1.21'),
            (g, f'10 cm vs no-pose p (n={n10})', R[10]['pose_vs_nopose']['p'], '0.07'),
            (g, '10 cm baseline loss', R[10]['degradation']['rel'], '14.1'),
            (g, '30 cm baseline loss', R[30]['degradation']['rel'], '32.3'),
            (g, 'clean rung content', R[0]['content']['rel'], '0.41'),
            (g, 'clean rung content p', R[0]['content']['p'], '0.60'),
            (g, 'clean rung vs no-pose', R[0]['pose_vs_nopose']['rel'], '1.97'),
            (g, '10 cm tracker position error (cm)', e10['pos_err_median_m'] * 100, '7.5'),
            (g, '10 cm tracker velocity error', e10['vel_err_median_ms'], '0.23'),
            (g, '30 cm tracker position error (cm)', e30['pos_err_median_m'] * 100, '21'),
            (g, '30 cm tracker velocity error', e30['vel_err_median_ms'], '0.46'),
            (g, 'median pedestrian speed', e30['speed_median_ms'], '1.1'),
        ]
    HT = HN and {r['sigma_deg']: r for r in HN['rungs']}
    if HT:
        g = 'onboard-heading ladder, Table onboard'
        checks += [
            (g, 'sigma=15 vs no-pose', abs(HT[15]['pose_vs_nopose']['rel']), '0.39'),
            (g, 'sigma=30 vs no-pose', HT[30]['pose_vs_nopose']['rel'], '1.34'),
        ]
    PS = load('position_noise_subpopulation.json')
    if PS:
        g = 'onboard-track ladder by speed (Sec. IV-E)'
        w = PS['walking']
        checks += [
            (g, 'walking n', PS['_meta']['sizes']['walking'], '8{,}757'),
            (g, f'walking 10 cm content (n={w["pn10"]["content"]["n"]})', abs(w['pn10']['content']['rel']), '3.3'),
            (g, f'walking 10 cm p (n={w["pn10"]["content"]["n"]})', w['pn10']['content']['p'], '0.008'),
            (g, f'walking 30 cm content (n={w["pn30"]["content"]["n"]})', abs(w['pn30']['content']['rel']), '6.8'),
            (g, f'walking 30 cm p (n={w["pn30"]["content"]["n"]})', w['pn30']['content']['p'], '0.0002'),
            (g, 'standing n', PS['_meta']['sizes']['stationary'], '3{,}336'),
        ]
    SF = load('standing_finetune.json')
    if SF:
        g = 'standing-only fine-tune (Sec. V-B)'
        f5 = SF['stationary'][f'ft_epoch{SF["_meta"]["final_epoch"]}']
        checks += [
            (g, 'content after fine-tune', abs(f5['content']['rel']), '1.2'),
            (g, 'content p', f5['content']['p'], '0.28'),
            (g, 'change vs init (pp)', f5['did']['rel'], '1.5'),
            (g, 'change vs init p', f5['did']['p'], '0.36'),
        ]
    LC = load('label_causality.json')
    if LC:
        g = 'offboard labels (Sec. IV-E)'
        assert LC['velocity_is_exact_backward_difference']
        checks += [(g, 'pedestrian time steps checked', LC['n_timesteps'], '954{,}751')]
    if EF and EF.get('rungs'):
        g = 'early-fusion ladder at 25k, corrupted rungs (Sec. V-C)'
        R = {round(r['median_err_deg']): r for r in EF['rungs']}
        checks += [
            (g, f'5.0 deg rel (n={R[5]["n"]})', abs(R[5]['rel']), '19.4'),
            (g, f'10.0 deg rel (n={R[10]["n"]})', abs(R[10]['rel']), '15.5'),
        ]

    bad, group = [], None
    for g, label, val, written in checks:
        if g != group:
            print(f'\n{g}')
            print('-' * 74)
            group = g
        if isinstance(val, int):
            rounded_ok = written.replace('{,}', '') == str(val)
        else:
            # Round to the precision the PAPER uses, not to a fixed one decimal. A
            # hardcoded '%.1f' cannot express a two-decimal skill (0.46) or a whole
            # percentage (84), and reported both as mismatches when they were right.
            plain = written.replace('{,}', '')
            dp = len(plain.split('.')[1]) if '.' in plain else 0
            rounded_ok = f'{val:.{dp}f}' == plain
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
