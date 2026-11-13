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
    # --- retired 2026-11-12, after a review of the paper against the code -------------
    dict(pat=r'Degrading the box heading|Corrupting the box heading|box-heading quality|'
             r'However the heading is\s+degraded|onboard-heading\s+ladder|Nor does degrading it',
         why='the heading-noise runs rotate the agent-centric frame but leave the clean '
             'per-timestep heading in the channel (it reads -eps at the prediction frame, '
             'waymo_pose_dataset.py transform_trajs_to_center_coords), and the pose root is '
             'in world coordinates (AGENT_CENTRIC_POSE_ROT is unset in every headnoise '
             'config). They do not test a degraded heading',
         instead='a frame-rotation control; whether facing pays under an onboard-quality '
                 'heading is untested'),
    dict(pat=r'pose root,?\s+rotated into the same corrupted frame|map and pose root\s+all inherit',
         why='obj_poses is identical with and without heading noise; the root is never '
             'rotated into the frame in those configs',
         instead='the pose root stays in world coordinates'),
    dict(pat=r'fusion residual\s+is zero-initialised',
         why='the zero-initialised residual exists only under POSE_FUSER_RESIDUAL, which '
             'only the four fine-tune configs set; every from-scratch arm uses a plain MLP '
             'fuser with default initialisation',
         instead='drop the clause'),
    dict(pat=r'is absent below|track pay only where the sensor|and\s+only there \(Sec',
         why='the <30-returns stratum reads -0.3% at best epoch but -1.4 to -2.1% (n.s.) at '
             'fixed epochs 26/28/30, and the >=200 result is one of eight strata, unreplicated',
         instead='"appears to", with the caveat, wherever the survivor is stated'),
    dict(pat=r'minADE is\s+\$0\.091\$',
         why='0.091 m is the 2 s error of the eight-seed replication, while the oracle '
             'gain quoted beside it (1.4 cm, 14.7%) is taken from the 25k baseline, '
             'which reads 0.097 m',
         instead='0.097 m at 25k scenes'),
    dict(pat=r'queries by predicted likelihood: a better',
         why='the decoder keeps six modes by non-maximum suppression on the score with a '
             '2.5 m endpoint radius (NMS_DIST_THRESH), not by a plain top-six',
         instead='name the suppression and its radius'),
    dict(pat=r'fixed at \$5\{,\}171\$ scenes and \$16\{,\}486\$ pedestrians',
         why='the 5,171 scenes hold 17,218 predicted pedestrians; 732 have no valid future '
             'step and are not scored, which leaves 16,486 in 4,648 scenes',
         instead='give both counts'),
    dict(pat=r'clean pipeline lies somewhere between',
         why='nothing shows the fit adds body information: fitted poses score 92.7 mm '
             'against 87.6 mm pre-fit on the same 347 labelled frames',
         instead='state the MPJPE comparison'),
    dict(pat=r'pedestrian-resampling bootstrap gives a visibly|under a\s+pedestrian-resampling bootstrap \(\$p\{=\}0\.037|'
             r'pedestrian-resampling bootstrap makes the branch',
         why='the +-1.2% interval, p=0.037, p=0.016 and p=0.006 are all from the '
             'HIERARCHICAL bootstrap (multiseed_analysis.json pairs/*/hierarchical); the '
             'pedestrian-only one is far tighter',
         instead='"hierarchical bootstrap"'),

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
         instead='pose adds nothing to an offboard track; on a causal one the fitted pose mostly '
                 'restates the label and the pre-fit pose pays on well-resolved bodies'),
    dict(pat=r'this ladder does not degrade them',
         why='positions are degraded by the onboard-track ladder (Sec. IV-E)',
         instead='point to Sec. IV-E'),
    dict(pat=r'does not measurably improve pedestrian trajectory accuracy\.',
         why='unscoped: the null holds on offboard tracks, not on a causal tracker\'s',
         instead='scope it to WOMD\'s offboard tracks'),
    # --- retired 2026-11-12, when the pre-fit control landed --------------------------
    # The fitted poses come out of a motion-prior fit that matches the gait to the WOMD
    # box track, so they carry the offboard speed. On pre-fit poses the 30 cm content
    # effect is -1.34% (p=0.10) against -5.32%, and on walkers -0.8% against -6.8%.
    dict(pat=r'clean track cannot reach the model',
         why='it does, through the pose pipeline: body pose alone predicts the clean speed '
             'with R^2 0.94 after the fit and 0.44 before',
         instead='the label leak paragraph of Sec. IV-E'),
    dict(pat=r'not with a\s+causal one|the same body pays|Pose now pays|same body that adds '
             r'nothing to an offboard track adds\s+measurably',
         why='three quarters of the onboard-track gain is the fit restating the offboard '
             'track; pre-fit pose does not beat no-pose overall (+0.36%)',
         instead='pre-fit poses pay only on well-resolved bodies (-3.47% at >=200 returns)'),
    dict(pat=r'gait carries the\s+speed the tracker|where gait\s+acts|a stride\s+carries speed',
         why='the walker effect was the leak: -6.8% with fitted poses, -0.8% (p=0.36) pre-fit',
         instead='what survives follows LiDAR returns, not walking speed'),
    dict(pat=r'this one held',
         why='it held across seeds because a leak is as repeatable as a signal',
         instead='report the pre-fit control beside it'),
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


# Numbers that left the paper in the cuts to eight pages (2026-11-12, two rounds). Their results stay
# in experiments/ and their checks stay below, but a check whose number is no longer
# written can only pass by coincidence (the test is "the string occurs somewhere"), so
# these are skipped by name instead of being left to pass or fail at random. Remove an
# entry when its number returns to the text. An entry that matches no check is reported.
CUT = {
    ('facing (ladder units)', 'SMPL p25'),
    ('facing (ladder units)', 'SMPL p75'),
    ('facing (ladder units)', 'SMPL p90'),
    ('detection floor (Table II)', 'rung 0.0 MDE (n=3)'),
    ('detection floor (Table II)', 'rung 5.0 MDE (n=3)'),
    ('detection floor (Table II)', 'rung 10.0 MDE (n=5)'),
    ('detection floor (Table II)', 'high rungs MDE low'),
    ('detection floor (Table II)', 'high rungs MDE high'),
    ('detection floor (Table II)', 'high rungs rel most negative'),
    ('detection floor (Table II)', 'high rungs rel most positive'),
    ('detection floor (Table II)', 'rung 24.9 effect the 579 ladder should have expected'),
    ('detection floor (Table II)', 'rung 24.9 MDE at 579'),
    ('oracle ladder 25k (Table I)', 'persistent_range_pct low'),
    ('oracle ladder 25k (Table I)', 'persistent_range_pct high'),
    ('oracle ladder 25k (Table I)', 'amplitude_range_pct low'),
    ('oracle ladder 25k (Table I)', 'amplitude_range_pct high'),
    ('oracle ladder 25k (Table I)', 'half_life_range_deg low'),
    ('oracle ladder 25k (Table I)', 'half_life_range_deg high'),
    ('horizon decomposition (Sec. V-A)', 'abs gain widest mode-spread quartile'),
    ('horizon decomposition (Sec. V-A)', 'abs gain other quartiles low'),
    ('horizon decomposition (Sec. V-A)', 'abs gain other quartiles high'),
    ('per-joint recovery (Table V)', 'knee/ankle skill low'),
    ('per-joint recovery (Table V)', 'knee/ankle skill high'),
    ('per-joint recovery (Table V)', 'elbow skill low'),
    ('per-joint recovery (Table V)', 'elbow skill high'),
    # 2026-11-12, second cut: room for the table and figure of where the pre-fit gain sits
    ('detection floor (Table II)', 'rung 15.1 effect the 579 ladder should have expected'),
    ('heading-removal counterfactual (Sec. V-B)', 'movers DiD agent-level low'),
    ('heading-removal counterfactual (Sec. V-B)', 'movers DiD agent-level high'),
    ('heading-removal counterfactual (Sec. V-B)', 'stationary DiD agent-level low'),
    ('heading-removal counterfactual (Sec. V-B)', 'stationary DiD agent-level high'),
    ('heading-removal counterfactual (Sec. V-B)', 'top-decile turn DiD agent-level'),
    ('heading-removal counterfactual (Sec. V-B)', 'top-decile turn agent-level low'),
    ('heading-removal counterfactual (Sec. V-B)', 'top-decile turn agent-level high'),
    ('per-joint recovery (Table V)', 'leg spread ratio low'),
    ('per-joint recovery (Table V)', 'leg spread ratio high'),
    ('per-joint recovery (Table V)', 'arm spread ratio low'),
    ('per-joint recovery (Table V)', 'arm spread ratio high'),
    ('early-fusion oracle at 25k (Sec. IV-B)', 'theta=0 CI lo (n=3)'),
    ('early-fusion oracle at 25k (Sec. IV-B)', 'theta=0 CI hi (n=3)'),
    ('early-fusion ladder at 25k, corrupted rungs (Sec. V-C)', '5.0 deg rel (n=3)'),
    ('early-fusion ladder at 25k, corrupted rungs (Sec. V-C)', '10.0 deg rel (n=3)'),
    ('standing-only fine-tune (Sec. V-B)', 'content after fine-tune'),
    ('standing-only fine-tune (Sec. V-B)', 'content p'),
    ('standing-only fine-tune (Sec. V-B)', 'change vs init (pp)'),
    ('standing-only fine-tune (Sec. V-B)', 'change vs init p'),
    ('2 s headroom (Sec. V-F)', 'constant-velocity ADE at 2 s (m)'),
    ('oracle ladder 25k (Table I)', 'single-exp residual 1.5'),
    ('oracle ladder 25k (Table I)', 'single-exp residual 3.2'),
    ('oracle ladder 25k (Table I)', 'single-exp residual 0.3'),
    ('oracle ladder 25k (Table I)', 'single-exp residual 2.5'),
}


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
            if h['median_err_deg'] == 44.4:
                r = next(r for r in D25['rungs'] if r['median_err_deg'] == 44.4)
                checks += [(g, f'held-out 44.4 p (n={r["n"]})', r['p'], '0.013'),
                           (g, f'held-out 44.4 p, table (n={r["n"]})', r['p'], '0.0127'),
                           (g, f'held-out 44.4 CI low (n={r["n"]})', abs(r['lo']), '10.3'),
                           (g, f'held-out 44.4 CI high (n={r["n"]})', abs(r['hi']), '3.5')]
                # "about half of it is left"
                assert 0.45 <= h['slow_share'] < 0.60, h['slow_share']
                assert r['hi'] < 0, r['hi']    # "every rung converts, out to 44.4"
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
        # the baseline the gain is taken from, so the three figures share one source
        checks.append((g, '2 s baseline minADE (m)',
                       sum(x['minADE_2s']['base'] for x in r0) / len(r0), '0.097'))
    CV = load('cv_baseline.json')
    if CV:
        checks += [('2 s headroom (Sec. V-F)', 'constant-velocity ADE at 2 s (m)',
                    CV['cv_ade_2s'], '0.310'),
                   ('validation set (Sec. III-D)', 'predicted pedestrians',
                    CV['n_agents'], '17{,}218')]

    EF = load('detection_floor_ef25k.json')
    if EF and EF.get('rungs'):
        g = 'early-fusion oracle at 25k (Sec. IV-B)'
        r0 = min(EF['rungs'], key=lambda r: r['median_err_deg'])
        # n is in the label on purpose, so a change in the seed count shows up as a
        # mismatch.
        checks += [
            (g, f'theta=0 rel (n={r0["n"]})', abs(r0['rel']), '31.2'),
            (g, f'theta=0 p (n={r0["n"]})', r0['p'], '0.0002'),
            (g, f'theta=0 CI lo (n={r0["n"]})', abs(r0['lo']), '32.9'),
            (g, f'theta=0 CI hi (n={r0["n"]})', abs(r0['hi']), '29.5'),
        ]

    HN = load('heading_noise_ladder.json')
    if HN:
        # 2026-11-12: this is a frame-rotation control, not a degraded heading (the
        # heading channel still carries the clean heading in the rotated frame, and the
        # pose root is in world coordinates). The paper no longer quotes the median added
        # error, the recovery cap or the second learning rate, so they are not checked.
        g = 'frame-rotation control (Sec. V-B)'
        R = {r['sigma_deg']: r for r in HN['rungs']}
        checks += [
            (g, 'sigma=15 baseline loss', R[15]['degradation']['rel'], '5.8'),
            (g, 'sigma=30 baseline loss', R[30]['degradation']['rel'], '12.6'),
            (g, 'sigma=15 content', abs(R[15]['content']['rel']), '0.14'),
            (g, 'sigma=15 content p', R[15]['content']['p'], '0.90'),
            (g, 'sigma=30 content', abs(R[30]['content']['rel']), '0.11'),
            (g, 'sigma=30 content p', R[30]['content']['p'], '0.93'),
            (g, 'clean rung vs workstation (max |rel|)',
             max(abs(x['rel']) for x in HN['reproduction'].values()), '0.4'),
        ]
    HS = load('heading_noise_subpopulation.json')
    if HS:
        g = 'frame-rotation control, stationary (Sec. IV-D)'
        st = HS['stationary']
        checks += [
            (g, 'stationary n', HS['_meta']['sizes']['stationary'], '3{,}336'),
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
    PF = {r['noise_cm']: r for r in (PN or {}).get('prefit', [])}
    if 30 in PF and 'content' in PF[30]:
        g = 'pre-fit control at 30 cm (Sec. IV-E, Table onboard)'
        p = PF[30]
        n, c, d = p['n'], p['content'], p['content_prefit_minus_final']
        checks += [
            (g, f'content (n={n})', abs(c['rel']), '1.34'),
            (g, f'content, abstract (n={n})', abs(c['rel']), '1.3'),
            (g, f'content p (n={n})', c['p'], '0.10'),
            (g, f'content CI lo (n={n})', abs(c['lo']), '3.1'),
            (g, f'content CI hi (n={n})', c['hi'], '0.4'),
            (g, f'vs no-pose (n={n})', p['pose_vs_nopose']['rel'], '0.36'),
            (g, f'vs no-pose p (n={n})', p['pose_vs_nopose']['p'], '0.54'),
            (g, 'shuffled arms agree', p['prefit_shuffled_vs_final_shuffled']['rel'], '0.40'),
            (g, 'shuffled arms agree p', p['prefit_shuffled_vs_final_shuffled']['p'], '0.54'),
            (g, 'points the fit accounts for', d['rel'], '3.98'),
            (g, 'points the fit accounts for, p', d['p'], '0.0001'),
            (g, 'per-seed min', min(d['per_seed_rel']), '3.5'),
            (g, 'per-seed max', max(d['per_seed_rel']), '4.9'),
        ]
        assert p['seeds'] == [101, 202, 303, 404, 505], p['seeds']
        # "Three quarters of that gain is the label"
        assert 0.70 <= 1 - p['surviving_share'] <= 0.80, p['surviving_share']
        assert c['p'] > 0.05 and p['pose_vs_nopose']['rel'] > 0     # "does not beat no pose"
    PSF = (load('position_noise_subpopulation.json') or {}).get('prefit', {}).get('pn30')
    if PSF:
        g = 'pre-fit control by LiDAR returns (Sec. IV-E, Table onboard)'
        lo, m1, m2, hi, w = (PSF[k] for k in ('pts<30', 'pts30-100', 'pts100-200',
                                             'pts>=200', 'walking'))
        gr = PSF['returns_gradient']
        fe = [abs(x['pts>=200']['rel']) for x in PSF['fixed_epoch'].values()]
        checks += [
            (g, 'walkers, pre-fit content', abs(w['content']['rel']), '0.8'),
            (g, 'walkers, pre-fit content p', w['content']['p'], '0.36'),
            (g, '<30 returns content', abs(lo['content']['rel']), '0.34'),
            (g, '<30 returns content, text', abs(lo['content']['rel']), '0.3'),
            (g, '<30 returns p', lo['content']['p'], '0.72'),
            (g, '<30 returns vs no-pose', lo['pose_vs_nopose']['rel'], '1.82'),
            (g, '30-100 returns content', abs(m1['content']['rel']), '1.0'),
            (g, '100-200 returns content', abs(m2['content']['rel']), '1.4'),
            (g, '>=200 returns n', hi['n_agents'], '2{,}365'),
            (g, '>=200 returns content', abs(hi['content']['rel']), '3.47'),
            (g, '>=200 returns content, abstract', abs(hi['content']['rel']), '3.5'),
            (g, '>=200 returns p', hi['content']['p'], '0.010'),
            (g, '>=200 returns CI lo', abs(hi['content']['lo']), '5.6'),
            (g, '>=200 returns CI hi', abs(hi['content']['hi']), '1.4'),
            (g, '>=200 returns vs no-pose', abs(hi['pose_vs_nopose']['rel']), '1.99'),
            (g, '>=200 returns vs no-pose p', hi['pose_vs_nopose']['p'], '0.041'),
            (g, 'gradient, top minus bottom (pp)', abs(gr['prefit']['top_minus_bottom']['rel']), '3.1'),
            (g, 'gradient p', gr['prefit']['top_minus_bottom']['p'], '0.004'),
            (g, 'fixed-epoch min, >=200', min(fe), '3.2'),
            (g, 'fixed-epoch max, >=200', max(fe), '4.6'),
            (g, 'fitted, <30 returns', abs(lo['content_final_same_seeds']['rel']), '5.3'),
            (g, 'fitted, 30-100 returns', abs(m1['content_final_same_seeds']['rel']), '5.4'),
            (g, 'fitted, 100-200 returns', abs(m2['content_final_same_seeds']['rel']), '4.1'),
            (g, 'fitted, >=200 returns', abs(hi['content_final_same_seeds']['rel']), '6.0'),
            (g, 'fitted gradient p', gr['final']['top_minus_bottom']['p'], '0.41'),
        ]
        # "better than the bottom on every seed" and "at every fixed late epoch"
        assert all(x < 0 for x in gr['prefit']['top_minus_bottom']['per_seed'])
        assert all(x['pts>=200']['p'] < 0.05 for x in PSF['fixed_epoch'].values())
    LK = load('pose_fit_leak_speed.json')
    if LK:
        g = 'label leak through the pose fit (Sec. IV-E)'
        a = LK['A']['v_win']
        w30, w10 = LK['C']['v_win']['s30']['all'], LK['C']['v_win']['s10']['all']
        sparse = LK['A']['by_points']['pts<30']
        checks += [
            (g, 'pedestrians', LK['n'], '11{,}588'),
            (g, 'R2 fitted', a['all']['r2_final'], '0.94'),
            (g, 'R2 pre-fit', a['all']['r2_raw'], '0.44'),
            (g, 'R2 fitted, movers <30 returns', sparse['r2_final'], '0.78'),
            (g, '30 cm: share removed, fitted', w30['mse_removed']['tracker+final'] * 100, '71'),
            (g, '30 cm: share removed, stranger', w30['mse_removed']['tracker+final_shuf'] * 100, '1'),
            (g, '10 cm: share removed, fitted', w10['mse_removed']['tracker+final'] * 100, '21'),
        ]
        # The trees are fitted with threaded numerics, so a rerun moves the weak pre-fit
        # numbers in the second decimal (-0.03 and -0.05, 21.7% and 21.5%, 1.4% and 1.7% on
        # two runs). The paper states those three in words, checked here as ranges.
        assert sparse['r2_raw'] < 0, sparse['r2_raw']                        # "less than zero"
        assert 0.18 <= w30['mse_removed']['tracker+raw'] <= 0.25             # "about a fifth"
        assert 0 <= w10['mse_removed']['tracker+raw'] < 0.02                 # "under 2%"
    HT = HN and {r['sigma_deg']: r for r in HN['rungs']}
    if HT:
        g = 'frame-rotation control, Table onboard'
        checks += [
            (g, 'sigma=15 vs no-pose', abs(HT[15]['pose_vs_nopose']['rel']), '0.39'),
            (g, 'sigma=30 vs no-pose', HT[30]['pose_vs_nopose']['rel'], '1.34'),
        ]
    # --- added after the 2026-11-12 review of the paper against code and results ------
    MS = load('multiseed_analysis.json')
    if MS:
        from scipy import stats as _st
        import numpy as _np
        C = MS['cells']

        def _pair(a, b, key):
            A, B = C[a]['seeds'], C[b]['seeds']
            ss = sorted(set(A) & set(B))
            x = _np.array([A[q][key] for q in ss], float)
            y = _np.array([B[q][key] for q in ss], float)
            return (x.mean() / y.mean() - 1) * 100, float(_st.ttest_rel(x, y)[1])
        g = 'content test at epoch 30 (Sec. IV-B)'
        for name, a, b, rel, pv in (
                ('579', 'map_xattn_pe_fix', 'map_shufflepose_fix', '5.9', '0.16'),
                ('5k', 'scale5k_lr5_xattn_pe', 'scale5k_lr5_shufflepose', '4.7', '0.07'),
                ('25k', 'scale25k_xattn_pe', 'scale25k_shufflepose', '1.7', '0.60')):
            r_, p_ = _pair(a, b, 'last_eval_minADE')
            assert r_ < 0, (name, r_)                      # the paper prints them negative
            checks += [(g, f'{name} content', abs(r_), rel), (g, f'{name} p', p_, pv)]
        g = 'agent-frame-root content test (Sec. III-B)'
        r_, p_ = _pair('fa_agentrot', 'fa_agentrot_shuf', 'best_minADE')
        checks += [(g, 'content', r_, '0.68'), (g, 'p', p_, '0.49')]
        g = 'miss rate at 2 m, content (Sec. IV-A)'
        checks += [(g, '25k', _pair('scale25k_xattn_pe', 'scale25k_shufflepose', 'best_MR2m')[0], '2.2'),
                   (g, 'fine-tuned', _pair('ft_xattn_pe_fix', 'ft_shufflepose_fix', 'best_MR2m')[0], '4.3')]
    FO = load('facing_vs_observation_length.json')
    if FO:
        checks.append(('stationary onset (Sec. V-B)', 'onset events', FO['overall']['n'], '1{,}705'))
    MP = load('pose_accuracy_fit_vs_prefit.json')
    if MP:
        g = 'fitted vs pre-fit pose accuracy (Sec. IV-E)'
        d, J = MP['fitted_minus_prefit_mm']['root_aligned'], MP['per_joint_root_aligned_mm']
        checks += [(g, 'fitted MPJPE mm', MP['fitted']['root_aligned_mm']['mean'], '94.8'),
                   (g, 'pre-fit MPJPE mm', MP['prefit']['root_aligned_mm']['mean'], '90.7'),
                   (g, 'frames', MP['n_frames'], '1{,}789')]
        assert d['lo'] > 0                                 # "less accurate"
        # "the loss is at the knees and ankles": every leg joint loses more than any other
        gain = {k: v['fitted'] - v['prefit'] for k, v in J.items()}
        legs = [k for k in gain if 'KNEE' in k or 'ANKLE' in k]
        assert min(gain[k] for k in legs) > 10 > 4 > max(v for k, v in gain.items() if k not in legs)
    HD = load('horizon_decomposition.json')
    if HD:
        checks.append(('metric definition (Sec. III-C)', 'share with a full 8 s future (%)',
                       100 * HD['summary']['n_full_future'] / HD['summary']['n_paired'], '36'))
    TE = load('position_noise_tracker_error_0.3.json')
    if TE:
        checks.append(('simulated tracker (Sec. IV-E)', 'pedestrians measured', TE['agents'], '675'))
    if PN and PN.get('prefit'):
        g = 'pre-fit control at fixed epochs (Sec. IV-E)'
        fe = next(r for r in PN['prefit'] if r['noise_cm'] == 30)['fixed_epoch']
        for e, rel in (('26', '1.4'), ('28', '2.6'), ('30', '1.5')):
            assert fe[e]['content']['rel'] < 0
            checks.append((g, f'epoch {e} content', abs(fe[e]['content']['rel']), rel))
        checks.append((g, 'epoch 28 p', fe['28']['content']['p'], '0.02'))
        assert fe['26']['content']['p'] > 0.05 and fe['30']['content']['p'] > 0.05   # "the second only"
    PSr = load('position_noise_subpopulation.json')
    if PSr and PSr.get('prefit', {}).get('pn30'):
        g = 'pre-fit control by returns, robustness (Sec. IV-E)'
        c30 = PSr['prefit']['pn30']
        lo = [c30['fixed_epoch'][e]['pts<30'] for e in ('26', '28', '30')]
        hi = [c30['fixed_epoch'][e]['pts>=200'] for e in ('26', '28', '30')]
        gaps = [abs(h['rel'] - l['rel']) for h, l in zip(hi, lo)]
        assert all(x['p'] > 0.05 for x in lo)              # "(n.s.)"
        top = c30['pts>=200']
        checks += [
            (g, '<30 returns, fixed-epoch min', min(abs(x['rel']) for x in lo), '1.4'),
            (g, '<30 returns, fixed-epoch max', max(abs(x['rel']) for x in lo), '2.1'),
            (g, 'gap at fixed epochs, min', min(gaps), '1.7'),
            (g, 'gap at fixed epochs, max', max(gaps), '2.5'),
            (g, 'stranger arms differ in top stratum', top['shuffled_vs_final_shuffled']['rel'], '0.9'),
            (g, 'stranger arms differ, p', top['shuffled_vs_final_shuffled']['p'], '0.015'),
            (g, 'pre-fit pose vs fitted stranger', abs(top['pose_vs_final_shuffled']['rel']), '2.6'),
            (g, 'pre-fit pose vs fitted stranger, p', top['pose_vs_final_shuffled']['p'], '0.03'),
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

    rows = []     # (group, label, a table row rebuilt from the artifact, verbatim)
    DY = (load('position_noise_dynamics.json') or {})
    if DY.get('returns>=200'):
        g = 'pre-fit gain by tracker error and manoeuvre (Sec. IV-E, Table dyn)'
        hi, al = DY['returns>=200'], DY['all_returns']
        Q = ('Q1', 'Q2', 'Q3', 'Q4')
        man = ('cv_residual', 'heading_total_var', 'speed_range', 'max_accel')
        KEY, TREND = 'pose_vs_nopose', 'Q4_minus_Q1_vs_nopose'    # what Table dyn prints
        v, x = hi['tracker_vel_err'], hi['tracker_pos_err']
        fv = hi['full_history_only']['tracker_vel_err'][TREND]
        cvf = hi['cv_residual']
        q1 = [hi[a]['Q1'][KEY]['rel'] for a in man]
        # "largest in the lowest quartile of every measure"
        assert all(hi[a]['Q1'][KEY]['rel'] == min(hi[a][q][KEY]['rel'] for q in Q) for a in man)
        # against a stranger's pose: "ahead in all 24 quartiles"
        assert all(hi[a][q]['content']['rel'] < 0 for a in DY['_meta']['axes'] for q in Q)
        assert len({hi[a][q]['n_agents'] for a in hi if a in DY['_meta']['axes'] for q in Q}) == 1
        checks += [
            (g, 'pedestrians', hi['all']['n_agents'], '2{,}272'),
            (g, 'pedestrians per quartile', v['Q1']['n_agents'], '568'),
            (g, 'velocity error Q1', v['Q1'][KEY]['rel'], '0.4'),
            (g, 'velocity error Q4', abs(v['Q4'][KEY]['rel']), '3.3'),
            (g, 'velocity error trend (pp)', abs(v[TREND]['rel']), '3.7'),
            (g, 'velocity error trend p', v[TREND]['p'], '0.013'),
            (g, 'position error Q1', x['Q1'][KEY]['rel'], '0.1'),
            (g, 'position error Q4', abs(x['Q4'][KEY]['rel']), '5.0'),
            (g, 'position error trend (pp)', abs(x[TREND]['rel']), '5.1'),
            (g, 'position error trend p', x[TREND]['p'], '0.019'),
            (g, 'full histories, velocity trend', abs(fv['rel']), '4.0'),
            (g, 'full histories, velocity trend p', fv['p'], '0.019'),
            (g, 'all returns, velocity trend', abs(al['tracker_vel_err'][TREND]['rel']), '1.0'),
            (g, 'all returns, velocity trend p', al['tracker_vel_err'][TREND]['p'], '0.08'),
            (g, 'all returns, position trend', abs(al['tracker_pos_err'][TREND]['rel']), '1.9'),
            (g, 'all returns, position trend p', al['tracker_pos_err'][TREND]['p'], '0.10'),
            (g, 'manoeuvre Q1, smallest gain', min(abs(r) for r in q1), '3.4'),
            (g, 'manoeuvre Q1, largest gain', max(abs(r) for r in q1), '5.6'),
            (g, 'CV residual trend (pp)', cvf[TREND]['rel'], '3.8'),
            (g, 'CV residual trend p', cvf[TREND]['p'], '0.048'),
            (g, 'speed range trend (pp)', hi['speed_range'][TREND]['rel'], '4.1'),
            (g, 'speed range trend p', hi['speed_range'][TREND]['p'], '0.019'),
            (g, 'vs stranger, velocity trend (pp)', abs(v['Q4_minus_Q1']['rel']), '3.1'),
            (g, 'vs stranger, velocity trend p', v['Q4_minus_Q1']['p'], '0.001'),
            (g, 'vs stranger, position trend (pp)', abs(x['Q4_minus_Q1']['rel']), '5.5'),
            (g, 'vs stranger, position trend p', x['Q4_minus_Q1']['p'], '0.001'),
            (g, 'vs stranger, manoeuvre trend, smallest p',
             min(hi[a]['Q4_minus_Q1']['p'] for a in man), '0.19'),
            (g, 'fitted, velocity trend p', v['Q4_minus_Q1_fitted']['p'], '0.08'),
            (g, 'fitted, position trend p', x['Q4_minus_Q1_fitted']['p'], '0.87'),
            (g, 'fitted, CV residual Q1', abs(cvf['Q1']['content_fitted']['rel']), '8.4'),
            (g, 'fitted, CV residual Q4', abs(cvf['Q4']['content_fitted']['rel']), '3.5'),
            (g, 'fitted, CV residual trend p', cvf['Q4_minus_Q1_fitted']['p'], '0.013'),
        ]

        def cell(c):
            return f'${c["rel"]:+.1f}' + ('^{*}' if c['p'] < 0.05 else '') + '$'

        for a, name in (('tracker_vel_err', 'Velocity'), ('tracker_pos_err', 'Position'),
                        ('cv_residual', 'Constant-velocity residual'),
                        ('heading_total_var', 'Heading variation'),
                        ('speed_range', 'Speed range'), ('max_accel', 'Peak acceleration')):
            t = hi[a][TREND]
            pt = f'{t["p"]:.3f}' if t['p'] < 0.1 else f'{t["p"]:.2f}'
            rows.append((g, f'row: {name}', ' & '.join(
                [name] + [cell(hi[a][q][KEY]) for q in Q]
                + [f'${t["rel"]:+.1f}$ (${pt}$)']) + r' \\'))
    OX = load('onboard_examples.json')
    if OX:
        g = 'onboard examples (Fig. onboardex)'
        top, bot = OX['top_quartile'], OX['bottom_quartile']
        checks += [
            (g, 'pedestrians in the quartile', top['n'], '568'),
            (g, 'largest error: own on all seeds', top['own_all_seeds'], '43'),
            (g, 'largest error: stranger on all seeds', top['stranger_all_seeds'], '28'),
            (g, 'smallest error: own on all seeds', bot['own_all_seeds'], '33'),
            (g, 'smallest error: stranger on all seeds', bot['stranger_all_seeds'], '32'),
        ]
        assert len(OX['shown']) == 4 and OX['epoch'] == 30    # "four", "final epoch"
    GA = load('graphical_abstract.json')
    if GA:
        # Fig. gabs draws its bars from position_noise_dynamics.json. These tie what it
        # prints to its caption and to the sentence of Sec. IV-E that states the same.
        g = 'graphical abstract (Fig. gabs, Sec. IV-E)'
        al, up = GA['all'], GA['upper_half']
        checks += [
            (g, 'own vs no pose, all', abs(al['own_vs_nopose']['rel']), '2.1'),
            (g, 'own vs no pose, all, p', al['own_vs_nopose']['p'], '0.039'),
            (g, 'own vs no pose, upper half', abs(up['own_vs_nopose']['rel']), '4.0'),
            (g, 'own vs no pose, upper half, p', up['own_vs_nopose']['p'], '0.003'),
            (g, 'pedestrians', al['n_pedestrians'], '2{,}272'),
        ]
        # the half the graphic leaves out, which Sec. IV-E states beside the other
        lo = (load('position_noise_dynamics.json') or {}).get('returns>=200', {}).get(
            GA['axis'], {}).get('lower_half')
        if lo:
            checks += [(g, 'own vs no pose, lower half', lo['pose_vs_nopose']['rel'], '0.1'),
                       (g, 'own vs no pose, lower half, p', lo['pose_vs_nopose']['p'], '0.96')]
            assert lo['pose_vs_nopose']['rel'] > 0
        assert GA['noise_cm'] == 30 and GA['min_returns'] == 200 and GA['axis'] == 'tracker_vel_err'
        assert up['n_pedestrians'] * 2 == al['n_pedestrians']                  # "the half"
        # the asterisks it prints, and "a stranger's pose does not" lower minADE
        assert al['own_vs_nopose']['rel'] < 0 and up['own_vs_nopose']['rel'] < 0
        assert al['own_vs_nopose']['p'] < 0.05 and up['own_vs_nopose']['p'] < 0.05
        assert al['stranger_vs_nopose']['rel'] > 0 and up['stranger_vs_nopose']['rel'] > 0
        assert al['stranger_vs_nopose']['p'] >= 0.05 and up['stranger_vs_nopose']['p'] >= 0.05
        # "the example is from Fig. onboardex", and the order its three lines show
        if OX:
            e = OX['shown']['abcd'.index(GA['example']['panel'])]
            assert (e['scenario_id'], e['object_id']) == (GA['example']['scenario_id'],
                                                          GA['example']['object_id'])
        d = GA['example']['minADE_drawn']
        assert d['own'] < d['nopose'] < d['stranger'], d
    PM = load('pointcloud_mesh_examples.json')
    if PM:
        # Fig. pcmesh cites no number beyond the stratum cuts. What Sec. IV-E says about
        # its two end panels has to hold for the frames that are drawn.
        shown = {e['stratum']: e for e in PM['shown']}
        assert PM['strata_cut_at'] == [30, 100, 200] and len(shown) == 4
        assert shown['<30']['frame_returns'] < 30 and shown['>=200']['frame_returns'] >= 200
        # heights as a fraction of the drawn body: knee about 0.28, chin about 0.87.
        # "between thigh and shoulder, none on the head or the lower legs"
        assert 0.30 <= shown['<30']['returns_height_min']
        assert shown['<30']['returns_height_max'] <= 0.86
        # "from head to foot"
        assert shown['>=200']['returns_height_min'] <= 0.10
        assert shown['>=200']['returns_height_max'] >= 0.90
        assert all(lo <= shown[k]['mean_returns'] < hi for k, lo, hi in
                   (('<30', 0, 30), ('30-100', 30, 100), ('100-200', 100, 200),
                    ('>=200', 200, float('inf'))))

    skipped = [c for c in checks if (c[0], c[1]) in CUT]
    stale = sorted(CUT - {(c[0], c[1]) for c in checks})
    checks = [c for c in checks if (c[0], c[1]) not in CUT]

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

    if rows:
        print('\ntable rows rebuilt from the artifacts')
        print('-' * 74)
    for g, label, row in rows:
        ok = row in tex
        print(f'  {label:<40}{"OK" if ok else "NOT IN main.tex"}')
        if not ok:
            bad.append((g, label, row, 'that row', 'NOT IN main.tex'))

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
    print(f'{len(skipped)} checks skipped: their numbers were cut from the eight-page version')
    for g, label in stale:
        print(f'  stale CUT entry, no such check: [{g}] {label}')
    if bad:
        print(f'{len(bad)} MISMATCH(ES):')
        for g, label, val, written, status in bad:
            print(f'  [{g}] {label}: artifact {val}, paper writes {written} -- {status}')
    if retired:
        print(f'{len(retired)} RETIRED CLAIM(S) PRESENT (listed above)')
    if bad or retired:
        return 1
    print(f'all {len(checks)} number checks and {len(rows)} table rows pass; '
          f'no retired claims present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
