"""
Graphical abstract: what a pedestrian's own body pose is worth on an onboard track.

The paper's positive result in one picture, the part of it that replicated on held-out
scenes. Under a simulated onboard tracker (30 cm of detection noise), with poses that
never saw the track (pre-fit), on pedestrians with at least 200 mean LiDAR returns:

  left    one pedestrian's LiDAR returns and the pre-fit body estimated from them
  middle  that pedestrian's forecast from three models, with its own pose, with a
          stranger's pose and without pose, against the label and the tracked history.
          One line per model: the seed whose error is the median of its five. The
          pedestrian is the first example of fig_onboard_examples, the largest gain of own
          pose over a stranger's, so it is selected on the effect (see that script). It
          was fixed before the no-pose predictions were looked at. Predictions are the
          stored final-epoch ones
  right   minADE with the own pose against a stranger's pose, in percent, by quartile of
          the tracker's velocity error at prediction time, on the held-out validation
          scenes at the final epoch (heldout_replication.py). Until 2026-11-13 this panel
          showed both pose arms against no pose on the original scenes at the best
          epoch (-2.1% and -4.0%). That gain is absent on the held-out scenes and on the
          original ones at the final epoch, so it is not the picture of the paper.

No number is typed in here. The bars are read from experiments/heldout_replication.json,
the example from experiments/onboard_examples_drawn.pkl (make_onboard_examples.py). An
asterisk marks a paired p below 0.05, uncorrected.

Run with an env that has smplx:
  /home/erik/anaconda3/envs/lidar-hmr/bin/python paper/iv2027/make_graphical_abstract.py
Writes paper/iv2027/fig_graphical_abstract.pdf and experiments/graphical_abstract.json.
"""
from __future__ import annotations

import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import smplx
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from make_onboard_examples import C_INK, C_OWN, C_STRANGER, C_TRACK
from make_qualitative_pointclouds import PC, SMPL_DIR, draw, frame_of, vertices

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
POPULATION, AXIS = 'heldout', 'velocity_error'
EXAMPLE = 0                      # panel (a) of fig_onboard_examples, its largest gain
# Third categorical hue, validated against the other two for colour-vision deficiency.
C_NOPOSE = '#1baf7a'
ARMS = (('own', 'own pose', C_OWN), ('stranger', "stranger's pose", C_STRANGER),
        ('nopose', 'no pose', C_NOPOSE))


def readings() -> dict:
    full = json.load(open(os.path.join(REPO, 'experiments', 'heldout_replication.json')))
    pop, meta = full[POPULATION], full['_meta']
    quart = pop[f'quartiles_of_tracker_{AXIS}']

    def pair(cell, key):
        return dict(rel=cell[key]['rel'], p=cell[key]['p'])

    whole = pop['with_a_second_of_future']         # the pedestrians the quartiles cut
    return dict(noise_cm=meta['noise_cm'], min_returns=meta['min_returns'], epoch=meta['epoch'],
                n_seeds=len(meta['seeds']), population=POPULATION, axis=AXIS,
                all=dict(n_pedestrians=whole['n_agents'], n_scenes=whole['n_scenes'],
                         own_vs_stranger=pair(whole, 'content'),
                         own_vs_nopose=pair(whole, 'own_vs_nopose'),
                         stranger_vs_nopose=pair(whole, 'stranger_vs_nopose')),
                quartiles=[dict(n_pedestrians=c['n_agents'], own_vs_stranger=pair(c, 'content'),
                                own_vs_nopose=pair(c, 'own_vs_nopose')) for c in quart['quartiles']],
                trend=dict(own_vs_stranger=pair(quart['Q4_minus_Q1'], 'content'),
                           own_vs_nopose=pair(quart['Q4_minus_Q1'], 'own_vs_nopose')))


def median_seed(ade: list) -> int:
    """The seed whose error is the median of an odd number of seeds."""
    return int(np.argsort(ade)[len(ade) // 2])


def main() -> None:
    r = readings()
    ex = pickle.load(open(os.path.join(REPO, 'experiments', 'onboard_examples_drawn.pkl'),
                          'rb'))[EXAMPLE]
    key = (ex['scenario_id'], ex['object_id'])
    pc = pickle.load(open(PC, 'rb'))
    body = frame_of(key, pc[key]['n_mean'])       # the frame closest to its mean returns
    line = {a: median_seed(ex['ade'][a]) for a, _, _ in ARMS}
    r['example'] = dict(scenario_id=key[0], object_id=key[1], panel='abcd'[EXAMPLE],
                        mean_returns=float(pc[key]['n_mean']),
                        frame_returns=int(len(body['points'])),
                        minADE_mean={a: float(np.mean(ex['ade'][a])) for a, _, _ in ARMS},
                        minADE_drawn={a: float(ex['ade'][a][line[a]]) for a, _, _ in ARMS})
    json.dump(r, open(os.path.join(REPO, 'experiments', 'graphical_abstract.json'), 'w'), indent=1)
    print(json.dumps(r, indent=1))

    plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'pdf.fonttype': 42,
                         'font.serif': ['Liberation Serif', 'DejaVu Serif']})
    fig = plt.figure(figsize=(3.45, 1.34))

    # left: the pedestrian's own body, as the estimator saw it
    model = smplx.create(SMPL_DIR, model_type='smpl', gender='neutral')
    ax = fig.add_axes([-0.09, -0.02, 0.34, 1.02], projection='3d')
    ax.patch.set_alpha(0.0)
    draw(ax, vertices(model, body['smpl']), model.faces.astype(np.int64), body['points'],
         alpha=0.42, point_size=0.6)
    fig.text(0.0, 0.985, 'Own pose', ha='left', va='top', weight='bold')
    fig.text(0.0, 0.885, f'{len(body["points"])} returns', ha='left', va='top', fontsize=6.5)

    # middle: its forecast on the tracked history, one line per model
    ax = fig.add_axes([0.185, 0.245, 0.365, 0.625])
    for a, _, col in reversed(ARMS):
        ax.plot(*np.vstack([[0, 0], ex[a][line[a]]]).T, color=col, lw=1.2,
                solid_capstyle='round', zorder=3)
    ax.plot(*ex['tracked'].T, ls='none', marker='o', ms=2.2, mfc=C_TRACK, mec='white', mew=0.3,
            zorder=4)
    ax.plot(*ex['label_history'].T, color=C_INK, lw=0.8, zorder=5)
    ax.plot(*np.vstack([[0, 0], ex['truth']]).T, color=C_INK, lw=1.6, solid_capstyle='round',
            zorder=6)
    ax.plot(0, 0, 'o', ms=3.6, mfc=C_INK, mec='white', mew=0.7, zorder=7)
    ax.margins(0.08)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color('#c9c8c2')
    ax.add_artist(AnchoredSizeBar(ax.transData, 2.0, '2 m', 'lower left', pad=0.25,
                                  borderpad=0.3, sep=2, frameon=False, color=C_INK,
                                  size_vertical=0, fontproperties=FontProperties(size=6)))
    fig.text(0.185, 0.985, 'Forecast on an onboard track', ha='left', va='top', weight='bold')
    handles = [Line2D([], [], color=col, lw=1.2, label=lab) for _, lab, col in ARMS]
    handles += [Line2D([], [], color=C_INK, lw=1.6, label='label'),
                Line2D([], [], ls='none', marker='o', ms=2.2, mfc=C_TRACK, mec='white',
                       mew=0.3, label='tracked history')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=3,
               frameon=False, fontsize=6.5, handlelength=1.3, columnspacing=0.8,
               handletextpad=0.4, labelspacing=0.22, borderaxespad=0.0, borderpad=0.1)

    # right: own pose against a stranger's, by how wrong the tracker is (held-out scenes)
    ax = fig.add_axes([0.60, 0.085, 0.395, 0.70])
    ax.set_xlim(-6.6, 3.2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    bh, ys = 0.125, (0.80, 0.62, 0.44, 0.26)
    names = ('lowest', '', '', 'highest')
    ax.text(-6.6, 0.955, 'by tracker velocity error', ha='left', va='center', fontsize=6.5)
    for y, cell, name in zip(ys, r['quartiles'], names):
        v, p = cell['own_vs_stranger']['rel'], cell['own_vs_stranger']['p']
        ax.barh(y, v, height=bh, color=C_OWN, linewidth=0, zorder=2)
        txt = ('0.0' if abs(v) < 0.05 else f'{v:+.1f}'.replace('-', '−')) + ('*' if p < 0.05 else '')
        ax.text(min(v, 0) - 0.18, y, txt, va='center', ha='right')
        ax.text(0.3, y, name, va='center', ha='left', fontsize=6.5)
    # the baseline, beside the bars only
    ax.plot([0, 0], [ys[-1] - bh / 2 - 0.035, ys[0] + bh / 2 + 0.035], color=C_STRANGER, lw=1.2,
            solid_capstyle='butt', zorder=3)
    ax.text(0, 0.115, "stranger's pose", ha='center', va='top', fontsize=6.5)
    ax.text(-6.6, 0.115, '← lower', ha='left', va='top', fontsize=6.5)
    fig.text(0.60, 0.985, "Own vs. stranger's pose", ha='left', va='top', weight='bold')
    fig.text(0.60, 0.885, 'minADE change (%), held-out scenes', ha='left', va='top',
             fontsize=6.5)

    out = os.path.join(HERE, 'fig_graphical_abstract')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=300)
    print('wrote', out + '.pdf')


if __name__ == '__main__':
    main()
