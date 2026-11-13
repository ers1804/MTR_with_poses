"""
Qualitative figure for the onboard-track result (Sec. IV-E).

Shows pedestrians on which own pose beats a stranger's under the simulated 30 cm tracker,
with pre-fit poses. The examples are selected on the effect, so the rule is fixed here and
stated in the caption, together with how often the opposite happens:

  population   pedestrians with at least 200 mean LiDAR returns and the manoeuvre
               covariates of position_noise_dynamics.py (10 valid future steps), scored
               here on the stored epoch-30 predictions of the ten pre-fit runs
  stratum      top quartile of tracker velocity error at prediction time, the axis the
               content effect grows along (experiments/position_noise_dynamics.json)
  candidates   a complete 8 s future, own pose better than the stranger's on all five seeds
  shown        the N largest mean gains

Predictions are the stored ones, epoch 30, so the win counts written to the JSON are
epoch-30 counts and are not comparable cell by cell with the best-epoch table. Each line
is the best of six modes of one seed.
Panels are drawn in the pedestrian's label frame at prediction time.

Usage:  python paper/iv2027/make_onboard_examples.py [--n 4]
Writes paper/iv2027/fig_onboard_examples.pdf and experiments/onboard_examples.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
METRICS = os.environ.get('METRICS', '/home/erik/big1/heading_noise_metrics')
SCEN = os.environ.get(
    'SCEN', '/home/erik/big1/waymo_agent_batch_25k/processed_scenarios_validation')
SEEDS = (101, 202, 303, 404, 505)
ARMS = {'own': 'cross_attn_pe_with_map', 'stranger': 'cross_attn_pe_with_map_shufflepose'}
MIN_RETURNS, CUR, FULL_FUTURE = 200, 10, 80
# Two categorical hues, validated for colour-vision deficiency on white. Truth and tracker
# are ink and grey, so hue only ever separates the two arms.
C_OWN, C_STRANGER, C_INK, C_TRACK = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8985'


def load(arm: str, seed: int) -> dict:
    cfg = f'{ARMS[arm]}_prefit_posnoise30_scale25k'
    path = os.path.join(METRICS, f'mtr+pose_data_{cfg}', f'PF_25k_{cfg}_lr25_s{seed}', 'eval',
                        'eval_with_train', 'epoch_30', 'result.pkl')
    out = {}
    for item in pickle.load(open(path, 'rb')):
        for r in (item if isinstance(item, list) else [item]):
            gt = r['gt_trajs']
            v = gt[CUR + 1:, 9] > 0
            if not v.any():
                continue
            e = np.linalg.norm(r['pred_trajs'][:, v] - gt[CUR + 1:, :2][v][None], axis=-1).mean(-1)
            out[(str(r['scenario_id']), str(r['object_id']))] = dict(
                gt=gt, mode=r['pred_trajs'][int(e.argmin())], ade=float(e.min()))
    return out


def label_track(key: tuple) -> np.ndarray:
    info = pickle.load(open(glob.glob(os.path.join(SCEN, f'sample_{key[0]}_*.pkl'))[0], 'rb'))
    ids = [str(x) for x in info['track_infos']['object_id']]
    return np.asarray(info['track_infos']['trajs'])[ids.index(key[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    n_show = ap.parse_args().n

    pc = pickle.load(open(os.path.join(REPO, 'experiments', 'val_agent_pointcount_25k.pkl'), 'rb'))
    man = pickle.load(open(os.path.join(REPO, 'experiments', 'eval_agent_manoeuvre.pkl'), 'rb'))
    trk = pickle.load(open(os.path.join(REPO, 'experiments', 'val_tracker_state_posnoise30.pkl'), 'rb'))
    R = {(a, s): load(a, s) for a in ARMS for s in SEEDS}

    keys = sorted(k for k in pc if pc[k]['n_mean'] >= MIN_RETURNS and k in man and k in trk
                  and all(k in x for x in R.values()))
    ade = {a: np.array([[R[a, s][k]['ade'] for s in SEEDS] for k in keys]) for a in ARMS}
    wins = (ade['own'] < ade['stranger']).sum(1)
    gain = (ade['own'] - ade['stranger']).mean(1)
    ve = np.array([trk[k]['vel_err'] for k in keys])
    q1, q3 = np.percentile(ve, [25, 75])
    top, bottom = ve > q3, ve <= q1
    full = np.array([man[k]['n_valid_future'] >= FULL_FUTURE for k in keys])
    cand = np.flatnonzero(top & full & (wins == len(SEEDS)))
    shown = cand[np.argsort(gain[cand])][:n_show]

    facts = dict(
        epoch=30, n_population=len(keys), velocity_error_q3_ms=float(q3),
        content_epoch30_pct=float(np.mean((ade['own'].mean(0) / ade['stranger'].mean(0) - 1) * 100)),
        all=dict(own_all_seeds=int((wins == 5).sum()), stranger_all_seeds=int((wins == 0).sum())),
        top_quartile=dict(n=int(top.sum()), own_all_seeds=int((top & (wins == 5)).sum()),
                          stranger_all_seeds=int((top & (wins == 0)).sum())),
        bottom_quartile=dict(n=int(bottom.sum()), own_all_seeds=int((bottom & (wins == 5)).sum()),
                             stranger_all_seeds=int((bottom & (wins == 0)).sum())),
        n_candidates=int(len(cand)),
        shown=[dict(scenario_id=keys[i][0], object_id=keys[i][1],
                    own_minADE=float(ade['own'][i].mean()),
                    stranger_minADE=float(ade['stranger'][i].mean()),
                    tracker_vel_err=float(ve[i]), tracker_pos_err=float(trk[keys[i]]['pos_err']),
                    speed0=float(man[keys[i]]['speed0']), returns=float(pc[keys[i]]['n_mean']))
               for i in shown])
    json.dump(facts, open(os.path.join(REPO, 'experiments', 'onboard_examples.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in facts.items() if k != 'shown'}, indent=1))

    plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'pdf.fonttype': 42,
                         'font.serif': ['Liberation Serif', 'DejaVu Serif']})
    ncol = 2
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.45, 0.98 * nrow + 0.26))
    t = (np.arange(30) + 1) / 10.0
    for ax, i, tag in zip(np.atleast_1d(axes).ravel(), shown, 'abcdefgh'):
        k = keys[i]
        g, c = R['own', SEEDS[0]][k]['gt'], label_track(k)
        o, h = c[CUR, :2].astype(np.float64), float(c[CUR, 6])
        rot = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

        def T(p):
            return (np.asarray(p, np.float64) - o) @ rot.T

        fv, pv = g[CUR + 1:, 9] > 0, g[:CUR + 1, 9] > 0
        truth = T(g[CUR + 1:, :2][fv])
        for arm, col in (('stranger', C_STRANGER), ('own', C_OWN)):
            for s in SEEDS:
                p = T(R[arm, s][k]['mode'][fv])
                ax.plot(*np.vstack([[0, 0], p]).T, color=col, lw=0.9, alpha=0.9,
                        solid_capstyle='round', zorder=3 if arm == 'own' else 2)
        # the tracker: what it reported, and where its velocity says the pedestrian is going
        # (3 s of constant velocity, kept out of the panel's extent)
        ax.plot(*T(g[:CUR + 1, :2][pv]).T, ls='none', marker='o', ms=2.2, mfc=C_TRACK,
                mec='white', mew=0.3, zorder=4)
        ax.plot(*T(c[:CUR + 1, :2][c[:CUR + 1, 9] > 0]).T, color=C_INK, lw=0.8, zorder=5)
        ax.plot(*np.vstack([[0, 0], truth]).T, color=C_INK, lw=1.6, solid_capstyle='round', zorder=6)
        ax.plot(0, 0, 'o', ms=3.6, mfc=C_INK, mec='white', mew=0.7, zorder=7)
        ax.margins(0.07)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot(*T(g[CUR, :2][None] + g[CUR, 7:9][None] * t[:, None]).T, color=C_TRACK, lw=0.8,
                ls=(0, (3, 2)), zorder=1, scalex=False, scaley=False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('#c9c8c2')
        ax.set_title(f'({tag})  own {ade["own"][i].mean():.2f} m, stranger '
                     f'{ade["stranger"][i].mean():.2f} m', loc='left', fontsize=6.5, pad=2,
                     color=C_INK)
        ax.add_artist(AnchoredSizeBar(ax.transData, 2.0, '2 m', 'lower left', pad=0.25,
                                      borderpad=0.3, sep=2, frameon=False, color=C_INK,
                                      size_vertical=0, fontproperties=FontProperties(size=6)))
    for ax in np.atleast_1d(axes).ravel()[len(shown):]:
        ax.axis('off')

    handles = [Line2D([], [], color=C_OWN, lw=1.2, label='own pose'),
               Line2D([], [], color=C_STRANGER, lw=1.2, label="stranger's pose"),
               Line2D([], [], color=C_INK, lw=1.6, label='label'),
               Line2D([], [], color=C_TRACK, lw=0.8, ls=(0, (3, 2)), marker='o', ms=2.2,
                      mfc=C_TRACK, mec='white', mew=0.3, label='tracker')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=6.5,
               handlelength=1.8, columnspacing=1.1, handletextpad=0.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.36 / fig.get_figheight(),
                        wspace=0.04, hspace=0.20)
    out = os.path.join(HERE, 'fig_onboard_examples.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
