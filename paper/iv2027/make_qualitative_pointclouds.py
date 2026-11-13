"""
LiDAR returns against the pre-fit SMPL body, one pedestrian per return stratum.

Shows what the pose estimator has to work with in each stratum of the pre-fit content
test (Sec. IV-E, tools/scripts/position_noise_subpopulation.py). Each panel overlays the
LiDAR returns inside one pedestrian's box on the body LiDAR-HMR estimated from them in
that frame. That body is the pre-fit pose of the paper: the per-frame output before the
motion-prior stage and before the root is snapped to the direction of travel
(amass_format_motion/raw, the files tools/scripts/export_prefit_poses.py exports).

The examples are not picked by eye:

  population   the scored validation pedestrians of experiments/val_agent_pointcount_25k.pkl
  strata       mean returns over the observed second, cut at 30, 100 and 200
  shown        per stratum, the pedestrian whose mean is closest to the stratum median
               (the first in sorted key order on a tie, the next closest if it has no
               estimate), at the observed frame whose return count is closest to that
               mean among the frames with an estimate

Panels share one metric scale and are turned so the body's widest horizontal extent lies
across the view.

Run with an env that has smplx:
  /home/erik/anaconda3/envs/lidar-hmr/bin/python paper/iv2027/make_qualitative_pointclouds.py
Writes paper/iv2027/fig_pointcloud_mesh.pdf and experiments/pointcloud_mesh_examples.json.
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
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
WOMD = os.environ.get(
    'WOMD_POSE', '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed')
EXTR = os.path.join(WOMD, 'waymo_motion_extracted', 'validation')
RAW = os.path.join(WOMD, 'amass_format_motion', 'raw', 'validation')
SMPL_DIR = '/home/erik/ssd2/Waymo-3DSkelMo/LiDAR-HMR/smplx_models'
PC = os.path.join(REPO, 'experiments', 'val_agent_pointcount_25k.pkl')
STRATA = (('<30', 0, 30), ('30-100', 30, 100), ('100-200', 100, 200), ('>=200', 200, np.inf))
# The body takes the blue of the own-pose arm in fig_onboard_examples, the returns are ink.
C_BODY, C_POINT, C_INK = '#2a78d6', '#0b0b0b', '#0b0b0b'


def frame_of(key: tuple, target: float):
    """The observed frame of one pedestrian whose return count is closest to target,
    among frames the estimator ran on. None if there is no such frame."""
    raw = os.path.join(RAW, key[0], f'{key[1]}.npz')
    if not os.path.exists(raw):
        return None
    d = np.load(raw, allow_pickle=True)
    smt = d['smpl_timestamps'].astype(np.int64)
    recs = [r for r in pickle.load(open(os.path.join(EXTR, f'{key[0]}.pkl'), 'rb'))
            if str(r['basic_info']['object_id']) == key[1]]
    best = None
    for r in sorted(recs, key=lambda r: int(r['basic_info']['time_stamp'])):
        j = np.flatnonzero(smt == int(r['basic_info']['time_stamp']))
        pts = np.asarray(r['world_frame']['pedestrian_pcd'], np.float64).reshape(-1, 3)
        if len(j) == 0 or len(pts) == 0:
            continue
        cost = abs(len(pts) - target)
        if best is None or cost < best[0]:
            best = (cost, dict(row=int(j[0]), points=pts,
                               range_m=float(np.linalg.norm(r['vehicle_frame']['center'][:2])),
                               time_stamp=int(r['basic_info']['time_stamp']),
                               smpl={k: d[k][int(j[0])] for k in
                                     ('betas', 'pose_body', 'root_orient', 'trans')}))
    return None if best is None else best[1]


def select(pc: dict) -> list:
    keys = sorted(pc)
    n_mean = np.array([pc[k]['n_mean'] for k in keys])
    out = []
    for name, lo, hi in STRATA:
        idx = np.flatnonzero((n_mean >= lo) & (n_mean < hi))
        med = float(np.median(n_mean[idx]))
        for i in idx[np.argsort(np.abs(n_mean[idx] - med), kind='stable')]:
            fr = frame_of(keys[i], n_mean[i])
            if fr is not None:
                break
        else:
            raise RuntimeError(f'no pedestrian with an estimate in stratum {name}')
        out.append(dict(stratum=name, n_pedestrians=int(len(idx)), median_mean_returns=med,
                        key=keys[i], mean_returns=float(n_mean[i]), **fr))
    return out


def vertices(model, s: dict) -> np.ndarray:
    def t(x):
        return torch.tensor(np.asarray(x)[None], dtype=torch.float32)
    with torch.no_grad():
        o = model(betas=t(s['betas']), body_pose=t(s['pose_body']),
                  global_orient=t(s['root_orient']), transl=t(s['trans']))
    return o.vertices[0].numpy().astype(np.float64)


def draw(ax, V: np.ndarray, faces: np.ndarray, P: np.ndarray) -> None:
    """Body-centred frame at one metric scale, the widest horizontal extent across the view."""
    c = V.mean(0)
    Vc, Pc = V - c, P - c
    H = Vc[:, :2]
    w, Q = np.linalg.eigh(H.T @ H)
    major = Q[:, int(np.argmax(w))]
    ang = np.arctan2(major[1], major[0]) - np.pi / 2
    ca, sa = np.cos(-ang), np.sin(-ang)
    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Vc, Pc = Vc @ R.T, Pc @ R.T

    tri = Poly3DCollection(Vc[faces], alpha=0.26, linewidths=0)
    tri.set_facecolor(C_BODY)
    tri.set_edgecolor('none')
    ax.add_collection3d(tri)
    ax.scatter(Pc[:, 0], Pc[:, 1], Pc[:, 2], s=1.6, c=C_POINT, depthshade=False,
               edgecolors='none', zorder=10)
    # every body stands on the same line, so a shorter one is not drawn floating
    rh, zh = 0.62, 1.05
    z0 = Vc[:, 2].min() - 0.10
    ax.set_xlim(-rh, rh)
    ax.set_ylim(-rh, rh)
    ax.set_zlim(z0, z0 + 2 * zh)
    ax.set_box_aspect((1, 1, zh / rh))
    ax.view_init(elev=6, azim=-88)
    ax.set_axis_off()


def render(panels: list, faces: np.ndarray, out: str) -> None:
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'pdf.fonttype': 42,
                         'font.serif': ['Liberation Serif', 'DejaVu Serif']})
    n = len(panels)
    fig = plt.figure(figsize=(3.45, 1.12))
    for i, p in enumerate(panels):
        # a 3D axes keeps wide margins around its box, so each one overlaps its neighbours
        ax = fig.add_axes([i / n - 0.10, -0.15, 1 / n + 0.20, 1.30], projection='3d')
        ax.patch.set_alpha(0.0)
        draw(ax, p['vertices'], faces, p['points'])
        fig.text((i + 0.5) / n, 0.995, f'{len(p["points"])} returns, {p["range_m"]:.0f} m',
                 ha='center', va='top', fontsize=7, color=C_INK)
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=300)
    print('wrote', out + '.pdf')


def main() -> None:
    panels = select(pickle.load(open(PC, 'rb')))
    model = smplx.create(SMPL_DIR, model_type='smpl', gender='neutral')
    for p in panels:
        p['vertices'] = vertices(model, p['smpl'])
        # where on the drawn body the returns sit: 0 at its lowest vertex, 1 at its highest
        z = p['vertices'][:, 2]
        p['height'] = (p['points'][:, 2] - z.min()) / (z.max() - z.min())
    facts = [dict(stratum=p['stratum'], n_pedestrians=p['n_pedestrians'],
                  median_mean_returns=p['median_mean_returns'], scenario_id=p['key'][0],
                  object_id=p['key'][1], mean_returns=p['mean_returns'],
                  frame_returns=int(len(p['points'])), frame_time_stamp=p['time_stamp'],
                  range_m=p['range_m'], returns_height_min=float(p['height'].min()),
                  returns_height_max=float(p['height'].max())) for p in panels]
    json.dump(dict(strata_cut_at=[30, 100, 200], shown=facts),
              open(os.path.join(REPO, 'experiments', 'pointcloud_mesh_examples.json'), 'w'),
              indent=1)
    print(json.dumps(facts, indent=1))
    render(panels, model.faces.astype(np.int64), os.path.join(HERE, 'fig_pointcloud_mesh'))


if __name__ == '__main__':
    main()
