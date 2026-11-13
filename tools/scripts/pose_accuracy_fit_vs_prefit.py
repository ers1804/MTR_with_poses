#!/usr/bin/env python
"""Is the fitted pose more accurate than the pre-fit pose? Scored against keypoint labels.

WOMD has no keypoint labels, so this runs on the Waymo perception validation set, where the
same pipeline (Waymo-3DSkelMo) was run and LiDAR keypoints are labelled. Every labelled
pedestrian frame is scored twice, on the same frame:

  pre-fit   the per-frame LiDAR-HMR output (amass/raw), at the label's exact timestamp
  fitted    the pipeline's final output after the motion-prior fit (final_30fps)

Protocol: neutral SMPL with zero shape, the 13 joints Waymo and SMPL share, per-frame mean
over the labelled joints, mean over frames. Three alignments are reported. The paper quotes
the one aligned at the hip midpoint. The unaligned one is the protocol of the pipeline's own
evaluation (Waymo-3DSkelMo evaluation/metrics.py), which is why its published figures are
larger. Frames without both hips labelled cannot be aligned and are dropped.

The script does not import the pipeline's evaluation code. Two guards run with it: the
joint map with left and right swapped must score far worse, and every scored frame must
match a fitted frame at the same timestamp.

Run with an env that has smplx (the fitted poses are read from the NAS):
  PYTHONNOUSERSITE=1 /home/erik/anaconda3/envs/lidar-hmr/bin/python \
      tools/scripts/pose_accuracy_fit_vs_prefit.py [workspace]
Writes experiments/pose_accuracy_fit_vs_prefit.json for the default workspace, the 113
perception segments of /home/erik/big1/mpjpe_eval113. Any other workspace is printed only.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np
import smplx
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_WS = '/home/erik/big1/mpjpe_eval113'
FINAL = '/home/erik/NAS/publicdatasets/generated/final_30fps/validation'
SMPL_DIR = '/home/erik/ssd2/Waymo-3DSkelMo/LiDAR-HMR/smplx_models'
OUT = os.path.join(REPO, 'experiments', 'pose_accuracy_fit_vs_prefit.json')
# Waymo KeypointType value: (name, SMPL joint index, Waymo value of the mirrored joint)
JOINTS = {5: ('LEFT_SHOULDER', 16, 13), 6: ('LEFT_ELBOW', 18, 14), 7: ('LEFT_WRIST', 20, 15),
          8: ('LEFT_HIP', 1, 16), 9: ('LEFT_KNEE', 4, 17), 10: ('LEFT_ANKLE', 7, 18),
          13: ('RIGHT_SHOULDER', 17, 5), 14: ('RIGHT_ELBOW', 19, 6), 15: ('RIGHT_WRIST', 21, 7),
          16: ('RIGHT_HIP', 2, 8), 17: ('RIGHT_KNEE', 5, 9), 18: ('RIGHT_ANKLE', 8, 10),
          20: ('HEAD', 15, 20)}
L_HIP, R_HIP = 8, 16
PAIR = {w: s for w, (_, s, _) in JOINTS.items()}
PAIR_SWAPPED = {w: PAIR[m] for w, (_, _, m) in JOINTS.items()}
N_BOOT = 4000


def smpl_joints(model, d: dict, i: int) -> np.ndarray:
    def t(x):
        return torch.tensor(np.asarray(x, np.float32)[None])
    with torch.no_grad():
        o = model(betas=torch.zeros(1, 10), body_pose=t(d['pose_body'][i]),
                  global_orient=t(d['root_orient'][i]), transl=t(d['trans'][i]))
    return o.joints[0, :24].numpy().astype(np.float64)


def rigid_align(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    Pc, Gc = P - P.mean(0), G - G.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Gc)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return Pc @ R + G.mean(0)


def errors(J: np.ndarray, kp: np.ndarray, types: list, pair: dict) -> dict:
    """Per-joint error in metres under the three alignments, for the labelled shared joints."""
    ids = [t for t in types if t in pair]
    G = np.array([kp[types.index(t)] for t in ids])
    P = np.array([J[pair[t]] for t in ids])
    g_root = (kp[types.index(L_HIP)] + kp[types.index(R_HIP)]) / 2
    p_root = (J[pair[L_HIP]] + J[pair[R_HIP]]) / 2
    return dict(ids=ids, unaligned=np.linalg.norm(P - G, axis=1),
                root_aligned=np.linalg.norm((P - p_root) - (G - g_root), axis=1),
                rigid_aligned=np.linalg.norm(rigid_align(P, G) - G, axis=1))


def collect(ws: str, model) -> tuple[list, dict]:
    rows, n = [], dict(labelled=0, without_both_hips=0, without_prefit=0, without_fitted=0)
    for pkl in sorted(glob.glob(os.path.join(ws, 'extracted', 'validation', '*.pkl'))):
        seg = os.path.basename(pkl)[:-4]
        files: dict = {}
        for r in pickle.load(open(pkl, 'rb')):
            kp, ty = r['world_frame'].get('keypoints'), r['world_frame'].get('keypoint_type')
            if kp is None or ty is None:
                continue
            n['labelled'] += 1
            types = [int(t) for t in ty]
            if L_HIP not in types or R_HIP not in types:
                n['without_both_hips'] += 1
                continue
            oid, ts = str(r['basic_info']['object_id']), int(r['basic_info']['time_stamp'])
            if oid not in files:
                files[oid] = [dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None
                              for p in (os.path.join(ws, 'amass', 'raw', 'validation', seg,
                                                     f'{oid}.npz'),
                                        os.path.join(FINAL, seg, f'{oid}.npz'))]
            raw, fit = files[oid]
            i = [] if raw is None else np.flatnonzero(raw['smpl_timestamps'].astype(np.int64) == ts)
            if len(i) == 0:
                n['without_prefit'] += 1
                continue
            j = [] if fit is None else np.flatnonzero(fit['30fps_timestamps'].astype(np.int64) == ts)
            if len(j) == 0:
                n['without_fitted'] += 1
                continue
            kp = np.asarray(kp, np.float64)
            Jr, Jf = smpl_joints(model, raw, int(i[0])), smpl_joints(model, fit, int(j[0]))
            rows.append(dict(ped=(seg, oid), prefit=errors(Jr, kp, types, PAIR),
                             fitted=errors(Jf, kp, types, PAIR),
                             swapped=errors(Jr, kp, types, PAIR_SWAPPED)))
    return rows, n


def per_frame_mm(rows: list, stage: str, key: str) -> np.ndarray:
    return np.array([r[stage][key].mean() for r in rows]) * 1000


def summarise(rows: list, n: dict, ws: str) -> dict:
    peds = sorted({r['ped'] for r in rows})
    of = {p: k for k, p in enumerate(peds)}
    ped = np.array([of[r['ped']] for r in rows])
    members = [np.flatnonzero(ped == k) for k in range(len(peds))]
    rng = np.random.default_rng(0)

    def stage(name: str) -> dict:
        out = {}
        for key in ('root_aligned', 'unaligned', 'rigid_aligned'):
            v = per_frame_mm(rows, name, key)
            out[key + '_mm'] = dict(mean=float(v.mean()), median=float(np.median(v)))
        return out

    def paired(key: str) -> dict:
        d = per_frame_mm(rows, 'fitted', key) - per_frame_mm(rows, 'prefit', key)
        bs = [d[np.concatenate([members[k] for k in rng.integers(len(peds), size=len(peds))])].mean()
              for _ in range(N_BOOT)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                    n_fitted_worse=int((d > 0).sum()))

    def joint(w: int) -> dict:
        a = [r['prefit']['root_aligned'][r['prefit']['ids'].index(w)] for r in rows
             if w in r['prefit']['ids']]
        b = [r['fitted']['root_aligned'][r['fitted']['ids'].index(w)] for r in rows
             if w in r['fitted']['ids']]
        return dict(n=len(a), prefit=float(np.mean(a) * 1000), fitted=float(np.mean(b) * 1000))

    return dict(
        _meta=dict(workspace=ws, fitted_root=FINAL, n_boot=N_BOOT,
                   note='Waymo perception validation, frames with LiDAR keypoint labels and both '
                        'hips. Neutral SMPL, zero shape, 13 shared joints. Per-frame mean over '
                        'labelled joints, then over frames. root_aligned subtracts the hip '
                        'midpoint on both sides. unaligned is the protocol of the pipeline\'s '
                        'own evaluation. The paired interval resamples pedestrians.'),
        n_labelled=n['labelled'], n_without_both_hips=n['without_both_hips'],
        n_without_prefit=n['without_prefit'], n_without_fitted=n['without_fitted'],
        n_frames=len(rows), n_pedestrians=len(peds), n_segments=len({p[0] for p in peds}),
        prefit=stage('prefit'), fitted=stage('fitted'),
        prefit_left_right_swapped_root_aligned_mm=float(
            per_frame_mm(rows, 'swapped', 'root_aligned').mean()),
        fitted_minus_prefit_mm={k: paired(k) for k in ('root_aligned', 'unaligned',
                                                       'rigid_aligned')},
        per_joint_root_aligned_mm={JOINTS[w][0]: joint(w) for w in JOINTS})


def main() -> None:
    ws = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WS
    model = smplx.create(SMPL_DIR, model_type='smpl', gender='neutral')
    rows, n = collect(ws, model)
    res = summarise(rows, n, ws)
    # a wrong joint map would not show in the stage comparison, so it is guarded here
    assert res['prefit_left_right_swapped_root_aligned_mm'] > 2 * res['prefit']['root_aligned_mm']['mean']
    print(json.dumps(res, indent=1))
    if os.path.abspath(ws) == DEFAULT_WS:
        json.dump(res, open(OUT, 'w'), indent=1)
        print('wrote', OUT)


if __name__ == '__main__':
    main()
