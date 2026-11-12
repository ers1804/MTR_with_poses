"""Export the PRE-FIT poses in the layout WaymoPoseDataset reads.

Why. The final poses come out of a NeMF latent optimisation whose loss ties the gait to
the box track: the root translation is predicted from the local body motion by a
pretrained global-motion predictor and matched to the WOMD track with weight 10
(Waymo-3DSkelMo NeMF/src/application.py, latent_optimization). WOMD tracks are offboard
labels, so the fitted body carries the clean track's speed. On the onboard-track ladder
that is exactly the quantity the simulated tracker loses, so a pose-versus-shuffled gain
there cannot be read as body information until the poses are free of the track.

What. The per-frame LiDAR-HMR output, before any motion prior and before the pipeline's
travel-direction snapping of the root. It never sees the track beyond the box the points
were cropped with. `amass_format_motion/raw` holds it (bit-identical to the LiDAR-HMR
result files), one row per frame the estimator ran on.

The export mirrors an existing final export file for file, so the scene index, the agent
set and the shuffled arm's donor permutation are identical to the final-pose runs:

    <out>/<split>/<scene>/<id>.npz
        root_orient (N, 3)   LiDAR-HMR, world frame
        pose_body   (N, 69)  LiDAR-HMR
        betas       (10,)
        trans       (N, 3)   the FINAL export's rows at the same timestamps (only read
                             when a scene has no processed WOMD track, as before)
        waymo_timestamps (N,)

    python tools/scripts/export_prefit_poses.py \
        --final /home/erik/big1/waymo_motion_poses/scale_25k \
        --out   /home/erik/big1/waymo_motion_poses/scale_25k_prefit
"""
import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np

RAW = '/home/erik/NAS/publicdatasets/waymo_motion/v_1_3_0/uncompressed/amass_format_motion/raw'


def one_scene(args):
    final_root, out_root, raw_root, split, scene = args
    c = Counter()
    src = os.path.join(final_root, split, scene)
    dst = os.path.join(out_root, split, scene)
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if not name.endswith('.npz'):
            continue
        fin = np.load(os.path.join(src, name))
        raw = np.load(os.path.join(raw_root, split, scene, name))
        ts = raw['smpl_timestamps'].astype(np.int64)
        fts = fin['waymo_timestamps'].astype(np.int64)
        n = raw['pose_body'].shape[0]
        # Every one of these held on all 162,841 files when the export was written. They
        # are what makes the file a drop-in, so fail loudly instead of writing a mismatch.
        assert n >= 1 and len(ts) == n, (scene, name, n, len(ts))
        assert np.isin(ts, fts).all(), (scene, name, 'pre-fit frame outside the final rows')
        assert np.allclose(raw['betas'][0], fin['betas'], atol=1e-5), (scene, name, 'betas')
        rows = np.searchsorted(fts, ts)
        assert np.array_equal(fts[rows], ts), (scene, name, 'unsorted timestamps')
        tmp = os.path.join(dst, name + '.tmp.npz')
        np.savez(tmp,
                 trans=fin['trans'][rows].astype(np.float64),
                 root_orient=raw['root_orient'].astype(np.float32),
                 pose_body=raw['pose_body'].astype(np.float32),
                 betas=fin['betas'].astype(np.float32),
                 waymo_timestamps=ts)
        os.replace(tmp, os.path.join(dst, name))
        c['files'] += 1
        c['rows_prefit'] += n
        c['rows_final'] += len(fts)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default=RAW)
    ap.add_argument('--splits', nargs='+', default=['validation', 'training'])
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()

    manifest = dict(final=a.final, raw=a.raw, splits={})
    for split in a.splits:
        scenes = sorted(d for d in os.listdir(os.path.join(a.final, split))
                        if os.path.isdir(os.path.join(a.final, split, d)))
        tot = Counter()
        with Pool(a.workers) as p:
            for i, c in enumerate(p.imap_unordered(
                    one_scene, [(a.final, a.out, a.raw, split, s) for s in scenes], chunksize=16)):
                tot += c
                if i % 5000 == 0:
                    print(split, i, dict(tot), flush=True)
        tot['scenes'] = len(scenes)
        manifest['splits'][split] = dict(tot)
        print(split, 'DONE', dict(tot), flush=True)
    with open(os.path.join(a.out, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


if __name__ == '__main__':
    main()
