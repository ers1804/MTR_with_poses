#!/usr/bin/env python
"""Per-joint recoverability of LiDAR-based human mesh recovery against GT keypoints.

WHY THIS AND NOT MPJPE
----------------------
A single MPJPE number cannot distinguish two very different failure modes:

  (a) the estimator tracks the joint but imprecisely  -> error is large, information is present
  (b) the estimator ignores the input and emits its PRIOR (roughly the mean human pose)
      -> error can still look respectable for a joint that does not move much, while the
         output carries NO information about that particular person

Downstream users care about (b) and MPJPE hides it. This script separates them, per joint, by
scoring the estimator against the strongest trivial competitor: ALWAYS PREDICTING THE MEAN
ROOT-ALIGNED GT POSE. Concretely, per joint:

  mpjpe_mm     mean || pred - gt ||            (root-aligned at the hip midpoint)
  gt_spread    mean || gt - mean(gt) ||        how much the joint genuinely varies
  pred_spread  mean || pred - mean(pred) ||    how much the estimator lets it vary
  spread_ratio pred_spread / gt_spread         ~0 => collapsed onto the prior
  mean_pose_mm mean || mean(gt) - gt ||        error of the constant mean-pose predictor
  SKILL        1 - mpjpe / mean_pose_mm        >0 beats the mean pose; <=0 adds nothing
  r_xyz        per-axis Pearson correlation between prediction and GT

A joint with SKILL <= 0 is not measured by the model, whatever its MPJPE says.

Protocol follows full_pipeline_mpjpe.py: 13 COMMON_JOINTS, hip-midpoint root alignment, and
TIMESTAMP-based frame matching (index-based matching was the 2026-08-20 harness bug that inflated
MPJPE from 87.6 to 160.6 mm). Evaluates the LiDAR-HMR stage with zero betas, matching the
convention under which 87.6 mm was reported.

Env: lidar-hmr, PYTHONNOUSERSITE=1.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import smplx
import torch

sys.path.insert(0, "/home/erik/ssd2/Waymo-3DSkelMo")
from evaluation.metrics import (COMMON_JOINTS, SMPL_IDS, WAYMO_IDS,  # noqa: E402
                                load_waymo_tracks, run_smpl_sequence)

WORK = os.environ.get("MPJPE_WORK", "/home/erik/big1/mpjpe_eval")
PRED = f"{WORK}/amass/raw/validation"
_TAG = "" if WORK.rstrip("/").endswith("mpjpe_eval") else "_" + os.path.basename(WORK.rstrip("/"))
OUT = ("/home/erik/ssd2/gitprojects/MTR_with_poses/experiments/"
       f"per_joint_recoverability{_TAG}.json")


def main() -> int:
    model = smplx.create("/home/erik/ssd2/Waymo-3DSkelMo/LiDAR-HMR/smplx_models",
                         model_type="smpl", gender="neutral")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    segs = sorted(f[:-4] for f in os.listdir(f"{WORK}/extracted/validation") if f.endswith(".pkl"))
    print(f"segments: {len(segs)}", flush=True)

    gt_by_joint: dict[str, list] = {n: [] for n in COMMON_JOINTS}
    pr_by_joint: dict[str, list] = {n: [] for n in COMMON_JOINTS}
    # Frames within one pedestrian are highly correlated, so a frame-level bootstrap
    # understates the CI. Record which pedestrian each sample came from so the CI can
    # also be computed by resampling PEDESTRIANS (the independent unit).
    pid_by_joint: dict[str, list] = {n: [] for n in COMMON_JOINTS}
    n_frames = n_ped = 0

    for si, ctx in enumerate(segs):
        try:
            tracks = load_waymo_tracks(f"{WORK}/extracted", "validation", ctx)
        except FileNotFoundError:
            continue
        for oid, tr in tracks.items():
            path = os.path.join(PRED, ctx, f"{oid}.npz")
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            if "smpl_timestamps" not in d:
                continue
            t30 = d["smpl_timestamps"].astype(np.int64)
            if len(t30) != len(d["trans"]):
                continue
            J = run_smpl_sequence(model, dev, d["trans"], d["root_orient"], d["pose_body"])
            used = False
            for entry in tr:
                kp = entry["world_frame"]["keypoints"]
                kt = entry["world_frame"]["keypoint_type"]
                if kp is None or kt is None:
                    continue
                ktl = kt.tolist()
                if WAYMO_IDS["LEFT_HIP"] not in ktl or WAYMO_IDS["RIGHT_HIP"] not in ktl:
                    continue
                ts = int(entry["basic_info"]["time_stamp"])
                jidx = int(np.argmin(np.abs(t30 - ts)))
                if abs(int(t30[jidx]) - ts) > 60_000:      # >60 ms: no matching frame
                    continue
                Jj = J[jidx]
                w_root = (kp[ktl.index(WAYMO_IDS["LEFT_HIP"])]
                          + kp[ktl.index(WAYMO_IDS["RIGHT_HIP"])]) / 2
                s_root = (Jj[SMPL_IDS["LEFT_HIP"]] + Jj[SMPL_IDS["RIGHT_HIP"]]) / 2
                for name in COMMON_JOINTS:
                    wid = WAYMO_IDS[name]
                    if wid not in ktl:
                        continue
                    gt_by_joint[name].append(np.asarray(kp[ktl.index(wid)], float) - w_root)
                    pr_by_joint[name].append(np.asarray(Jj[SMPL_IDS[name]], float) - s_root)
                    pid_by_joint[name].append(n_ped)
                n_frames += 1
                used = True
            n_ped += int(used)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(segs)} segments, {n_frames} matched frames", flush=True)

    print(f"\nmatched frames {n_frames} over {n_ped} pedestrians\n")
    rows = []
    for name in COMMON_JOINTS:
        if len(gt_by_joint[name]) < 50:
            continue
        G = np.stack(gt_by_joint[name]); P = np.stack(pr_by_joint[name])
        mpjpe = float(np.linalg.norm(P - G, axis=1).mean()) * 1000
        gs = float(np.linalg.norm(G - G.mean(0), axis=1).mean()) * 1000
        ps = float(np.linalg.norm(P - P.mean(0), axis=1).mean()) * 1000
        base = float(np.linalg.norm(G.mean(0) - G, axis=1).mean()) * 1000
        r = [float(np.corrcoef(P[:, k], G[:, k])[0, 1]) for k in range(3)]
        # Two bootstrap CIs on SKILL. The frame-level one treats every matched frame as
        # independent, which it is not (frames within a pedestrian are correlated), so it
        # is anti-conservative; the pedestrian-clustered one resamples whole pedestrians
        # and is the interval to quote.
        rng = np.random.default_rng(0)
        bs = np.empty(2000)
        idx_all = np.arange(len(G))
        for b in range(2000):
            i = rng.choice(idx_all, len(idx_all), replace=True)
            Gb, Pb = G[i], P[i]
            m = np.linalg.norm(Pb - Gb, axis=1).mean()
            bb = np.linalg.norm(Gb.mean(0) - Gb, axis=1).mean()
            bs[b] = 1 - m / bb if bb else np.nan
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])

        pid = np.asarray(pid_by_joint[name])
        uniq = np.unique(pid)
        by_ped = {u: np.where(pid == u)[0] for u in uniq}
        rng2 = np.random.default_rng(0)
        bsc = np.empty(2000)
        for b in range(2000):
            pick = rng2.choice(uniq, len(uniq), replace=True)
            i = np.concatenate([by_ped[u] for u in pick])
            Gb, Pb = G[i], P[i]
            m = np.linalg.norm(Pb - Gb, axis=1).mean()
            bb = np.linalg.norm(Gb.mean(0) - Gb, axis=1).mean()
            bsc[b] = 1 - m / bb if bb else np.nan
        clo, chi = np.nanpercentile(bsc, [2.5, 97.5])
        rows.append(dict(joint=name, n=len(G), mpjpe_mm=mpjpe, gt_spread_mm=gs,
                         pred_spread_mm=ps, spread_ratio=ps / gs if gs else float("nan"),
                         mean_pose_mm=base, skill=1 - mpjpe / base if base else float("nan"),
                         skill_lo=float(lo), skill_hi=float(hi),
                         n_ped=int(len(uniq)),
                         skill_ped_lo=float(clo), skill_ped_hi=float(chi),
                         r_x=r[0], r_y=r[1], r_z=r[2]))

    rows.sort(key=lambda d: -d["skill"])
    print(f"{'joint':<16}{'n':>6}{'MPJPE':>8}{'meanpose':>10}{'SKILL':>8}{'95% CI':>18}"
          f"{'GTspr':>8}{'PRspr':>8}{'ratio':>7}{'max|r|':>8}")
    print("(95% CI resamples PEDESTRIANS, not frames)")
    print("-" * 101)
    for d in rows:
        ci = f"[{d['skill_ped_lo']:+.3f},{d['skill_ped_hi']:+.3f}]"
        print(f"{d['joint']:<16}{d['n']:>6}{d['mpjpe_mm']:>8.1f}{d['mean_pose_mm']:>10.1f}"
              f"{d['skill']:>+8.3f}{ci:>18}{d['gt_spread_mm']:>8.1f}{d['pred_spread_mm']:>8.1f}"
              f"{d['spread_ratio']:>7.2f}{max(abs(d['r_x']),abs(d['r_y']),abs(d['r_z'])):>8.2f}")
    ov = float(np.mean([d["mpjpe_mm"] for d in rows]))
    ovb = float(np.mean([d["mean_pose_mm"] for d in rows]))
    print("-" * 101)
    print(f"{'OVERALL':<16}{'':>6}{ov:>8.1f}{ovb:>10.1f}{1-ov/ovb:>+8.3f}")
    print("\nSKILL > 0: beats always-predict-the-mean-pose.  SKILL <= 0: adds nothing for that joint.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(n_frames=n_frames, n_pedestrians=n_ped, overall_mpjpe_mm=ov,
                   overall_mean_pose_mm=ovb, overall_skill=1 - ov / ovb, per_joint=rows),
              open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
