#!/usr/bin/env python
"""Compare per-joint recoverability between the 40-segment and 113-segment evaluations.

A reviewer's first question about a small-n measurement is whether it moves when n grows. This
prints the two runs side by side, per joint, and flags any joint whose SKILL point estimate falls
outside the OTHER run's 95% CI -- i.e. a genuine shift rather than tightening.

Usage: python compare_recoverability_runs.py
"""
from __future__ import annotations

import json
import os
import sys

E = "/home/erik/ssd2/gitprojects/MTR_with_poses/experiments"
A = f"{E}/per_joint_recoverability.json"           # 40 segments
B = f"{E}/per_joint_recoverability_mpjpe_eval113.json"   # 113 segments


def main() -> int:
    if not os.path.exists(B):
        print(f"113-segment result not present yet: {B}")
        return 1
    a, b = json.load(open(A)), json.load(open(B))
    ja = {d["joint"]: d for d in a["per_joint"]}
    jb = {d["joint"]: d for d in b["per_joint"]}
    print(f"40-seg : {a['n_frames']} frames, {a['n_pedestrians']} pedestrians, "
          f"overall MPJPE {a['overall_mpjpe_mm']:.1f} mm, skill {a['overall_skill']:+.3f}")
    print(f"113-seg: {b['n_frames']} frames, {b['n_pedestrians']} pedestrians, "
          f"overall MPJPE {b['overall_mpjpe_mm']:.1f} mm, skill {b['overall_skill']:+.3f}")
    print(f"  n grew {b['n_frames']/max(a['n_frames'],1):.2f}x\n")

    print(f"{'joint':<16}{'skill@40':>10}{'skill@113':>11}{'delta':>8}"
          f"{'CI@113':>18}{'CIwidth 40->113':>17}  flag")
    print("-" * 92)
    shifted = []
    for jn in [d["joint"] for d in b["per_joint"]]:
        if jn not in ja:
            continue
        x, y = ja[jn], jb[jn]
        wa = x["skill_hi"] - x["skill_lo"]
        wb = y["skill_hi"] - y["skill_lo"]
        # does each point estimate sit inside the other's CI?
        ok = (x["skill_lo"] <= y["skill"] <= x["skill_hi"]) and \
             (y["skill_lo"] <= x["skill"] <= y["skill_hi"])
        if not ok:
            shifted.append(jn)
        print(f"{jn:<16}{x['skill']:>+10.3f}{y['skill']:>+11.3f}{y['skill']-x['skill']:>+8.3f}"
              f"{'['+format(y['skill_lo'],'+.3f')+','+format(y['skill_hi'],'+.3f')+']':>18}"
              f"{wa:>8.3f}->{wb:<7.3f}  {'SHIFTED' if not ok else ''}")
    print("-" * 92)
    zero = [d["joint"] for d in b["per_joint"] if d["skill_lo"] <= 0 <= d["skill_hi"]]
    print(f"\njoints whose 113-segment CI contains zero (= not measured): {zero or 'none'}")
    if shifted:
        print(f"joints that genuinely SHIFTED between runs: {shifted}")
        print("  -> the paper table and text must be updated to the 113-segment values.")
    else:
        print("no joint shifted outside the other run's CI: the 40-segment table is confirmed,")
        print("  and the larger run only tightens it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
