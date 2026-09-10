#!/usr/bin/env python
"""Pre-registered readout of the 30-epoch AVO confirmation.

Written 2026-09-10 BEFORE any confirmation result existed, so the analysis cannot be
tuned to the numbers it will see. Reads the 15 result JSONs produced by
slurm/52_avo_confirm.sh (3 arms x 5 seeds) and reports, in this fixed order:

  1. correct vs shuffled  -- THE DECISIVE TEST. Same architecture, same capacity, same
     regularisation; the only difference is whether a pedestrian receives its OWN body
     motion. Anything that survives here is pose CONTENT.
  2. pose vs no-pose      -- the user-facing goal ("is pose better than no pose"), but
     confounded by architecture/capacity, which is exactly why (1) exists.
  3. shuffled vs no-pose  -- the branch-capacity tax on its own.

Each test is reported four ways, and the PRIMARY endpoint is fixed in advance:

  PRIMARY : FINAL split, fixed last epoch. No selection of any kind anywhere.
  also    : FINAL split, epoch chosen on DEV (selection on dev only -- never on FINAL)
  also    : DEV split, fixed last epoch
  also    : DEV split, best epoch on DEV (the optimistic bound; double-dips)

DECISION RULE (pre-registered in research-log.md before launch): the pose-content effect
is real only if correct-vs-shuffled is NEGATIVE with p < 0.05 on the FINAL split at 30
epochs. A dev-only or best-epoch-only effect does NOT count.

Usage:  python avo_confirmation_analysis.py [--dir /path/to/avo_confirm] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Seeds 202-606 launched first; 707/808/909 added 2026-09-10 on a power calculation made
# BEFORE any 30-epoch result existed (n=5 could only detect >=1.64% at 80% power, while the
# effect under test is ~1.2%). Not optional stopping -- no confirmation data had been seen.
SEEDS = (202, 303, 404, 505, 606, 707, 808, 909)
ARMS = {"nopose": "CONF30_nopose", "pose": "CONF30_pose", "shuffled": "CONF30_shuf"}
N_BOOT = 20000
ALPHA = 0.05

# (a, b) -> a is the treatment, b the reference. Negative rel% means a is BETTER (lower minADE).
TESTS = (
    ("pose", "shuffled", "DECISIVE: pose content (capacity cancels)"),
    ("pose", "nopose", "user-facing goal (confounded by architecture)"),
    ("shuffled", "nopose", "branch-capacity tax alone"),
)

ENDPOINTS = (
    ("minADE_final_last", "FINAL / fixed last epoch", True),
    ("minADE_final_bestdev", "FINAL / epoch chosen on DEV", False),
    ("minADE_dev_last", "dev   / fixed last epoch", False),
    ("minADE_dev_bestdev", "dev   / best epoch on DEV", False),
)


def load(result_dir: str) -> dict:
    """{arm: {seed: result_json}} -- refuses to proceed on a partial or failed set."""
    out: dict = {}
    missing, failed = [], []
    for arm, tag in ARMS.items():
        out[arm] = {}
        for s in SEEDS:
            p = os.path.join(result_dir, f"{tag}_s{s}.json")
            if not os.path.exists(p):
                missing.append(f"{tag}_s{s}")
                continue
            try:
                r = json.load(open(p))
            except json.JSONDecodeError:
                failed.append(f"{tag}_s{s} (unparseable)")
                continue
            if r.get("status") != "ok":
                failed.append(f"{tag}_s{s} ({r.get('reason', 'failed')})")
                continue
            out[arm][s] = r
    if missing or failed:
        print("INCOMPLETE -- not analysing a partial confirmation.")
        for m in missing:
            print(f"  missing: {m}")
        for f in failed:
            print(f"  failed : {f}")
        print(f"\n({sum(len(v) for v in out.values())}/{len(ARMS) * len(SEEDS)} usable)")
        sys.exit(2)
    return out


def paired(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Paired-by-seed comparison.

    The reported p is the PAIRED t-TEST p, not the percentile-bootstrap p. At n=5 the
    bootstrap p is badly anti-conservative: if all five per-seed deltas share a sign, a
    resampled mean essentially never crosses zero, so it returns ~1/N_BOOT no matter how
    small the effect is (verified on synthetic data 2026-09-10: t=-2.72, true p=.053,
    bootstrap p=.0001). Over-confident paired bootstraps have already manufactured several
    dead results in this project, so the t p-value is the gate and the bootstrap interval
    is kept only as a descriptive spread.
    """
    from scipy import stats

    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd > 0:
        t = float(d.mean() / (sd / np.sqrt(n)))
        p = float(2 * stats.t.cdf(-abs(t), n - 1))
    else:
        t, p = float("nan"), 1.0
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    p_boot = 2.0 * min((boots >= 0).mean(), (boots <= 0).mean())
    ref = b.mean()
    return {
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "abs": float(d.mean()), "rel_pct": float(100 * d.mean() / ref),
        "ci_rel_pct": [float(100 * lo / ref), float(100 * hi / ref)],
        "t": t, "p": p,
        "p_boot_anticonservative": float(min(max(p_boot, 1.0 / N_BOOT), 1.0)),
        "improved": int((d < 0).sum()), "n": n,
        "per_seed_rel_pct": [float(100 * (x - y) / y) for x, y in zip(a, b)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/hnvme/workspace/v103fe12-mtr_smpl/avo_confirm")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    R = load(args.dir)
    rng = np.random.default_rng(0)

    eps = {arm: sorted({R[arm][s]["last_epoch"] for s in SEEDS}) for arm in ARMS}
    print("=" * 78)
    print("30-EPOCH AVO CONFIRMATION -- pre-registered readout")
    print("=" * 78)
    print(f"result dir : {args.dir}")
    print(f"seeds      : {list(SEEDS)}")
    for arm in ARMS:
        n_dev = {R[arm][s]["per_epoch"][str(R[arm][s]["last_epoch"])]["n_dev"] for s in SEEDS}
        n_fin = {R[arm][s]["per_epoch"][str(R[arm][s]["last_epoch"])]["n_final"] for s in SEEDS}
        print(f"  {arm:<9} last_epoch{eps[arm]}  n_dev={sorted(n_dev)}  n_final={sorted(n_fin)}")
    if any(len(v) > 1 for v in eps.values()):
        print("  WARNING: arms did not all reach the same last epoch -- fixed-epoch test is not clean.")

    print("\nper-seed minADE (FINAL split, fixed last epoch)")
    print("  seed   " + "".join(f"{a:>12}" for a in ARMS))
    for s in SEEDS:
        print(f"  {s}  " + "".join(f"{R[a][s]['minADE_final_last']:>12.4f}" for a in ARMS))

    results: dict = {}
    for key, label, is_primary in ENDPOINTS:
        vals = {arm: np.array([R[arm][s][key] for s in SEEDS]) for arm in ARMS}
        print("\n" + "-" * 78)
        print(f"{label}{'   <-- PRIMARY ENDPOINT' if is_primary else ''}")
        print("-" * 78)
        for a, b, why in TESTS:
            st = paired(vals[a], vals[b], rng)
            sig = "SIGNIFICANT" if st["p"] < ALPHA else "n.s."
            print(f"  {a:<8} vs {b:<8} {st['rel_pct']:+6.2f}%  "
                  f"CI [{st['ci_rel_pct'][0]:+.2f}%, {st['ci_rel_pct'][1]:+.2f}%]  "
                  f"t={st['t']:+5.2f}  p={st['p']:.4f}  "
                  f"{st['improved']}/{st['n']}  {sig}")
            print(f"      ({why})")
            results.setdefault(key, {})[f"{a}_vs_{b}"] = st

    # ---- the pre-registered verdict, primary endpoint only -------------------------
    prim = results["minADE_final_last"]["pose_vs_shuffled"]
    real = prim["rel_pct"] < 0 and prim["p"] < ALPHA
    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICT (FINAL split, fixed last epoch, correct vs shuffled)")
    print("=" * 78)
    print(f"  effect {prim['rel_pct']:+.2f}%   p={prim['p']:.4f} (paired t)   "
          f"{prim['improved']}/{prim['n']} seeds improved")
    if real:
        print("  --> POSE CONTENT CONFIRMED at the full 30-epoch recipe on held-out data.")
        print("      The project's standing null does not survive; paper/one-pager need revision.")
    else:
        print("  --> NOT CONFIRMED. The 12-epoch proxy effect does not survive full training")
        print("      on held-out data; it was a convergence-rate artifact, not pose content.")
        print("      The null stands and the existing write-up remains correct.")

    goal = results["minADE_final_last"]["pose_vs_nopose"]
    print(f"\n  secondary (user-facing goal) pose vs no-pose: {goal['rel_pct']:+.2f}%, "
          f"p={goal['p']:.4f}, {goal['improved']}/{goal['n']}")

    payload = {"verdict_content_confirmed": bool(real), "alpha": ALPHA,
               "primary_endpoint": "minADE_final_last / pose_vs_shuffled",
               "raw": {a: {str(s): R[a][s][k] for s in SEEDS for k in ()} for a in ARMS},
               "per_seed": {a: {str(s): {k: R[a][s][k] for k, _, _ in ENDPOINTS} for s in SEEDS}
                            for a in ARMS},
               "tests": results}
    if args.json:
        json.dump(payload, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0 if real else 1


if __name__ == "__main__":
    sys.exit(main())
