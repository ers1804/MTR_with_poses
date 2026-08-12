#!/usr/bin/env python
"""Where does pose conditioning help? (professor follow-up #3)

Answers "which pedestrians benefit from pose" WITHOUT selecting on the outcome:

  1. Per-agent scenario descriptors are computed from TRAJECTORIES ONLY (never from
     pose, never from model error), in two families:
       past_*   -- computable at inference time  => a DEPLOYABLE selector
       future_* -- uses the ground-truth future  => CHARACTERIZATION only (oracle)
  2. Evaluated pedestrians are split into a discovery half (A) and a confirmation
     half (B) by a deterministic hash of (scenario_id, object_id).
  3. The full heuristic x threshold scan runs on A only. The single best subgroup is
     then reported on B, untouched, with a hierarchical (seed + pedestrian) bootstrap.
  4. The complete scan is printed, so the multiplicity is visible rather than hidden.

Selecting the subgroup on the pooled set would inflate the effect by construction --
exactly the error this paper documents elsewhere.

Usage: python subgroup_analysis.py [--pair map_nopose:map_xattn_pe] [--out FILE]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_multiseed import (CELLS, OUT_ROOT, SEEDS, hierarchical_bootstrap,  # noqa: E402
                               load_agent_metrics, run_summary)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_CACHE = os.path.join(REPO, 'experiments', 'val_agent_dynamics.pkl')
NUM_PAST = 11
DT = 0.1


# ─────────────────────────── scenario descriptors ───────────────────────────
def agent_features(traj):
    """Dynamics descriptors for one agent from its (91, 10) world trajectory.

    Columns: [cx, cy, cz, dx, dy, dz, heading, vx, vy, valid].
    `past_*` use only steps < NUM_PAST (deployable); `future_*` use the GT future.
    """
    valid = traj[:, -1] > 0
    past, fut = traj[:NUM_PAST], traj[NUM_PAST:]
    pv, fv = valid[:NUM_PAST], valid[NUM_PAST:]
    if pv.sum() < 3 or fv.sum() < 2:
        return None

    f = {}
    # ---- past-only (deployable at inference) ----
    pxy = past[pv][:, :2]
    pspeed = np.linalg.norm(np.diff(pxy, axis=0), axis=-1) / DT
    f['past_speed'] = float(pspeed.mean())
    f['past_speed_change'] = float(pspeed.max() - pspeed.min()) if len(pspeed) > 1 else 0.0
    if len(pxy) >= 3:                      # turning already visible in the past
        d = np.diff(pxy, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f['past_turn'] = float(np.abs(np.angle(np.exp(1j * (ang[-1] - ang[0])))))
    else:
        f['past_turn'] = 0.0

    # ---- future (oracle characterization) ----
    fxy = fut[fv][:, :2]
    last_p = pxy[-1]
    v_last = (pxy[-1] - pxy[-2]) / DT if len(pxy) >= 2 else np.zeros(2)
    n_f = len(fxy)
    # deviation from a constant-velocity extrapolation = "how non-trivial is this future"
    t = np.arange(1, n_f + 1) * DT
    cv = last_p[None, :] + v_last[None, :] * t[:, None]
    f['future_cv_error'] = float(np.linalg.norm(fxy - cv, axis=-1).mean())
    fspeed = np.linalg.norm(np.diff(fxy, axis=0), axis=-1) / DT if n_f > 1 else np.array([0.0])
    f['future_speed_change'] = float(np.abs(fspeed - f['past_speed']).max())
    if n_f >= 3:
        d = np.diff(fxy, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f['future_turn'] = float(np.abs(np.angle(np.exp(1j * (ang[-1] - ang[0])))))
    else:
        f['future_turn'] = 0.0
    f['future_displacement'] = float(np.linalg.norm(fxy[-1] - last_p))
    return f


def build_feature_table(logger_cfg):
    """Iterate the validation split once and cache per-agent descriptors."""
    if os.path.exists(FEATURE_CACHE):
        with open(FEATURE_CACHE, 'rb') as fh:
            return pickle.load(fh)

    sys.path.insert(0, REPO)                      # for `mtr`
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    import logging
    from mtr.config import cfg, cfg_from_yaml_file
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset

    log = logging.getLogger('subgroup'); log.addHandler(logging.NullHandler())
    cfg_from_yaml_file(logger_cfg, cfg)
    ds = WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG, training=False, logger=log)

    feats = {}
    for i in range(len(ds)):
        try:
            d = ds[i]
        except Exception:
            continue
        for k in range(len(d['center_objects_id'])):
            key = (str(d['scenario_id'][k]), str(d['center_objects_id'][k]))
            fe = agent_features(d['center_gt_trajs_src'][k])
            if fe is not None:
                feats[key] = fe
        if (i + 1) % 500 == 0:
            print(f'  ...{i+1}/{len(ds)} scenes, {len(feats)} agents', flush=True)

    os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
    with open(FEATURE_CACHE, 'wb') as fh:
        pickle.dump(feats, fh)
    return feats


# ─────────────────────────── per-seed metric loading ───────────────────────────
def cell_per_seed(cell):
    out = {}
    for seed in SEEDS:
        rd = os.path.join(OUT_ROOT, CELLS[cell], f'MS_{cell}_s{seed}')
        s = run_summary(rd)
        if s is None:
            continue
        m = load_agent_metrics(rd, s['best_epoch'])
        if m:
            out[seed] = {k: v[0] for k, v in m.items()}   # minADE
    return out


def split_half(key):
    """Deterministic 50/50 assignment; independent of any model output."""
    h = hashlib.md5(f'{key[0]}|{key[1]}'.encode()).hexdigest()
    return 'A' if int(h[:8], 16) % 2 == 0 else 'B'


def subgroup_effect(a_ps, b_ps, keys, n_boot=10000):
    """Hierarchical bootstrap of (b - a) restricted to `keys`."""
    a_r = {s: {k: v for k, v in d.items() if k in keys} for s, d in a_ps.items()}
    b_r = {s: {k: v for k, v in d.items() if k in keys} for s, d in b_ps.items()}
    return hierarchical_bootstrap(a_r, b_r, n_boot=n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='map_nopose:map_xattn_pe')
    ap.add_argument('--cfg', default=os.path.join(
        REPO, 'tools', 'cfgs', 'waymo', 'mtr+pose_data_cross_attn_pe_with_map.yaml'))
    ap.add_argument('--min-frac', type=float, default=0.05,
                    help='smallest subgroup considered, as a fraction of the split')
    ap.add_argument('--out', default=os.path.join(REPO, 'experiments', 'subgroup_analysis.json'))
    args = ap.parse_args()
    ref, treat = args.pair.split(':')

    print('building per-agent scenario descriptors (cached after first run)...')
    feats = build_feature_table(args.cfg)
    a_ps, b_ps = cell_per_seed(ref), cell_per_seed(treat)
    common = set.intersection(*[set(d) for d in list(a_ps.values()) + list(b_ps.values())])
    keys = sorted(common & set(feats))
    print(f'{len(keys)} evaluated pedestrians with descriptors; '
          f'{len(a_ps)}/{len(b_ps)} seeds for {ref}/{treat}')

    A = [k for k in keys if split_half(k) == 'A']
    B = [k for k in keys if split_half(k) == 'B']
    print(f'discovery split A: {len(A)}   confirmation split B: {len(B)}')

    feat_names = sorted(next(iter(feats.values())).keys())
    quantiles = [0.5, 0.6, 0.7, 0.8, 0.9]          # keep the top (1-q) fraction
    results = {'pair': args.pair, 'n_A': len(A), 'n_B': len(B), 'scan': [], }

    overall = subgroup_effect(a_ps, b_ps, set(keys))
    results['overall'] = overall
    print(f"\nOVERALL (all {len(keys)} peds): {overall['rel_diff_pct']:+.2f}% "
          f"p={overall['p_boot_two_sided']:.4f}")

    print('\n=== DISCOVERY SCAN (split A only) ===')
    print(f"{'feature':<22} {'keep':<7} {'n_A':>6} {'rel%':>8} {'p':>8}")
    best = None
    for fn in feat_names:
        vals = np.array([feats[k][fn] for k in A])
        for q in quantiles:
            thr = float(np.quantile(vals, q))
            sub = {k for k, v in zip(A, vals) if v >= thr}
            if len(sub) < args.min_frac * len(A):
                continue
            r = subgroup_effect(a_ps, b_ps, sub, n_boot=2000)
            if r is None:
                continue
            row = {'feature': fn, 'quantile': q, 'threshold': thr,
                   'n': len(sub), 'rel_diff_pct': r['rel_diff_pct'],
                   'p': r['p_boot_two_sided']}
            results['scan'].append(row)
            print(f"{fn:<22} top{1-q:<4.0%} {len(sub):>6} {r['rel_diff_pct']:>+8.2f} "
                  f"{r['p_boot_two_sided']:>8.4f}")
            if best is None or row['rel_diff_pct'] < best['rel_diff_pct']:
                best = row

    print(f"\nBEST ON A: {best['feature']} >= {best['threshold']:.3f} "
          f"(top {1-best['quantile']:.0%}, n={best['n']}): {best['rel_diff_pct']:+.2f}%")

    # ---- confirmation: apply the A-derived rule verbatim to B ----
    fn, thr = best['feature'], best['threshold']
    subB = {k for k in B if feats[k][fn] >= thr}
    conf = subgroup_effect(a_ps, b_ps, subB)
    restB = set(B) - subB
    rest = subgroup_effect(a_ps, b_ps, restB)
    results['selected_rule'] = {'feature': fn, 'threshold': thr, 'quantile': best['quantile']}
    results['confirmation_B'] = {'n': len(subB), **conf}
    results['complement_B'] = {'n': len(restB), **rest}

    print('\n=== CONFIRMATION (held-out split B, rule fixed on A) ===')
    print(f"  subgroup   n={len(subB):>5}  {conf['rel_diff_pct']:+.2f}%  "
          f"CI[{100*conf['ci95_lo']/ (conf['mean_diff']/(conf['rel_diff_pct']/100)):+.2f},"
          f"{100*conf['ci95_hi']/ (conf['mean_diff']/(conf['rel_diff_pct']/100)):+.2f}]%  "
          f"p={conf['p_boot_two_sided']:.4f}")
    print(f"  complement n={len(restB):>5}  {rest['rel_diff_pct']:+.2f}%  "
          f"p={rest['p_boot_two_sided']:.4f}")

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nfull JSON: {args.out}')


if __name__ == '__main__':
    main()
