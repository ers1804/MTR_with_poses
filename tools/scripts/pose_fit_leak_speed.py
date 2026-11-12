#!/usr/bin/env python
"""Does the fitted pose carry the clean track's speed? A check that needs no training.

The pose pipeline ends in a motion-prior fit (Waymo-3DSkelMo, NeMF latent_optimization)
whose loss predicts the root translation from the local body motion and matches it to
the WOMD box track with weight 10. That track is an offboard label. If the fit pulls the
gait toward a stride that matches the track, the fitted body states the clean speed, and
on the onboard-track ladder that is exactly what the simulated tracker loses.

A: predict the clean speed from BODY pose alone (root orientation and translation are
   excluded), for the pre-fit pose (per-frame LiDAR-HMR, never saw the track) and the
   final one. By LiDAR returns too: a pose that knows the speed where the sensor could
   not see the legs has been told it.
C: the same on top of the ladder's own simulated tracker, as the share of the tracker's
   speed error each pose stream removes. Two targets: the one-second window speed and
   the one-step speed at prediction time.
Gradient-boosted trees, 5-fold grouped by scene. Shuffled final poses are the control.

Reads the cache pose_fit_leak_pairs.py writes. Writes experiments/pose_fit_leak_speed.json.
Not bit-reproducible: threaded numerics move the weak (pre-fit) numbers in the second
decimal between runs. The strong ones (final-pose R^2, shares removed) hold to +-0.005.

Usage: python tools/scripts/pose_fit_leak_speed.py [pairs.npz]
"""
import os
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/home/erik/big1/pose_fit_leak/pairs_val.npz')
OUT = os.path.join(REPO, 'experiments', 'pose_fit_leak_speed.json')
N = len(d['scene'])
groups = np.unique(d['scene'], return_inverse=True)[1]
rng0 = np.random.default_rng(0)


def six_d(aa):
    n, t, _ = aa.shape
    R = Rotation.from_rotvec(aa.reshape(-1, 3).astype(np.float64)).as_matrix()
    return R[:, :, :2].transpose(0, 2, 1).reshape(n, t, -1).astype(np.float32)


def pose_feats(body):
    x = six_d(body)                                   # (N, 11, 138)
    summ = np.concatenate([x.mean(1), x.std(1), np.abs(np.diff(x, axis=1)).mean(1)], axis=1)
    flat = StandardScaler().fit_transform(x.reshape(len(x), -1))
    pc = PCA(64, random_state=0).fit_transform(flat)  # unsupervised, no target involved
    return np.concatenate([summ, pc], axis=1).astype(np.float32)


c = d['center'][:, :, :2]
v_t0 = np.linalg.norm(c[:, 10] - c[:, 9], axis=1) / 0.1
v_win = np.linalg.norm(c[:, 10] - c[:, 0], axis=1) / 1.0
hd = d['heading'][:, 10, :2]
yaw = np.arctan2(hd[:, 1], hd[:, 0])

F_raw, F_fin = pose_feats(d['raw_body']), pose_feats(d['fin_body'])
perm = rng0.permutation(N)
F_fin_shuf = F_fin[perm]


def track(s, sigma_a=1.0, sigma_v0=1.5, dt=0.1, seed=1):
    """The ladder's filter (waymo_pose_dataset._track_past_causally), all 11 frames valid."""
    T = 11
    z = c + np.random.default_rng(seed).normal(0.0, s, c.shape)
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sigma_a ** 2
    Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = q; Q[np.ix_([1, 3], [1, 3])] = q
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
    Rm = np.eye(2) * s ** 2
    x = np.concatenate([z[:, 0], np.zeros((N, 2))], axis=1)
    P = np.tile(np.diag([s ** 2, s ** 2, sigma_v0 ** 2, sigma_v0 ** 2]), (N, 1, 1))
    xs = [x.copy()]
    for t in range(1, T):
        x = x @ F.T
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (z[:, t] - x @ H.T)[..., None])[..., 0]
        P = (np.eye(4) - K @ H) @ P
        xs.append(x.copy())
    xs = np.stack(xs, 1)                               # (N, 11, 4)
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    rot = lambda a: np.stack([a[..., 0] * cs[:, None] - a[..., 1] * sn[:, None],
                              a[..., 0] * sn[:, None] + a[..., 1] * cs[:, None]], -1)
    vel = rot(xs[:, :, 2:4]); pos = rot(xs[:, :, 0:2] - xs[:, 10:11, 0:2])
    sp = np.linalg.norm(xs[:, :, 2:4], axis=-1)
    feats = np.concatenate([vel.reshape(N, -1), pos.reshape(N, -1), sp, sp[:, -3:].mean(1, keepdims=True)], axis=1)
    return feats.astype(np.float32), sp[:, 10]


def cv_pred(X, y, train_mask=None):
    pred = np.full(N, np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def r2(y, p, m):
    return float(1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())


def rmse(y, p, m):
    return float(np.sqrt(((y[m] - p[m]) ** 2).mean()))


def boot_diff(y, pa, pb, m, stat, B=1000):
    """Scene-cluster bootstrap of stat(a) - stat(b)."""
    idx = np.where(m)[0]; g = groups[idx]; ug = np.unique(g)
    by = {u: idx[g == u] for u in ug}
    rs = np.random.default_rng(7); out = []
    for _ in range(B):
        ii = np.concatenate([by[u] for u in rs.choice(ug, len(ug))])
        mm = np.ones(len(ii), bool)
        out.append(stat(y[ii], pa[ii], mm) - stat(y[ii], pb[ii], mm))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


pops = {'all': np.ones(N, bool), 'movers(>0.5)': v_win > 0.5, 'walk(0.8-1.6)': (v_win >= 0.8) & (v_win <= 1.6),
        'standing(<0.3)': v_win < 0.3}
npts = d['npts']
strata = {'pts<30': npts < 30, 'pts30-100': (npts >= 30) & (npts < 100), 'pts100-200': (npts >= 100) & (npts < 200),
          'pts>=200': npts >= 200}
res = dict(n=int(N), pops={k: int(v.sum()) for k, v in pops.items()}, strata={k: int(v.sum()) for k, v in strata.items()})

# ---------------- A: speed from pose alone
print('=== A. clean speed from body pose alone (R^2, 5-fold by scene) ===', flush=True)
res['A'] = {}
for tname, y in (('v_win', v_win), ('v_t0', v_t0)):
    P = {k: cv_pred(X, y) for k, X in (('raw', F_raw), ('final', F_fin), ('final_shuffled', F_fin_shuf))}
    Pm = {k: cv_pred(X, y, train_mask=pops['movers(>0.5)']) for k, X in (('raw', F_raw), ('final', F_fin))}
    res['A'][tname] = {}
    for pn, m in pops.items():
        use = Pm if pn in ('movers(>0.5)', 'walk(0.8-1.6)') else P
        row = dict(r2_raw=r2(y, use['raw'], m), r2_final=r2(y, use['final'], m),
                   rmse_raw=rmse(y, use['raw'], m), rmse_final=rmse(y, use['final'], m),
                   rmse_const=float(y[m].std()),
                   r2_diff_ci=boot_diff(y, use['final'], use['raw'], m, r2))
        if pn == 'all':
            row['r2_final_shuffled'] = r2(y, P['final_shuffled'], m)
        res['A'][tname][pn] = row
        print(f"{tname:6s} {pn:15s} n={m.sum():5d}  R2 raw {row['r2_raw']:+.3f}  final {row['r2_final']:+.3f}  "
              f"diff CI [{row['r2_diff_ci'][0]:+.3f},{row['r2_diff_ci'][1]:+.3f}]   RMSE raw {row['rmse_raw']:.3f} "
              f"final {row['rmse_final']:.3f} (const {row['rmse_const']:.3f})", flush=True)
    print(f"{tname:6s} shuffled-final control R2 {res['A'][tname]['all']['r2_final_shuffled']:+.3f}", flush=True)
    if tname == 'v_win':
        res['A']['by_points'] = {}
        for sn, sm in strata.items():
            m = sm & pops['movers(>0.5)']
            row = dict(n=int(m.sum()), r2_raw=r2(y, Pm['raw'], m), r2_final=r2(y, Pm['final'], m),
                       rmse_raw=rmse(y, Pm['raw'], m), rmse_final=rmse(y, Pm['final'], m))
            res['A']['by_points'][sn] = row
            print(f"   movers, {sn:10s} n={row['n']:5d}  R2 raw {row['r2_raw']:+.3f} final {row['r2_final']:+.3f}  "
                  f"RMSE raw {row['rmse_raw']:.3f} final {row['rmse_final']:.3f}", flush=True)

# ---------------- C: on top of the simulated tracker
res['C'] = {}
for tname, y in (('v_win', v_win), ('v_t0', v_t0)):
    print(f'\n=== C. clean {tname}, on top of the simulated causal tracker (RMSE m/s) ===', flush=True)
    res['C'][tname] = {}
    for s in (0.1, 0.3):
        Ft, sp_hat = track(s)
        sets = {'tracker': Ft, 'tracker+raw': np.concatenate([Ft, F_raw], 1), 'tracker+final': np.concatenate([Ft, F_fin], 1),
                'tracker+final_shuf': np.concatenate([Ft, F_fin_shuf], 1),
                'tracker+raw+final': np.concatenate([Ft, F_raw, F_fin], 1)}
        P = {k: cv_pred(X, y) for k, X in sets.items()}
        res['C'][tname][f's{int(s*100)}'] = {}
        for pn, m in pops.items():
            e = {k: rmse(y, p, m) for k, p in P.items()}
            e['tracker_raw_estimate'] = rmse(y, sp_hat, m)
            gain = {k: 1 - (e[k] / e['tracker']) ** 2 for k in ('tracker+raw', 'tracker+final', 'tracker+final_shuf', 'tracker+raw+final')}
            ci = boot_diff(y, P['tracker+raw'], P['tracker+final'], m, rmse)
            res['C'][tname][f's{int(s*100)}'][pn] = dict(rmse=e, mse_removed=gain, rmse_raw_minus_final_ci=ci)
            print(f"s={int(s*100):2d}cm {pn:15s} filter {e['tracker_raw_estimate']:.3f} | learned tracker {e['tracker']:.3f} "
                  f"+raw {e['tracker+raw']:.3f} ({gain['tracker+raw']*100:+.1f}% MSE) +final {e['tracker+final']:.3f} "
                  f"({gain['tracker+final']*100:+.1f}%) +shuf {e['tracker+final_shuf']:.3f} ({gain['tracker+final_shuf']*100:+.1f}%) "
                  f"+both {e['tracker+raw+final']:.3f} | raw-final CI [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
        if s == 0.3:
            res['C'][tname]['s30_by_points'] = {}
            for sn, sm in strata.items():
                m = sm & pops['movers(>0.5)']
                e = {k: rmse(y, P[k], m) for k in ('tracker', 'tracker+raw', 'tracker+final')}
                res['C'][tname]['s30_by_points'][sn] = dict(n=int(m.sum()), rmse=e)
                print(f"   movers, {sn:10s} n={m.sum():5d} tracker {e['tracker']:.3f} +raw {e['tracker+raw']:.3f} "
                      f"({(1-(e['tracker+raw']/e['tracker'])**2)*100:+.1f}%) +final {e['tracker+final']:.3f} "
                      f"({(1-(e['tracker+final']/e['tracker'])**2)*100:+.1f}%)", flush=True)

json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
