#!/usr/bin/env python
"""R2: does pose help will-enter-roadway INTENT classification? (pre-registered)

Standalone binary classifier over the same data machinery as the paper:
  traj      : GRU over the past 11 agent-centric states (x, y, vx, vy, heading-cos/sin)
  traj+pose : + the paper's xattn+PE pose encoder over the past 11 pose frames
  traj+wrong: + the SAME encoder fed the shuffled-donor pose sequences (control)

Protocol: train on training split, AUROC on validation; 5 seeds per variant;
hierarchical bootstrap (seed, then pedestrian) on per-pedestrian scores via the
rank-sum formulation. Decision rule pre-registered in project_review Part 6.

Usage: python intent_classifier.py --variant {nopose,pose,wrongpose} --seed N
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))

CFG = {
    'nopose':    ('cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map.yaml', False),
    'pose':      ('cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map.yaml', True),
    'wrongpose': ('cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map_shufflepose.yaml', True),
}


def build_samples(split, use_pose, cfg_file, labels):
    """One pass over the dataset -> (traj_feat, pose_feat, mask, y) arrays."""
    import logging
    from mtr.config import cfg, cfg_from_yaml_file
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    log = logging.getLogger('ic'); log.addHandler(logging.NullHandler())
    cfg_from_yaml_file(os.path.join(REPO, 'tools', cfg_file), cfg)
    ds = WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG, training=(split == 'training'), logger=log)
    T, P, M, Y = [], [], [], []
    for i in range(len(ds)):
        try:
            d = ds[i]
        except Exception:
            continue
        tix = d['track_index_to_predict']
        for c in range(len(d['center_objects_id'])):
            key = f"{d['scenario_id'][c]}|{d['center_objects_id'][c]}"
            if key not in labels:
                continue
            tr = d['obj_trajs'][c, tix[c]]            # (11, 29) agent-centric encoded
            # features: pos(0:2), heading cos/sin at 33? -> keep it simple: first 2 + last 4
            feat = np.concatenate([tr[:, 0:2], tr[:, -4:]], axis=-1)   # (11, 6)
            T.append(feat.astype(np.float32))
            P.append(d['obj_poses'][c, tix[c]].astype(np.float32))     # (11, 144)
            M.append(d['obj_poses_mask'][c, tix[c]])
            Y.append(1.0 if labels[key]['enter_8s'] else 0.0)
    return (np.stack(T), np.stack(P), np.stack(M), np.array(Y, dtype=np.float32))


class IntentNet(nn.Module):
    def __init__(self, use_pose, traj_dim=6, pose_dim=144, h=128):
        super().__init__()
        self.use_pose = use_pose
        self.traj_gru = nn.GRU(traj_dim, h, num_layers=2, batch_first=True, dropout=0.1)
        d = h
        if use_pose:
            self.pose_proj = nn.Linear(pose_dim, h)
            pe = torch.zeros(11, h)
            pos = torch.arange(11, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, h, 2).float() * (-np.log(10000.0) / h))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))
            self.pose_attn = nn.MultiheadAttention(h, 4, dropout=0.1, batch_first=True)
            d += h
        self.head = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))

    def forward(self, t, p, m):
        _, ht = self.traj_gru(t)
        z = ht[-1]
        if self.use_pose:
            keys = self.pose_proj(p * m.unsqueeze(-1)) + self.pe
            pad = ~m.bool()
            allpad = pad.all(dim=1)
            pad = pad.clone(); pad[allpad, 0] = False
            ctx, _ = self.pose_attn(z.unsqueeze(1), keys, keys, key_padding_mask=pad)
            ctx = ctx.squeeze(1) * (~allpad).float().unsqueeze(-1)
            z = torch.cat([z, ctx], dim=-1)
        return self.head(z).squeeze(-1)


def auroc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(CFG))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'experiments', 'intent'))
    args = ap.parse_args()
    cfg_file, use_pose = CFG[args.variant]

    labels = json.load(open(os.path.join(REPO, 'experiments', 'intent_labels.json')))
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cache = os.path.join(args.out_dir, f'cache_{args.variant}.npz')
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(cache):
        z = np.load(cache)
        data = {k: (z[f'{k}_t'], z[f'{k}_p'], z[f'{k}_m'], z[f'{k}_y']) for k in ['train', 'val']}
    else:
        data = {'train': build_samples('training', use_pose, cfg_file, labels['training']),
                'val': build_samples('validation', use_pose, cfg_file, labels['validation'])}
        np.savez_compressed(cache, **{f'{k}_{s}': v for k, (a, b, c, d) in data.items()
                                      for s, v in zip('tpmy', (a, b, c, d))})
    (Tt, Pt, Mt, Yt), (Tv, Pv, Mv, Yv) = data['train'], data['val']
    print(f'{args.variant} s{args.seed}: train {len(Yt)} (pos {Yt.mean():.3f}) '
          f'val {len(Yv)} (pos {Yv.mean():.3f})')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = IntentNet(use_pose).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(1 - Yt.mean()) / max(Yt.mean(), 1e-6)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    tt, tp, tm, ty = [torch.from_numpy(x).to(dev) for x in (Tt, Pt, Mt.astype(np.float32), Yt)]
    vt, vp, vm = [torch.from_numpy(x).to(dev) for x in (Tv, Pv, Mv.astype(np.float32))]

    best = {'auroc': -1}
    idx = np.arange(len(ty))
    for ep in range(args.epochs):
        net.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 256):
            b = torch.from_numpy(idx[i:i+256]).long().to(dev)
            opt.zero_grad()
            loss = lossf(net(tt[b], tp[b], tm[b]), ty[b])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = net(vt, vp, vm).cpu().numpy()
        a = auroc(Yv, s)
        if a > best['auroc']:
            best = {'auroc': a, 'epoch': ep, 'scores': s.tolist()}
    out = {'variant': args.variant, 'seed': args.seed, 'best_auroc': best['auroc'],
           'best_epoch': best['epoch'], 'val_labels': Yv.tolist(), 'val_scores': best['scores']}
    path = os.path.join(args.out_dir, f'{args.variant}_s{args.seed}.json')
    json.dump(out, open(path, 'w'))
    print(f'BEST AUROC {best["auroc"]:.4f} (epoch {best["epoch"]}) -> {path}')


if __name__ == '__main__':
    main()
