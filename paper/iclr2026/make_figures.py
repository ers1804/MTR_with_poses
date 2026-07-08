"""
Generate paper figures (post-audit, seed-replicated version).
Outputs: fig_architecture.pdf, fig_ablation.pdf, fig_training_curves.pdf

Figures 2 and 3 read the seed-replicated MS_* run logs directly, so they stay
in sync with the experimental record.
"""

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ─── style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE   = '#2166ac'
ORANGE = '#d6604d'
GREEN  = '#4dac26'
GRAY   = '#999999'
LBLUE  = '#92c5de'
LORANGE= '#f4a582'

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(REPO, 'output', 'waymo')

# ─── log parsing (same logic as tools/scripts/analyze_multiseed.py) ─────────
EPOCH_RE = re.compile(r'Performance of EPOCH (\d+)')
ADE_RE = re.compile(r'^minADE:\s*([\d.]+)')


def parse_log(run_dir):
    """{epoch: minADE} from a run's training log(s)."""
    per_epoch = {}
    for log in sorted(glob.glob(os.path.join(run_dir, 'log_train_*.txt'))):
        cur_epoch = None
        with open(log) as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    cur_epoch = int(m.group(1))
                    continue
                m = ADE_RE.match(line)
                if m and cur_epoch is not None:
                    per_epoch[cur_epoch] = float(m.group(1))
                    cur_epoch = None
    return per_epoch


def cell_runs(cfg, cell, seeds):
    out = []
    for s in seeds:
        d = os.path.join(OUT_ROOT, cfg, f'MS_{cell}_s{s}')
        pe = parse_log(d)
        if pe:
            out.append(pe)
    return out


def cell_best_values(cfg, cell, seeds):
    return [min(pe.values()) for pe in cell_runs(cfg, cell, seeds)]


S5 = [101, 202, 303, 404, 505]
S3 = [101, 202, 303]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════

def draw_box(ax, xy, w, h, label, color='white', ec='black', fontsize=8, bold=False):
    x, y = xy
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=color, edgecolor=ec, linewidth=1.0)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True)


def draw_arrow(ax, x0, y0, x1, y1, color='black', lw=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

draw_box(ax, (0.1, 2.3), 1.2, 0.8, 'Agent\npositions\n(past 11)', color='#e8f4f8', ec=BLUE)
draw_box(ax, (0.1, 0.9), 1.2, 0.8, 'SMPL poses\n(past 11)\n6D repr.', color='#fef3e2', ec=ORANGE)

draw_box(ax, (1.8, 2.1), 1.6, 1.2, 'MTR Backbone\n(Transformer\nencoder)', color='#e8f4f8', ec=BLUE)
draw_arrow(ax, 1.3, 2.7, 1.8, 2.7, color=BLUE)

draw_box(ax, (1.8, 0.7), 1.6, 1.0, 'Pose Encoder\n(GRU or\nXAttn+PE, 256-d)', color='#fef3e2', ec=ORANGE)
draw_arrow(ax, 1.3, 1.3, 1.8, 1.3, color=ORANGE)

draw_box(ax, (4.0, 1.6), 1.4, 1.2, 'Concat +\nMLP\nProjection', color='#f0f0f0', ec=GRAY)
draw_arrow(ax, 3.4, 2.7, 4.0, 2.1, color=BLUE)
draw_arrow(ax, 3.4, 1.2, 4.0, 1.9, color=ORANGE)

draw_box(ax, (6.0, 1.6), 1.5, 1.2, 'MTR Decoder\n(6 refinement\nlayers)', color='#e8f4f8', ec=BLUE)
draw_arrow(ax, 5.4, 2.2, 6.0, 2.2, color='black')

draw_box(ax, (8.2, 2.4), 1.6, 0.8, '6 trajectory\nmodes\n(minADE)', color='#d4edda', ec=GREEN, bold=True)
draw_box(ax, (8.2, 1.0), 1.6, 0.8, 'Future pose\nprediction\n(aux. task)', color='#fff3cd', ec=ORANGE)
draw_arrow(ax, 7.5, 2.6, 8.2, 2.8, color=BLUE)
draw_arrow(ax, 7.5, 1.8, 8.2, 1.4, color=ORANGE)

# Auxiliary loss: winner-takes-all L1 (the only active pose supervision;
# the geodesic term is inert in the 10fps condition — see Sec. 4.3 of the paper)
ax.text(9.0, 0.55, r'$\mathcal{L}_{\rm wta}$', ha='center', va='center',
        fontsize=10, color=ORANGE)
ax.annotate('', xy=(9.0, 0.68), xytext=(9.0, 1.0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))

ax.text(1.35, 0.35, 'SMPL\n6D repr.', ha='center', va='center',
        fontsize=7, color=ORANGE, style='italic')
ax.text(5.0, 0.3, 'WTA-L1 aux. gradients are the only\npath shaping shared features',
        ha='center', va='center', fontsize=7, color=ORANGE, style='italic')

ax.set_title('MTR+Pose architecture: pose branch fused with MTR agent features',
             fontsize=8, pad=4)

plt.tight_layout(pad=0.4)
plt.savefig('fig_architecture.pdf', bbox_inches='tight')
plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_architecture.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — (a) seed-replicated supervision ablation; (b) WTA-L1 weight,
#            single-seed sweep against measured seed noise
# ═══════════════════════════════════════════════════════════════════════════

baseline_v = cell_best_values('mtr+pose_data_no_pose', 'baseline', S5)
noaux_v    = cell_best_values('mtr+pose_data_geo_pure', 'geo_pure', S5)
wta_v      = (cell_best_values('mtr+pose_data_geo_only', 'wta01', S5)
              + cell_best_values('mtr+pose_data_gmm_only', 'gmm_only', S3))  # same config, pooled
mpjpe_v    = cell_best_values('mtr+pose_data_mpjpe_only', 'mpjpe', S3)
full_v     = cell_best_values('mtr+pose_data', 'full', S3)
xattnpe_v  = cell_best_values('mtr+pose_data_cross_attn_pe', 'xattn_pe', S3)

print("panel (a) data:")
for name, v in [('baseline', baseline_v), ('no aux', noaux_v), ('WTA pooled', wta_v),
                ('mpjpe', mpjpe_v), ('full', full_v), ('xattn_pe', xattnpe_v)]:
    print(f"  {name:<11} n={len(v)} mean={np.mean(v):.4f} values={[f'{x:.4f}' for x in v]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

configs = ['No pose\n(baseline)', 'No aux\n(GRU)', 'WTA-L1\n(GRU, 8 runs)',
           'MPJPE+WTA\n(GRU)', 'All\n(GRU)']
values  = [baseline_v, noaux_v, wta_v, mpjpe_v, full_v]
colors  = [GRAY, LBLUE, LORANGE, LORANGE, LORANGE]

rng = np.random.RandomState(0)
for i, (v, c) in enumerate(zip(values, colors)):
    v = np.asarray(v)
    mean = v.mean()
    # mean bar (thin) + per-seed points
    ax1.hlines(mean, i - 0.28, i + 0.28, color='black', linewidth=1.6, zorder=4)
    jitter = rng.uniform(-0.14, 0.14, size=len(v))
    ax1.scatter(i + jitter, v, s=18, facecolor=c, edgecolor='black',
                linewidth=0.5, zorder=3, alpha=0.95)

bm = np.mean(baseline_v)
ax1.axhline(bm, color=GRAY, linestyle='--', linewidth=0.8, zorder=1)
ax1.text(4.45, bm + 0.003, 'baseline\nmean', fontsize=6, color=GRAY, ha='right')

ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(configs, fontsize=6.5)
ax1.set_ylabel('minADE $\\downarrow$')
ax1.set_title('(a) Supervision ablation (per-seed)')
ax1.set_ylim(0.60, 0.92)

# panel (b): single-seed draft sweep vs seed noise at w=0.1
weights      = [0.05, 0.10, 0.20]
draft_sweep  = [0.6786, 0.6231, 0.6660]   # draft single runs 011 / 010 / 012

# baseline seed range band
ax2.axhspan(min(baseline_v), max(baseline_v), color=GRAY, alpha=0.18, zorder=0)
ax2.axhline(bm, color=GRAY, linestyle='--', linewidth=0.8, zorder=1)

# the 8 replicates at w=0.1 (same config as the draft's w=0.1 point)
jitter = rng.uniform(-0.004, 0.004, size=len(wta_v))
ax2.scatter(0.10 + jitter, wta_v, s=16, facecolor='white', edgecolor=ORANGE,
            linewidth=0.9, zorder=3, label='replicates at $w{=}0.1$ (8 seeds)')

ax2.plot(weights, draft_sweep, 'o-', color=BLUE, linewidth=1.2, markersize=5,
         zorder=4, label='draft sweep (1 seed)')

ax2.set_xlabel('WTA-L1 weight $w_{\\rm wta}$')
ax2.set_ylabel('minADE $\\downarrow$')
ax2.set_title('(b) Weight sweep vs. seed noise')
ax2.set_xlim(0.02, 0.24)
ax2.set_ylim(0.60, 0.92)
ax2.set_xticks(weights)
ax2.set_xticklabels(['0.05', '0.10', '0.20'])
ax2.legend(loc='upper right', frameon=False, fontsize=6.5)
ax2.text(0.225, min(baseline_v) - 0.012, 'baseline seed range', fontsize=6,
         color=GRAY, ha='right')

plt.tight_layout(pad=0.5)
plt.savefig('fig_ablation.pdf', bbox_inches='tight')
plt.savefig('fig_ablation.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Training dynamics: mean curve + min-max band over seeds
# ═══════════════════════════════════════════════════════════════════════════

CELLS_FIG3 = [
    ('mtr+pose_data_no_pose',        'baseline', S5, GRAY,   '--', 'MTR baseline (no pose)'),
    ('mtr+pose_data_cross_attn',     'xattn',    S3, ORANGE, '-.', 'Cross-attn, no PE'),
    ('mtr+pose_data_cross_attn_pe',  'xattn_pe', S3, GREEN,  ':',  'Cross-attn + sinus. PE'),
    ('mtr+pose_data_geo_only',       'wta01',    S5, BLUE,   '-',  'GRU + WTA-L1'),
]

fig, ax = plt.subplots(figsize=(5.5, 2.8))

for cfg, cell, seeds, color, ls, label in CELLS_FIG3:
    runs = cell_runs(cfg, cell, seeds)
    epochs = sorted(set.intersection(*[set(r) for r in runs]))
    arr = np.array([[r[e] for e in epochs] for r in runs])  # (n_seeds, n_epochs)
    mean = arr.mean(axis=0)
    ax.fill_between(epochs, arr.min(axis=0), arr.max(axis=0),
                    color=color, alpha=0.15, linewidth=0)
    ax.plot(epochs, mean, color=color, linewidth=1.5, linestyle=ls,
            label=f'{label} ({len(runs)} seeds)')

ax.set_xlabel('Training epoch')
ax.set_ylabel('Validation minADE $\\downarrow$')
ax.set_title('Training dynamics: mean over seeds, band = per-seed range')
ax.set_xlim(1, 30)
ax.set_ylim(0.58, 1.05)   # early epochs (>1.0) clipped for readability
ax.legend(loc='upper right', frameon=False, fontsize=6.5)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.text(1.5, 0.595, 'y-axis clipped at 1.05 (epochs 1–4 start near 2.0)',
        fontsize=6, color='gray', style='italic')

plt.tight_layout(pad=0.4)
plt.savefig('fig_training_curves.pdf', bbox_inches='tight')
plt.savefig('fig_training_curves.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_training_curves.pdf")

print("\nAll figures generated.")
