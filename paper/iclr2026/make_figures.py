"""
Generate all paper figures for ICLR 2026 submission.
Outputs: fig_architecture.pdf, fig_ablation.pdf, fig_training_curves.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

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

# ── Inputs ──────────────────────────────────────────────────────────────────
# Trajectory input
draw_box(ax, (0.1, 2.3), 1.2, 0.8, 'Agent\npositions\n(past 11)', color='#e8f4f8', ec=BLUE)
# Pose input
draw_box(ax, (0.1, 0.9), 1.2, 0.8, 'SMPL poses\n(past 11)\n6D repr.', color='#fef3e2', ec=ORANGE)
# Map input
draw_box(ax, (0.1, 3.0), 0.0, 0.0, '', color='white', ec='white')  # spacer

# ── MTR Backbone ─────────────────────────────────────────────────────────
draw_box(ax, (1.8, 2.1), 1.6, 1.2, 'MTR Backbone\n(Transformer\nencoder)', color='#e8f4f8', ec=BLUE)
draw_arrow(ax, 1.3, 2.7, 1.8, 2.7, color=BLUE)

# ── GRU Pose Encoder ─────────────────────────────────────────────────────
draw_box(ax, (1.8, 0.7), 1.6, 1.0, 'GRU Pose\nEncoder\n(256-d)', color='#fef3e2', ec=ORANGE)
draw_arrow(ax, 1.3, 1.3, 1.8, 1.3, color=ORANGE)

# ── Fusion ───────────────────────────────────────────────────────────────
draw_box(ax, (4.0, 1.6), 1.4, 1.2, 'Concat +\nMLP\nProjection', color='#f0f0f0', ec=GRAY)
# arrows into fusion
draw_arrow(ax, 3.4, 2.7, 4.0, 2.1, color=BLUE)
draw_arrow(ax, 3.4, 1.2, 4.0, 1.9, color=ORANGE)

# ── MTR Decoder ──────────────────────────────────────────────────────────
draw_box(ax, (6.0, 1.6), 1.5, 1.2, 'MTR Decoder\n(6 refinement\nlayers)', color='#e8f4f8', ec=BLUE)
draw_arrow(ax, 5.4, 2.2, 6.0, 2.2, color='black')

# ── Outputs ──────────────────────────────────────────────────────────────
draw_box(ax, (8.2, 2.4), 1.6, 0.8, '6 trajectory\nmodes\n(minADE)', color='#d4edda', ec=GREEN, bold=True)
draw_box(ax, (8.2, 1.0), 1.6, 0.8, 'Future pose\nprediction\n(aux. task)', color='#fff3cd', ec=ORANGE)
draw_arrow(ax, 7.5, 2.6, 8.2, 2.8, color=BLUE)
draw_arrow(ax, 7.5, 1.8, 8.2, 1.4, color=ORANGE)

# ── Losses ────────────────────────────────────────────────────────────────
ax.text(9.0, 0.55, r'$\mathcal{L}_{\rm geo}$', ha='center', va='center',
        fontsize=10, color=ORANGE)
ax.annotate('', xy=(9.0, 0.68), xytext=(9.0, 1.0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))

# ── Labels ────────────────────────────────────────────────────────────────
ax.text(1.35, 0.35, 'SMPL\n6D repr.', ha='center', va='center',
        fontsize=7, color=ORANGE, style='italic')
ax.text(5.0, 0.3, 'Geodesic loss\nshapes GRU encoder', ha='center', va='center',
        fontsize=7, color=ORANGE, style='italic')

# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title('MTR+Pose architecture: GRU pose branch fused with MTR agent features',
             fontsize=8, pad=4)

plt.tight_layout(pad=0.4)
plt.savefig('fig_architecture.pdf', bbox_inches='tight')
plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_architecture.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation Results
# Two panels: (a) loss ablation bar chart, (b) geo weight sensitivity
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))

# ── Panel (a): Loss ablation ───────────────────────────────────────────────
configs  = ['No pose\n(baseline)', 'MPJPE\nonly', 'GMM NLL\nonly', 'Full\n(all 3)', 'Geo only\n(ours)']
minADEs  = [0.6745,                 0.6576,         0.6467,           0.6532,          0.6231]
colors   = [GRAY, LORANGE, LBLUE, LBLUE, BLUE]
bars = ax1.bar(configs, minADEs, color=colors, edgecolor='black', linewidth=0.6, width=0.6)
# highlight best
bars[4].set_edgecolor(BLUE)
bars[4].set_linewidth(1.5)

ax1.set_ylim(0.60, 0.695)
ax1.set_ylabel('minADE $\\downarrow$')
ax1.set_title('(a) Pose supervision loss ablation')
ax1.axhline(0.6745, color=GRAY, linestyle='--', linewidth=0.8, label='No-pose baseline')

# annotate values
for bar, v in zip(bars, minADEs):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.4f}',
             ha='center', va='bottom', fontsize=7)

# delta annotation
ax1.annotate('', xy=(4.3, 0.6231), xytext=(4.3, 0.6745),
             arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.2))
ax1.text(4.48, 0.649, '−7.6%', fontsize=7, color=BLUE)

ax1.tick_params(axis='x', labelsize=7)

# ── Panel (b): Geo weight sensitivity ─────────────────────────────────────
weights = [0.05, 0.10, 0.20]
ades    = [0.6786, 0.6231, 0.6660]

ax2.plot(weights, ades, 'o-', color=BLUE, linewidth=1.5, markersize=6, markerfacecolor=BLUE)
ax2.axhline(0.6745, color=GRAY, linestyle='--', linewidth=0.8, label='No-pose baseline')

# shade near-baseline region
ax2.fill_between([0.04, 0.22], [0.670, 0.670], [0.68, 0.68], alpha=0.08, color=GRAY)

ax2.set_xlabel('Geodesic loss weight $w_{\\rm geo}$')
ax2.set_ylabel('minADE $\\downarrow$')
ax2.set_title('(b) Geodesic loss weight sensitivity')
ax2.set_xlim(0.03, 0.23)
ax2.set_ylim(0.610, 0.695)
ax2.set_xticks(weights)
ax2.set_xticklabels(['0.05', '0.10\n(best)', '0.20'])

for w, v in zip(weights, ades):
    offset = -0.004 if w == 0.10 else 0.002
    ax2.text(w, v + offset, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

ax2.legend(loc='upper right', frameon=False)
ax2.tick_params(axis='x', labelsize=7)

plt.tight_layout(pad=0.5)
plt.savefig('fig_ablation.pdf', bbox_inches='tight')
plt.savefig('fig_ablation.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Training Curves (minADE vs epoch)
# Real per-epoch validation minADE from training logs
# ═══════════════════════════════════════════════════════════════════════════

# Real data extracted from training logs
# H3_geo_only: log_train_20260402-121052.txt — best 0.6231 at epoch 20
epochs_22 = np.arange(1, 23)
geo_only = np.array([2.0209, 1.7540, 1.3268, 0.9699, 0.7982, 0.7116, 0.6655, 0.6860,
                     0.6413, 0.6856, 0.6876, 0.6530, 0.6712, 0.6511, 0.6539, 0.6415,
                     0.6281, 0.6497, 0.6405, 0.6231, 0.6379, 0.6379])

# H2_v2_real_traj (no pose): log_train_20260402-091913.txt — best 0.6745 at epoch 11
baseline = np.array([2.0849, 1.6412, 1.2154, 1.0657, 0.9204, 0.7891, 0.6900, 0.6997,
                     0.7160, 0.6950, 0.6745, 0.6823, 0.7068, 0.6835, 0.6873, 0.6916,
                     0.6793, 0.6853, 0.6891, 0.6814, 0.7006, 0.7006])

# H5_cross_attn_geo_only: log_train_20260402-142149.txt — best 0.6765 at epoch 18
cross_attn = np.array([2.1170, 1.7228, 1.2051, 0.9088, 0.8141, 0.8101, 0.7746, 0.6908,
                       0.6837, 0.6793, 0.7466, 0.6952, 0.7294, 0.7304, 0.7301, 0.7108,
                       0.6889, 0.6765, 0.6777, 0.7199, 0.7002, 0.7002])

# H5b_cross_attn_pe: log_train_20260403-131827.txt — best 0.6337 at epoch 24
# Evaluated at every 2 epochs (1-20) then every epoch (20-29)
cross_attn_pe_epochs = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
cross_attn_pe = np.array([2.1271, 1.7347, 1.1778, 0.9001, 0.7585, 0.6970, 0.6964, 0.6982,
                          0.6644, 0.6758, 0.6365, 0.6491, 0.6498, 0.6447, 0.6337, 0.7420,
                          0.6559, 0.6436, 0.6401, 0.6517, 0.6751])

fig, ax = plt.subplots(figsize=(5.5, 2.8))

ax.plot(epochs_22,          baseline,   color=GRAY,   linewidth=1.5, label='MTR baseline (no pose), best=0.6745', linestyle='--')
ax.plot(epochs_22,          cross_attn, color=ORANGE, linewidth=1.5, label='Cross-attn, no PE, best=0.6765',       linestyle='-.')
ax.plot(cross_attn_pe_epochs, cross_attn_pe, color=GREEN, linewidth=1.5, label='Cross-attn + sinus. PE, best=0.6337', linestyle=':')
ax.plot(epochs_22,          geo_only,   color=BLUE,   linewidth=1.5, label='GRU + geo (ours), best=0.6231')

# Mark best epochs
ax.axvline(11, color=GRAY,   linewidth=0.7, linestyle=':', alpha=0.5)
ax.axvline(20, color=BLUE,   linewidth=0.7, linestyle=':', alpha=0.6)
ax.axvline(24, color=GREEN,  linewidth=0.7, linestyle=':', alpha=0.6)

ax.annotate('ep.20\n0.6231', xy=(20, 0.6231), xytext=(16.5, 0.637),
            fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
ax.annotate('ep.24\n0.6337', xy=(24, 0.6337), xytext=(25.5, 0.646),
            fontsize=6.5, color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.8))

ax.set_xlabel('Training epoch')
ax.set_ylabel('Validation minADE $\\downarrow$')
ax.set_title('Training dynamics: temporal encoding comparison')
ax.set_xlim(1, 30)
ax.legend(loc='upper right', frameon=False, fontsize=6.5)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

plt.tight_layout(pad=0.4)
plt.savefig('fig_training_curves.pdf', bbox_inches='tight')
plt.savefig('fig_training_curves.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved fig_training_curves.pdf")

print("\nAll figures generated.")
