import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
})

# Ablation data: C = Pr * Pg for GNN-only vs H-DeepPruner
# Order: Facebook, Wiki, Deezer, Slashdot, Twitter, DBLP, YouTube, Skitter
datasets = ['Facebook', 'Wiki', 'Deezer', 'Slashdot', 'Twitter', 'DBLP', 'YouTube', 'Skitter']

gnn_only = {
    'MaxCover': [0.777, 0.863, 0.763, 0.857, 0.852, 0.790, 0.844, 0.863],
    'MaxCut':   [0.765, 0.864, 0.754, 0.857, 0.852, 0.797, 0.855, 0.999],
    'IM':       [0.460, 0.750, 0.413, 0.740, 0.691, 0.516, 0.625, 0.601],
}

hdeeppruner = {
    'MaxCover': [0.936, 0.958, 0.993, 0.987, 0.969, 0.969, 0.997, 0.959],
    'MaxCut':   [0.879, 0.966, 0.993, 0.997, 0.971, 0.963, 0.981, 0.667],
    'IM':       [0.874, 0.934, 0.961, 0.994, 0.959, 0.850, 0.932, 0.883],
}

problems = ['MaxCover', 'MaxCut', 'IM']
labels = ['Maximum Cover', 'Maximum Cut', 'Influence Maximization']
markers = ['o', 's', 'D']
colors = ['#2196F3', '#FF9800', '#4CAF50']

fig, ax = plt.subplots(figsize=(6, 5.5))

# Diagonal reference line
ax.plot([0, 1.05], [0, 1.05], 'k--', alpha=0.4, linewidth=1, zorder=1)

for prob, label, marker, color in zip(problems, labels, markers, colors):
    ax.scatter(gnn_only[prob], hdeeppruner[prob], marker=marker, color=color,
               edgecolors='black', s=70, linewidths=0.6, label=label, zorder=3)

ax.set_xlabel('GNN-Only $C$')
ax.set_ylabel('H-DeepPruner (GNN + NMCTS) $C$')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.6, 1.05)
ax.set_aspect('equal')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/ablation_scatter.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/ablation_scatter.pdf")
