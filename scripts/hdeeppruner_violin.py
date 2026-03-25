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

# Combined metric C = Pr * Pg for each method across all datasets
# Order: Facebook, Wiki, Deezer, Slashdot, Twitter, DBLP, YouTube, Skitter

data = {
    'MaxCover': {
        'H-DeepPruner': [0.936, 0.958, 0.993, 0.987, 0.969, 0.969, 0.997, 0.959],
        'GCOMB':        [0.065, 0.030, 0.129, 0.020, 0.170, 0.030, 0.070, 0.100],
        'COMBHelper':   [0.384, 0.453, 0.728, 0.984, 0.565, 0.182, 0.957, None],  # Skitter timeout
        'LeNSE':        [0.068, 0.372, 0.734, 0.676, 0.326, 0.891, 0.776, 0.683],
    },
    'MaxCut': {
        'H-DeepPruner': [0.879, 0.966, 0.993, 0.997, 0.971, 0.963, 0.981, 0.671],
        'GCOMB':        [0.772, 0.883, 0.842, 0.626, 0.622, 0.640, 0.531, 0.423],
        'COMBHelper':   [0.154, 0.701, 0.375, 0.788, 0.554, 0.183, 0.970, None],  # Skitter timeout
        'LeNSE':        [0.070, 0.383, 0.722, 0.614, 0.474, 0.914, 0.780, 0.692],
    },
    'Influence Maximization': {
        'H-DeepPruner': [0.874, 0.934, 0.961, 0.994, 0.959, 0.850, 0.932, 0.883],
        'GCOMB':        [0.694, 0.872, 0.443, 0.947, 0.902, 0.854, 0.924, 0.874],
        'COMBHelper':   [0.326, 0.605, 0.108, 0.995, 0.493, 0.019, 0.970, None],  # Skitter timeout
        'LeNSE':        [0.088, 0.490, 0.739, 0.744, 0.386, 0.862, 0.728, 0.767],
    },
}

methods = ['H-DeepPruner', 'GCOMB', 'COMBHelper', 'LeNSE']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
problems = list(data.keys())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for ax, problem in zip(axes, problems):
    # Filter out None values
    plot_data = []
    for m in methods:
        vals = [v for v in data[problem][m] if v is not None]
        plot_data.append(vals)

    parts = ax.violinplot(plot_data, positions=range(len(methods)),
                          showmeans=True, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)

    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)

    # Overlay individual points
    for i, (m, vals) in enumerate(zip(methods, plot_data)):
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=colors[i],
                   edgecolors='black', s=30, zorder=3, linewidths=0.5)

    ax.set_title(problem)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Combined Metric ($C = P_r \\cdot P_g$)')

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/hdp_violin.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/hdp_violin.pdf")