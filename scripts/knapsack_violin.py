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
        'QuickPrune': [0.960, 0.926, 0.874, 0.894, 0.869, 0.845, 0.915, 0.905],
        'Top-$k$':    [0.543, 0.744, 0.913, 0.569, 0.594, 0.715, 0.520, 0.815],
        'GNNPruner':  [0.328, 0.870, 0.973, 0.833, 0.478, 0.997, 0.363, 0.989],
    },
    'MaxCut': {
        'QuickPrune': [0.957, 0.953, 0.959, 0.969, 0.950, 0.950, 0.970, 0.970],
        'Top-$k$':    [0.967, 0.506, 0.799, 0.669, 0.290, 0.430, 0.730, 1.000],
        'GNNPruner':  [0.483, 0.184, 0.975, 0.440, 0.022, 0.999, 0.941, 0.999],
    },
    'Influence Maximization': {
        'QuickPrune': [0.909, 0.848, 0.982, 0.901, 0.911, 0.996, 0.917, 0.829],
        'Top-$k$':    [0.909, 0.877, 0.982, 0.999, 0.968, 0.996, 0.950, 0.939],
        'GNNPruner':  [0.116, 0.703, 0.455, 0.621, 0.140, 1.000, 0.324, 0.140],
    },
}

methods = ['QuickPrune', 'Top-$k$', 'GNNPruner']
colors = ['#2196F3', '#FF9800', '#4CAF50']
problems = list(data.keys())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for ax, problem in zip(axes, problems):
    plot_data = [data[problem][m] for m in methods]
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
    ax.set_xticklabels(methods)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Combined Metric ($C = P_r \\cdot P_g$)')

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/quickprune/knapsack_violin.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/quickprune/knapsack_violin.pdf")
