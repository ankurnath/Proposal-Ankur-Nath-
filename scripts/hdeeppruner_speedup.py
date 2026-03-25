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

# Estimated from original figure
datasets = ['Facebook', 'Wiki', 'Deezer', 'Slashdot', 'Twitter', 'DBLP', 'YouTube', 'Skitter']

speedup = {
    'H-DeepPruner': [3.5, 3.0, 12.5, 8.0, 5.0, 40.0, 9.5, 6.0],
    'GCOMB':        [1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0],
    'LeNSE':        [1.5, 2.5, 1.0,  2.0, 2.5, 13.0, 4.0, 1.0],
    'COMBHelper':   [1.0, 1.5, 3.0,  3.0, 1.0, 1.0,  5.0, 3.0],
}

methods = ['H-DeepPruner', 'LeNSE', 'COMBHelper', 'GCOMB']
colors = ['#2196F3', '#E91E63', '#FF9800', '#4CAF50']

# Sort datasets by H-DeepPruner speed-up
order = np.argsort(speedup['H-DeepPruner'])
sorted_datasets = [datasets[i] for i in order]

fig, ax = plt.subplots(figsize=(7, 4.5))

bar_height = 0.18
y_pos = np.arange(len(datasets))

for j, (method, color) in enumerate(zip(methods, colors)):
    vals = [speedup[method][i] for i in order]
    ax.barh(y_pos + j * bar_height, vals, height=bar_height, color=color,
            edgecolor='white', linewidth=0.5, label=method)

ax.set_xscale('log')
ax.set_xlabel('Speed-up $S$ (log scale)')
ax.set_yticks(y_pos + 1.5 * bar_height)
ax.set_yticklabels(sorted_datasets)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/speedup_new.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/speedup_new.pdf")