import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
})

budgets = [1, 10, 25, 50, 75, 100]
datasets = ['Facebook', 'Wiki', 'Deezer', 'Slashdot', 'Twitter', 'DBLP', 'YouTube', 'Skitter']

data = np.array([
    [1.084, 0.990, 0.978, 0.955, 0.932, 0.966],
    [1.063, 1.014, 1.001, 1.011, 0.968, 0.969],
    [0.948, 0.928, 0.967, 0.986, 0.985, 0.996],
    [1.011, 1.057, 0.961, 0.976, 1.019, 0.986],
    [1.024, 0.990, 0.950, 0.976, 0.982, 0.918],
    [0.615, 0.659, 0.763, 0.810, 0.852, 0.835],
    [1.034, 1.005, 1.022, 0.992, 0.957, 0.958],
    [0.980, 1.042, 0.962, 0.995, 0.929, 0.949],
])

# Custom colormap: deep coral -> warm white -> teal
custom_colors = ['#d73027', '#fc8d59', '#fee08b', '#ffffbf',
                 '#d9ef8b', '#91cf60', '#1a9850']
cmap = LinearSegmentedColormap.from_list('custom', custom_colors, N=256)

fig, ax = plt.subplots(figsize=(7.5, 4))

im = ax.pcolormesh(data, cmap=cmap, vmin=0.6, vmax=1.1, edgecolors='white', linewidth=2)

# Annotate each cell
for i in range(len(datasets)):
    for j in range(len(budgets)):
        val = data[i, j]
        text_color = 'white' if val < 0.72 else 'black'
        ax.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center',
                fontsize=11, color=text_color, fontweight='bold')

ax.set_xticks(np.arange(len(budgets)) + 0.5)
ax.set_xticklabels([f'$k={b}$' for b in budgets])
ax.set_yticks(np.arange(len(datasets)) + 0.5)
ax.set_yticklabels(datasets)
ax.invert_yaxis()

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(length=0)

cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=25)
cbar.set_label('$P_r$', rotation=0, labelpad=15, fontsize=14)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/multibudget.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/multibudget.pdf")