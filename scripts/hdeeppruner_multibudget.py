import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
})

budgets = [1, 10, 25, 50, 75, 100]

data = {
    'Facebook': [1.084, 0.990, 0.978, 0.955, 0.932, 0.966],
    'Wiki':     [1.063, 1.014, 1.001, 1.011, 0.968, 0.969],
    'Deezer':   [0.948, 0.928, 0.967, 0.986, 0.985, 0.996],
    'Slashdot': [1.011, 1.057, 0.961, 0.976, 1.019, 0.986],
    'Twitter':  [1.024, 0.990, 0.950, 0.976, 0.982, 0.918],
    'DBLP':     [0.615, 0.659, 0.763, 0.810, 0.852, 0.835],
    'YouTube':  [1.034, 1.005, 1.022, 0.992, 0.957, 0.958],
    'Skitter':  [0.980, 1.042, 0.962, 0.995, 0.929, 0.949],
}

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'd']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
          '#795548', '#607D8B', '#FF5722']

fig, ax = plt.subplots(figsize=(7, 4.5))

for (name, vals), marker, color in zip(data.items(), markers, colors):
    ax.plot(budgets, vals, marker=marker, color=color, label=name,
            linewidth=1.5, markersize=6)

# Reference line at Pr = 1
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('Budget $k$')
ax.set_ylabel('Pruning Approximation Ratio $P_r$')
ax.set_xticks(budgets)
ax.set_ylim(0.55, 1.12)
ax.legend(ncol=2, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/multibudget.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/multibudget.pdf")
