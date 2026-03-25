import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
})

budgets = [1, 10, 25, 50, 75, 100]
datasets = ['Facebook', 'Wiki', 'Deezer', 'Slashdot', 'Twitter', 'DBLP', 'YouTube', 'Skitter']

data = np.array([
    [1.084, 0.990, 0.978, 0.955, 0.932, 0.966],  # Facebook
    [1.063, 1.014, 1.001, 1.011, 0.968, 0.969],  # Wiki
    [0.948, 0.928, 0.967, 0.986, 0.985, 0.996],  # Deezer
    [1.011, 1.057, 0.961, 0.976, 1.019, 0.986],  # Slashdot
    [1.024, 0.990, 0.950, 0.976, 0.982, 0.918],  # Twitter
    [0.615, 0.659, 0.763, 0.810, 0.852, 0.835],  # DBLP
    [1.034, 1.005, 1.022, 0.992, 0.957, 0.958],  # YouTube
    [0.980, 1.042, 0.962, 0.995, 0.929, 0.949],  # Skitter
])

fig, ax = plt.subplots(figsize=(7, 4))

cmap = plt.cm.RdYlGn
im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.6, vmax=1.1)

# Annotate each cell with its value
for i in range(len(datasets)):
    for j in range(len(budgets)):
        val = data[i, j]
        text_color = 'white' if val < 0.75 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='medium')

ax.set_xticks(range(len(budgets)))
ax.set_xticklabels([f'$k={b}$' for b in budgets])
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('$P_r$', rotation=0, labelpad=15)

plt.tight_layout()
plt.savefig('/home/grads/a/anath/Proposal-Ankur-Nath-/figures/hdeeppruner/multibudget.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved to figures/hdeeppruner/multibudget.pdf")
