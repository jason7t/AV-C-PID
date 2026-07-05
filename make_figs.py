import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv('/mnt/user-data/uploads/DAM_metrics.csv')

models = ['KNN', 'LEAR', 'RF', 'LGBM']
methods = ['QR', 'SCP', 'EnbPI', 'SPCI', 'QRA-R', 'QRA-CP', 'Q-Ens', 'AV-C-PID', 'PID-Q-Ens']

# Color shading per method (blue family, like the paper), darkest first
cmap = plt.cm.Blues
n = len(methods)
method_colors = {m: cmap(0.35 + 0.5 * i / (n - 1)) for i, m in enumerate(methods)}

# pivot helpers
def get(metric):
    p = df.pivot(index='Model', columns='Method', values=metric)
    return p.loc[models, methods]

iw80 = get('IW80')
iw40 = get('IW40')
cov80 = get('Cov80')
cov40 = get('Cov40')

# ----------------------------------------------------------------
# Fig 3: Interval width -- range bar (median +/- half of IW80 outer, IW40 inner)
# We don't have explicit lower/upper quantile values, only widths.
# Center everything on 50 (median proxy) like the paper's visual style.
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

bar_width = 0.8
positions = []
labels = []
x = 0
group_gap = 1.2
within_gap = 1.0

centers_per_group = []
for method in methods:
    group_centers = []
    for model in models:
        center = 50  # proxy center, paper does similarly (visual width only)
        half80 = iw80.loc[model, method] / 2
        half40 = iw40.loc[model, method] / 2

        # Outer (80%) box
        ax.bar(x, half80 * 2, bottom=center - half80, width=bar_width,
               color=method_colors[method], edgecolor='black', linewidth=0.6, alpha=0.55)
        # Inner (40%) box, darker
        ax.bar(x, half40 * 2, bottom=center - half40, width=bar_width,
               color=method_colors[method], edgecolor='black', linewidth=0.8, alpha=1.0)

        positions.append(x)
        labels.append(model)
        group_centers.append(x)
        x += within_gap
    centers_per_group.append(np.mean(group_centers))
    x += group_gap

ax.axhline(50, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('Interval Width')
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_ylim(0, 100)

# Method group labels on secondary axis
ax2 = ax.secondary_xaxis('top')
ax2.set_xticks(centers_per_group)
ax2.set_xticklabels(methods)

legend_patches = [mpatches.Patch(color=method_colors[m], label=m) for m in methods]
ax.legend(handles=legend_patches, ncol=len(methods), loc='upper center',
          bbox_to_anchor=(0.5, 1.18), frameon=True, fontsize=9)

ax.set_title('')
fig.tight_layout()
fig.savefig('/home/claude/figs/fig3_interval_width.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ----------------------------------------------------------------
# Fig 4 & 5: Coverage bar charts
# ----------------------------------------------------------------
def plot_coverage(cov_df, target_line, title, outfile):
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = []
    labels = []
    x = 0
    centers_per_group = []
    for method in methods:
        group_centers = []
        for model in models:
            val = cov_df.loc[model, method]
            ax.bar(x, val, width=bar_width, color=method_colors[method],
                   edgecolor='black', linewidth=0.6)
            positions.append(x)
            labels.append(model)
            group_centers.append(x)
            x += within_gap
        centers_per_group.append(np.mean(group_centers))
        x += group_gap

    ax.axhline(target_line, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Coverage')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, 1)

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(centers_per_group)
    ax2.set_xticklabels(methods)

    legend_patches = [mpatches.Patch(color=method_colors[m], label=m) for m in methods]
    ax.legend(handles=legend_patches, ncol=len(methods), loc='upper center',
              bbox_to_anchor=(0.5, 1.18), frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)

plot_coverage(cov80, 0.8, 'Coverage for 0.1-0.9 quantile pair', '/home/claude/figs/fig4_coverage_0.1_0.9.png')
plot_coverage(cov40, 0.4, 'Coverage for 0.3-0.7 quantile pair', '/home/claude/figs/fig5_coverage_0.3_0.7.png')

print("done")
