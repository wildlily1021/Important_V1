# Regenerate PDFs with requested modifications:
# - Add 'C40_C21' into first-order group.
# - Adjust group sample sizes to shape aggregated density:
#   aim: density peaks at alpha=1.0 and alpha=1.5 around ~1.0; peak at 2.0 around ~1.5 (slightly higher).
# - Ensure PDF font type uses TrueType (Type 42) to avoid Type-3.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Force TrueType embedding to avoid Type-3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Font settings: Times New Roman preferred, fallback to serif
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

np.random.seed(2025)

# Define groups with corrected name and added C40_C21 into first-order
first_order = [
    'C21_C20', 'C42_C21', 'C60_C21', 'C60_C42', 'C61_C60',
    'C62_C42', 'C63_C40', 'C63_C41', 'C63_C42', 'C40_C21'
]
uncertain = [
    'C40_C20','C41_C40','C42_C20','C60_C20','C60_C40',
    'C61_C20','C61_C21','C62_C21','C62_C41','C62_C60'
]
second_order = [
    'C41_C20','C42_C40','C42_C41','C60_C41','C61_C40','C61_C41','C61_C42',
    'C62_C20','C62_C40','C62_C61','C63_C20','C63_C21','C63_C61'
]

# Colors
first_col = 'tab:blue'
uncertain_col = (149/255, 207/255, 149/255)  # 1.5 group
second_col = (234/255, 147/255, 147/255)   # 2.0 group

# To shape density peaks, control total sample contributions and std deviation.
# We'll create per-combo samples but also weight group totals by varying n_boot.
n_boot_first = 180    # larger to make alpha=1 peak prominent (~1.0)
n_boot_uncertain = 160  # make alpha=1.5 peak similar (~1.0)
n_boot_second = 260   # keep 2 peak slightly higher (~1.5) but not too dominant

alpha_vals = {}
for name in first_order:
    alpha_vals[name] = np.random.normal(loc=1.00, scale=0.10, size=n_boot_first)
for name in uncertain:
    alpha_vals[name] = np.random.normal(loc=1.50, scale=0.12, size=n_boot_uncertain)
for name in second_order:
    alpha_vals[name] = np.random.normal(loc=2.00, scale=0.12, size=n_boot_second)

# Add slight outliers
for k, arr in alpha_vals.items():
    idx = np.random.choice(len(arr), size=max(1, int(0.015 * len(arr))), replace=False)
    arr[idx] += np.random.choice([-0.6, 0.6], size=len(idx))
    alpha_vals[k] = arr

# Aggregated density via histogram smoothing
all_alphas = np.concatenate([alpha_vals[k] for k in alpha_vals.keys()])
bins = np.linspace(-0.5, 3.0, 300)
hist, edges = np.histogram(all_alphas, bins=bins, density=True)
xs = 0.5 * (edges[:-1] + edges[1:])
smooth = gaussian_filter1d(hist, sigma=2.0)

plt.figure(figsize=(8, 4.5))
plt.plot(xs, smooth, lw=2, color='tab:blue')
plt.fill_between(xs, smooth, alpha=0.25, color='tab:blue')

# rug colored by group membership
alpha_points = []
alpha_colors = []
for name in first_order:
    alpha_points.append(alpha_vals[name])
    alpha_colors.extend([first_col]*len(alpha_vals[name]))
for name in uncertain:
    alpha_points.append(alpha_vals[name])
    alpha_colors.extend([uncertain_col]*len(alpha_vals[name]))
for name in second_order:
    alpha_points.append(alpha_vals[name])
    alpha_colors.extend([second_col]*len(alpha_vals[name]))
alpha_points = np.concatenate(alpha_points)
alpha_colors = np.array(alpha_colors)

plt.scatter(alpha_points, np.full_like(alpha_points, -0.002), s=6, alpha=0.35, color=alpha_colors)
plt.axvline(1.0, color='k', linestyle='--', linewidth=1)
plt.axvline(1.5, color='gray', linestyle=':', linewidth=0.8)
plt.axvline(2.0, color='k', linestyle='--', linewidth=1)
plt.xlim(-0.5, 3.0)
plt.ylim(bottom=-0.01)
plt.ylabel('Density')
plt.xlabel('Estimated slope α (all candidate combos)')
plt.title('Aggregated distribution of α across candidate combinations (simulated)')

pct_below1 = np.mean(all_alphas < 1.0) * 100
pct_1_2 = np.mean((all_alphas >= 1.0) & (all_alphas <= 2.0)) * 100
pct_above2 = np.mean(all_alphas > 2.0) * 100
txt = f'Pct <1: {pct_below1:.1f}%  |  1≤α≤2: {pct_1_2:.1f}%  |  >2: {pct_above2:.1f}%'
plt.gca().text(0.02, 0.95, txt, transform=plt.gca().transAxes, fontsize=11,
               va='top', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

agg_pdf = 'alpha_aggregated_updated.pdf'
plt.tight_layout()
plt.savefig(agg_pdf, dpi=600, bbox_inches='tight', format='pdf')
plt.close()

# Two-panel: First-order (left) and Uncertain (right)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, group, title, col in zip(axes, [first_order, uncertain],
                                 ['First-order ratios', 'Uncertain (~1.5) ratios'],
                                 [first_col, uncertain_col]):
    for i, name in enumerate(group):
        arr = alpha_vals[name]
        xpos = np.full_like(arr, i) + (np.random.rand(len(arr)) - 0.5) * 0.18
        ax.scatter(xpos, arr, s=8, alpha=0.45, color=col)
        med = np.median(arr); q1 = np.percentile(arr, 25); q3 = np.percentile(arr, 75)
        ax.plot([i, i], [q1, q3], color='k', lw=2)
        ax.plot(i, med, marker='_', color='tab:red', markersize=12, markeredgewidth=3)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(2.0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xticks(range(len(group)))
    ax.set_xticklabels(group, rotation=45, ha='right')
    ax.set_xlim(-0.6, len(group) - 0.4)
    ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel('combination')

axes[0].set_ylabel('alpha')
panel12_pdf = 'alpha_first_uncertain_updated.pdf'
plt.suptitle('Grouped alpha distributions: First-order and Uncertain groups', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(panel12_pdf, dpi=600, bbox_inches='tight', format='pdf')
plt.close()

# Second-order combined two-panel with second_col color
sec = second_order.copy()
half = len(sec) // 2
sec1 = sec[:half]
sec2 = sec[half:]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, group, title in zip(axes, [sec1, sec2], ['Second-order ratios (part A)', 'Second-order ratios (part B)']):
    for i, name in enumerate(group):
        arr = alpha_vals[name]
        xpos = np.full_like(arr, i) + (np.random.rand(len(arr)) - 0.5) * 0.18
        ax.scatter(xpos, arr, s=8, alpha=0.45, color=second_col)
        med = np.median(arr); q1 = np.percentile(arr, 25); q3 = np.percentile(arr, 75)
        ax.plot([i, i], [q1, q3], color='k', lw=2)
        ax.plot(i, med, marker='_', color='tab:red', markersize=12, markeredgewidth=3)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(2.0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xticks(range(len(group)))
    ax.set_xticklabels(group, rotation=45, ha='right')
    ax.set_xlim(-0.6, len(group) - 0.4)
    ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel('combination')
axes[0].set_ylabel('alpha')
second_pdf = 'alpha_second_combined_updated.pdf'
plt.suptitle('Grouped alpha distributions: Second-order ratios', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(second_pdf, dpi=600, bbox_inches='tight', format='pdf')
plt.close()

agg_pdf, panel12_pdf, second_pdf

