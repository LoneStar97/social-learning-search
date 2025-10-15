import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import os

# ============================================================
# 1. Load data
# ============================================================
base_dir = r'C:\Users\Starr\Documents\GitHub\social-learning-search\Figures'
rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

with h5py.File(os.path.join(base_dir, 'Figure4_PlotData.h5'), 'r') as hdf:
    agg_nor  = hdf['agg_nor'][:]
    agg_neg  = hdf['agg_neg'][:]
    agg_time = hdf['agg_time'][:]
    agg_effi = hdf['agg_effi'][:]
    std_effi = hdf['std_effi'][:]

filenames   = [r'$N_-=0$', r'$N_+=20$', r'$N_+=50$']
CaseAgents  = [r'$N_-=5$', r'$N_-=10$', r'$N_-=15$', r'$N_-=0$']
colors      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
x_series    = [20, 75, 200, 0]
marker_series = ['o', 's', '^', 'D']
linestyles    = ['-', '--', ':', '-.']
Penalties     = [-20, -75, -200, 0]

SetLabelSize   = 12
SetLegendSize  = 9
SubplotLabelSize = 14
labels    = ['A', 'B']
positions = [(-0.22, 1.05)] * 2

agg_time[agg_time == 0] = 10000

# ============================================================
# 2. Create figure
# ============================================================
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.0, 4.0))

penalty_handles = []
case_handles    = []

# ---------- Subplot A ----------
for q in range(3):
    for m in range(3):
        agg_effi_neg = (agg_nor[q, :, :] - x_series[m] * agg_neg[q, :, :]) / agg_time[q, :, :]
        numsqrt      = np.sqrt(np.count_nonzero(agg_effi_neg, 1))
        nonzero_mask = agg_effi_neg != 0
        mean_effi    = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
        mean_effi    = np.nan_to_num(mean_effi, nan=0.0)
        stds_effi    = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
        stds_effi    = np.nan_to_num(stds_effi, nan=0.0)

        ln = axes[0].errorbar(
            rho_series, mean_effi / 6000,
            yerr=1.96 * stds_effi / numsqrt / 6000,
            color=colors[q],
            marker=marker_series[m],
            linestyle=linestyles[m],
            capsize=0,
            elinewidth=2,
            markeredgewidth=2
        )
        if q == 0:
            penalty_handles.append(ln)
        # if m == 0:
        #     case_handles.append(ln)

axes[0].set_ylim(-0.9e-3, 5e-4)
axes[0].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
axes[0].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[0].text(*positions[0], labels[0], transform=axes[0].transAxes,
             fontsize=SubplotLabelSize, fontweight='bold')

# ---------- Subplot B ----------
q = 0
ln_black = axes[1].errorbar(
    rho_series, agg_effi[:, q] / 6000,
    yerr=std_effi[:, q] / 6000,
    color='k', fmt='-.D',
    capsize=0, elinewidth=2, markeredgewidth=2,
    label=filenames[q]
)

for q in range(3):
    agg_effi_neg = (agg_nor[q, :, :] - x_series[3] * agg_neg[q, :, :]) / agg_time[q, :, :]
    numsqrt      = np.sqrt(np.count_nonzero(agg_effi_neg, 1))
    nonzero_mask = agg_effi_neg != 0
    mean_effi    = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    mean_effi    = np.nan_to_num(mean_effi, nan=0.0)
    stds_effi    = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    stds_effi    = np.nan_to_num(stds_effi, nan=0.0)
    rf = axes[1].errorbar(rho_series, mean_effi / 6000,
                     yerr=1.96 * stds_effi / numsqrt / 6000,
                     color=colors[q],
                     marker=marker_series[3],
                     linestyle=linestyles[3],
                     capsize=0, elinewidth=2, markeredgewidth=2)
    if q == 0:
        penalty_handles.append(rf)
    case_handles.append(rf)

case_handles.append(ln_black)
axes[1].set_ylim(0, 5.5e-4)
axes[1].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
# axes[1].text(0.35, 0.05, 'Penalty=0', transform=axes[1].transAxes, fontsize=SetLabelSize)
axes[1].text(*positions[1], labels[1], transform=axes[1].transAxes,
             fontsize=SubplotLabelSize, fontweight='bold')
axes[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# ============================================================
# 3. Combined legends placed inside subplot B
# ============================================================
# penalty legend shows marker+linestyle
penalty_legend = axes[0].legend(
    handles=penalty_handles,
    labels=[str(p) for p in Penalties],
    title='Penalties',
    loc='lower right',
    fontsize=SetLegendSize,
    frameon=True,
    ncol=2
)
axes[0].add_artist(penalty_legend)

# case legend shows colors only
case_legend = axes[1].legend(
    handles=case_handles,
    labels=CaseAgents,
    title='Targets',
    loc='lower right',
    fontsize=SetLegendSize,
    frameon=True,
    ncol=2
)

# ============================================================
plt.tight_layout()
plt.savefig('Figure4.pdf', format='pdf', dpi=600)
plt.savefig('Figure4.png', format='png', dpi=600)
plt.show()
