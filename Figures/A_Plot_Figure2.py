# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 07:19 AM EDT 2025

@author: Starr
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import os

# Define constants
base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\6000T'
filepaths = ['10P600', '20P300', '50P120']
numfile = len(filepaths)
rho_series = None  # Will be loaded from file

rho_series = [0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
# Load plotting data from HDF5
with h5py.File(os.path.join(base_dir, 'A_Figure2_PlotData.h5'), 'r') as hdf:
    agg_effi = hdf['agg_effi'][:]
    std_effi = hdf['std_effi'][:]
    agg_fe = hdf['agg_fe'][:]
    std_fe = hdf['std_fe'][:]
    agg_ft = hdf['agg_ft'][:]
    std_ft = hdf['std_ft'][:]
    agg_f1 = hdf['agg_f1'][:]
    std_f1 = hdf['std_f1'][:]
    tw_len = hdf['tw_len'][:]
    tw_times = hdf['tw_times'][:]
    std_tw_len = hdf['std_tw_len'][:]
    std_tw_times = hdf['std_tw_times'][:]
    d_tw_times_agg = hdf['d_tw_times_agg'][:]
    tw_len_diff = hdf['tw_len_diff'][:]
    std_tw_len_diff = hdf['std_tw_len_diff'][:]

#%%
filenames = [r'$N_+=10$', r'$N_+=20$', r'$N_+=50$']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

SetLabelSize = 12
SetLegendSize = 9
SubplotLabelSize = 14
SetLineWidth = 1
Setticksize = 8

# Define subplot labels
labels = ['A', 'B', 'C', 'D']
positions = [(-0.28, 1.04)] * 4  # Adjust placement to align along the ylabel

# Define the final figure width: 17.8 cm ≈ 7.0 inches.
# For a 2×2 grid, we choose a square layout (7.0 inches x 7.0 inches).
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.0, 7.0))
# Subfigure A: Efficiency vs rho
for q in range(numfile):
    axes[0, 0].errorbar(rho_series, agg_effi[:, q]/6000, yerr=std_effi[:, q]/6000, fmt='-o',
                        capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
axes[0, 0].set_ylim(0, 6e-4)
axes[0, 0].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
axes[0, 0].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
axes[0, 0].legend(loc='upper left', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[0, 0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axes[0, 0].text(*positions[0], labels[0], transform=axes[0, 0].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
axes[0, 0].vlines(x=rho_series[7], ymin=0, ymax=agg_effi[7, 0]/6000, color=colors[0], linestyle='--', linewidth=2) # indicate the optimal rho
axes[0, 0].vlines(x=rho_series[6], ymin=0, ymax=agg_effi[6, 1]/6000, color=colors[1], linestyle='--', linewidth=2) # indicate the optimal rho
axes[0, 0].vlines(x=rho_series[5], ymin=0, ymax=agg_effi[5, 2]/6000, color=colors[2], linestyle='--', linewidth=2) # indicate the optimal rho
axins = inset_axes(axes[0, 0], width="100%", height="100%", loc='upper right',
                   bbox_to_anchor=(0.67, 0.7, 0.33, 0.3), bbox_transform=axes[0, 0].transAxes)
for q in range(numfile):
    axins.plot(agg_fe[:, q], agg_effi[:, q]/6000, 'o', markersize=4, linestyle='', label=filenames[q])
axins.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
axins.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axins.get_yaxis().get_offset_text().set_visible(False)
axins.set_xlabel(r'$f_{\mu=3}$', fontsize=SetLegendSize)
axins.set_ylabel(r'$\eta$', fontsize=SetLegendSize)
axins.tick_params(axis='both', labelsize=Setticksize)

# Subfigure B: Time ratio of behaviors
q = 1
axes[0, 1].errorbar(rho_series, agg_ft[:, q], yerr=std_ft[:, q], fmt='--k', capsize=0, elinewidth=2, markeredgewidth=1)
axes[0, 1].errorbar(rho_series, agg_fe[:, q]+agg_ft[:, q], yerr=std_fe[:, q], fmt='--k', capsize=0, elinewidth=2, markeredgewidth=1)
axes[0, 1].fill_between(rho_series, 0, agg_ft[:, q], color='lightblue', alpha=0.8, label=r'$f_t$')
axes[0, 1].fill_between(rho_series, agg_ft[:, q], agg_fe[:, q]+agg_ft[:, q], color='orange', alpha=0.3, label=r'$f_{\mu = 3}$')
axes[0, 1].fill_between(rho_series, agg_fe[:, q]+agg_ft[:, q], agg_f1[:, q]+agg_fe[:, q]+agg_ft[:, q],
                       color='lightgreen', alpha=0.8, label=r'$f_{\mu = 1.1}$')
axes[0, 1].vlines(x=rho_series[6], ymin=agg_ft[6, 1], ymax=agg_fe[6,1]+agg_ft[6, 1], color=colors[1], linestyle='--', linewidth=2)  # indicate the optimal rho
axes[0, 1].set_xlim([0, 0.7])
axes[0, 1].set_ylim([0, 1.0])
axes[0, 1].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
axes[0, 1].set_ylabel('Time ratio', fontsize=SetLabelSize)
axes[0, 1].legend(loc='lower right', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[0, 1].text(*positions[1], labels[1], transform=axes[0, 1].transAxes, fontsize=SubplotLabelSize, fontweight='bold')

# Subfigure C: Targeted walk length vs times
for q in range(numfile):
    # axes[1, 0].errorbar(tw_len[:, q], tw_times[:, q], xerr=std_tw_len[:, q], yerr=std_tw_times[:, q],
    #                     fmt='-o', capsize=0, elinewidth=2, markeredgewidth=1, label=filenames[q])
    axes[1, 0].errorbar(tw_len[:, q], tw_times[:, q], xerr=std_tw_len[:, q], yerr=std_tw_times[:, q],
                        fmt='-o', capsize=0, elinewidth=2, markeredgewidth=1)
axes[1, 0].vlines(x=tw_len[7,0], ymin=0, ymax=tw_times[7,0], color=colors[0], linestyle='--', linewidth=2, label=filenames[0]+r' $\rho^*$') # indicate the optimal rho
axes[1, 0].vlines(x=tw_len[6,1], ymin=0, ymax=tw_times[6,1], color=colors[1], linestyle='--', linewidth=2, label=filenames[1]+r' $\rho^*$') # indicate the optimal rho
axes[1, 0].vlines(x=tw_len[5,2], ymin=0, ymax=tw_times[5,2], color=colors[2], linestyle='--', linewidth=2, label=filenames[2]+r' $\rho^*$') # indicate the optimal rho
axes[1, 0].set_ylim([0, 6e-2])
# axes[1, 0].set_xlabel(r'$\left\langle l \right\rangle$', fontsize=SetLabelSize)
# axes[1, 0].set_ylabel(r'$\left\langle \mathrm{TWs} \right\rangle$', fontsize=SetLabelSize)
axes[1, 0].set_ylabel('Counts of targeted walks/Total time', fontsize=SetLabelSize)
axes[1, 0].set_xlabel('Average length of targeted walks', fontsize=SetLabelSize)
axes[1, 0].legend(loc='lower right', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[1, 0].tick_params(axis='both', labelsize=Setticksize)
axes[1, 0].text(*positions[2], labels[2], transform=axes[1, 0].transAxes, fontsize=SubplotLabelSize, fontweight='bold')

# Subfigure D: Derivative of times vs length
for q in range(numfile):
    axes[1, 1].errorbar(tw_len[:, q], d_tw_times_agg[:, q], xerr=std_tw_len[:, q],
                        fmt='-o', capsize=0, elinewidth=2, markeredgewidth=1, label=filenames[q])
axes[1, 1].vlines(x=tw_len[7,0], ymin=-5, ymax=d_tw_times_agg[7,0], color=colors[0], linestyle='--', linewidth=2)  # indicate the optimal rho
axes[1, 1].vlines(x=tw_len[6,1], ymin=-5, ymax=d_tw_times_agg[6,1], color=colors[1], linestyle='--', linewidth=2)  # indicate the optimal rho
axes[1, 1].vlines(x=tw_len[5,2], ymin=-5, ymax=d_tw_times_agg[5,2], color=colors[2], linestyle='--', linewidth=2)  # indicate the optimal rho
axes[1, 1].set_ylim([-5, 15])
# axes[1, 1].set_xlabel(r'$\left\langle l \right\rangle$', fontsize=SetLabelSize)
# axes[1, 1].set_ylabel(r'$\mathrm{d\left\langle TWs \right\rangle}/\mathrm{d}\left\langle l \right\rangle$', fontsize=SetLabelSize)
# axes[1, 1].legend(loc='upper right', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[1, 1].set_xlabel('Average length of targeted walks', fontsize=SetLabelSize)
axes[1, 1].set_ylabel('Slope', fontsize=SetLabelSize)
axes[1, 1].tick_params(axis='both', labelsize=Setticksize)
axes[1, 1].text(*positions[3], labels[3], transform=axes[1, 1].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
# Improve layout spacing
plt.tight_layout()

# Save the figure in a high-quality PDF format (vector graphics)
plt.savefig('Figure2.Effi_mod.pdf', format='pdf', dpi=600)
plt.savefig('Figure2.Effi_mod.png', format='png', dpi=600)
# Optionally display the figure
plt.show()