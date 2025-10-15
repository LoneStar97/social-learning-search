# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:49:58 2025

@author: Starr
"""

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.signal import welch
from scipy.interpolate import interp1d
import json

rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
Rv = 0.01
NumCases = 1000
base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\6000T'
filepaths = ['10P600']
numfile = len(filepaths)

# common_freq_points = 500
# common_freq_series_effi = np.zeros([numfile, common_freq_points-2, np.size(rho_series)])
# mean_psd_series_effi = np.zeros_like(common_freq_series_effi)
# common_freq_series_fe = np.zeros_like(common_freq_series_effi)
# mean_psd_series_fe = np.zeros_like(common_freq_series_effi)
# common_freq_series_ft = np.zeros_like(common_freq_series_effi)
# mean_psd_series_ft = np.zeros_like(common_freq_series_effi)
#
# mean_mag_list  = np.zeros_like(common_freq_series_effi)
# mean_mag_points = np.zeros_like(common_freq_series_effi)


eta_ins_all = {rho: [] for rho in rho_series} # Initialize storage for eta_ins across all rho values

for m in range(numfile):
    data_dir = os.path.join(base_dir, filepaths[m])  # Data directory within the project
    for k in range(np.size(rho_series)):
        rho = rho_series[k]
        file = data_dir + '\\Rv0.01_rho='+str(rho)+'_A50.h5'
        id_range = range(NumCases)
        # fe_list = []
        # ft_list = []
        # effi_list = []
        with h5py.File(file, 'r') as hdf:
            # List all groups
            # print("Keys: %s" % hdf.keys())
            fe = np.zeros(NumCases) #　ratio of exploiter mu = 3　
            ft = np.zeros(NumCases) # ratio of agents performing targeted walk
            effi = np.zeros(NumCases)
            agg_tw = []
            for i in id_range:
                groupname = 'case_' + str(i)
                if groupname not in hdf:
                    continue
                group = hdf[groupname]
                # Get a group or dataset
                group = hdf.get(groupname)
                numtar = group.get('foods')[:]
                time = group.get('ticks')[:]
                
                # subgroup = group["subgroup_mu"]
                # # Access and read a dataset within the subgroup
                # mugroup = subgroup['mu'][:]
                # twgroup = group["tw_step_lengths"]
                # tw = twgroup['step'][:]
                
                # ft_inst = np.zeros(np.size(tw,0))
                # fe_inst = np.zeros(np.size(mugroup,0))
                # for j in range(np.size(mugroup,0)):
                #     mu_inst = mugroup[j,:]
                #     fe_inst[j] = np.count_nonzero(mu_inst == 3)/np.size(mu_inst)
                #     tw_inst = tw[j,:]
                #     ft_inst[j] = np.count_nonzero(tw_inst)/np.size(tw_inst)
                    
                numtar_diff = numtar[1:np.size(numtar)] - numtar[0:np.size(numtar)-1]
                time_diff = time[1:np.size(time)] - time[0:np.size(time)-1]
                eta_ins = numtar_diff/time_diff
                eta_ins_all[rho].extend(eta_ins)  # Store eta_ins for this rho
                # Append the array to the list
                # effi_list.append(eta_ins)
                # fe_list.append(fe_inst)
                # ft_list.append(ft_inst)
                
        # # Step 1: Calculate PSD for each time series
        # psd_list_effi = []
        # freqs_list_effi = []
        # psd_list_fe = []
        # freqs_list_fe = []
        # psd_list_ft = []
        # freqs_list_ft = []
        #
        # for ts in effi_list:
        #     freqs, psd = welch(ts, nperseg=len(ts) // 8)  # Welch's method to compute PSD
        #     freqs_list_effi.append(freqs)
        #     psd_list_effi.append(psd)
        #
        # # Step 2: Define a common frequency grid
        # common_freqs = np.linspace(min([freqs[0] for freqs in freqs_list_effi]), max([freqs[-1] for freqs in freqs_list_effi]), num=common_freq_points)
        #
        # # Step 3: Interpolate each PSD onto the common frequency grid
        # interpolated_psds = []
        #
        # for jj in range(NumCases):
        #     # Interpolate each PSD to the common frequency grid
        #     interp_psd = interp1d(freqs_list_effi[jj], psd_list_effi[jj], kind='linear', bounds_error=False, fill_value=0)
        #     psd_on_common = interp_psd(common_freqs)
        #     interpolated_psds.append(psd_on_common)
        #
        # # Step 4: Compute the mean PSD across all time series
        # mean_psd = np.mean(interpolated_psds, axis=0)
        # mean_psd = mean_psd[1:len(mean_psd)-1]
        # total_power = np.sum(mean_psd)
        # psd_normalized = mean_psd / total_power / (common_freqs[2]-common_freqs[1]) #normalization
        #
        # common_freq_series_effi[m,:,k] = common_freqs[1:len(common_freqs)-1]
        # mean_psd_series_effi[m,:,k] = psd_normalized

#%% provide timeseries
# Load JSON 
with open(os.path.join(data_dir, "time_series_effi.json"), "r") as json_file:
    loaded_data = json.load(json_file)
# Convert back to Pandas DataFrame
loaded_series = {key: pd.DataFrame(value) for key, value in loaded_data.items()}

#%% provide 3d plot showing the PDF of number of collected targets by each agent (for equality)
# Initialize data storage
tar_col_data = [[[] for _ in range(len(rho_series))] for _ in range(len(filepaths))]

# Load data from files
for m in range(numfile): #range(len(filepaths)):
    data_dir = os.path.join(base_dir, filepaths[m])
    for k, rho in enumerate(rho_series):
        file = os.path.join(data_dir, f'Rv0.01_rho={rho}_A50.h5')
        id_range = range(NumCases)
        
        with h5py.File(file, 'r') as hdf:
            for i in id_range:
                groupname = f'case_{i}'
                group = hdf.get(groupname)
                if group is None:
                    continue
                tar_count = group.get('target_counts')[:]
                tar_col_data[m][k].append(tar_count[-1, :])  # Last time step data

# Convert to numpy array for processing
tar_col_data_np = np.array(tar_col_data, dtype=object)
#%% check two modes of behaviors with higher rho and lower rho
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
import matplotlib
# matplotlib.use('cairo')  # Use the Agg backend for non-interactive rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern']

SetLabelSize = 12
SetLegendSize = 9
SubplotLabelSize = 14
SetLineWidth = 1
Setticksize = 8

# # Create figure and define GridSpec layout
# fig = plt.figure(figsize=(7.0, 5.0))  # Square layout for PNAS
# gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])

# Define subplot labels
labels = ['A', 'B', 'C', 'B', 'C', 'D']
positions = [(-0.28, 1.04), (-0.28, 1.04), (-0.28, 1.04), (-0.3, 1.04), (-0.28, 1.04), (-0.3, 1.04)]   # Align with y-axis in upper-left
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f7b6d2']
colors = plt.cm.viridis(np.linspace(0, 1, len(rho_series)))  # Use a color map for distinct colors

# Create figure and define GridSpec
fig = plt.figure(figsize=(7.0, 6.0))
main_gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.25, wspace=0.35)  # Left column (A,B,C) vs. right (D)

# Left column: Stack A, B, C vertically
left_gs = main_gs[0].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
ax1 = fig.add_subplot(left_gs[0])  # A
ax5 = fig.add_subplot(left_gs[1])  # B
ax2 = fig.add_subplot(left_gs[2])  # C
ax3 = fig.add_subplot(main_gs[0,1])  # D

# Right column: Full space for D
ax4 = fig.add_subplot(main_gs[1,1])  # F
ax6 = fig.add_subplot(main_gs[1,0])  # E

# Plot data and add subplot labels
ax1.plot(loaded_series["rho_0.01"].Time, loaded_series["rho_0.01"].Effi_Inst,color=colors[0], linewidth=SetLineWidth, label=r'$\rho$=0.01')
# ax1.set_xlabel('Timesteps', fontsize=SetLabelSize)
# ax1.set_ylabel(r'$\eta_i$', fontsize=SetLabelSize)
ax1.set_ylim(0, 23)
# ax1.set_xlim(0, 4000)
ax1.legend(loc='upper left', fontsize=SetLegendSize)
ax1.text(*positions[0], labels[0], transform=ax1.transAxes, fontsize=SubplotLabelSize, fontweight='bold')

ax5.plot(loaded_series["rho_0.2"].Time, loaded_series["rho_0.2"].Effi_Inst,color=colors[4], linewidth=SetLineWidth, label=r'$\rho$=0.15')
ax5.set_ylabel(r'$\eta_i$', fontsize=SetLabelSize)
ax5.set_ylim(0, 23)
# ax5.set_xlim(0, 4000)
ax5.legend(loc='upper left', fontsize=SetLegendSize)
# ax5.text(*positions[1], labels[1], transform=ax5.transAxes, fontsize=SubplotLabelSize, fontweight='bold')

ax2.plot(loaded_series["rho_0.7"].Time, loaded_series["rho_0.7"].Effi_Inst,color=colors[-1], linewidth=SetLineWidth, label=r'$\rho$=0.7')
ax2.set_xlabel('Timesteps', fontsize=SetLabelSize)
# ax2.set_ylabel(r'$\eta_i$', fontsize=SetLabelSize)
ax2.set_ylim(0, 23)
# ax2.set_xlim(0, 4000)
ax2.legend(loc='upper left', fontsize=SetLegendSize)
# ax2.text(*positions[2], labels[2], transform=ax2.transAxes, fontsize=SubplotLabelSize, fontweight='bold')


# Plot PDFs for each rho
for ii, rho in enumerate(rho_series):
    data = np.array(eta_ins_all[rho])
    if len(data) == 0:
        print(f"No data for rho={rho}")
        continue
    # Calculate histogram bins and density
    counts, bin_edges = np.histogram(data, bins=np.arange(data.min(), data.max() + 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove zero values from counts and corresponding bin_centers
    valid_mask = counts > 0
    counts = counts[valid_mask]
    bin_centers = bin_centers[valid_mask]

    ax3.plot(bin_centers, counts, color=colors[ii], alpha=1, linewidth=SetLineWidth)
    ax3.scatter(bin_centers, counts, s=30, color=colors[ii], alpha=1)
ax3.set_yscale('log')
# ax3.set_xscale('log')
ax3.set_ylabel('PDF', fontsize=SetLabelSize)
ax3.set_xlabel(r'$\eta_i$', fontsize=SetLabelSize)
ax3.text(*positions[3], labels[3], transform=ax3.transAxes, fontsize=SubplotLabelSize, fontweight='bold')

# Load the data
with open(os.path.join(data_dir,'iet_pdf_data.json'), 'r') as f:
    data = json.load(f)
for k, rho in enumerate(rho_series):
    rho_str = str(rho)
    bin_centers = np.array(data['curves'][rho_str]['bin_centers'])
    counts = np.array(data['curves'][rho_str]['counts'])
    ax6.scatter(bin_centers, counts, s=30, color=colors[k], alpha=1)
    ax6.plot(bin_centers, counts, label=f'ρ={rho}', color=colors[k], linewidth=SetLineWidth)
ax6.set_xlabel(r'$\tau$', fontsize=SetLabelSize)
ax6.set_ylabel('PDF', fontsize=SetLabelSize)
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.text(*positions[4], labels[4], transform=ax6.transAxes, fontsize=SubplotLabelSize, fontweight='bold')

burstiness_factor=[0.34, 0.32, 0.23, 0.16, 0.14, 0.13, 0.14, 0.17, 0.21, 0.26, 0.32, 0.38, 0.42, 0.42]
ax6_inset = inset_axes(ax6, width="100%", height="100%", loc='upper right', borderpad=1,
                      bbox_to_anchor=(0.7, 0.7, 0.3, 0.3), bbox_transform=ax6.transAxes)
ax6_inset.scatter(rho_series, burstiness_factor, s=15, color='red')
# ax4_inset.plot(rho_series, std_devs, color='red', linestyle='--', linewidth=2, label=r'Std Dev of $N$')
ax6_inset.set_xlabel(r'$\rho$', fontsize=SetLabelSize)
ax6_inset.set_ylabel(r'$B$', fontsize=SetLabelSize)
ax6_inset.set_ylim(min(burstiness_factor) * 0.8, max(burstiness_factor) * 1.15)

# Plot PDF as scatter points and lines for each rho
std_devs = []  # Store standard deviations for each rho
for k, rho in enumerate(np.array(rho_series)):  # Iterate over rho values
    all_agent_data = [case_data for case_data in tar_col_data[0][k]]  # Choose one filepath (e.g., index 2)
    flattened_agent_data = [value for case_data in all_agent_data for value in case_data]  # Flatten list
    print(np.std(flattened_agent_data))
    if flattened_agent_data:  # Skip if no data
        # Calculate histogram bins and densities
        counts, bin_edges = np.histogram(np.array(flattened_agent_data), bins=np.geomspace(1, np.max(np.array(flattened_agent_data))+1, 15), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers

        # Remove zero values from counts and corresponding bin_centers
        valid_mask = counts > 0
        counts = counts[valid_mask]
        bin_centers = bin_centers[valid_mask]

        # Create 3D scatter points and line
        ax4.scatter(bin_centers, counts, s=30, color=colors[k], alpha=1) #label=f'$ρ$ = {rho}',
        ax4.plot(bin_centers, counts, color=colors[k], alpha=1, linewidth=SetLineWidth)
        # Compute and store standard deviation
        std_dev = np.std(flattened_agent_data)
        std_devs.append(std_dev)

# ax4.plot(std_devs, rho_series, np.zeros_like(rho_series), color='red', linewidth=2, linestyle='--', label=r'Std Dev of $N$')
# Customize 3D plot
ax4.set_xlabel(r'$N$', fontsize=SetLabelSize)
ax4.set_ylabel('PDF', fontsize=SetLabelSize)
# ax4.tick_params(axis='both', labelsize=Setticksize)  # 'both' affects x and y axes

# Create a ScalarFormatter
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # Use scientific notation for numbers outside 10^-3 to 10^3
# Apply the formatter to the y-axis
ax4.yaxis.set_major_formatter(formatter)
ax4.xaxis.set_major_formatter(formatter)
# Optional: Adjust the offset text (e.g., "1e-3" at the axis) to look like "10^{-3}"
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_xticks([1, 500])
# ax4.set_xlim(8e0,8e2)
ax4.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# Adjust the font size of the offset text (scientific notation)
ax4.yaxis.offsetText.set_fontsize(Setticksize)  # Make the exponent smaller
ax4.xaxis.offsetText.set_fontsize(Setticksize)  # Make the exponent smaller
ax4.set_xscale('log')
ax4.set_yscale('log')

ax4_inset = inset_axes(ax4, width="100%", height="100%", loc='upper right', borderpad=1,
                      bbox_to_anchor=(0.32, 0.32, 0.3, 0.3), bbox_transform=ax4.transAxes)
ax4_inset.scatter(rho_series, std_devs, s=15, color='red')
# ax4_inset.plot(rho_series, std_devs, color='red', linestyle='--', linewidth=2, label=r'Std Dev of $N$')
ax4_inset.set_xlabel(r'$\rho$', fontsize=SetLabelSize)
ax4_inset.set_ylabel(r'$\sigma$', fontsize=SetLabelSize)
ax4_inset.set_ylim(min(std_devs) * 0.8, max(std_devs) * 1.15)

# ax.set_title('3D PDF of Targets Collected by Agents Across ρ', fontsize=14)
# ax4.view_init(elev=20, azim=30)  # Adjust 3D view angle
# ax4.set_box_aspect([1.25, 1.5, 1])  # Equal aspect ratio


# ax4.legend(loc='upper left', fontsize=8, title=r'$\rho$ Values')
# ax4.grid(False)  # Disable the grid
ax4.text(*positions[5], labels[5], transform=ax4.transAxes, fontsize=SubplotLabelSize, fontweight='bold')

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=min(rho_series), vmax=max(rho_series))
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
# Manually place the label at the top of the colorbar
cbar.ax.text(0.5, 1.05, r'$\rho$', fontsize=SetLabelSize, ha='center', va='bottom', transform=cbar.ax.transAxes)
cbar.ax.tick_params(labelsize=SetLegendSize)  # Adjust tick size

# Add colorbar to the right of subfigure D (ax4)
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(rho_series), vmax=max(rho_series))),
#                    ax=ax4,
#                    orientation='vertical',
#                    pad=0.1,  # pad controls the distance from the subplot
#                    shrink=0.7,  # Reduce the length of the colorbar (default is 1.0)
#                    aspect=30)   # Controls width relative to length (higher value = thinner)
# cbar.set_label(r'$\rho$', fontsize=SetLabelSize)
# cbar.ax.tick_params(labelsize=SetLegendSize)

# Adjust layout
# plt.tight_layout()

# # Add colorbar at the top of the figure
# cbar_ax = ax4.add_axes([0.12, 1.0, 0.7, 0.02])  # [left, bottom, width, height]

# # Create colorbar with the same colormap
# norm = plt.Normalize(vmin=min(rho_series), vmax=max(rho_series))  # Normalize rho values
# sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)  # Use same colormap
# sm.set_array([])  # Empty array needed for colorbar

# # Add colorbar to figure
# cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
# cbar.set_label(r'$\rho$', fontsize=SetLabelSize)  # Set colorbar label
# cbar.ax.tick_params(labelsize=SetLegendSize)  # Adjust tick size

# Save the figure in a high-quality PDF format (vector graphics)
plt.savefig('Figure3.TimeVarying.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('Figure3.TimeVarying.png', format='png', dpi=600, bbox_inches='tight')
# Show figure
plt.show()

#%%
from PIL import Image
import os

# Define paths
save_dir = r'D:\OneDrive - University of Pittsburgh\CollectiveForagingData\6000T'  # Adjust to your writable directory
png_path = os.path.join(save_dir, 'Figure3.TimeVarying.png')
pdf_path = os.path.join(save_dir, 'Figure3.TimeVarying_converted.pdf')

# Convert PNG to PDF
img = Image.open(png_path)
img.save(pdf_path, format='PDF', resolution=600)  # Maintain 600 DPI for PNAS

# Close the plot
plt.close()
plt.show()

# # plot power spectral density in subfigure
# check_rho_id = [1, 2, 9, 10]
# for p in range(np.size(check_rho_id)):
#     frqs = common_freq_series_effi[1,:,check_rho_id[p]]
#     psd = mean_psd_series_effi[1,:,check_rho_id[p]]
#     ax3.plot(frqs[2:len(frqs)-4], psd[2:len(psd)-4], '-', color=colors[p], linewidth=SetLineWidth, label=r'$\rho$='+str(rho_series[check_rho_id[p]]))
# ax3.set_xlabel(r'$f$', fontsize=SetLabelSize)
# ax3.set_ylabel(r'$PSD$', fontsize=SetLabelSize)
# ax3.set_yscale('log')
# # ax3.legend(loc='upper right', ncol=2, fontsize=SetLegendSize, frameon=True)
# ax3.text(*positions[2], labels[2], transform=ax3.transAxes, fontsize=SubplotLabelSize, fontweight='bold')


# # 2D plot ===========================================
# # Plot PDF as scatter points and lines for each rho
# for k, rho in enumerate(np.array(rho_series)[check_rho_id]):  # Iterate over rho values
#     all_agent_data = [case_data for case_data in tar_col_data[0][k]]  # Choose one filepath (e.g., index 2)
#     flattened_agent_data = [value for case_data in all_agent_data for value in case_data]  # Flatten list
#     flattened_data = np.array(flattened_agent_data)
#     flattened_data = flattened_data[flattened_data<800]
#     if flattened_agent_data:  # Skip if no data
#         # Calculate histogram bins and densities
#         counts, bin_edges = np.histogram(flattened_data, bins=20, density=True)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
        
#         # Create 3D scatter points and line
#         ax4.scatter(bin_centers, counts, color=colors[k], alpha=1) #label=f'$ρ$ = {rho}',
#         ax4.plot(bin_centers, counts, color=colors[k], alpha=1, linewidth=SetLineWidth)

# # Customize 3D plot
# ax4.set_xlabel(r'$N_T$', fontsize=SetLabelSize)
# ax4.set_ylabel('PDF', fontsize=SetLabelSize)
# ax4.set_yscale('log')
# # ax4.set_xscale('log')
# ax4.text(*positions[2], labels[2], transform=ax4.transAxes, fontsize=SubplotLabelSize, fontweight='bold')