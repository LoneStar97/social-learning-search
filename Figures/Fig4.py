# -*- coding: utf-8 -*-
"""
Created on Thu MarE 13 08:29:08 2025

@author: Starr
"""
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
Rv = 0.01
NumCases = 1000
# base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\6000T'
base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\NegativePatch'
filepaths = ['10P600']
numfile = len(filepaths)
agg_effi = np.zeros([np.size(rho_series), numfile])
std_effi = np.zeros_like(agg_effi)
agg_fe = np.zeros_like(agg_effi)
std_fe = np.zeros_like(agg_effi)
agg_f1 = np.zeros_like(agg_effi)
std_f1 = np.zeros_like(agg_effi)
agg_ft = np.zeros_like(agg_effi)
std_ft = np.zeros_like(agg_effi)
tw_len = np.zeros_like(agg_effi) # length of targeted walk
std_tw_len = np.zeros_like(agg_effi)
n_bins = 200

# agg_nor = np.zeros_like(agg_effi) # length of targeted walk
# agg_neg = np.zeros_like(agg_effi) # length of targeted walk

for m in range(numfile):
    data_dir = os.path.join(base_dir, filepaths[m])  # Data directory within the project
    for k in range(np.size(rho_series)):
        rho = rho_series[k]
        file = data_dir + '\\Rv0.01_rho='+str(rho)+'_A50.h5'
        id_range = range(NumCases)
        with h5py.File(file, 'r') as hdf:
            existing_groups = list(hdf.keys())
            # print(f"Existing groups: {existing_groups}")
            # Count the number of groups
            num_groups = len(existing_groups)
            print(f"Number of groups: {num_groups}")
            fe = np.zeros(NumCases) #　ratio of exploiter mu = 3　
            ft = np.zeros(NumCases) # ratio of agents performing targeted walk
            f1 = np.zeros(NumCases) # ratio of explorer mu = 1.1　
            effi = np.zeros(NumCases)
            agg_tw = []
            for i in id_range:
                groupname = 'case_' + str(i)
                # Get a group or dataset
                if hdf.get(groupname) == None:
                    fe[i-id_range[0]] = 1000 # False cases
                    f1[i-id_range[0]] = 1000
                    ft[i-id_range[0]] = 1000
                    continue
                else:
                    group = hdf.get(groupname)
                numtar = group.get('foods')
                time = group.get('ticks')
                # numNegtive = group.get('NumNegTar')
                NumCellNoTar = group.get('NvNoTar')
                NumCellTar = group.get('NvTar')
                # Convert dataset to a NumPy array or access the data
                numtar = numtar[:]
                time = time[:]
                # numNegtive = numNegtive[:]


                subgroup = group["subgroup_mu"]
                # Access and read a dataset within the subgroup
                mugroup = subgroup['mu'][:]

                twgroup = group["tw_step_lengths"]
                # Access and read a dataset within the subgroup
                tw = twgroup['step'][:]

                for kk in range(np.size(tw,1)):
                    arrary = tw[:,kk]
                    max_values = []
                    temp = []
                    for value in arrary:
                        if value != 0:
                            temp.append(value)
                        else:
                            if temp:
                                max_values.append(max(temp))
                                temp = []
                    if temp:
                        max_values.append(max(temp))
                    agg_tw.append(max_values)

                # Flatten the list of lists into a single list
                flattened_list = [item for sublist in agg_tw for item in sublist]
                # Convert the flattened list to a NumPy one-dimensional array
                agg_tw_len = np.array(flattened_list)


                mugroup[tw !=0] = 100 # recognize targeted walk
                fe[i-id_range[0]] = np.count_nonzero(mugroup == 3)/np.size(mugroup) # mu=3
                f1[i-id_range[0]] = np.count_nonzero(mugroup == 1.1)/np.size(mugroup) # mu = 1.1
                ft[i-id_range[0]] = np.count_nonzero(mugroup == 100)/np.size(mugroup) # targeted walk

                effi[i-id_range[0]] = numtar[-1]/time[-1]
        agg_effi[k,m] = np.mean(effi[effi!=0])
        std_effi[k,m] = 1.96 * np.std(effi[effi!=0])/np.sqrt(len((effi[effi!=0])))
        agg_fe[k,m] = np.mean(fe[fe!=1000])
        std_fe[k,m] = 1.96 * np.std(fe[fe!=1000])/np.sqrt(len((fe[fe!=1000])))
        agg_f1[k,m] = np.mean(f1[f1!=1000])
        std_f1[k,m] = 1.96 * np.std(f1[f1!=1000])/np.sqrt(len((f1[f1!=1000])))
        agg_ft[k,m] = np.mean(ft[ft!=1000])
        std_ft[k,m] = 1.96 * np.std(ft[ft!=1000])/np.sqrt(len((ft[ft!=1000])))

        # agg_nor[k,m] = numtar[-1]
        # agg_neg[k,m] = np.size(numNegtive,0)


# #%% efficiency for cases
# NumCases2 = 1000
# # base_dir = 'C:\\Users\\ZEL45\\OneDrive - University of Pittsburgh\\collective_search_fake_negativeT'
# base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\6000T_FakeNegative'
# NPatches = [5, 10, 20]
# dirlist = ['5P1200', '10P600', '20P300']
# agg_effi2 = np.zeros([np.size(rho_series), np.size(NPatches)])
# std_effi2 = np.zeros_like(agg_effi2)
# agg_fe2 = np.zeros_like(agg_effi2)
# std_fe2 = np.zeros_like(agg_effi2)
# agg_f12 = np.zeros_like(agg_effi2)
# std_f12 = np.zeros_like(agg_effi2)
# agg_ft2 = np.zeros_like(agg_effi2)
# std_ft2 = np.zeros_like(agg_effi2)
#
# for m in range(np.size(NPatches)):
#     data_dir = os.path.join(base_dir, str(dirlist[m]))  # Data directory within the project
#     for k in range(np.size(rho_series)):
#         rho = rho_series[k]
#         file = data_dir + '\\Rv0.01_rho='+str(rho)+'_A50.h5'
#         id_range = range(NumCases2)
#         with h5py.File(file, 'r') as hdf:
#             existing_groups = list(hdf.keys())
#             # print(f"Existing groups: {existing_groups}")
#             # Count the number of groups
#             num_groups = len(existing_groups)
#             print(f"Number of groups: {num_groups}")
#             fe = np.zeros(NumCases2) #　ratio of exploiter mu = 3
#             f1 = np.zeros(NumCases2) # ratio of explorer mu = 1.1　
#             ft = np.zeros(NumCases2) # ratio of agents performing targeted walk
#             effi = np.zeros(NumCases2)
#             agg_tw = []
#             for i in id_range:
#                 groupname = 'case_' + str(i)
#                 # Get a group or dataset
#                 if hdf.get(groupname) == None:
#                     fe[i-id_range[0]] = 1000 # False cases
#                     ft[i-id_range[0]] = 1000
#                     f1[i-id_range[0]] = 1000
#                     continue
#                 else:
#                     group = hdf.get(groupname)
#                 numtar = group.get('foods')
#                 time = group.get('ticks')
#
#                 numtar = numtar[:]
#                 time = time[:]
#
#                 subgroup = group["subgroup_mu"]
#                 # Access and read a dataset within the subgroup
#                 mugroup = subgroup['mu'][:]
#
#                 twgroup = group["tw_step_lengths"]
#                 # Access and read a dataset within the subgroup
#                 tw = twgroup['step'][:]
#
#                 effi[i-id_range[0]] = numtar[-1]/time[-1]
#                 mugroup[tw !=0] = 100 # recognize targeted walk
#                 fe[i-id_range[0]] = np.count_nonzero(mugroup == 3)/np.size(mugroup) # mu=3
#                 f1[i-id_range[0]] = np.count_nonzero(mugroup == 1.1)/np.size(mugroup) # mu = 1.1
#                 ft[i-id_range[0]] = np.count_nonzero(mugroup == 100)/np.size(mugroup) # targeted walk
#
#         agg_effi2[k,m] = np.mean(effi[effi!=0])
#         std_effi2[k,m] = 1.96 * np.std(effi[effi!=0])/np.sqrt(len((effi[effi!=0])))
#         agg_fe2[k,m] = np.mean(fe[fe!=1000])
#         std_fe2[k,m] = 1.96 * np.std(fe[fe!=1000])/np.sqrt(len((fe[fe!=1000])))
#         agg_f12[k,m] = np.mean(f1[f1!=1000])
#         std_f12[k,m] = 1.96 * np.std(f1[f1!=1000])/np.sqrt(len((f1[f1!=1000])))
#         agg_ft2[k,m] = np.mean(ft[ft!=1000])
#         std_ft2[k,m] = 1.96 * np.std(ft[ft!=1000])/np.sqrt(len((ft[ft!=1000])))
#%% efficiency for cases with different number of agents
# base_dir = 'C:\\Users\\ZEL45\\OneDrive - University of Pittsburgh\\collective_search_fake_negativeT'
base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\NegativePatch'

NPatches = [5, 10, 15]
# dirlist = ['10P600_5NP1200', '10P600_10NP600', '10P600_15NP400']
# NPatches = [2, 5, 10, 1, 2, 3, 5, 10, 15]
dirlist = ['10P600_2NP300', '10P600_5NP120', '10P600_10NP60', '10P600_1NP6000','10P600_2NP3000','10P600_3NP2000','10P600_5NP1200', '10P600_10NP600', '10P600_15NP400']
agg_effi3 = np.zeros([np.size(rho_series), np.size(NPatches)])
std_effi3 = np.zeros_like(agg_effi3)
agg_fe3 = np.zeros_like(agg_effi3)
std_fe3 = np.zeros_like(agg_effi3)
agg_f13 = np.zeros_like(agg_effi3)
std_f13 = np.zeros_like(agg_effi3)
agg_ft3 = np.zeros_like(agg_effi3)
std_ft3 = np.zeros_like(agg_effi3)

agg_neg = np.zeros([np.size(NPatches), np.size(rho_series), 1000]) # number of collected negative targets
agg_nor = np.zeros_like(agg_neg) # number of collected normal targets
agg_time = np.zeros_like(agg_neg) # search time
NumCases3 = 1000
rho_series3 = rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

for m in range(np.size(NPatches)):
    data_dir = os.path.join(base_dir, str(dirlist[m]))  # Data directory within the project
    for k in range(np.size(rho_series3)):
        rho = rho_series3[k]
        file = data_dir + '\\Rv0.01rho=' + str(rho) + '_A50Neg.h5'
        # if m <= 5:
        #     file = data_dir + '\\Rv0.01rho='+str(rho)+'_A50Neg_Updated.h5'
        # else:
        #     file = data_dir + '\\Rv0.01rho=' + str(rho) + '_A50Neg.h5'
        id_range = range(NumCases3)
        with h5py.File(file, 'r') as hdf:
            existing_groups = list(hdf.keys())
            # print(f"Existing groups: {existing_groups}")          
            # Count the number of groups
            num_groups = len(existing_groups)
            print(f"Number of groups: {num_groups}")
            fe = np.zeros(NumCases3) #　ratio of exploiter mu = 3
            f1 = np.zeros(NumCases3) # ratio of explorer mu = 1.1　
            ft = np.zeros(NumCases3) # ratio of agents performing targeted walk
            effi = np.zeros(NumCases3)
            agg_tw = []
            for i in id_range:
                groupname = 'case_' + str(i)
                # Get a group or dataset
                if hdf.get(groupname) == None:
                    fe[i-id_range[0]] = 1000 # False cases
                    ft[i-id_range[0]] = 1000
                    f1[i-id_range[0]] = 1000
                    continue
                else:
                    group = hdf.get(groupname)
                numtar = group.get('foods')
                time = group.get('ticks')
                
                numtar = numtar[:]
                time = time[:]
                
                subgroup = group["subgroup_mu"]
                # Access and read a dataset within the subgroup
                mugroup = subgroup['mu'][:]
                
                twgroup = group["tw_step_lengths"]
                # Access and read a dataset within the subgroup
                tw = twgroup['step'][:]
                
                numNegtive = group.get('NumNegTar')
                
                effi[i-id_range[0]] = numtar[-1]/time[-1]
                mugroup[tw !=0] = 100 # recognize targeted walk
                fe[i-id_range[0]] = np.count_nonzero(mugroup == 3)/np.size(mugroup) # mu=3
                f1[i-id_range[0]] = np.count_nonzero(mugroup == 1.1)/np.size(mugroup) # mu = 1.1
                ft[i-id_range[0]] = np.count_nonzero(mugroup == 100)/np.size(mugroup) # targeted walk  
                
                agg_neg[m, k, i-id_range[0]] = np.size(numNegtive,0)
                agg_nor[m, k, i-id_range[0]] = numtar[-1]
                agg_time[m, k, i-id_range[0]] = time[-1]
                
        agg_effi3[k,m] = np.mean(effi[effi!=0])
        std_effi3[k,m] = 1.96 * np.std(effi[effi!=0])/np.sqrt(len((effi[effi!=0])))
        agg_fe3[k,m] = np.mean(fe[fe!=1000])
        std_fe3[k,m] = 1.96 * np.std(fe[fe!=1000])/np.sqrt(len((fe[fe!=1000])))
        agg_f13[k,m] = np.mean(f1[f1!=1000])
        std_f13[k,m] = 1.96 * np.std(f1[f1!=1000])/np.sqrt(len((f1[f1!=1000])))
        agg_ft3[k,m] = np.mean(ft[ft!=1000])
        std_ft3[k,m] = 1.96 * np.std(ft[ft!=1000])/np.sqrt(len((ft[ft!=1000])))
#%% combined figure
from matplotlib.ticker import ScalarFormatter
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# filenames = ['5Patches', '10Patches', r'$20P\times 500$', r'$10P\times 500$', r'$5P\times 500$']
# filenames = [r'$N_+=5, N_-=0$', r'$N_+=10, N_-=0$', r'$N_+=20, N_-=0$']
filenames = [r'$N_+=10$', r'$N_+=20$', r'$N_+=50$']
CaseAgents = [r'$N_+=10, N_-=5$', r'$N_+=10, N_-=10$', r'$N_+=10, N_-=15$']
# CaseAgents = [r'$N_+=10, N_-=1$', r'$N_+=10, N_-=2$', r'$N_+=10, N_-=3$', r'$N_+=10, N_-=5$', r'$N_+=10, N_-=10$', r'$N_+=10, N_-=15$']
# CaseAgents = [r'$N_-=2*300$', r'$N_-=5\times 120$', r'$N_-=10\times 60$', r'$N_-=1\times 6000$', r'$N_-=2\times 3000$', r'$N_-=3\times 2000$', r'$N_-=5\times 1200$', r'$N_-=10\times 600$', r'$N_+=10, N_-=15\times 400$']
CaseFake = [r'$N_+=5, Fake$', r'$N_+=10, Fake$', r'$N_+=20, Fake$']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf' ]

SetLabelSize = 12
SetLegendSize = 7
SubplotLabelSize = 14

# Define subplot labels
labels = ['A', 'B', 'C', 'D']
positions = [(-0.22, 1.05)] * 4  # Adjust placement to align along the ylabel
norfac1_series = [0.25, 0.5, 0.75]
norfac2_series = [0.25, 0.5, 1]

x_series = [20, 75, 200, 0]

# Define the final figure width: 17.8 cm ≈ 7.0 inches. 
# For a 2×2 grid, we choose a square layout (7.0 inches x 7.0 inches).
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.0, 7.0))
agg_time[agg_time == 0] = 10000
# Plot on the first subplot: efficiency with rho for different number of agents
for q in range(3):
    # axes[0,0].errorbar(rho_series, agg_effi[:,q]/norfac1_series[q], yerr=std_effi[:,q], color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
    agg_effi_neg = (agg_nor[q,:,:]-x_series[0]*agg_neg[q,:,:])/agg_time[q,:,:]
    numsqrt = np.sqrt(np.count_nonzero(agg_effi_neg,1))
    nonzero_mask = agg_effi_neg!= 0 
    mean_effi = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    mean_effi = np.nan_to_num(mean_effi, nan=0.0)
    stds_effi = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    stds_effi = np.nan_to_num(stds_effi, nan=0.0)
    axes[0,0].errorbar(rho_series, mean_effi/6000, yerr=1.96*stds_effi/numsqrt/6000, color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=CaseAgents[q])

# axes[0,0].set_yticks([0, 0.5, 1, 1.5, 2])
# axes[0,0].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
axes[0,0].set_ylim(0, 5.5e-4)
axes[0,0].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
axes[0,0].legend(loc='lower right', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[0,0].text(*positions[0], labels[0], transform=axes[0,0].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
axes[0,0].text(0.35,0.85, 'Penalty=-20', transform=axes[0,0].transAxes, fontsize=SetLabelSize)
axes[0,0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

for q in range(3):
    # axes[0,0].errorbar(rho_series, agg_effi[:,q]/norfac1_series[q], yerr=std_effi[:,q], color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
    agg_effi_neg = (agg_nor[q,:,:]-x_series[1]*agg_neg[q,:,:])/agg_time[q,:,:]
    numsqrt = np.sqrt(np.count_nonzero(agg_effi_neg,1))
    nonzero_mask = agg_effi_neg!= 0 
    mean_effi = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    mean_effi = np.nan_to_num(mean_effi, nan=0.0)
    stds_effi = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    stds_effi = np.nan_to_num(stds_effi, nan=0.0)
    axes[0,1].errorbar(rho_series, mean_effi/6000, yerr=1.96*stds_effi/numsqrt/6000, color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=CaseAgents[q])
# axes[0,1].set_ylim(0, 2.2)
# axes[0,1].set_yticks([0, 0.5, 1, 1.5, 2])
# axes[0,1].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
# axes[0,1].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
# axes[0,1].legend(loc='upper left', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[0,1].text(*positions[1], labels[1], transform=axes[0,1].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
axes[0,1].text(0.35,0.05, 'Penalty=-75', transform=axes[0,1].transAxes, fontsize=SetLabelSize)
axes[0,1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

for q in range(3):
    # axes[0,0].errorbar(rho_series, agg_effi[:,q]/norfac1_series[q], yerr=std_effi[:,q], color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
    agg_effi_neg = (agg_nor[q,:,:]-x_series[2]*agg_neg[q,:,:])/agg_time[q,:,:]
    numsqrt = np.sqrt(np.count_nonzero(agg_effi_neg,1))
    nonzero_mask = agg_effi_neg!= 0 
    mean_effi = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    mean_effi = np.nan_to_num(mean_effi, nan=0.0)
    stds_effi = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    stds_effi = np.nan_to_num(stds_effi, nan=0.0)
    axes[1,0].errorbar(rho_series, mean_effi/6000, yerr=1.96*stds_effi/numsqrt/6000, color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=CaseAgents[q])
# axes[1,0].set_ylim(-0.6, 1.6)
# axes[1,0].set_yticks([-0.5, 0, 0.5, 1, 1.5])
axes[1,0].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
axes[1,0].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
# axes[1,0].legend(loc='upper left', ncol=1, fontsize=SetLegendSize, frameon=True)
axes[1,0].text(*positions[2], labels[2], transform=axes[1,0].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
axes[1,0].text(0.35,0.05, 'Penalty=-200', transform=axes[1,0].transAxes, fontsize=SetLabelSize)
axes[1,0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Plot on the second subplot: efficiency with rho for different number of patches
q=0
axes[1,1].errorbar(rho_series, agg_effi[:,q]/6000, yerr=std_effi[:,q]/6000, color = 'k', fmt='--^', capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
for q in range(3):
    # axes[0,0].errorbar(rho_series, agg_effi[:,q]/norfac1_series[q], yerr=std_effi[:,q], color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2, label=filenames[q])
    agg_effi_neg = (agg_nor[q,:,:]-x_series[3]*agg_neg[q,:,:])/agg_time[q,:,:]
    numsqrt = np.sqrt(np.count_nonzero(agg_effi_neg,1))
    nonzero_mask = agg_effi_neg!= 0
    mean_effi = np.nanmean(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    mean_effi = np.nan_to_num(mean_effi, nan=0.0)
    stds_effi = np.nanstd(np.where(nonzero_mask, agg_effi_neg, np.nan), axis=1)
    stds_effi = np.nan_to_num(stds_effi, nan=0.0)
    axes[1,1].errorbar(rho_series, mean_effi/6000, yerr=1.96*stds_effi/numsqrt/6000, color = colors[q], fmt='-o', capsize=0, elinewidth=2, markeredgewidth=2)
# for m in range(np.size(NPatches)):
#     axes[1,1].errorbar(rho_series, agg_effi2[:,m]/6000, yerr=std_effi2[:,m]/6000, color = colors[m], fmt='--s', capsize=0, elinewidth=2, markeredgewidth=2, label=CaseFake[m])
axes[1,1].set_ylim(0, 5.5e-4)
# axes[1,1].set_yticks([0, 0.5, 1, 1.5, 2])
axes[1,1].set_xlabel(r'$\rho$', fontsize=SetLabelSize)
# axes[1,1].set_ylabel(r'$\eta$', fontsize=SetLabelSize)
axes[1,1].legend(loc='upper right', ncol=2, fontsize=SetLegendSize, frameon=True)
axes[1,1].text(0.35,0.05, 'Penalty=0', transform=axes[1,1].transAxes, fontsize=SetLabelSize)
axes[1,1].text(*positions[3], labels[3], transform=axes[1,1].transAxes, fontsize=SubplotLabelSize, fontweight='bold')
axes[1,1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# # Add inset
# axins = inset_axes(axes[0, 1], width="100%", height="100%", loc='upper right',
#                    bbox_to_anchor=(0.65, 0.65, 0.35, 0.35), bbox_transform=axes[0, 1].transAxes)
# for q in range(numfile):
#     axins.plot(agg_fe2[:,q], agg_effi2[:,q], 'o',    # 'o' sets the marker to circle
#          markersize=4,  # adjust the marker size as needed
#          # markerfacecolor='red',  # fill color of the markers
#          linestyle='', label=filenames[q])
# # axes[1, 0].set_ylim(0, 0.6)
# axins.set_xlim(0, 0.3)
# axins.set_xlabel(r'$f_{\mu=3}$', fontsize=SetLegendSize)
# axins.set_ylabel(r'$\eta$', fontsize=SetLegendSize)

# Improve layout spacing
plt.tight_layout()

# Save the figure in a high-quality PDF format (vector graphics)
plt.savefig('Figure4.pdf', format='pdf', dpi=600)
plt.savefig('Figure4.png', format='png', dpi=600)
# Optionally display the figure
plt.show()

#%% save the data for plotting
base_dir = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\NegativePatch'
with h5py.File(os.path.join(base_dir, 'A_Figure3_PlotData.h5'), 'w') as hdf:
    hdf.create_dataset('agg_nor', data=agg_nor) # for subfigure A
    hdf.create_dataset('agg_neg', data=agg_neg) # 3*14*1000
    hdf.create_dataset('agg_time', data=agg_time) # 3*14*1000
    hdf.create_dataset('agg_effi', data=agg_effi)  # 18*3
    hdf.create_dataset('std_effi', data=std_effi)  # 18*3
