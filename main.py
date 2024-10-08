# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:58:19 2022

@author: arr43
"""


# %% Imports
import os
import datetime
import itertools
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.cluster import adjusted_mutual_info_score
from clustering_utils import *
import copy


# %% Defaults

# Project dir wherever this script is
projectdir = os.path.dirname(os.path.realpath(__file__))

# Data dir
datadir = os.path.join(projectdir, 'data')

# Report dir for saving figures
reportdir = os.path.join(projectdir, 'report')

# Master datasheet
masterdatapath = os.path.join(datadir, 'MasterDataSheet.xlsx')

# kinematics data
kindatapath = os.path.join(datadir, 'AllCurves_ptavgs.npy')

# # Opensim model path
# modelpath = r'C:\Users\arr43\Documents\OpenSim\4.3\Models\Gait2392_Simbody\gait2392_simbody_custom.osim'
#
# # Sto file header template
# stoheaderpath = r'X:\Health\ResearchProjects\EPreatoni\EG-FH1095\Admin\sto_header.sto'
#
# # Opensim paths
# modelpath = r'C:\Users\arr43\Documents\OpenSim\4.3\Models\Gait2392_Simbody\gait2392_simbody_custom.osim'
# genscalsetuppath = r'C:\Users\arr43\Documents\OpenSim\4.3\Models\Gait2392_Simbody\Generic_Scale_setup.xml'

# Matplotlib style
matplotlib.use('Qt5Agg')
matplotlib.style.use('default')

# Saving filename
savefilename = 'PCA_speed_by_speed_clustering'
savingkw = 'Clust'

# Stages of interest
stages = ['STG_02', 'STG_03', 'STG_04']

# Speeds of interest
speeds = [9 + int(stage[-2:]) for stage in stages]

# Wanted variables for clustering
wantedvars = ['STRIDEFREQ', 'DUTYFACTOR', 'RCOM_2', 'RTRUNK2PELVIS_0', 'RPELV_ANG_0', 'RHIP_0', 'RKNEE_0', 'RANK_0']
discvars = ['STRIDEFREQ', 'DUTYFACTOR']
contvars = ['RCOM_2', 'RTRUNK2PELVIS_0', 'RPELV_ANG_0', 'RHIP_0', 'RKNEE_0', 'RANK_0']

# For figure decoration
stgtitles = ['11 km/h', '12 km/h', '13 km/h', 'Multispeed']
kinematics_titles = {'STRIDEFREQ': 'Stride frequency',
                      'DUTYFACTOR': 'Duty factor',
                      'RCOM_2': 'vCOM',
                      'RTRUNK2PELVIS_0': 'Trunk-pelvis',
                      'RHIP_0': 'Hip',
                      'RPELV_ANG_0': 'Pelvis tilt',
                      'RKNEE_0': 'Knee',
                      'RANK_0': 'Ankle',
                      }

# Ylims for bottom axs in reconstruction quality figures
recbot_ylims = [[-0.125, 0.125], [-0.025, 0.025], [-0.0125, 0.0125], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]]

# Short ylabels for kinematics
short_ylabels = ['Hz/leg', 'CT/ST', '< D - U >', '< F - E >', '< A - P >', '< E - F >', '< E - F >', '< P - D >']

# Labels for final figs
kinematics_ylabels = {'STRIDEFREQ': 'Hz/leg',
                      'DUTYFACTOR': 'CT/ST',
                      'RCOM_2': 'Position (m/leg) \n< Down - Up >',
                      'RTRUNK2PELVIS_0': '${\Theta}$ (°) \n< Flex - Ext >',
                      'RHIP_0': '${\Theta}$ (°) \n< Ext - Flex >',
                      'RPELV_ANG_0': '${\Theta}$ (°) \n< Ant - Post >',
                      'RKNEE_0': '${\Theta}$ (°) \n< Ext - Flex >',
                      'RANK_0': '${\Theta}$ (°) \n< Plantar - Dorsi >',
                      }

# Wanted scores with the optimum for the plot
wanted_scores = {'Silhouette': '1',
                'Calinski-Harabasz': 'largest',
                 'Davies-Bouldin': '0'}

# Acceptable errors
acceptable_errors = {'STRIDEFREQ': 0.05,
                        'DUTYFACTOR': 0.01,
                        'RCOM_2': 0.005,
                        'RTRUNK2PELVIS_0': 2,
                        'RPELV_ANG_0': 2,
                        'RHIP_0': 2,
                        'RKNEE_0': 2,
                        'RANK_0': 2}

# Demographics, anthropometrics and physiological variables and titles
demoanthrophysvars_titles = {'Age': 'Age',
                             'Height': 'Height',
                             'Mass': 'Mass',
                             'TrunkLgth': 'Trunk length',
                             'PelvWidth': 'Pelvis width',
                             'LegLgth_r': 'Leg length',
                             'ThiLgth_r': 'Thigh length',
                             'ShaLgth_r': 'Shank length',
                             'FootLgth_r': 'Foot length',
                             'LT': 'LT',
                             'VO2peakkg': 'VO2peak',
                             'RE': 'Running Economy',
                             'RELT': 'Running Economy LT',
                             'RunningDaysAWeek': 'Weekly runs',
                             'KmAWeek': 'Weekly volume',
                             'Time10Ks': '10k time'
                            }

# Names and units for figures
demoanthrophysvars_ylabels = {'Sex': 'Females (%)',
                          'Age': 'years',
                          'Height': 'm',
                          'Mass': 'kg',
                          'TrunkLgth': 'm',
                          'LegLgth_r': 'm',
                          'PelvWidth': 'm',
                          'ThiLgth_r': 'm',
                          'ShaLgth_r': 'm',
                          'FootLgth_r': 'm',
                          'LT': 'km/h',
                          'VO2peakkg': 'ml/min/kg',
                          'RunningDaysAWeek': 'count',
                          'KmAWeek': 'km',
                          'Time10Ks': 'mm:ss',
                          'RE': 'kcal/min/kg',
                          }

# physvars_ylabels = {'LT': 'km/h',
#                     'VO2peakkg': 'ml/min/kg',
#                     'RunningDaysAWeek': 'count',
#                     'KmAWeek': 'km',
#                     'Time10Ks': 'mm:ss',
#                     }
#
# revars_ylabels = {'RE': 'kcal/min/kg',
#                   'RELT': 'kcal/km/kg'}

# Speed linestyles
speedcolours = ['C0', 'C6', 'C3']

# # Opensim stoheader
# stoheaderpath = r'X:\Health\ResearchProjects\EPreatoni\EG-FH1095\Admin\sto_header.sto'
#
# # Default column names for opensim model 3DGaitModel12392
# osimvarnames = ['time', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
#             'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
#             'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
#             'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
#
# # Corresponding varnames
# corrvarnames = ['tsec', 'RPELV_ANG_0', '_', '_', '_', 'RCOM_2', '_',
#                 'RHIP_0','_', '_', 'RKNEE_0', 'RANK_0', '_',
#                 '_', '_', '_', '_', '_', '_',
#                 '_', '_', 'RTRUNK2PELVIS_0', '_', '_']


# %% Data loading and wrangling

# Load mastersheet
master = pd.read_excel(masterdatapath, index_col=1, header=1)

# Have VO2max called VO2peakkg for correctness
master['VO2peakkg'] = master['VO2max']

# EE is in kcal/min, convert to kcal/km
for speed in speeds:

    master[f'EE{speed}kg'] = master[f'EEJW{speed}'] / master['Mass']
    master[f'EE{speed}km'] = master[f'EEJW{speed}'] * 60 / speed
    master[f'EE{speed}kmkg'] = master[f'EE{speed}km'] / master['Mass']
    master[f'LT{speed}pctge'] = speed / master['LT'] * 100

# Get 10k times which are datetime.time in seconds
master['Time10Ks'] = master['Time10K'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Get LT+1 speed for all participants
ltplus1 = np.round(master['LT'] + 1)
ltplus1 = np.round(master['LT'])

# Get participants who have LT+1 speed of at least 13 km/h
pts_physok = ltplus1.loc[ltplus1 >= 13].index.to_list()
sub13 = master.loc[pts_physok]

# Load kinematic data
data = np.load(kindatapath, allow_pickle=True).item()

clustdata = {}
footstrikedata = {}
rawstridelen = []

for stgi, (stage, speed) in enumerate(zip(stages, speeds)):

    # Get pt codes
    pts_kinemok = list(data[stage].keys())

    # Include only pts that were below ltplus1 speed and who had data
    pts = sorted(list(set(pts_physok) & set(pts_kinemok)))

    # Get foot angle
    footstrikedata[stage] = np.empty((len(pts), 201))
    footstrikedata[stage][:] = np.nan

    for pti, pt in enumerate(pts):
        footstrikedata[stage][pti, :] = data[stage][pt]['RFOOT_0']

    # Get rid of unwanted vars
    unwanted = [key for key in list(data[stage][pts[0]].keys()) if key not in wantedvars]
    for pt in pts:
        for poop in unwanted:
            data[stage][pt].pop(poop)

    # Get average patterns
    clustdata[stage] = {}

    # Stack each participant's avge pattern together
    for pt in pts:

        # Get raw length of stride
        rawstridelen.append(1/(data[stage][pt]['STRIDEFREQ'] * master['LegLgth_r'].loc[pt]))

        for key, values in data[stage][pt].items():

            if key not in clustdata[stage].keys():
                clustdata[stage][key] = values
            else:
                clustdata[stage][key] = np.vstack([clustdata[stage][key], values])

# Create vartracker
vartracker = {}
for stage in stages:
    vartracker[stage] = [[key] * values.shape[1] for key, values in clustdata[stage].items()]
    vartracker[stage] = list(itertools.chain(*vartracker[stage]))

# Horizontally stack all stages in multispeed
if len(stages) > 1:
    clustdata['multispeed'] = {}
    vartracker['multispeed'] = []

    for key in clustdata[stages[0]].keys():
        clustdata['multispeed'][key] = np.hstack([clustdata[stage][key] for stage in stages])
        vartracker['multispeed'].extend([[f'{key}_{int(speeds[stgi])}'] * clustdata[stage][key].shape[1] for stgi, stage in enumerate(stages)])

    # Unnest vartracker['multispeed']
    vartracker['multispeed'] = list(itertools.chain(*vartracker['multispeed']))

# Get all rawstridelen
rawstridelen = np.hstack(rawstridelen)

# Get mean and std of rawstridelen in frames
meanstridelen = np.mean(rawstridelen) * 200
stdstridelen = np.std(rawstridelen) * 200

# print mean and std of rawstridelen for reporting in paper
print(f'Mean stride length: {meanstridelen} frames')
print(f'Std stride length: {stdstridelen} frames')


# %% Perform clustering for each speed using participants up to ltplus1 speed

# Create holders
dr_scores = pd.DataFrame(columns=['n_components99', 'mse', 'rmse'])
clust_scores = pd.DataFrame(columns=['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Gap'])
dendros = []
labels = []
ptlabels = {}
refcolours = []
linkmats = []
ami = {}


#%% Perform PCA and clustering analysis for each stage

scores_by_k = {}
stat_comparison = {}

for stgi, stage in enumerate(clustdata.keys()):

    scores_by_k[stage] = {scorename: pd.DataFrame() for scorename in wanted_scores.keys()}

    # Concatenate data
    X = np.hstack([clustdata[stage][key] for key in clustdata[stage].keys()])

    # Standardise data
    scaler = CustomScaler()
    Xz = scaler.fit_transform(X, vartracker=vartracker[stage])

    # Apply PCA
    pca = PCA(n_components=0.99)
    pcaed = pca.fit_transform(Xz)

    # Reconstruction quality
    yhat = pca.inverse_transform(pcaed)
    mse = mean_squared_error(Xz, yhat)
    rmse = mean_squared_error(Xz, yhat, squared=False)

    # Get row wise mse
    ptmse = mean_squared_error(Xz.T, yhat.T, multioutput='raw_values')

    # Sorted ptmse
    ptmsesorted = np.sort(ptmse)

    # Get index of pt with median mse to be the representative pt
    medianptidx = np.where(ptmse == ptmsesorted[len(ptmsesorted) // 2])[0][0]

    if stage != 'multispeed':

        # Create grid figure for reconstruction quality assessment
        recfig, recaxs, gridshape = make_splitgrid(2, int(len(wantedvars)/2), figsize=(11, 5.75))

        # Plot ground truth curve data in top axs
        for vari, varname in enumerate(wantedvars):

            # Get varidx
            varidx = np.where(np.array(vartracker[stage]) == varname)[0]

            if len(varidx) == 1:
                recaxs['topaxs'][vari].plot(X[medianptidx, varidx], '-o', color='k')
            else:
                recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(X[medianptidx, varidx])),
                                            X[medianptidx, varidx], color='k')

            recaxs['topaxs'][vari].set_title(kinematics_titles[varname])
            recaxs['topaxs'][vari].set_ylabel(short_ylabels[vari])
            recaxs['topaxs'][vari].spines['right'].set_visible(False)
            recaxs['topaxs'][vari].spines['top'].set_visible(False)
            recaxs['topaxs'][vari].set_xticks([])

        # Back scale the predicted data
        yhat_original = scaler.inverse_transform(yhat)

        # Get errors by variable
        recerrors = yhat_original - X
        recmeanerrors = np.mean(recerrors, axis=0)
        rec25pctileerrors = np.quantile(recerrors, 0.025, axis=0)
        rec975pctileerrors = np.quantile(recerrors, 0.975, axis=0)
        recstderrors = np.std(recerrors, axis=0)

        # Plot data in top axs and reconstruction error in bottom axs
        for vari, varname in enumerate(wantedvars):

            # Get varidx
            varidx = np.where(np.array(vartracker[stage]) == varname)[0]

            if len(varidx) == 1:
                recaxs['topaxs'][vari].plot(yhat_original[medianptidx, varidx], '-o', color=speedcolours[stgi])

                # Plot  as a point with errorbar
                recaxs['bottomaxs'][vari].vlines(x=1, ymin=rec25pctileerrors[varidx],
                                                 ymax=rec975pctileerrors[varidx], color=speedcolours[stgi])
                recaxs['bottomaxs'][vari].plot(1, recmeanerrors[varidx], 'o', color=speedcolours[stgi])

            else:
                recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(yhat_original[medianptidx, varidx])),
                                            yhat_original[medianptidx, varidx],
                                            color=speedcolours[stgi])

                # Plot 95% of the errors
                recaxs['bottomaxs'][vari].plot(np.linspace(0, 100, len(recmeanerrors[varidx])), recmeanerrors[varidx],
                                               color=speedcolours[stgi])
                recaxs['bottomaxs'][vari].fill_between(np.linspace(0, 100, len(recmeanerrors[varidx])),
                                                       rec25pctileerrors[varidx], rec975pctileerrors[varidx],
                                                       alpha=0.5, color=speedcolours[stgi], edgecolor='none')

                # Set xlims
                recaxs['topaxs'][vari].set_xlim([0, 100])
                recaxs['bottomaxs'][vari].set_xlim([0, 100])

        # Decorate recfig

        # Get ylims of stride frequency and duty factor
        ylims = recaxs['topaxs'][0].get_ylim()
        recaxs['topaxs'][0].set_ylim(ylims[0] - 0.05 * ylims[1], ylims[1] + 0.05 * ylims[1])
        ylims = recaxs['topaxs'][1].get_ylim()
        recaxs['topaxs'][1].set_ylim(ylims[0] - 0.05*ylims[1], ylims[1] + 0.05*ylims[1])

        # Round current yticks in duty factor top axs figure
        yticks = recaxs['topaxs'][1].get_yticks()
        recaxs['topaxs'][1].set_yticklabels(np.round(yticks, 3))
        recaxs['bottomaxs'][0].set_xticks([1], ['Direct PCA 99'])
        recaxs['bottomaxs'][0].set_xlim([0.5, 1.5])
        recaxs['bottomaxs'][1].set_xticks([1], ['Direct PCA 99'])
        recaxs['bottomaxs'][1].set_xlim([0.5, 1.5])

        for vari, (var, ax) in enumerate(zip(acceptable_errors.keys(), recaxs['bottomaxs'])):

            # Add horizontal lines to indicate acceptable errors
            ax.hlines(y=acceptable_errors[var],
                      xmin=ax.get_xlim()[0],
                      xmax=ax.get_xlim()[1],
                      color='k', linestyle=':')
            ax.hlines(y=-acceptable_errors[var],
                      xmin=ax.get_xlim()[0],
                      xmax=ax.get_xlim()[1],
                      color='k', linestyle=':')

            # Remove right spines
            ax.spines['right'].set_visible(False)

            # Set ylims
            ax.set_ylim(recbot_ylims[vari])

            # Set yticks
            ax.set_yticks([-acceptable_errors[var], acceptable_errors[var]])

            print(f'{var} ylims: {ax.get_ylim()}')

        # Set ylabels
        recaxs['bottomaxs'][0].set_ylabel('Error')
        recaxs['bottomaxs'][4].set_ylabel('Error')

        recaxs['topaxs'][-1].legend(['Ground Truth', 'Direct PCA 99'],
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0),
                                    ncol=5,
                                    bbox_transform=recfig.transFigure,
                                    frameon=False)
        plt.subplots_adjust(bottom=0.11)
        recfig.suptitle(f'Single speed PCA {speeds[stgi]} km/h')

        # Save figure
        recfig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_recquality.png'), dpi=300, bbox_inches='tight')
        plt.close(recfig)

    elif stage == 'multispeed':

        # Back scale the predicted data
        yhat_original = scaler.inverse_transform(yhat)

        # Go through unique speeds included
        for speedi, speed in enumerate(speeds):

            # Create grid figure for reconstruction quality assessment
            recfig, recaxs, gridshape = make_splitgrid(2, int(len(wantedvars) / 2), figsize=(11, 5.75))

            # Plot ground truth curve data in top axs
            for vari, varname in enumerate(wantedvars):

                # Get varidx
                varidx = np.where(np.array(vartracker[stage]) == f'{varname}_{str(speed)}')[0]

                if len(varidx) == 1:
                    recaxs['topaxs'][vari].plot(X[medianptidx, varidx], '-o', color='k')
                else:
                    recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(X[medianptidx, varidx])),
                                                X[medianptidx, varidx], color='k')

                recaxs['topaxs'][vari].set_title(kinematics_titles[varname])
                recaxs['topaxs'][vari].set_ylabel(short_ylabels[vari])
                recaxs['topaxs'][vari].spines['right'].set_visible(False)
                recaxs['topaxs'][vari].spines['top'].set_visible(False)
                recaxs['topaxs'][vari].set_xticks([])

            # Get errors by variable
            recerrors = yhat_original - X
            recmeanerrors = np.mean(recerrors, axis=0)
            rec25pctileerrors = np.quantile(recerrors, 0.025, axis=0)
            rec975pctileerrors = np.quantile(recerrors, 0.975, axis=0)
            recstderrors = np.std(recerrors, axis=0)

            # Plot data in top axs and reconstruction error in bottom axs
            for vari, varname in enumerate(wantedvars):

                # Get varidx
                varidx = np.where(np.array(vartracker[stage]) == f'{varname}_{str(speed)}')[0]

                if len(varidx) == 1:
                    recaxs['topaxs'][vari].plot(yhat_original[medianptidx, varidx], '-o', color=speedcolours[speedi])

                    # Plot  as a point with errorbar
                    recaxs['bottomaxs'][vari].vlines(x=1, ymin=rec25pctileerrors[varidx],
                                                     ymax=rec975pctileerrors[varidx], color=speedcolours[speedi])
                    recaxs['bottomaxs'][vari].plot(1, recmeanerrors[varidx], 'o', color=speedcolours[speedi])

                else:
                    recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(yhat_original[medianptidx, varidx])),
                                                yhat_original[medianptidx, varidx],
                                                color=speedcolours[speedi])

                    # Plot 95% of the errors
                    recaxs['bottomaxs'][vari].plot(np.linspace(0, 100, len(recmeanerrors[varidx])), recmeanerrors[varidx],
                                                   color=speedcolours[speedi])
                    recaxs['bottomaxs'][vari].fill_between(np.linspace(0, 100, len(recmeanerrors[varidx])),
                                                           rec25pctileerrors[varidx], rec975pctileerrors[varidx],
                                                           alpha=0.5, color=speedcolours[speedi], edgecolor='none')

                    # Set xlims
                    recaxs['topaxs'][vari].set_xlim([0, 100])
                    recaxs['bottomaxs'][vari].set_xlim([0, 100])

            # Decorate recfig

            # Get ylims of stride frequency and duty factor
            ylims = recaxs['topaxs'][0].get_ylim()
            recaxs['topaxs'][0].set_ylim(ylims[0] - 0.05 * ylims[1], ylims[1] + 0.05 * ylims[1])
            ylims = recaxs['topaxs'][1].get_ylim()
            recaxs['topaxs'][1].set_ylim(ylims[0] - 0.05 * ylims[1], ylims[1] + 0.05 * ylims[1])

            # Round current yticks in duty factor top axs figure
            yticks = recaxs['topaxs'][1].get_yticks()
            recaxs['topaxs'][1].set_yticklabels(np.round(yticks, 3))

            recaxs['bottomaxs'][0].set_xticks([1], ['Direct PCA 99'])
            recaxs['bottomaxs'][0].set_xlim([0.5, 1.5])
            recaxs['bottomaxs'][1].set_xticks([1], ['Direct PCA 99'])
            recaxs['bottomaxs'][1].set_xlim([0.5, 1.5])

            for vari, (var, ax) in enumerate(zip(acceptable_errors.keys(), recaxs['bottomaxs'])):
                # Add horizontal lines to indicate acceptable errors
                ax.hlines(y=acceptable_errors[var],
                          xmin=ax.get_xlim()[0],
                          xmax=ax.get_xlim()[1],
                          color='k', linestyle=':')
                ax.hlines(y=-acceptable_errors[var],
                          xmin=ax.get_xlim()[0],
                          xmax=ax.get_xlim()[1],
                          color='k', linestyle=':')

                # Remove right spines
                ax.spines['right'].set_visible(False)

                # Set ylims
                ax.set_ylim(recbot_ylims[vari])

                # Set yticks
                ax.set_yticks([-acceptable_errors[var], acceptable_errors[var]])

                print(f'{var} ylims: {ax.get_ylim()}')

            # Set ylabels
            recaxs['bottomaxs'][0].set_ylabel('Error')
            recaxs['bottomaxs'][4].set_ylabel('Error')

            recaxs['topaxs'][-1].legend(['Ground Truth', 'Direct PCA 99'],
                                        loc='lower center',
                                        bbox_to_anchor=(0.5, 0),
                                        ncol=5,
                                        bbox_transform=recfig.transFigure,
                                        frameon=False)
            plt.subplots_adjust(bottom=0.11)
            recfig.suptitle(f'Multispeed PCA {str(speed)} km/h')

            # Save figure
            recfig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_{str(speed)}_recquality.png'), dpi=300, bbox_inches='tight')
            plt.close(recfig)

    # Store dr scores
    dr_scores.loc[stage] = {'n_components99': pcaed.shape[1], 'mse': mse, 'rmse': rmse}

    # Hierarchical clustering analysis to choose number of clusters
    HCA = HierarchClusteringAnalysisTool(pcaed)

    # Save individual scores for wearables study
    if stage == 'multispeed':

        sampsilh = pd.DataFrame(data=np.array([pts, HCA.Silhouette_samples[HCA.n_clusters]]).T, columns=['ptcode', 'Silhouette'])
        sampsilh.to_csv(os.path.join(reportdir, 'Multispeed_ptsilhouette.csv'), index=False)

    # Store scores
    for scorename in wanted_scores.keys():
        scores_by_k[stage][scorename] = HCA.scores[scorename]

    # Apply hierarchical clustering with final choice
    dendrofig = plt.figure(figsize=(6.5, 1.5))
    clustlabels, dendro, score_table, linkmat, _ = hierarch_clust(pcaed, HCA.n_clusters, pts)
    dendroax = plt.gca()
    dendroax.set_title(f'Recoloring {stage}. Try to colour-match the leaves and bottom vert lines')

    # Store results
    clust_scores.loc[stage] = HCA.scores.loc[HCA.n_clusters]
    dendros.append(dendro)
    linkmats.append(linkmat)

    # Make plots a bit prettier
    if len(dendros) == 1:

        # Add bottom leaves to dendrogram
        append_bottom_leaves_dendrogram(dendroax)

        # Set title adding the Silhouette score
        dendroax.set_title(f'{stgtitles[stgi]} '
                           f'(Silh = {np.round(score_table.loc["Silhouette"].values[0], 3)})')

    # Get participant labels and colours from dendrogram
    colourid = pd.DataFrame({'ptcode': dendros[-1]['ivl'], 'colourcode': dendros[-1]['leaves_color_list']}).sort_values(
        by=['ptcode'], ignore_index=True)
    colourid['clustlabel'] = 0
    colours = np.sort(colourid['colourcode'].unique())
    for label, colourcode in enumerate(colours):
        colourid.loc[colourid['colourcode'] == colourcode, 'clustlabel'] = label

    if len(dendros) == 1:

        # Write legend. This is currently disgusting but it seems to work

        # Temporary legend
        templegend = list(colourid['colourcode'].unique())
        templegend.insert(0, '_nolegend')
        legend = dendroax.legend(templegend, frameon=False)

        # Get handles and their colours
        handles = legend.legendHandles
        handlecolours = [handle.get_color() for handle in handles]

        # Get labels which should be the same as the CN colourcode in matplotlib
        leglabels = [label.get_text() for label in legend.get_texts()]
        leglabelcolours = [mcolors.to_rgba(leglabel) for leglabel in leglabels]

        # Get indices of leglabelcolours in handlecolours
        idcs = []
        for handlecolour in handlecolours:
            for leglabi, leglabelcolour in enumerate(leglabelcolours):
                if np.all(handlecolour == leglabelcolour):
                    idcs.append(leglabi)
                    break

        # Get the colourcode again and order them correctly
        finalcolours = list(colourid['colourcode'].unique())
        orderedcolours = [finalcolours[idx] for idx in idcs]

        # Get count of pts in each cluster
        clustcount = [
            len(colourid.loc[colourid['colourcode'] == colourlabel]) for colourlabel in orderedcolours]

        # Subtract 1 from the digit in finalcolours
        orderedcolours = [orderedcolours[:-1] + str(int(orderedcolours[-1]) - 1) for orderedcolours in
                          orderedcolours]

        # Add count of pts in each cluster
        orderedcolours = [f'{orderedcolour} ({clustcounti})' for orderedcolour, clustcounti in
                          zip(orderedcolours, clustcount)]

        # Set the legend correctly now
        orderedcolours.insert(0, '_nolegend')
        dendroax.legend(orderedcolours, frameon=False)

        plt.tight_layout()

        # TSNE plot
        tsnefig = tsne_plot(pcaed, [5, 10, 15, 20], colourid['colourcode'], labels=colourid['ptcode'],
                            title=f'TSNE_{stage}')

        # Store results here and there. Note there might be some of these you don't need and can delete
        labels.append(colourid['clustlabel'].tolist())
        ptlabels[stage] = colourid
        clustdata[stage]['ptlabels'] = colourid

    # Make colours consistent with previous iteration
    elif len(dendros) > 1:

        append_bottom_leaves_dendrogram(dendroax, labelcolour=ptlabels[stages[stgi - 1]])

        # Transition analysis from previous to current partition
        trans_analysis = TransitionAnalysis(ptlabels[stages[stgi - 1]], colourid, dendroax)
        trans_analysis.dendroax.set_title(f'{stgtitles[stgi]} '
                                          f'(Silh = {np.round(score_table.loc["Silhouette"].values[0], 3)}, '
                                          f'AMI = {np.round(trans_analysis.ami, 3)})')

        # Write legend. This is currently disgusting but it seems to work

        # Temporary legend
        templegend = trans_analysis.currentcolourlabels['colourcode'].to_list()
        templegend.insert(0, '_nolegend')
        legend = trans_analysis.dendroax.legend(templegend, frameon=False)

        # Get handles and their colours
        handles = legend.legendHandles
        handlecolours = [handle.get_color() for handle in handles]

        # Get labels which should be the same as the CN colourcode in matplotlib
        leglabels = [label.get_text() for label in legend.get_texts()]
        leglabelcolours = [mcolors.to_rgba(leglabel) for leglabel in leglabels]

        # Get indices of leglabelcolours in handlecolours
        idcs = []
        for handlecolour in handlecolours:
            for leglabi, leglabelcolour in enumerate(leglabelcolours):
                if np.all(handlecolour == leglabelcolour):
                    idcs.append(leglabi)
                    break

        # Get the colourcode again and order them correctly
        finalcolours = trans_analysis.currentcolourlabels['colourcode'].to_list()
        orderedcolours = [finalcolours[idx] for idx in idcs]

        # Get count of pts in each cluster
        clustcount = [len(trans_analysis.curr_colourid.loc[trans_analysis.curr_colourid['colourcode'] == colourlabel]) for
                      colourlabel in orderedcolours]

        # Subtract 1 from the digit in finalcolours
        orderedcolours = [orderedcolours[:-1] + str(int(orderedcolours[-1]) - 1) for orderedcolours in orderedcolours]

        # Add count of pts in each cluster
        orderedcolours = [f'{orderedcolour} ({clustcounti})' for orderedcolour, clustcounti in
                          zip(orderedcolours, clustcount)]

        # Set the legend correctly now
        orderedcolours.insert(0, '_nolegend')
        trans_analysis.dendroax.legend(orderedcolours, frameon=False)

        plt.tight_layout()

        # Colour edge according to previous speed for easier visualisation of betrayers
        tsnefig = tsne_plot(pcaed, [5, 10, 15, 20], colourid['colourcode'], ringcolours=ptlabels[stages[stgi - 1]]['colourcode'],
                  labels=colourid['ptcode'], title=f'{stage} TSNE')

        # Store labels and ptlabels
        labels.append(trans_analysis.curr_colourid['clustlabel'].tolist())
        ptlabels[stage] = trans_analysis.curr_colourid

        # Store colourid in clustdata
        clustdata[stage]['ptlabels'] = trans_analysis.curr_colourid

    # Save and close figures
    HCA.scorefig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_HCA_scores.png'), dpi=300, bbox_inches='tight')
    dendrofig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_finaldendro.png'), dpi=300, bbox_inches='tight')
    tsnefig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close(HCA.scorefig)
    plt.close(HCA.dendrofig)
    plt.close(dendrofig)
    plt.close(tsnefig)

    # Get unique clustlabels and corresponding colour
    uniqclustlabels = natsort.natsorted(np.unique(clustdata[stage]['ptlabels']['clustlabel']))
    uniqclustcolours = [clustdata[stage]['ptlabels']['colourcode'].loc[
                            clustdata[stage]['ptlabels']['clustlabel'] == x].iloc[0] for x in uniqclustlabels]

    # Stat comparison of single speed clustering TODO. The plotting part is not ready for more than two groups
    if stage != 'multispeed':
        stat_comparison[stage] = {'0D': {}, '1D': {}}
        stat_comparison[stage]['0D'] = comparison_0D_contvar_indgroups({key: clustdata[stage][key] for key in discvars},
                                                                        clustdata[stage]['ptlabels']['clustlabel'],
                                                                        f'{savingkw}_{stage}',
                                                                        reportdir,
                                                                        uniqclustcolours)

        stat_comparison[stage]['1D'] = comparison_1D_contvar_indgroups({key: clustdata[stage][key] for key in contvars},
                                                                        clustdata[stage]['ptlabels']['clustlabel'],
                                                                        f'{savingkw}_{stage}',
                                                                        reportdir,
                                                                        uniqclustcolours)

        # Plot all variables with significant differences indicated
        kinfig, kinaxs = plt.subplots(2, 4, figsize=(11, 4.5))
        kinaxs = kinaxs.flatten()

        avgeto = []
        for clusti, uniqclust in enumerate(uniqclustlabels):
            clustidcs = np.where(clustdata[stage]['ptlabels']['clustlabel'] == uniqclust)[0]
            avgeto.append(np.round(np.mean(clustdata[stage]['DUTYFACTOR'][clustidcs, :]) * 100, 1))

        for vari, varname in enumerate(wantedvars):

            if varname in discvars:

                # Violin plot
                sns.violinplot(ax=kinaxs[vari],
                               x=clustdata[stage][f'ptlabels']['clustlabel'].values,
                               y=clustdata[stage][varname].flatten(),
                               palette=uniqclustcolours)

                # Xticks
                kinaxs[vari].set_xticks([0, 1], ['C0', 'C1'])

                # Get key which is not normality
                stat_test = [key for key in stat_comparison[stage]['0D'][varname].keys() if key != 'normality'][0]

                # Add asterisk to the title to indicate significant differences
                if stat_comparison[stage]['0D'][varname][stat_test]['p'] < 0.05:
                    kinaxs[vari].set_title(f'{kinematics_titles[varname]} *')
                else:
                    kinaxs[vari].set_title(f'{kinematics_titles[varname]}')

            elif varname in contvars:

                groups = []

                for clusti, uniqclust in enumerate(uniqclustlabels):
                    clustidcs = np.where(clustdata[stage][f'ptlabels']['clustlabel'] == uniqclust)[0]
                    groups.append(clustdata[stage][varname][clustidcs, :])

                    # SPM plot
                    spm1d.plot.plot_mean_sd(groups[-1], x=np.linspace(0, 100, groups[-1].shape[1]),
                                            linecolor=uniqclustcolours[clusti], facecolor=uniqclustcolours[clusti],
                                            ax=kinaxs[vari])

                # Add vertical line at avge toe off (outside the previous loop so we can get the final ylimits)
                for clusti, clustavgeto in enumerate(avgeto):
                    kinaxs[vari].axvline(x=clustavgeto, color=uniqclustcolours[clusti], linestyle=':')

                # Add patch to indicate significant differences
                spmtest = list(stat_comparison[stage]['1D'][varname].keys())[0]
                if stat_comparison[stage]['1D'][varname][spmtest].h0reject:

                    # Scaler for sigcluster endpoints
                    tscaler = kinaxs[vari].get_xlim()[1] / (groups[0].shape[1] - 1)

                    for sigcluster in stat_comparison[stage]['1D'][varname][spmtest].clusters:
                        ylim = kinaxs[vari].get_ylim()
                        kinaxs[vari].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                             (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                             ylim[1] - ylim[0], color='grey', alpha=0.5, linestyle=''))

                # title
                kinaxs[vari].set_title(kinematics_titles[varname])

            # Add ylabel
            kinaxs[vari].set_ylabel(kinematics_ylabels[varname])

            # Rid off top and right spines
            kinaxs[vari].spines['right'].set_visible(False)
            kinaxs[vari].spines['top'].set_visible(False)

        # Legend
        kinaxs[-1].legend(['C0', 'C1'], loc='lower right', frameon=False)

        # Suptitle
        kinfig.suptitle(f'{stgtitles[stgi]} kinematics')
        plt.tight_layout()

        # Save and close
        kinfig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_kinematics.png'), dpi=300, bbox_inches='tight')
        plt.close(kinfig)

# Save multispeed ptlabels to file
clustdata['multispeed']['ptlabels'].to_csv(os.path.join(reportdir, f'{savingkw}_multispeed_ptlabels.csv'), index=False)

# %% Score analysis

# Create figure
scorefig, scoreaxs = plt.subplots(1, len(wanted_scores), figsize=(11, 2))

for scorei, scorename in enumerate(wanted_scores.keys()):
    for stgi, stage in enumerate(clustdata.keys()):
        if stage != 'multispeed':
            scoreaxs[scorei].plot(scores_by_k[stage][scorename].index, scores_by_k[stage][scorename].values, '-o',
                                  color=speedcolours[stgi])
        else:
            scoreaxs[scorei].plot(scores_by_k[stage][scorename].index, scores_by_k[stage][scorename].values, '-o',
                                  color='k')

    scoreaxs[scorei].set_xlabel('k')
    scoreaxs[scorei].set_title(scorename)
    scoreaxs[scorei].spines['right'].set_visible(False)
    scoreaxs[scorei].spines['top'].set_visible(False)
    scoreaxs[scorei].text(0.98, 0.98, f'optimum: {wanted_scores[scorename]}', ha='right', va='top',
                          transform=scoreaxs[scorei].transAxes)

scoreaxs[-1].legend(stgtitles,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0),
                    ncol=len(stgtitles),
                    bbox_transform=scorefig.transFigure,
                    frameon=False)
plt.subplots_adjust(bottom=0.35)
scorefig.savefig(os.path.join(reportdir, f'{savingkw}_score_analysis.png'), dpi=300, bbox_inches='tight')
plt.close(scorefig)

# %% Consistency assessment

# Get results for all speeds in one single dataframe
multispeed = copy.deepcopy(ptlabels[stages[0]])
multispeed = multispeed.rename(
    columns={'clustlabel': 'clustlabel_' + stages[0], 'colourcode': 'colourcode_' + stages[0]})

for key, df in ptlabels.items():

    if key != stages[0]:

        # Add suffix for the stage to the two last columns
        df = df.rename(columns={'clustlabel': 'clustlabel_' + key, 'colourcode': 'colourcode_' + key})

        # Check whether there are pts missing speeds wrt the initial speed
        missingpts = multispeed['ptcode'][~multispeed['ptcode'].isin(df['ptcode'])]

        # if there are missing pts, create pad
        if len(missingpts) > 0:
            df = pd.concat([df, missingpts.to_frame()], ignore_index=True)
        multispeed = pd.merge(multispeed, df, on='ptcode')

# make pt code be the index column
multispeed = multispeed.set_index('ptcode')
multispeed = multispeed.drop(columns=multispeed.filter(regex='colourcode').columns)

# Create all possible combinations of speeds
speedcombs = list(itertools.combinations(multispeed.columns, 2))

# Calculate agreement for each combination
ami = {}
for comb in speedcombs:
    ami['_v_'.join(comb)] = adjusted_mutual_info_score(multispeed[comb[0]], multispeed[comb[1]])

#%% Print out results summary

print('Dimensionality reduction compression and reconstruction quality:\n')
print(dr_scores)

print('\n\nClustering quality:\n')
print(clust_scores)

print('\n\nPartition agreeement:\n')
print(pd.DataFrame(data=ami.values(), index=ami.keys(), columns=['AMI']))

# Return counts of participants in each cluster for each speed
for stage in clustdata.keys():
    print(f'\n{stage}:\n')
    print(clustdata[stage]['ptlabels']['clustlabel'].value_counts())

#%% Multispeed stat comparison

# Mastersheet with only selected pts and cluster labels
selmaster = master.loc[clustdata['multispeed']['ptlabels']['ptcode']]
selmaster['clustlabel'] = clustdata['multispeed']['ptlabels']['clustlabel'].values

stat_comparison['multispeed'] = {'demophysanthro': {}, '0D': {}, '1D': {}}

# Demographics, anthropometrics and physiological variables ignoring EE
stat_comparison['multispeed']['demophysanthro'] = comparison_0D_contvar_indgroups({key: master[key].loc[clustdata['multispeed']['ptlabels']['ptcode']].values for key in demoanthrophysvars_titles.keys() if 'RE' not in key},
                                                                                  clustdata[stage]['ptlabels']['clustlabel'],
                                                                                  f'{savingkw}_{stage}',
                                                                                  reportdir,
                                                                                  uniqclustcolours)

# Get EE data into a dataframe
redf = pd.DataFrame()
redf['EE'] = np.concatenate([selmaster[f'EE{speed}kg'].values for speed in speeds])
redf['speed'] = np.concatenate([[int(speed)] * len(pts) for speed in speeds])
redf['clustlabel'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, len(stages))
redf['ptcode'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, len(stages))

# Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
stat_comparison['multispeed']['demophysanthro']['EE'] = {}
stat_comparison['multispeed']['demophysanthro']['EE']['ANOVA2onerm'] = pg.mixed_anova(dv='EE',
                                                                             within='speed',
                                                                             subject='ptcode',
                                                                             between='clustlabel',
                                                                             data=redf,
                                                                             effsize='np2',
                                                                             correction=True)

# Get avge EE for each cluster at each speed
avgeedf = redf.groupby(['clustlabel', 'speed']).mean().reset_index()

# Run post-hoc tests
stat_comparison['multispeed']['demophysanthro']['EE']['posthocs'] = pg.pairwise_tests(dv='EE', within='speed',
                                                                                      subject='ptcode', between='clustlabel',
                                                                                      data=redf, padjust='bonf', effsize='cohen')

# RE with relative LT speeds. Create numpy array of nans
reltdf = np.empty((len(pts), 3))
reltdf[:] = np.nan

for speedi, speed in enumerate(speeds):
    for pti, pt in enumerate(pts):
        if selmaster[f'LT{speed}pctge'].loc[pt] > 70 and selmaster[f'LT{speed}pctge'].loc[pt] <= 80:
            reltdf[pti, 0] = selmaster[f'EE{speed}kmkg'].loc[pt]
        elif selmaster[f'LT{speed}pctge'].loc[pt] > 80 and selmaster[f'LT{speed}pctge'].loc[pt] <= 90:
            reltdf[pti, 1] = selmaster[f'EE{speed}kmkg'].loc[pt]
        elif selmaster[f'LT{speed}pctge'].loc[pt] > 90 and selmaster[f'LT{speed}pctge'].loc[pt] <= 100:
            reltdf[pti, 2] = selmaster[f'EE{speed}kmkg'].loc[pt]

# Convert to dataframe
reltdf = pd.DataFrame(data=reltdf.flatten(), columns=['EEkmkg'])
reltdf['LTpctge'] = np.concatenate([[int(speed)] * len(pts) for speed in [75, 85, 95]])
reltdf['clustlabel'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, 3)
reltdf['ptcode'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, 3)

# Make a table with number of females
femcount = []

# Make figures
demoanthrophysfig, demoanthrophysaxs = plt.subplots(3, 5, figsize=(11, 6))
demoanthrophysaxs = demoanthrophysaxs.flatten()

for vari, varname in enumerate([key for key in demoanthrophysvars_ylabels.keys() if key != 'RE']):

    if varname == 'Sex':

        fempctge = []
        sextable = []
        for clusti, uniqclust in enumerate(uniqclustlabels):

            # Get pts in that cluster
            clustmaster = master.loc[clustdata['multispeed']['ptlabels']['ptcode'].loc[clustdata['multispeed']['ptlabels']['clustlabel'] == uniqclust]]
            fempctge.append(len(clustmaster.loc[clustmaster['Sex'] == 'Female']) / len(clustmaster) * 100)

            # Get number of women and men in that cluster
            sextable.append([len(clustmaster.loc[clustmaster['Sex'] == 'Female']), len(clustmaster.loc[clustmaster['Sex'] == 'Male'])])

        # Add chi square test
        stat_comparison['multispeed']['demophysanthro'][varname] = {}
        stat_comparison['multispeed']['demophysanthro'][varname]['chi_test'] = {}
        stat_comparison['multispeed']['demophysanthro'][varname]['chi_test']['chi_sq'], \
        stat_comparison['multispeed']['demophysanthro'][varname]['chi_test']['p'], _, _ = stats.chi2_contingency(sextable)

        # Bar plot
        sns.barplot(ax=demoanthrophysaxs[vari], x=uniqclustlabels, y=fempctge, palette=uniqclustcolours)

        # Set xticks
        demoanthrophysaxs[vari].set_xticks([0, 1], ['C0', 'C1'])

    elif varname == 'RunningDaysAWeek':

        # Count plot
        sns.countplot(ax=demoanthrophysaxs[vari],
                      x=master[varname].loc[clustdata['multispeed']['ptlabels']['ptcode']],
                      hue=clustdata['multispeed']['ptlabels']['clustlabel'].values,
                      palette=uniqclustcolours)

        # Remove legend
        demoanthrophysaxs[vari].get_legend().remove()

        # Remove xlabel
        demoanthrophysaxs[vari].set_xlabel('')

    else:

            # Violin plot
            sns.violinplot(ax=demoanthrophysaxs[vari],
                           x=clustdata['multispeed']['ptlabels']['clustlabel'].values,
                           y=master[varname].loc[clustdata['multispeed']['ptlabels']['ptcode']].values,
                           palette=uniqclustcolours)

            # Xticks
            demoanthrophysaxs[vari].set_xticks([0, 1], ['C0', 'C1'])

    # Yticks for Time10Ks
    if varname == 'Time10Ks':

        # Convert to datetime and keep just mm:ss
        yticks = [str(datetime.timedelta(seconds=x)) for x in demoanthrophysaxs[vari].get_yticks()]
        yticks = [x[x.find(':') + 1:] for x in yticks]

        # Set new ticks
        demoanthrophysaxs[vari].set_yticklabels(yticks)

    # Ylabels
    demoanthrophysaxs[vari].set_ylabel(demoanthrophysvars_ylabels[varname])

    # Spines off
    demoanthrophysaxs[vari].spines['right'].set_visible(False)
    demoanthrophysaxs[vari].spines['top'].set_visible(False)

    # Title
    if varname in demoanthrophysvars_titles.keys():
        title = demoanthrophysvars_titles[varname]
    elif varname == 'Sex':
        title = 'Sex'

    if varname in stat_comparison['multispeed']['demophysanthro'].keys():

        # Get key which is not normality
        stat_test = [key for key in stat_comparison['multispeed']['demophysanthro'][varname].keys() if key != 'normality'][0]

        # Add asterisk to indicate significant differences
        if stat_comparison['multispeed']['demophysanthro'][varname][stat_test]['p'] < 0.05:
            demoanthrophysaxs[vari].set_title(f'{title} *')
        else:
            demoanthrophysaxs[vari].set_title(title)

    else:
        demoanthrophysaxs[vari].set_title(title)

plt.tight_layout()

# Save and close
demoanthrophysfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_demophysanthro.png'), dpi=300, bbox_inches='tight')
plt.close(demoanthrophysfig)

# RE variables
refig, reaxs = plt.subplots(1, 3, figsize=(11, 3))
reaxs = reaxs.flatten()

# For simplicity of calling
mixed_anova = stat_comparison['multispeed']['demophysanthro']['EE']['ANOVA2onerm']
posthocs = stat_comparison['multispeed']['demophysanthro']['EE']['posthocs']

# Plot results
for speedi, speed in enumerate(speeds):

    # Violin plot
    sns.violinplot(ax=reaxs[speedi],
                   x='clustlabel',
                   y='EE',
                   data=redf.loc[redf['speed'] == speed],
                   palette=uniqclustcolours,
                   legend=False)

    # Xticks
    reaxs[speedi].set_xticks([0, 1], ['C0', 'C1'])

    # Add stats in xlabel
    if mixed_anova['p-unc'].loc[mixed_anova['Source'] == 'clustlabel'].values < 0.05:
        t = np.round(
            posthocs['T'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[
                0], 2)
        d = np.round(
            posthocs['cohen'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[
                0],
            2)
        if posthocs['p-corr'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[
            0] < 0.001:
            reaxs[speedi].set_xlabel(f'C: t = {t}, p < 0.001, d = {d}', fontsize=10)
        else:
            p = np.round(posthocs['p-corr'].loc[
                             (posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[0],
                         3)
            reaxs[speedi].set_xlabel(f'C: t = {t}, p = {p}, d = {d}', fontsize=10)
    else:
        reaxs[speedi].set_xlabel(' ', fontsize=10)

    # y label
    if speedi == 0:
        reaxs[speedi].set_ylabel(demoanthrophysvars_ylabels['RE'])
    else:
        reaxs[speedi].set_ylabel('')

    # Get rid of spines
    reaxs[speedi].spines['right'].set_visible(False)
    reaxs[speedi].spines['top'].set_visible(False)

    # Add title
    reaxs[speedi].set_title(f'{speed} km/h')

# Same y limits
ylims = [ax.get_ylim() for ax in reaxs]
for ax in reaxs:
    ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

# Set suptitle
if mixed_anova['p-unc'].loc[mixed_anova['Source'] == 'clustlabel'].values < 0.001:
    strstats = f'C: F = {np.round(mixed_anova["F"].values[0], 2)}, p < 0.001'
else:
    strstats = f'C: F = {np.round(mixed_anova["F"].values[0], 2)}, p = {np.round(mixed_anova["p-unc"].values[0], 3)}'

if mixed_anova['p-unc'].loc[mixed_anova['Source'] == 'speed'].values < 0.001:
    strstats += f'; S: F = {np.round(mixed_anova["F"].values[1], 2)}, p < 0.001'
else:
    strstats += f'; S: F = {np.round(mixed_anova["F"].values[1], 2)}, p = {np.round(mixed_anova["p-unc"].values[1], 3)}'

if mixed_anova['p-unc'].loc[mixed_anova['Source'] == 'Interaction'].values < 0.001:
    strstats += f'; CxS: F = {np.round(mixed_anova["F"].values[2], 2)}, p < 0.001'
else:
    strstats += f'; CxS: F = {np.round(mixed_anova["F"].values[2], 2)}, p = {np.round(mixed_anova["p-unc"].values[2], 2)}'

# Set title
refig.suptitle(f'Running economy\n{strstats}')

# Save and close
plt.tight_layout()
refig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_RE_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
plt.close(refig)

# # RE variables
# refig, reaxs = plt.subplots(1, 1, figsize=(6, 2.5))
# sns.violinplot(ax=reaxs, x='speed', y='EE', hue='clustlabel', data=redf, palette=uniqclustcolours)
#
# # Append km/h to each xtick
# reaxs.set_xticklabels([f'{speed} km/h' for speed in speeds])
# reaxs.set_xlabel('')
# reaxs.set_ylabel(demoanthrophysvars_ylabels['RE'])
# reaxs.spines['right'].set_visible(False)
# reaxs.spines['top'].set_visible(False)
# reaxs.set_title('Running Economy')
#
# # Legend
# reaxs.legend(loc='lower center',
#              bbox_to_anchor=(0.5, 0),
#              ncol=2,
#              bbox_transform=refig.transFigure,
#              frameon=False)
# plt.subplots_adjust(bottom=0.25)
#
# # Get legend
# legend = reaxs.get_legend()
#
# # Change legend labels
# for clusti, uniqclust in enumerate(uniqclustlabels):
#     legend.get_texts()[clusti].set_text(f'C{clusti}')
#
# # Save and close
# plt.tight_layout()
# refig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_RE.png'), dpi=300, bbox_inches='tight')
# plt.close(refig)

#%% Kineamtic comparison

# 0D variables: 2 way ANOVA with one RM factor (speed) and one between factor (cluster)

for vari, varname in enumerate(discvars):
    discvarfig, discvaraxs = plt.subplots(1, 3, figsize=(11, 3))
    discvaraxs = discvaraxs.flatten()

    stat_comparison['multispeed']['0D'][varname] = {}

    # Get data
    df = pd.DataFrame()
    df[varname] = np.concatenate(clustdata['multispeed'][varname].T)
    df['speed'] = np.concatenate([[int(speeds[stgi])] * clustdata['multispeed'][varname].shape[0] for stgi, stage in enumerate(stages)])
    df['clustlabel'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, len(stages))
    df['ptcode'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, len(stages))

    # Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    mixed_anova = pg.mixed_anova(dv=varname,
                                 within='speed',
                                 subject='ptcode',
                                 between='clustlabel',
                                 data=df,
                                 effsize='np2',
                                 correction=True)
    stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm'] = mixed_anova

    # Run post-hoc tests
    posthocs = pg.pairwise_tests(dv=varname, within='speed', subject='ptcode', between='clustlabel', data=df, padjust='bonf', effsize='cohen')
    stat_comparison['multispeed']['0D'][varname]['posthocs'] = posthocs

    # Plot results
    for speedi, speed in enumerate(speeds):

        # Violin plot
        sns.violinplot(ax=discvaraxs[speedi],
                       x='clustlabel',
                       y=varname,
                       data=df.loc[df['speed'] == speed],
                       palette=uniqclustcolours,
                       legend=False)

        # Xticks
        discvaraxs[speedi].set_xticks([0, 1], ['C0', 'C1'])

        # Add stats in xlabel
        if mixed_anova['p-unc'].loc[mixed_anova['Source'] == 'clustlabel'].values < 0.05:
            t = np.round(
                posthocs['T'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[
                    0], 2)
            d = np.round(posthocs['cohen'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[0],
                         2)
            if posthocs['p-corr'].loc[(posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[
                0] < 0.001:
                discvaraxs[speedi].set_xlabel(f'C: t = {t}, p < 0.001, d = {d}', fontsize=11)
            else:
                p = np.round(posthocs['p-corr'].loc[
                         (posthocs['Contrast'] == 'speed * clustlabel') & (posthocs['speed'] == speed)].values[0], 3)
                discvaraxs[speedi].set_xlabel(f'C: t = {t}, p = {p}, d = {d}', fontsize=11)
        else:
            discvaraxs[speedi].set_xlabel(' ', fontsize=11)

        # y label
        if speedi == 0:
            discvaraxs[speedi].set_ylabel(kinematics_ylabels[varname])
        else:
            discvaraxs[speedi].set_ylabel('')

        # Get rid of spines
        discvaraxs[speedi].spines['right'].set_visible(False)
        discvaraxs[speedi].spines['top'].set_visible(False)

        # Add title
        discvaraxs[speedi].set_title(f'{speed} km/h')

    # Same y limits
    ylims = [ax.get_ylim() for ax in discvaraxs]
    for ax in discvaraxs:
        ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

    # Set suptitle
    if stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['p-unc'].loc[stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['Source'] == 'clustlabel'].values < 0.001:
        strstats = f'C: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[0], 2)}, p < 0.001'
    else:
        strstats = f'C: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[0], 2)}, p = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["p-unc"].values[0], 3)}'

    if stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['p-unc'].loc[stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['Source'] == 'speed'].values < 0.001:
        strstats += f'; S: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[1], 2)}, p < 0.001'
    else:
        strstats += f'; S: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[1], 2)}, p = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["p-unc"].values[1], 3)}'

    if stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['p-unc'].loc[stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['Source'] == 'Interaction'].values < 0.001:
        strstats += f'; CxS: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[2], 2)}, p < 0.001'
    else:
        strstats += f'; CxS: F = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["F"].values[2], 2)}, p = {np.round(stat_comparison["multispeed"]["0D"][varname]["ANOVA2onerm"]["p-unc"].values[2], 2)}'

    # Set title
    discvarfig.suptitle(f'{kinematics_titles[varname]}\n{strstats}')

    # Save and close
    plt.tight_layout()
    discvarfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{varname}_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
    plt.close(discvarfig)

# 1D variables: SPM 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
# 1D continuous variables using SPM
speedfig, speedaxs = plt.subplots(2, 3, figsize=(11, 4.5))
speedaxs = speedaxs.flatten()

# Get avge toe off for full speed and for each cluster based on duty factor for the plots
avgeto = {}
speedavgeto = []
for stage in stages:
    avgeto[stage] = []
    for clusti, uniqclust in enumerate(uniqclustlabels):
        clustidcs = np.where(clustdata[stage]['ptlabels']['clustlabel'] == uniqclust)[0]
        avgeto[stage].append(np.round(np.mean(clustdata[stage]['DUTYFACTOR'][clustidcs, :])*100, 1))

    speedavgeto.append(np.round(np.mean(clustdata[stage]['DUTYFACTOR'])*100, 1))

for vari, contvar in enumerate(contvars):

    stat_comparison['multispeed']['1D'][contvar] = {}

    # Initialise data holder
    group = []
    speed = []
    subject = []
    Y = []
    Ydiff = []

    # Prepare data for SPM and SPM mean and std plots
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    basegrid = fig.add_gridspec(2, 1)
    topgrid = basegrid[0].subgridspec(1, len(stages))
    bottomgrid = basegrid[1].subgridspec(1, len(stages) - 1)

    upperaxs = []
    loweraxs = []

    for stgi, stage in enumerate(stages):

        # Append group speed and subject
        group.append(clustdata['multispeed']['ptlabels']['clustlabel'].values)
        speed.append(np.ones(clustdata['multispeed'][varname].shape[0]) * stgi)
        subject.append(np.arange(len(clustdata['multispeed']['ptlabels']['clustlabel'].values)))
        Y.append(clustdata[stage][contvar])

        # Create axis
        upperaxs.append(fig.add_subplot(topgrid[0, stgi]))

        # Plot mean and std curves
        for labi, lab in enumerate(np.sort(np.unique(group[-1]))):

            # Top row: group by group for each speed
            spm1d.plot.plot_mean_sd(Y[-1][np.where(group[-1] == lab)[0], :],
                                    x=np.linspace(0, 100, Y[-1].shape[1]),
                                    linecolor=uniqclustcolours[labi], facecolor=uniqclustcolours[labi],
                                    ax=upperaxs[stgi])

        # Add vertical line at avge toe off (outside the previous loop so we can get the final ylimits)
        for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
            upperaxs[stgi].axvline(x=avgeto[stage][labi], color=uniqclustcolours[labi], linestyle=':')

        # xlabel. This will then get filled with stats if post-hocs were performed and if not it ensures they are all the same size
        upperaxs[stgi].set_xlabel(' ')

        # Speed figure
        spm1d.plot.plot_mean_sd(Y[-1], x=np.linspace(0, 100, Y[-1].shape[1]),
                                linecolor=speedcolours[stgi], facecolor=speedcolours[stgi],
                                ax=speedaxs[vari])

        if stgi > 0:

            # Calculate change from one speed to another
            Ydiff.append(Y[-1] - Y[-2])

            # Plot it by group
            loweraxs.append(fig.add_subplot(bottomgrid[0, stgi - 1]))

            # Add horizontal line at 0
            loweraxs[-1].axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)

            for uni in np.sort(np.unique(group)):
                spm1d.plot.plot_mean_sd(Ydiff[-1].T[:, group[stgi] == uni].T,
                                        x=np.linspace(0, 100, Ydiff[-1].shape[1]),
                                        linecolor=colours[uni], facecolor=colours[uni],
                                        ax=loweraxs[-1])

            # Add vertical line at avge toe off between speeds (outside the previous loop so we can get the final ylimits)
            for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                loweraxs[-1].axvline(x=np.mean([avgeto[stages[stgi - 1]][labi], avgeto[stage][labi]]),
                                     color=uniqclustcolours[labi], linestyle=':')

            # Set title
            loweraxs[-1].set_title(f'{speeds[stgi]} wrt {speeds[stgi - 1]} km/h')

            # xlabel. This will then get filled with stats if post-hocs were performed and if not it ensures they are all the same size
            loweraxs[-1].set_xlabel(' ')

            # Spines off
            loweraxs[-1].spines['right'].set_visible(False)
            loweraxs[-1].spines['top'].set_visible(False)

            # Legend
            loweraxs[-1].legend(['_nolegend_', 'C0', 'C1'],
                                loc='lower center',
                                bbox_to_anchor=(0.5, 0),
                                ncol=2,
                                bbox_transform=fig.transFigure,
                                frameon=False)
            plt.subplots_adjust(bottom=0.11)

        # Title
        upperaxs[stgi].set_title(stgtitles[stgi])

        # Get rid of spines
        upperaxs[stgi].spines['right'].set_visible(False)
        upperaxs[stgi].spines['top'].set_visible(False)

    # ylabels
    upperaxs[0].set_ylabel(kinematics_ylabels[contvar])
    loweraxs[0].set_ylabel('${\Delta}$')

    # Get ylims for all upperaxs
    ylims = [ax.get_ylim() for ax in upperaxs]

    # Set ylims for all upperaxs
    for ax in upperaxs:
        ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

    # Get ylims for all loweraxs
    ylims = [ax.get_ylim() for ax in loweraxs]

    # Set ylims for all loweraxs
    for ax in loweraxs:
        ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

    # add vertical lines at avge toe off to speed figure
    for spavgetoi, spavgeto in enumerate(speedavgeto):
        speedaxs[vari].axvline(x=spavgeto, color=speedcolours[spavgetoi], linestyle=':')

    # title and ylabel for speed figure
    speedaxs[vari].spines['right'].set_visible(False)
    speedaxs[vari].spines['top'].set_visible(False)
    speedaxs[vari].set_title(kinematics_titles[contvar])
    speedaxs[vari].set_ylabel(kinematics_ylabels[contvar])

    # Conduct SPM analysis
    spmlist = spm1d.stats.nonparam.anova2onerm(np.concatenate(Y, axis=0), np.concatenate(group), np.concatenate(speed),
                                               np.concatenate(subject))
    stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'] = spmlist.inference(alpha=0.05, iterations=1000)

    # Post hoc tests and figures
    stat_comparison['multispeed']['1D'][contvar]['posthocs'] = {}

    # Add patches to speed figure if there is an effect of speed
    if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][1].h0reject:

        # Scaler for sigcluster endpoints
        tscaler = speedaxs[vari].get_xlim()[1] / (Y[0].shape[1] - 1)

        # Add patches to speed figure
        for sigcluster in stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][1].clusters:
            ylim = speedaxs[vari].get_ylim()
            speedaxs[vari].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                   (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                   ylim[1] - ylim[0], color='grey', alpha=0.5, linestyle=''))

    # Add title to speed figure
    statsstr = f'F* = {np.round(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][1].zstar, 2)}'

    speedaxs[vari].set_title(f'{kinematics_titles[contvar]}\n{statsstr}')

    # Follow up with post-hoc tests if cluster effects are found
    if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].h0reject:

        stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'] = {}

        # For each speed
        for spi, (groupi, Yi) in enumerate(zip(group, Y)):

            stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'][stages[spi]] = {}

            # SnPM ttest
            snpm = spm1d.stats.nonparam.ttest2(Yi[groupi == 0, :], Yi[groupi == 1, :], )
            snpmi = snpm.inference(alpha=0.05 / len(Y), two_tailed=True, iterations=1000)

            # Add snpmi to dictionary
            stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'][stages[spi]]['snpm_ttest2'] = snpmi

            # Add stats to xlabel
            statsstr = f't* = {np.round(snpmi.zstar, 2)}'
            if len(snpmi.p) == 1:
                if snpmi.p[0] < 0.001:
                    statsstr += f', p < 0.001'
                else:
                    statsstr += f', p = {np.round(snpmi.p[0], 3)}'
            elif len(snpmi.p) > 1:
                statsstr += ', p = ['
                for i, p in enumerate(snpmi.p):
                    if i > 0:
                        statsstr += ', '
                    if p < 0.001:
                        statsstr += f'< 0.001'
                    else:
                        statsstr += f'{np.round(p, 3)}'
                statsstr += ']'

            upperaxs[spi].set_xlabel(statsstr, fontsize=10)

            # Plot
            plt.figure()
            snpmi.plot()
            snpmi.plot_threshold_label(fontsize=8)
            snpmi.plot_p_values(size=10)
            plt.gcf().suptitle(f'{contvar}_posthoc_{stages[spi]}')

            # Save figure and close it TODO. fix naming for saving
            plt.savefig(os.path.join(reportdir, f'{savingkw}_{contvar}_posthoc_{stages[spi]}.png'))
            plt.close(plt.gcf())

            # Add patches to upperaxs if significant diffs are found
            if snpmi.h0reject:

                # Scaler for sigcluster endpoints
                tscaler = upperaxs[spi].get_xlim()[1] / (Y[0].shape[1] - 1)

                for sigcluster in snpmi.clusters:
                    ylim = upperaxs[spi].get_ylim()
                    upperaxs[spi].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                          (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                          ylim[1] - ylim[0], color='grey', alpha=0.5,
                                                          linestyle=''))

def write_spm_stats_str(spmobj):

    statsstr = f'{np.round(spmobj.zstar, 2)}'
    if len(spmobj.p) == 1:
        if spmobj.p[0] < 0.001:
            statsstr += f', p < 0.001'
        else:
            statsstr += f', p = {np.round(spmobj.p[0], 3)}'
    elif len(spmobj.p) > 1:
        statsstr += ', p = ['
        for i, p in enumerate(spmobj.p):
            if i > 0:
                statsstr += ', '
            if p < 0.001:
                statsstr += f'< 0.001'
            else:
                statsstr += f'{np.round(p, 3)}'
        statsstr += ']'

    return statsstr

    # Interaction effect
    if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].h0reject:

        stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'] = {}

        # Calculate change in conditions
        for condi in range(len(stages) - 1):

            # SnPM ttest
            snpm = spm1d.stats.nonparam.ttest2(Ydiff[condi][group[condi] == 0, :], Ydiff[condi][group[condi] == 1, :], )
            snpmi = snpm.inference(alpha=0.05 / len(range(len(stages) - 1)), two_tailed=True, iterations=1000)

            # Add snpmi to dictionary
            stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'][f'{speeds[condi + 1]}_wrt_{speeds[condi]}'] = {}
            stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'][f'{speeds[condi + 1]}_wrt_{speeds[condi]}']['snpm_ttest2'] = snpmi

            # Add stats to xlabel
            statsstr = f't* = {np.round(snpmi.zstar, 2)}'
            if len(snpmi.p) == 1:
                if snpmi.p[0] < 0.001:
                    statsstr += f', p < 0.001'
                else:
                    statsstr += f', p = {np.round(snpmi.p[0], 3)}'
            elif len(snpmi.p) > 1:
                statsstr += ', p = ['
                for i, p in enumerate(snpmi.p):
                    if i > 0:
                        statsstr += ', '
                    if p < 0.001:
                        statsstr += f'< 0.001'
                    else:
                        statsstr += f'{np.round(p, 3)}'
                statsstr += ']'

            loweraxs[condi].set_xlabel(statsstr, fontsize=10)

            # Plot
            plt.figure()
            snpmi.plot()
            snpmi.plot_threshold_label(fontsize=8)
            snpmi.plot_p_values(size=10)
            plt.gcf().suptitle(f'{contvar}_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}')

            # Save figure and close it
            plt.savefig(os.path.join(reportdir, f'{savingkw}_{contvar}_interact_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}.png'))
            plt.close(plt.gcf())

            # Add patches to upperaxs if significant diffs are found
            if snpmi.h0reject:

                # Scaler for sigcluster endpoints
                tscaler = loweraxs[condi].get_xlim()[1] / (Ydiff[0].shape[1] - 1)

                for sigcluster in snpmi.clusters:
                    ylim = loweraxs[condi].get_ylim()
                    loweraxs[condi].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                            (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                            ylim[1] - ylim[0], color='grey', alpha=0.5,
                                                            linestyle=''))

    # Supttitle
    # Write cluster effect string
    statstr = f'C: F* = {np.round(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][0].zstar, 2)}'
    if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].h0reject:

        if len(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].p) == 1:
            if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].p[0] < 0.001:
                statstr += f', p < 0.001'
            else:
                statstr += f', p = {np.round(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][0].p[0], 3)}'
        elif len(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].p) > 1:
            statstr += ', p = ['
            for i, p in enumerate(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].p):
                if i > 0:
                    statstr += ', '
                if p < 0.001:
                    statstr += f'< 0.001'
                else:
                    statstr += f'{np.round(p, 3)}'
            statstr += ']'

    # Write interaction effect string
    statstr += f'; CxS: F* = {np.round(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][2].zstar, 2)}'
    if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].h0reject:
        if len(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].p) == 1:
            if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].p[0] < 0.001:
                statstr += f', p < 0.001'
            else:
                statstr += f', p = {np.round(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][2].p[0], 3)}'
        elif len(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].p) > 1:
            statstr += ', p = ['
            for i, p in enumerate(stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].p):
                if i > 0:
                    statstr += ', '
                if p < 0.001:
                    statstr += f'< 0.001'
                else:
                    statstr += f'{np.round(p, 3)}'
            statstr += ']'

    fig.suptitle(f'{kinematics_titles[contvar]}\n{statstr}')

    # Save and close
    plt.subplots_adjust(bottom=0.13)
    plt.tight_layout()
    fig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{contvar}_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Add legend to last plot in speed figure
speedaxs[-1].legend(speeds, frameon=False)

# Save speed figure
plt.tight_layout()
speedfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_kinematics_by_speed.png'), dpi=300, bbox_inches='tight')
plt.close(speedfig)

#%% Create visualisation files for Opensim

# Cluster models
create_clust_opensim_models(master.loc[clustdata['multispeed']['ptlabels']['ptcode']],
                            clustdata['multispeed']['ptlabels'],
                            modelpath=modelpath,
                            genscalsetuppath=genscalsetuppath,
                            reportdir=reportdir)

# Kinematics files
sf = 200
tlen = np.shape(clustdata['STG_02']['RCOM_2'])[-1]
tsec = np.arange(tlen) * 1/sf

# Loop through speeds ignoring multispeed
for stg in [x for x in stages if x != 'multispeed']:

    # Write file for each cluster
    for clust in np.sort(np.unique(clustdata['multispeed']['ptlabels']['clustlabel'].values)):

        # Create data holder
        dataholder = pd.DataFrame(data=np.zeros((tlen, len(osimvarnames))), columns=osimvarnames)
        dataholder['time'] = tsec
        for osimvar, corrvar in zip(osimvarnames[1:], corrvarnames[1:]):

            if corrvar != '_':
                # Add mean to dataholder
                dataholder[osimvar] = np.mean(clustdata[stg][corrvar][clustdata['multispeed']['ptlabels']['clustlabel'] == clust, :], axis=0)

            # Flip sign of knee flexion
            if 'knee' in osimvar:
                dataholder[osimvar] = -dataholder[osimvar]

        # Save to file
        dataholder.to_csv('tempdataholder.txt', sep='\t', index=False)

        # Read motion data back in
        with open('tempdataholder.txt') as f:
            motiondata = f.readlines()

        os.remove('tempdataholder.txt')

        # Write sto header file
        with open(stoheaderpath) as f:
            header = f.readlines()

        # Overwrite the rows and columns lines
        rowidx = [i for i, s in enumerate(header) if 'nRows' in s][0]
        header[rowidx] = 'nRows=' + str(len(tsec)) + ' '
        colidx = [i for i, s in enumerate(header) if 'nColumns' in s][0]
        header[colidx] = 'nColumns=' + str(len(osimvarnames)) + ' '

        stofile = header + motiondata

        # Write file
        stofilename = f'Clust_{clust}_{stg}.sto'
        with open(os.path.join(reportdir, stofilename), 'w') as f:
            for item in stofile:
                f.write("%s\n" % item)
