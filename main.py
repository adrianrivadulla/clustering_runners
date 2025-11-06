"""
===============================
Replicate Rivadulla et al. (2024) results
===============================

This script replicates the results from the paper Rivadulla et al. (2024).

The data loading and wrangling part loads the kinematics data and the master datasheet,
and prepares the data for the clustering analysis.

The clustering analysis part performs PCA and hierarchical clustering analysis for each stage (speed),
and then performs a multispeed clustering analysis. This part is interactive, and the user can see the
dendrogram and internal validity scores for each stage to decide the number of clusters to use. Additionally,
users will have to match up the colouring of the clusters between partitions at different stages to make the
visualisations coherent. The results are saved in the report directory.

In addition to the clustering analysis, the script performs a consistency assessment between the clustering partitions
using the adjusted mutual information score.

Finally, the script performs a statistical comparison of the kinematics variables between the clusters for each stage,
and for the multispeed condition. Additionally, the demographics, anthropometrics, physiological variables and
running economy are compared between the clusters for the multispeed condition.

The results are saved in the report directory. Users shouldn't need to modify the script but can do so in the Defalts
section if minor changes to variables of interest, thresholds, etc. are needed.

Author: Adrian R Rivadulla


TODO.

"""


# %% Imports
import os
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.colors as mcolors
from sklearn.metrics.cluster import adjusted_mutual_info_score
from clustering_utils import *
from data_wrangling_utils import *
import copy


# %% Defaults
# Project dir wherever this script is
projectdir = os.path.dirname(os.path.realpath(__file__))

# Data dir
datadir = os.path.join(projectdir, 'data')

# Report dir for saving figures
reportdir = os.path.join(projectdir, 'report')

# Create it if it doesn't exist
if not os.path.exists(reportdir):
    os.makedirs(reportdir)
else:
    print(f'Report directory found. Existing files will be overwritten.')

# Master datasheet
masterdatapath = os.path.join(datadir, 'MasterDataSheet.xlsx')

# kinematics data
kindatapath = os.path.join(datadir, 'AllCurves_ptavgs.npy')

# Matplotlib style
matplotlib.use('Qt5Agg')
matplotlib.style.use('default')

# Update default rcParams
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Saving keyword
savingkw = 'Clust'

# Stages of interest
stages = ['STG_02', 'STG_03', 'STG_04']

# Speeds of interest
speeds = [9 + int(stage[-2:]) for stage in stages]

# Wanted variables for clustering
discvars = ['STRIDEFREQ', 'DUTYFACTOR']
contvars = ['RCOM_2', 'RTRUNK2PELVIS_0', 'RPELV_ANG_0', 'RHIP_0', 'RKNEE_0', 'RANK_0']
wantedvars = discvars + contvars

# For figure decoration
stg_titles = ['11 km/h', '12 km/h', '13 km/h', 'Multispeed']
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

# Speed linestyles
speedcolours = ['C0', 'C6', 'C3']

# recquality colours
recqualcolours = [['C0'], ['C6'], ['C3'], ['C0', 'C6', 'C3']]

# %% Data loading and wrangling
master, clustdata, vartracker, pts = load_mastersheet_and_kinematics(masterdatapath,
                                                                     kindatapath,
                                                                     stages,
                                                                     speeds,
                                                                     wantedvars,
                                                                     discvars,
                                                                     contvars)

# %% Perform PCA and clustering analysis for each stage

# Preallocate data holders
dr_scores = {}
clust_scores = pd.DataFrame(columns=['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Gap'])
dendros = []
ptlabels = {}
scores_by_k = {}
stat_comparison = {}

for stgi, stage in enumerate(clustdata.keys()):

    # Preallocate scores for each stage
    scores_by_k[stage] = {scorename: pd.DataFrame() for scorename in wanted_scores.keys()}

    # For figure creation and saving
    figinfo = {'reportdir': reportdir,
                'savingkw': savingkw,
                'colour': recqualcolours[stgi],
                'acceptable_errors': acceptable_errors,
                'kinematics_titles': kinematics_titles,
                'short_ylabels': short_ylabels,
                'recbot_ylims': recbot_ylims}

    # Prepare data, perform PCA and run reconstruction analysis
    pcaed, dr_scores[stage] = pca_dimensionality_reduction(clustdata[stage], vartracker[stage], stage, figinfo)

    # Hierarchical clustering analysis
    HCA = HierarchClusteringAnalysisTool(pcaed, labels=pts)

    # Rename datalabels column in HCA.colourid as ptcode to make it meaningful for current analysis
    HCA.colourid = HCA.colourid.rename(columns={'datalabels': 'ptcode'})

    # Store results
    for scorename in wanted_scores.keys():
        scores_by_k[stage][scorename] = HCA.scores[scorename]
    clust_scores.loc[stage] = HCA.scores.loc[HCA.n_clusters]
    dendros.append(HCA.dendro)

    # Decorate plots
    if len(dendros) == 1:
        # Add bottom leaves to dendrogram
        append_bottom_leaves_dendrogram(HCA.dendroax)

        # Set title adding the Silhouette score
        HCA.dendroax.set_title(f'{stg_titles[stgi]} '
                           f'(Silh = {np.round(HCA.finalscore_table.loc["Silhouette"].values[0], 3)})')

    if len(dendros) == 1:

        # Add legend to dendrogram including count of pts in each cluster
        add_dendro_legend(HCA.dendroax, HCA.colourid)

        # Store results
        ptlabels[stage] = HCA.colourid
        clustdata[stage]['ptlabels'] = HCA.colourid

    # Make colours consistent with previous iteration
    elif len(dendros) > 1:

        append_bottom_leaves_dendrogram(HCA.dendroax, labelcolour=ptlabels[stages[stgi - 1]])

        # Transition analysis from previous to current partition
        trans_analysis = TransitionAnalysis(ptlabels[stages[stgi - 1]], HCA.colourid, HCA.dendrofig, HCA.dendroax)
        trans_analysis.dendroax.set_title(f'{stg_titles[stgi]} '
                                          f'(Silh = {np.round(HCA.finalscore_table.loc["Silhouette"].values[0], 3)}, '
                                          f'AMI = {np.round(trans_analysis.ami, 3)})')

        # Write legend
        add_dendro_legend(HCA.dendroax, trans_analysis.curr_colourid)

        # Store labels and ptlabels
        ptlabels[stage] = trans_analysis.curr_colourid

        # Store colourid in clustdata
        clustdata[stage]['ptlabels'] = trans_analysis.curr_colourid

    # Save and close figures
    HCA.scorefig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_HCA_scores.png'), dpi=300, bbox_inches='tight')
    HCA.dendrofig.savefig(os.path.join(reportdir, f'{savingkw}_{stage}_finaldendro.png'), dpi=300, bbox_inches='tight')
    plt.close(HCA.scorefig)
    plt.close(HCA.dendrofig)
    plt.close(HCA.dendrofig)

    # Get unique clustlabels and corresponding colour
    uniqclustlabels = natsort.natsorted(np.unique(clustdata[stage]['ptlabels']['clustlabel']))
    uniqclustcolours = [clustdata[stage]['ptlabels']['colourcode'].loc[
                            clustdata[stage]['ptlabels']['clustlabel'] == x].iloc[0] for x in uniqclustlabels]


    #%% Stat comparison of single speed clustering
    if stage != 'multispeed':

        # For figure creation and saving
        figinfo = {'reportdir': reportdir,
                   'savingkw': savingkw,
                   'study_title': stg_titles[stgi],
                   'kinematics_titles': kinematics_titles,
                   'kinematics_ylabels': kinematics_ylabels}

        stat_comparison[stage] = single_speed_kinematics_comparison(clustdata[stage],
                                                                    discvars,
                                                                    contvars,
                                                                    figinfo)

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
    scoreaxs[scorei].text(0.98, 0.98, f'optimum: {wanted_scores[scorename]}', ha='right', va='top',
                          transform=scoreaxs[scorei].transAxes)

scoreaxs[-1].legend(stg_titles,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0),
                    ncol=len(stg_titles),
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

# %% Print out results summary

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

# %% Multispeed stat comparison

# Mastersheet with only selected pts and cluster labels
selmaster = master.loc[clustdata['multispeed']['ptlabels']['ptcode']]
selmaster['clustlabel'] = clustdata['multispeed']['ptlabels']['clustlabel'].values

stat_comparison['multispeed'] = {'demophysanthro': {}, '0D': {}, '1D': {}}

#%% Demographics, anthropometrics and physiological variables ignoring EE
stat_comparison['multispeed']['demophysanthro'] = comparison_0D_contvar_indgroups(
    {key: master[key].loc[clustdata['multispeed']['ptlabels']['ptcode']].values for key in
     demoanthrophysvars_titles.keys() if 'RE' not in key},
    clustdata['multispeed']['ptlabels']['clustlabel'],
    f'{savingkw}_{stage}',
    reportdir,
    uniqclustcolours)

# Make figures
demoanthrophysfig, demoanthrophysaxs = plt.subplots(3, 5, figsize=(11, 6))
demoanthrophysaxs = demoanthrophysaxs.flatten()

for vari, varname in enumerate([key for key in demoanthrophysvars_ylabels.keys() if key != 'RE']):

    if varname == 'Sex':

        fempctge = []
        sextable = []

        for clusti, uniqclust in enumerate(uniqclustlabels):

            # Get pts in that cluster
            clustmaster = master.loc[clustdata['multispeed']['ptlabels']['ptcode'].loc[
                clustdata['multispeed']['ptlabels']['clustlabel'] == uniqclust]]
            fempctge.append(len(clustmaster.loc[clustmaster['Sex'] == 'Female']) / len(clustmaster) * 100)

            # Get number of females and males in that cluster
            sextable.append([len(clustmaster.loc[clustmaster['Sex'] == 'Female']),
                             len(clustmaster.loc[clustmaster['Sex'] == 'Male'])])

        # Add chi square test
        stat_comparison['multispeed']['demophysanthro'][varname] = {}
        stat_comparison['multispeed']['demophysanthro'][varname]['chi_test'] = {}
        stat_comparison['multispeed']['demophysanthro'][varname]['chi_test']['chi_sq'], \
            stat_comparison['multispeed']['demophysanthro'][varname]['chi_test']['p'], _, _ = stats.chi2_contingency(
            sextable)

        # Bar plot
        sns.barplot(ax=demoanthrophysaxs[vari], x=uniqclustlabels, y=fempctge, palette=uniqclustcolours)

        # Set xticks
        demoanthrophysaxs[vari].set_xticks(demoanthrophysaxs[vari].get_xticks(),
                                           [f'C{int(x)}' for x in demoanthrophysaxs[vari].get_xticks()])

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
        demoanthrophysaxs[vari].set_xticks(demoanthrophysaxs[vari].get_xticks(),
                                           [f'C{int(x)}' for x in demoanthrophysaxs[vari].get_xticks()])

    # Yticks for Time10Ks
    if varname == 'Time10Ks':
        # Convert to datetime and keep just mm:ss
        yticks = [str(datetime.timedelta(seconds=x)) for x in demoanthrophysaxs[vari].get_yticks()]
        yticks = [x[x.find(':') + 1:] for x in yticks]

        # Set new ticks
        demoanthrophysaxs[vari].set_yticklabels(yticks)

    # Ylabels
    demoanthrophysaxs[vari].set_ylabel(demoanthrophysvars_ylabels[varname])

    # Title
    if varname in demoanthrophysvars_titles.keys():
        title = demoanthrophysvars_titles[varname]
    elif varname == 'Sex':
        title = 'Sex'

    if varname in stat_comparison['multispeed']['demophysanthro'].keys():

        # Get key which is not normality
        stat_test = \
        [key for key in stat_comparison['multispeed']['demophysanthro'][varname].keys() if key != 'normality'][0]

        # Add asterisk to indicate significant differences
        if stat_comparison['multispeed']['demophysanthro'][varname][stat_test]['p'] < 0.05:
            demoanthrophysaxs[vari].set_title(f'{title} *')
        else:
            demoanthrophysaxs[vari].set_title(title)

    else:
        demoanthrophysaxs[vari].set_title(title)

plt.tight_layout()

# Save and close
demoanthrophysfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_demophysanthro.png'), dpi=300,
                          bbox_inches='tight')
plt.close(demoanthrophysfig)

#%% RE variables

# Get EE data into a dataframe
redf = pd.DataFrame()
redf['EE'] = np.concatenate([selmaster[f'EE{speed}kg'].values for speed in speeds])
redf['speed'] = np.concatenate([[int(speed)] * len(pts) for speed in speeds])
redf['clustlabel'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, len(stages))
redf['ptcode'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, len(stages))

stat_comparison['multispeed']['demophysanthro']['EE'] = anova2onerm_0d_and_posthocs(redf,
                                                                                    dv='EE',
                                                                                    within='speed',
                                                                                    between='clustlabel',
                                                                                    subject='ptcode')

refig, reaxs = plt.subplots(1, 3, figsize=(11, 3))
reaxs = reaxs.flatten()

# Plot results
for speedi, speed in enumerate(speeds):

    # Violin plot
    sns.violinplot(ax=reaxs[speedi],
                   x='clustlabel',
                   y='EE',
                   data=redf.loc[redf['speed'] == speed],
                   palette=uniqclustcolours,
                   hue='clustlabel',
                   legend=False)

    # Add C at the start of each xtick
    reaxs[speedi].set_xticks(reaxs[speedi].get_xticks(),
                             [f'C{int(x)}' for x in reaxs[speedi].get_xticks()])

    # Add stats in xlabel
    if (stat_comparison['multispeed']['demophysanthro']['EE']['ANOVA2onerm']['p-unc'].loc[
        stat_comparison['multispeed']['demophysanthro']['EE']['ANOVA2onerm']['Source'] == 'clustlabel'].values
            < 0.05):

        statsstr = write_0Dposthoc_statstr(stat_comparison['multispeed']['demophysanthro']['EE']['posthocs'],
                                           'speed * clustlabel',
                                           'speed',
                                           speed)
        reaxs[speedi].set_xlabel(f'C: {statsstr}', fontsize=10)

    else:
        reaxs[speedi].set_xlabel(' ', fontsize=10)

    # y label
    if speedi == 0:
        reaxs[speedi].set_ylabel(demoanthrophysvars_ylabels['RE'])
    else:
        reaxs[speedi].set_ylabel('')

    # Add title
    reaxs[speedi].set_title(f'{speed} km/h')

# Same y limits
ylims = [ax.get_ylim() for ax in reaxs]
for ax in reaxs:
    ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

# Set suptitle
statsstr = write_0DmixedANOVA_statstr(stat_comparison['multispeed']['demophysanthro']['EE']['ANOVA2onerm'],
                                      between='clustlabel',
                                      within='speed',
                                      betweenlabel='C',
                                      withinlabel='S')

# Set title
refig.suptitle(f'Running economy\n{statsstr}')

# Save and close
plt.tight_layout()
refig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_RE_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
plt.close(refig)

# %% Kinematics comparison
figinfo = {'reportdir': reportdir,
           'savingkw': savingkw,
           'speedcolours': speedcolours,
           'kinematics_titles': kinematics_titles,
           'kinematics_ylabels': kinematics_ylabels,
           'stg_titles': stg_titles}

stat_comparison['multispeed'] = multispeed_kinematics_comparison(clustdata,
                                                                 stages,
                                                                 speeds,
                                                                 discvars,
                                                                 contvars,
                                                                 figinfo)
