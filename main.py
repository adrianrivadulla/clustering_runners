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

- Use demoanthrophys_analysis from research-utils and replace here
"""


# %% Imports
import config
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.metrics.cluster import adjusted_mutual_info_score
from clustering_utils import *
import copy
from research_utils.pipelines import run_0D_ANOVA2onerm, run_demoanthrophys_two_groups_comparisons, run_SPM_ANOVA2onerm, run_single_condition_comparison
from utils.data_processing import load_mastersheet_and_kinematics
from utils.vis import add_suffix_to_titles


# %% Defaults

# Saving keyword
savingkw = 'Clust'

# Stages of interest
stages = ['STG_02', 'STG_03', 'STG_04']

# Speeds of interest
speeds = [9 + int(stage[-2:]) for stage in stages]

# recquality colours
recqualcolours = [['C0'], ['C6'], ['C3'], ['C0', 'C6', 'C3']]

# %% Data loading and wrangling TODO. Integrate with mastersheet data prep from fatigue
master, clustdata, vartracker, pts = load_mastersheet_and_kinematics(config.masterdatapath,
                                                                     config.datapath,
                                                                     stages,
                                                                     speeds,
                                                                     config.wantedvars,
                                                                     config.discvars,
                                                                     config.contvars)

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
    scores_by_k[stage] = {scorename: pd.DataFrame() for scorename in config.wanted_scores.keys()}

    # For figure creation and saving
    figinfo = {'reportdir': config.reportdir,
                'savingkw': savingkw,
                'colour': recqualcolours[stgi],
                'acceptable_errors': config.acceptable_errors,
                'kinematics_titles': config.kinematics_titles,
                'short_ylabels': config.short_ylabels,
                'recbot_ylims': config.recbot_ylims}

    # Prepare data, perform PCA and run reconstruction analysis
    pcaed, dr_scores[stage] = pca_dimensionality_reduction(clustdata[stage], vartracker[stage], stage, figinfo)

    # Hierarchical clustering analysis
    # HCA = HierarchClusteringAnalysisTool(pcaed, labels=pts)
    HCA = HierarchClusteringAnalysisTool2(pcaed, labels=pts)

    # Rename datalabels column in HCA.colourid as ptcode to make it meaningful for current analysis
    HCA.colourid = HCA.colourid.rename(columns={'datalabels': 'ptcode'})

    # Store results
    for scorename in config.wanted_scores.keys():
        scores_by_k[stage][scorename] = HCA.scores[scorename]
    clust_scores.loc[stage] = HCA.scores.loc[HCA.n_clusters]
    dendros.append(HCA.dendro)

    # Decorate plots
    if len(dendros) == 1:
        # Add bottom leaves to dendrogram
        append_bottom_leaves_dendrogram(HCA.dendroax)

        # Set title adding the Silhouette score
        HCA.dendroax.set_title(f'{config.stg_titles[stgi]} '
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
        trans_analysis.dendroax.set_title(f'{config.stg_titles[stgi]} '
                                          f'(Silh = {np.round(HCA.finalscore_table.loc["Silhouette"].values[0], 3)}, '
                                          f'AMI = {np.round(trans_analysis.ami, 3)})')

        # Write legend
        add_dendro_legend(HCA.dendroax, trans_analysis.curr_colourid)

        # Store labels and ptlabels
        ptlabels[stage] = trans_analysis.curr_colourid

        # Store colourid in clustdata
        clustdata[stage]['ptlabels'] = trans_analysis.curr_colourid

    # Save and close figures
    HCA.scorefig.savefig(os.path.join(config.reportdir, f'{savingkw}_{stage}_HCA_scores.png'), dpi=300, bbox_inches='tight')
    HCA.dendrofig.savefig(os.path.join(config.reportdir, f'{savingkw}_{stage}_finaldendro.png'), dpi=300, bbox_inches='tight')
    plt.close(HCA.scorefig)
    plt.close(HCA.dendrofig)
    plt.close(HCA.dendrofig)

    # Get unique clustlabels and corresponding colour
    uniqclustlabels = natsort.natsorted(np.unique(clustdata[stage]['ptlabels']['clustlabel']))
    uniqclustcolours = [clustdata[stage]['ptlabels']['colourcode'].loc[
                            clustdata[stage]['ptlabels']['clustlabel'] == x].iloc[0] for x in uniqclustlabels]


    #%% Stat comparison of single speed clustering
    if stage != 'multispeed':

        # TODO. Revise why you are getting different stats in 0D
        stat_comparison[stage], normfigs, spmfigs, kinfig = run_single_condition_comparison(clustdata[stage],
                                                                    config.discvars,
                                                                    config.contvars,
                                                                 titles=config.kinematics_titles,
                                                                 ylabels=config.kinematics_ylabels,
                                                                 vline_var=clustdata[stage]['DUTYFACTOR'])

        a = 5

        # Save figures
        for var, fig in normfigs.items():
            fig.savefig(os.path.join(config.reportdir, f'{savingkw}_{stage}_{var}_QQplot.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        for var, fig in spmfigs.items():
            fig.savefig(os.path.join(config.reportdir, f'{savingkw}_{stage}_{var}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        kinfig.suptitle(f'{config.stg_titles[stgi]} kinematics')
        kinfig.savefig(os.path.join(config.reportdir, f'{savingkw}_{stage}_kinematics.png'), dpi=300, bbox_inches='tight')
        plt.close(kinfig)

# Save multispeed ptlabels to file
clustdata['multispeed']['ptlabels'].to_csv(os.path.join(config.reportdir, f'{savingkw}_multispeed_ptlabels.csv'), index=False)

# %% Score analysis

# Create figure
scorefig, scoreaxs = plt.subplots(1, len(config.wanted_scores), figsize=(11, 2))

for scorei, scorename in enumerate(config.wanted_scores.keys()):
    for stgi, stage in enumerate(clustdata.keys()):
        if stage != 'multispeed':
            scoreaxs[scorei].plot(scores_by_k[stage][scorename].index, scores_by_k[stage][scorename].values, '-o',
                                  color=config.speedcolours[stgi])
        else:
            scoreaxs[scorei].plot(scores_by_k[stage][scorename].index, scores_by_k[stage][scorename].values, '-o',
                                  color='k')

    scoreaxs[scorei].set_xlabel('k')
    scoreaxs[scorei].set_title(scorename)
    scoreaxs[scorei].text(0.98, 0.98, f'optimum: {config.wanted_scores[scorename]}', ha='right', va='top',
                          transform=scoreaxs[scorei].transAxes)

scoreaxs[-1].legend(config.stg_titles,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0),
                    ncol=len(config.stg_titles),
                    bbox_transform=scorefig.transFigure,
                    frameon=False)
plt.subplots_adjust(bottom=0.35)
scorefig.savefig(os.path.join(config.reportdir, f'{savingkw}_score_analysis.png'), dpi=300, bbox_inches='tight')
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

if len(uniqclustlabels) != 2:
    raise ValueError('To replicate the analysis in the paper, please select 2 clusters in the multispeed analysis.')

req_variables = (
    [key for key in config.demoanthrophysvars_titles if key != "RE"]
    + [f"EE{speed}kg" for speed in speeds]
    + ["clustlabel"]
)

stat_comparison["demoanthrophys"], demoanthrophysfig, normfigs, refig = run_demoanthrophys_two_groups_comparisons(
    selmaster[req_variables],
    grouping_var="clustlabel",
    re_speeds=speeds,
    titles=config.demoanthrophysvars_titles,
    ylabels=config.demoanthrophysvars_ylabels,
    group_names=[f'C{int(x)}' for x in range(len(uniqclustlabels))],
    group_colours=uniqclustcolours,
)

# Save and close
demoanthrophysfig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_demophysanthro.png'), dpi=300,
                          bbox_inches='tight')
plt.close(demoanthrophysfig)
for var, fig in normfigs.items():
    fig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_{var}QQplot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

refig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_RE_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
plt.close(refig)

# %% Kinematics comparison

# Get design factors for stats TODO. this is what you'll have to do for the function in research utils to work.
designfactors = {}
designfactors['group'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, len(stages))
designfactors['rm'] = np.concatenate(
    [[int(speeds[stgi])] * clustdata['multispeed'][config.discvars[0]].shape[0] for stgi, stage in enumerate(stages)])
designfactors['rm'] = designfactors['rm'].astype(str)
designfactors['ptids'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, len(stages))

# Run ANOVA2onrm for disc vars
discdata = {varname: np.concatenate(clustdata['multispeed'][varname].T) for varname in config.discvars}
figs, stat_comparison["multispeed"]["0D"] = run_0D_ANOVA2onerm(
    discdata,
    designfactors,
    between_factor="clustlabel",
    within_factor="speed",
    titles=config.kinematics_titles,
    ylabels=config.kinematics_ylabels,
    group_names=['C0', 'C1'],
    group_colours=uniqclustcolours,
    rm_names=['11', '12', '13'],
    rm_colours=config.speedcolours,
    between_label="C",
    within_label="S",
    within_vis=False
)

# Save figures adding km/h to the title
for var, fig in figs.items():
    add_suffix_to_titles(fig, ' km/h')
    fig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_{var}_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Run ANOVA2onrm for cont vars
contdata = {
    var: np.concatenate([values[var] for key, values in clustdata.items() if key != "multispeed"], axis=0)
    for var in config.contvars
}

# contdata = {
#     var: np.concatenate([values[var] for key, values in clustdata.items() if key != "multispeed"], axis=0)
#     for var in ['RCOM_2']
# }

stat_comparison["multispeed"]["1D"], kinspmfigs, kinfigs, kinrmfig = run_SPM_ANOVA2onerm(
    contdata,
    designfactors,
    spm_random_seed=45,
    titles=config.kinematics_titles,
    ylabels=config.kinematics_ylabels,
    group_names=['C0', 'C1'],
    group_colours=uniqclustcolours,
    rm_names=['11', '12', '13'],
    rm_colours=config.speedcolours,
    between_label="C",
    within_label="S",
    rm_fig_rows=2,
    rm_fig_cols=3,
    vline_var=discdata["DUTYFACTOR"],
    rm_spm_patches="anova2onerm",
)

# Save figures
for var, fig in kinfigs.items():
    # Add km/h to the title
    add_suffix_to_titles(fig, ' km/h')
    fig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_{var}_ANOVA2onerm.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)

for var, fig in kinspmfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_multispeed_{var}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

kinrmfig.savefig(os.path.join(config.reportdir, f'{savingkw}_multispeed_kinematics_by_speed.png'), dpi=300,
                 bbox_inches='tight')
plt.close(kinrmfig)
