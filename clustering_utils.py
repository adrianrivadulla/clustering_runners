# -*- coding: utf-8 -*-
"""

This module contains all the functions developed for the clustering study

Created on Tue February 2023

@author: arr43
"""

# %% Imports
import os
import natsort
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram
import sys
projectdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(projectdir, 'additional_modules'))
from gapstat import gapstat
import tkinter as tk
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
import mplcursors
from matplotlib import colors
from scipy import stats
# import statsmodels.api as sm
from scikit_posthocs import posthoc_ttest, posthoc_dunn
import pingouin as pg
import statsmodels.api as sm
import spm1d
# import opensim as osim #TODO create opensim_utils module
import matplotlib.colors as mcolors


# %% Utils


class HierarchClusteringAnalysisTool:

# TODO. Make GUI fit in any screen size

    """
    A GUI to choose the number of clusters for hierarchical clustering based on internal validity scores and visualisation.
    """

    def __init__(self, data, **kwargs):

        self.data = data
        self.kwargs = kwargs

        # kwargs
        figtitle = self.kwargs.get('figtitle', 'Hierarchical Clustering Analysis')
        datalabels = self.kwargs.get('labels', None)

        # Instantiate model
        hrcal_model = AgglomerativeClustering()

        # Choose number of clusters GUI
        # Display colours and let user choose the ralabelling
        # Create GUI
        master = tk.Tk()

        # Get the screen width and height
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        # Set the window dimensions to cover the full screen
        master.geometry(f"{int(screen_width)}x{int(screen_height)}")
        master.configure(bg='white')
        master.title('Choose number of clusters')
        master.attributes("-topmost", True)
        master.focus_force()

        # Figure with scores to decide n of clusters
        fig_width = min(screen_width/100*0.5, 5)  # Cap the width at 10 inches
        fig_height = min(screen_height/100*0.9, 6)  # Cap the height at 6 inches
        self.scorefig = plt.figure(figsize=(fig_width, fig_height))

        # Silhouette scores
        ax = plt.subplot(4, 2, 1)
        visualiser = KElbowVisualizer(hrcal_model, k=(2, 11), metric='silhouette', timings=False, locate_elbow=False)
        visualiser.fit(data)
        visualiser.show()
        plt.title('')
        plt.ylabel('Silhouette')
        plt.xlabel('')
        plt.grid()
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.text(0.98, 0.98, 'optimum: 1', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores = pd.DataFrame(visualiser.k_scores_, index=visualiser.k_values_, columns=['Silhouette'])

        # Calinski_harabasz
        ax = plt.subplot(4, 2, 3)
        visualiser = KElbowVisualizer(hrcal_model, k=(2, 11), metric='calinski_harabasz', timings=False, locate_elbow=False)
        visualiser.fit(data)
        visualiser.show()
        plt.title('')
        plt.ylabel('Calinski-Harabasz \nIndex')
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.xlabel('')
        plt.grid()
        plt.text(0.98, 0.98, 'optimum: largest', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores['Calinski-Harabasz'] = visualiser.k_scores_

        # Davies_Bouldin
        ax = plt.subplot(4, 2, 5)
        scores = [metrics.davies_bouldin_score(data, AgglomerativeClustering(n_clusters=k).fit_predict(data)) for k
                  in range(2, 11)]
        plt.plot(range(2, 11), scores, linestyle='-', marker='D', color='b')
        plt.title('')
        plt.xlabel('K')
        plt.ylabel('Davies-Bouldin \nIndex')
        plt.grid()
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.text(0.98, 0.98, 'optimum: 0', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores['Davies-Bouldin'] = scores

        # Gap statistic
        _, _, gapstats = gapstat(data, hrcal_model, max_k=10, calcStats=True)
        scores = gapstats['data'][:, list(gapstats['columns']).index('Gap')]
        ax = plt.subplot(4, 2, 7)
        plt.plot(gapstats['index'][1:-1], scores[1:-1], linestyle='-', marker='D', color='b')
        plt.ylabel('Gap')
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.grid()
        plt.text(0.98, 0.98, 'optimum: largest', ha='right', va='top', transform=ax.transAxes)
        plt.title('Internal validity scores')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Store scores
        self.scores['Gap'] = scores[1:-1]

        # Silhouette plots
        plotis = [2, 4, 6, 8]
        ks = [2, 3, 4, 5]

        self.Silhouette_samples = {}

        for ki, ploti in zip(ks, plotis):

            ax = plt.subplot(4, 2, ploti)
            if ploti == 2:
                ax.set_title('Silhouette analysis')
            # Set top and right spines invisible
            ax.spines[['top', 'right']].set_visible(False)

            templabels = AgglomerativeClustering(n_clusters=ki).fit_predict(data)

            # Get avge and sample Silhouette scores
            silh_avge = metrics.silhouette_score(data, templabels)
            silh_samp = metrics.silhouette_samples(data, templabels)

            # Store silhouette scores
            self.Silhouette_samples[ki] = silh_samp

            y_lower = 10

            for i in range(ki):

                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = silh_samp[templabels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = 'C' + str(i + 1)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers in the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # Make it pretty
            if ax.get_xlim()[0] > -0.1:
                ax.set_xlim(-0.1, ax.get_xlim()[1])
            ax.set_ylabel("Label")

            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silh_avge, color="red", linestyle="--")

        # X label silhouette score columns
        ax.set_xlabel("Silhouette scores")
        self.scorefig.suptitle(figtitle)
        plt.tight_layout()

        # Embed figures in tkinter
        # Internal validity scores and Silhouette analysis
        lefttopframe = tk.Frame(master)
        lefttopframe.grid(row=0, column=0)
        canvas = FigureCanvasTkAgg(self.scorefig, master=lefttopframe)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # String with instructions
        # number of cluster choice
        leftbottomframe = tk.Frame(master)
        leftbottomframe.grid(row=1, column=0)
        instructs = ('Set number of clusters in the drop menu on the right. '
                     'Silhouette, Calinski-Harabasz, Davies-Bouldin considered in the paper. '
                     'Priority given to Silhouette. '
                     'See paper for more details.')
        string = tk.Label(leftbottomframe, textvariable=tk.StringVar(leftbottomframe, instructs))
        string.grid(row=0, column=0)

        # Blank dendrogram
        righttopframe = tk.Frame()
        righttopframe.grid(row=0, column=1)
        self.dendrofig = plt.figure(figsize=(fig_width, fig_height))
        clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data)
        _, linkage = plot_dendrogram(clustdataholder,
                                     datalabels,
                                     color_threshold=0,
                                     orientation='left')
        plt.title('Dendrogram')

        canvas2 = FigureCanvasTkAgg(self.dendrofig, master=righttopframe)
        canvas2.draw()
        canvas2.get_tk_widget().pack()

        # number of cluster choice
        midrightframe = tk.Frame(master)
        midrightframe.grid(row=1, column=1)
        string = tk.Label(midrightframe, textvariable=tk.StringVar(midrightframe, 'N clusters:'))
        string.grid(row=0, column=0)
        n_cluster_choice = tk.StringVar(midrightframe)
        menu = tk.OptionMenu(midrightframe, n_cluster_choice, *range(2, 11))
        menu.grid(row=0, column=1)

        # ok button
        bottomrightframe = tk.Frame(master)
        bottomrightframe.grid(row=2, column=1)

        # when accept is clicked
        def accept():

            # Store selected number of clusters
            n_clusters = n_cluster_choice.get()
            self.n_clusters = int(n_clusters)

            # Apply hierarchical clustering with final choice of  n_clusters
            self.dendrofig = plt.figure(figsize=(6.5, 1.5))
            self.clustlabels, self.dendro, self.finalscore_table, self.linkmat, _ = hierarch_clust(self.data, self.n_clusters, self.kwargs['labels'])
            self.dendroax = plt.gca()

            # Close the window
            master.quit()
            master.destroy()

        # Create Accept button
        acceptbutton = tk.Button(bottomrightframe, text='Accept', command=accept)
        acceptbutton.grid(row=0, column=0)
        acceptbutton.focus_force()

        master.mainloop()


def hierarch_clust(X, n_clusters, datalabels):

    """
    Perform hierarchical clustering and return labels, dendrogram, scores and linkage matrix

    :param X: input data
    :param n_clusters: number of clusters
    :param datalabels: labels for the data
    :return:
    """

    # Fit model
    hrcal_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hrcal_model.fit_predict(X)

    # Calculate scores
    scorenames = ['Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin']
    scorelist = [metrics.silhouette_score(X, labels),
                 metrics.calinski_harabasz_score(X, labels),
                 metrics.davies_bouldin_score(X, labels)]
    scores = pd.DataFrame(data=scorelist, index=scorenames)

    # Plot dendrogram
    clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
    dendro, linkage_matrix = plot_dendrogram(clustdataholder,
                                             datalabels=datalabels,
                                             color_threshold=0)
    n_dendroclusters = 0
    shrink_factor = 0.05
    branch_height = np.max(linkage_matrix[:, 2])

    while n_dendroclusters != n_clusters:
        branch_height -= np.max(linkage_matrix[:, 2]) * shrink_factor
        clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
        dendro, linkage_matrix = plot_dendrogram(clustdataholder,
                                                 datalabels=datalabels,
                                                 color_threshold=branch_height)

        # Due to some weird behaviour in the dendrogram function possibly related to different versions of scipy
        try:
            n_dendroclusters = len(np.unique(dendro['leaves_color_list']))
        except:
            n_dendroclusters = len(np.unique(dendro['color_list']))

        # Keep readjusting shrinking factor until found
        if n_dendroclusters > n_clusters:
            shrink_factor *= 0.2
            branch_height = np.max(linkage_matrix[:, 2]) - np.max(linkage_matrix[:, 2]) *\
                            shrink_factor

    return labels, dendro, scores, linkage_matrix, branch_height


def plot_dendrogram(model, datalabels, color_threshold=None, orientation='top'):

    """ Plot dendrogram from hierarchical clustering model.
    """

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrofig = dendrogram(linkage_matrix,
                           labels=datalabels,
                           color_threshold=color_threshold,
                           orientation=orientation)

    return dendrofig, linkage_matrix


def append_bottom_leaves_dendrogram(dendroax, labelcolour=[]):

    ylim = dendroax.get_ylim()
    ylimlen = ylim[1] - ylim[0]

    # Colour bottom leaves according to colour in previous stage
    for xtick, ticklabel in zip(dendroax.get_xticks(), dendroax.get_xticklabels()):
        if not isinstance(labelcolour, pd.DataFrame):
            plt.vlines(xtick, 0 - 0.08 * ylimlen, 0 - 0.01 * ylimlen, color='k')
        else:
            plt.vlines(xtick, 0 - 0.08 * ylimlen, 0 - 0.01 * ylimlen, color=
                       labelcolour['colourcode'].loc[labelcolour['ptcode'] == ticklabel.get_text()].values[0])

    dendroax.set_ylim(ylim[0] - 0.09 * ylimlen, ylim[1])
    dendroax.set_axis_off()
    plt.draw()


def add_dendro_legend(dendroax, colourid):
    """
    Sets up a legend for a dendrogram axis using provided color labels and data. It also includes the number of points
    in each cluster in the legend.

    Parameters:
    - dendroax: The axis object containing the dendrogram where the legend is applied.
    - colourid: DataFrame or Series containing 'colourcode' entries for each cluster.
    """

    # Unique colourcodes for temporary legend
    # Get unique colourcodes
    templegend = list(colourid['colourcode'].unique())
    templegend.insert(0, '_nolegend')
    legend = dendroax.legend(templegend, frameon=False)

    # Get handles and their colours
    handlecolours = [handle.get_color() for handle in legend.legendHandles]

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
    clustcount = [len(colourid.loc[colourid['colourcode'] == colourlabel]) for colourlabel in orderedcolours]

    # Subtract 1 from the digit in finalcolours
    orderedcolours = [orderedcolours[:-1] + str(int(orderedcolours[-1]) - 1) for orderedcolours in orderedcolours]

    # Add count of pts in each cluster
    orderedcolours = [f'{orderedcolour} ({clustcounti})' for orderedcolour, clustcounti in
                      zip(orderedcolours, clustcount)]

    # Set the legend correctly now
    orderedcolours.insert(0, '_nolegend')
    dendroax.legend(orderedcolours, frameon=False)

    plt.tight_layout()


def tsne_plot(X, perplexities, colours, **kwargs):

    """
    Plots t-SNE for different perplexities.

    :param X: input data
    :param perplexities: list of perplexities
    :param colours: colours for each data point
    :param kwargs: it can be used to pass ringcolours, title and labels
    :return:
    """

    # Get kwargs
    ringcolours = kwargs.get('ringcolours', colours)
    title = kwargs.get('title', 't-SNE')
    labels = kwargs.get('labels', 'No labels added.'*len(X))

    # Create figure
    fig, axs = plt.subplots(1, len(perplexities), figsize=(12.38, 3.52))
    axs = axs.flat

    for plti, p in enumerate(perplexities):

        # TSNE data with perplexity p
        tsne = TSNE(perplexity=p, )
        tsned = tsne.fit_transform(X)

        # Plot data points
        axs[plti].scatter(tsned[:, 0], tsned[:, 1], c=colours, edgecolors=ringcolours, linewidths=2)

        # Make it pretty
        axs[plti].spines['top'].set_visible(False)
        axs[plti].spines['right'].set_visible(False)
        axs[plti].set_xlabel('Emb dim 1(.)')
        axs[plti].set_title(f'p = {p}')

        # Use mplcursors to display labels on hover
        try:
            mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
        except:
            print('mplcursors not available. Plot will not display labels on hover.')

    fig.suptitle(title)
    axs[0].set_ylabel('Emb dim 2(.)')
    fig.suptitle(title)
    fig.tight_layout()

    return fig

def pca_expvar_plot(pca, threhsolds, colours=['r'], threshlabels=[], highlighted=[], title='PCA explained variance'):

    """
    Plots the explained variance of a PCA model
    TODO: Make it look nice and more flexible.
    """

    # Visualise explained variance
    fig = plt.figure(figsize=[5.5, 4])

    # Plot thresholds first
    if len(threhsolds) != len(colours):
        print('Number of colours and thresholds do not match. Using red for all thresholds.')
        colours = ['r'] * len(threhsolds)

    if threshlabels == []:
        threshlabels = [str(int(thresh * 100)) + '%' for thresh in threhsolds]

    pcns = []
    for threshold, colour, threshlabel in zip(threhsolds, colours, threshlabels):
        pcns.append(np.where(np.cumsum(pca.explained_variance_ratio_) >= threshold)[0][0] + 1)
        plt.axhline(y=threshold, color=colour, linestyle=':', zorder=4)

        if threshold in highlighted:
            plt.vlines(x=pcns[-1], ymin=0, ymax=threshold, label=threshlabel,  color=colour, linestyle='-', linewidth=3, zorder=1)
        else:
            plt.vlines(x=pcns[-1], ymin=0, ymax=threshold, label=threshlabel,  color=colour, linestyle='-', zorder=1)

    # Plot explained variance
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5,
            align='center', label='Individual', zorder=3)
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid',
             label='Cumulative', color='k', zorder=3)

    # Set ylim
    plt.ylim([0, 1.03])

    # Set yticks
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # Scale to 100%
    plt.yticks(yticks, [int(ytick * 100) for ytick in yticks])
    plt.ylabel('Explained variance (%)')
    plt.xticks(pcns)
    plt.xlabel('PC index')
    plt.title(title)
    plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return fig


def corrmat_plot(array, figsize=(5, 5)):

    """
    Plots a correlation matrix
    """

    # Generate a correlation matrix
    corrmat = np.corrcoef(array, rowvar=False)

    # Plot the correlation matrix using a heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corrmat, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_title("Correlation Matrix")

    return fig, ax


class TransitionAnalysis:

    """
    A class to analyse transitions between two clustering partitions. It allows to recolour the dendrogram based on the
    transitions between the two partitions.
    """

    def __init__(self, prev_colourid, curr_colourid, dendroax):

        self.prev_colourid = prev_colourid
        self.curr_colourid = curr_colourid
        self.dendroax = dendroax

        # Get previous colour-label convention
        self.prevcolourlabels = self.prev_colourid[['colourcode', 'clustlabel']].drop_duplicates()

        # Get current colour-label convention
        self.currentcolourlabels = self.curr_colourid[['colourcode', 'clustlabel']].drop_duplicates()

        # Get participants that are in both speeds
        self.reppts = list(set(self.curr_colourid['ptcode']).intersection(set(self.prev_colourid['ptcode'])))

        # Get transitions
        self.jointdf, self.transitions = self.get_transitions()

        # Transition GUI
        transGUI = TransitionAnalysisGUI(self.transitions)
        recolour = transGUI.recolour

        # Get pts with each current label
        currcolourpts = []
        for lab in recolour.keys():
            currcolourpts.append(self.curr_colourid['ptcode'].loc[self.curr_colourid['clustlabel'] == int(lab)].to_list())

        # Replace labels and colours
        for colourptsidcs, (oldlab, newlab) in zip(currcolourpts, recolour.items()):
            self.curr_colourid['clustlabel'].loc[self.curr_colourid['ptcode'].isin(colourptsidcs)] = int(newlab)

            # Replace colour Cn colour code using old colours and then new colours as exception
            try:
                newcolour = self.prevcolourlabels['colourcode'].loc[self.prevcolourlabels['clustlabel'] == int(newlab)].values[0]
            except:
                newcolour = \
                self.currentcolourlabels['colourcode'].loc[self.currentcolourlabels['clustlabel'] == int(newlab)].values[0]

            self.curr_colourid['colourcode'].loc[self.curr_colourid['ptcode'].isin(colourptsidcs)] = newcolour

            # The n-1 first children in the axes are going to be the clusters
            self.dendroax._children[int(oldlab) + 1].set_color(newcolour)

        # Create updated transitions
        self.updatedjointdf, self.updatedtransitions = self.get_transitions()

        # Calculate AMI scores
        self.ami = metrics.adjusted_mutual_info_score(self.updatedjointdf['prev'], self.updatedjointdf['curr'])

    def get_transitions(self):

        # Get joint dataframe
        previous = self.prev_colourid.loc[self.prev_colourid['ptcode'].isin(self.reppts)]
        previous = previous.drop(columns='colourcode')
        previous = previous.rename(columns={'clustlabel': 'prev'})
        current = self.curr_colourid.loc[self.curr_colourid['ptcode'].isin(self.reppts)]
        current = current.drop(columns='colourcode')
        current = current.rename(columns={'clustlabel': 'curr'})
        jointdf = pd.merge(previous, current, on='ptcode')
        jointdf = jointdf.set_index('ptcode')

        # Count unique transitions
        un, count = np.unique(jointdf, axis=0, return_counts=True)
        transitions = pd.DataFrame({'prev': un[:, 0], 'post': un[:, 1], 'count': count})

        # Calculate count relative to count in prev and post cluster
        transitions['rel_count_prev'] = 0
        transitions['rel_count_post'] = 0

        for i, row in transitions.iterrows():
            transitions['rel_count_prev'].iloc[i] = row['count'] / np.sum(
                transitions['count'].loc[transitions['prev'] == row['prev']])
            transitions['rel_count_post'].iloc[i] = row['count'] / np.sum(
                transitions['count'].loc[transitions['post'] == row['post']])

        # Sort by count
        transitions = transitions.sort_values(by='count', ascending=False)

        return jointdf, transitions

class TransitionAnalysisGUI:

    """
    A GUI for the TransitionAnalysis class.
    Display colours and let user choose the ralabelling
    """

    def __init__(self, transitions):
        self.transitions = transitions

        # Create GUI
        self.master = tk.Tk()
        self.master.geometry('400x600')
        self.master.title('Colour matching')
        self.master.attributes("-topmost", True)
        self.master.focus_force()

        # Instructions frame
        topframe = tk.Frame(self.master)
        topframe.grid(row=0, column=0)

        # Display instructions
        instructstr = ('Indicate the colour replacements based on \n'
                       'the transitions below and the dendrogram on the right. \n'
                       'Vlines in the dendrogram represent the colour \n'
                       'of that datapoint in the previous clustering partition.')

        instructholder = tk.StringVar(topframe, instructstr)
        instructions = tk.Label(topframe, textvariable=instructholder)
        instructions.grid(row=0, column=0)

        # Display transitions
        top2frame = tk.Frame(self.master)
        top2frame.grid(row=1, column=0)
        transitionstring = '    ' + self.transitions.to_string(index=False)
        transitionholder = tk.StringVar(top2frame, transitionstring)
        dflabel = tk.Label(top2frame, textvariable=transitionholder)
        dflabel.grid(row=0, column=0)

        # Colour matching options
        midframe = tk.Frame(self.master)
        midframe.grid(row=2, column=0)

        # All the possible colours in the transitions
        posscolours = np.unique(pd.concat([self.transitions['prev'], self.transitions['post']]))

        self.currlabels = {}
        self.corrlabels = []

        for i, currlabel in enumerate(np.unique(self.transitions['post'])):

            # Write current colour in a string
            self.currlabels[currlabel] = tk.Label(midframe, textvariable=tk.StringVar(midframe, currlabel))
            self.currlabels[currlabel].grid(column=0, row=i)

            # Write possible colours to replace current colour
            corrlabel = tk.StringVar(midframe, currlabel)
            menu = tk.OptionMenu(midframe, corrlabel, *np.unique(posscolours))
            menu.grid(column=1, row=i)
            self.corrlabels.append(corrlabel)

        # Accept frame
        bottomframe = tk.Frame(self.master)
        bottomframe.grid(row=4, column=0)

        # Create Accept button
        acceptbutton = tk.Button(bottomframe, text='Accept', command=self.accept)
        acceptbutton.grid(row=0, column=0)
        acceptbutton.focus_force()

        tk.mainloop()

    # when accept is clicked
    def accept(self):

        # Store current label and correct label (new)
        self.recolour = {}
        for currlabel, corrlabel in zip(self.currlabels, self.corrlabels):
            self.recolour[str(currlabel)] = corrlabel.get()

        # Close the window
        self.master.quit()
        self.master.destroy()


class CustomScaler(BaseEstimator, TransformerMixin):

    """
    A custom standard scaler that can be used with sklearn pipelines. It allows to standardise data based on a
    variable tracker that indicates which variables belong to which group. This is useful when the data is not
    standardised only in a column by column basis but also in a group basis e.g., temporal data.
    """

    def fit(self, X, y=None, vartracker=None):

        self.vartracker_ = vartracker

        # Get vartracker
        if self.vartracker_ is None:
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
        else:
            self.vartracker_ = np.array(vartracker)

            # Get mean and std for each key in datadict
            self.mean_ = {var: np.mean(X[:, np.where(self.vartracker_ == var)]) for var in np.unique(self.vartracker_)}
            self.std_ = {var: np.std(X[:, np.where(self.vartracker_ == var)]) for var in np.unique(self.vartracker_)}

        return self

    def transform(self, X):

        Xz = np.zeros(X.shape)

        # Standardise data
        for var in np.unique(self.vartracker_):
            Xz[:, np.where(self.vartracker_ == var)] = (X[:, np.where(self.vartracker_ == var)] - self.mean_[var]) / \
                                                       self.std_[var]

        return Xz

    def fit_transform(self, X, y=None, vartracker=None):

        self.fit(X, y, vartracker)
        Xz = self.transform(X)

        return Xz

    def inverse_transform(self, Xz, y=None):

        X = np.zeros(Xz.shape)

        # Standardise data
        for var in np.unique(self.vartracker_):
            X[:, np.where(self.vartracker_ == var)] = Xz[:, np.where(self.vartracker_ == var)] * self.std_[var] + \
                                                       self.mean_[var]

        return X


def make_splitgrid(nrows, ncols=None, figsize=(19.2, 9.77)):

    if ncols is None:
        ncols = nrows

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, 1, hspace=0.3)
    axs = {'topaxs': [], 'bottomaxs': []}

    for row in gs:
        subgs = row.subgridspec(2, ncols, hspace=0, wspace=0.4)
        for subgdi, subgd in enumerate(subgs):
            if subgdi < subgs.ncols:
                axs['topaxs'].append(plt.subplot(subgd))
            else:
                axs['bottomaxs'].append(plt.subplot(subgd))

    rows_x_cols = [gs.nrows, subgs.ncols]

    return fig, axs, rows_x_cols


def comparison_0D_contvar_indgroups(datadict, grouping, title_kword, figdir, colours):

    """
    Compares 0D variables in different groups using parametric and non-parametric tests. It also plots Q-Q plots for
    normality.

    :param datadict: dictionary with data
    :param grouping: grouping of the data
    :param title_kword: title keyword for the plots
    :param figdir: directory to save figures
    :param colours: colours
    :return:
    """

    disc_comp = {}

    for key, values in datadict.items():

        disc_comp[key] = {}

        # Check for nans
        if np.any(np.isnan(values)):
            print(f'NaNs found in {key} and they will be removed.')

        # Get variable in groups
        holder = pd.DataFrame({key: np.squeeze(values)})
        holder['grouping'] = grouping
        groups = [holder.groupby(['grouping']).get_group(x)[key].dropna() for x in
                  np.sort(holder['grouping'].dropna().unique())]

        # Run normality tests
        disc_comp[key]['normality'] = {}
        fig, axes = plt.subplots(1, len(groups))
        fig.set_size_inches([11, 3.3])

        # test trigger
        param_route = 1

        for labi, group in enumerate(groups):
            disc_comp[key]['normality'][str(labi)] = {}
            disc_comp[key]['normality'][str(labi)]['W_stat'], disc_comp[key]['normality'][str(labi)][
                'p'] = stats.shapiro(group)

            # if there were violations of normality or homoscedasticity change trigger for tests later
            if disc_comp[key]['normality'][str(labi)]['p'] <= 0.05:
                param_route = 0

            # Q-Q plots
            sm.qqplot(group, ax=axes[labi], markeredgecolor=colours[labi], markerfacecolor=colours[labi], line='r',
                      fmt='k-')
            axes[labi].get_lines()[1].set_color('black')
            axes[labi].set_xlabel('Cluster ' + str(labi))

            if disc_comp[key]['normality'][str(labi)]['p'] < 0.001:
                axes[labi].set_title(
                    'W: ' + str(np.round(disc_comp[key]['normality'][str(labi)]['W_stat'], 3)) + '; p < 0.001')
            else:
                axes[labi].set_title(
                    'W: ' + str(np.round(disc_comp[key]['normality'][str(labi)]['W_stat'], 3)) + '; p = ' + str(
                        np.round(disc_comp[key]['normality'][str(labi)]['p'], 3)))

        fig.suptitle(title_kword + '_' + key)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, title_kword + '_' + key + '_' + 'QQplot.png'))
        plt.close(plt.gcf())

        # Parametric route
        if param_route:

            if len(groups) == 2:

                # Run heteroscedasticity tests
                disc_comp[key]['homoscedasticity'] = {}
                disc_comp[key]['homoscedasticity']['Levene_stat'], disc_comp[key]['homoscedasticity']['p'] = stats.levene(*groups)

                if disc_comp[key]['homoscedasticity']['p'] > 0.05:

                    # Independent standard t-test
                    disc_comp[key]['ttest_ind'] = {}
                    disc_comp[key]['ttest_ind']['t'], disc_comp[key]['ttest_ind']['p'] = stats.ttest_ind(*groups)

                else:

                    # Welch's t-test
                    disc_comp[key]['ttest_ind'] = {}
                    disc_comp[key]['ttest_ind']['welch_t'], disc_comp[key]['ttest_ind']['p'] = stats.ttest_ind(*groups, equal_var=False)

                # Get Cohen's d
                disc_comp[key]['ttest_ind']['Cohens_d'] = (np.mean(groups[0]) - np.mean(groups[1])) / \
                                                          np.sqrt(
                                                              (np.std(groups[0], ddof=1) ** 2 + np.std(groups[1], ddof=1) ** 2) / 2)

                # Get Hedge's g
                disc_comp[key]['ttest_ind']['Hedges_g'] = disc_comp[key]['ttest_ind']['Cohens_d'] * (
                        1 - (3 / (4 * (len(groups[0]) + len(groups[1]) - 2) - 1)))

            elif len(groups) > 2:

                # One-way ANOVA
                disc_comp[key]['ANOVA_1'] = {}
                disc_comp[key]['ANOVA_1']['F_stat'], disc_comp[key]['ANOVA_1']['p'] = stats.f_oneway(*groups)

                if disc_comp[key]['ANOVA_1']['p'] <= 0.05:
                    # Bonferroni post hoc tests
                    disc_comp[key]['Bonferroni_post_hoc'] = posthoc_ttest(groups, p_adjust='bonferroni')

        # Non-parametric route
        else:

            if len(groups) == 2:

                # Mann-Whitney U test
                disc_comp[key]['mann_whitney_U'] = {}
                disc_comp[key]['mann_whitney_U']['U_stat'], disc_comp[key]['mann_whitney_U']['p'] = stats.mannwhitneyu(
                    *groups)

            elif len(groups) > 2:

                # Kruskal
                disc_comp[key]['Kruskal'] = {}
                disc_comp[key]['Kruskal']['Hstat'], disc_comp[key]['Kruskal']['p'] = stats.kruskal(*groups)

                if disc_comp[key]['Kruskal']['p'] <= 0.05:
                    # Dunn post hoc tests
                    disc_comp[key]['Dunn_post_hoc'] = posthoc_dunn(groups, p_adjust='bonferroni')

    return disc_comp


def comparison_1D_contvar_indgroups(datadict, grouping, title_kword, figdir, colours):

    # Conduct traditional SPM1D non-param tests
    cont_comp = {}

    for key, values in datadict.items():

        cont_comp[key] = {}

        # Get variable in groups
        groups = [values[np.where(grouping == x)[0], :] for x in natsort.natsorted(np.unique(grouping))]

        if len(groups) == 2:

            # Non param ttest
            nonparam_ttest2 = spm1d.stats.nonparam.ttest2(groups[0], groups[1])
            cont_comp[key]['np_ttest2'] = nonparam_ttest2.inference(alpha=0.05, two_tailed=True, iterations=500)

            # Vis
            varfig = plt.figure(figsize=(10, 4))

            # Average and std patterns by group
            plt.subplot(1, 2, 1)
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(group, linecolor=colour, facecolor=colour)
            plt.title(key)

            plt.subplot(1, 2, 2)

            cont_comp[key]['np_ttest2'].plot()
            cont_comp[key]['np_ttest2'].plot_threshold_label(fontsize=8)
            cont_comp[key]['np_ttest2'].plot_p_values()
            plt.title(f'np_ttest2 {key}')

            plt.tight_layout()
            varfig.savefig(os.path.join(figdir, f'{title_kword}_{key}_np_ttest2.png'))
            plt.close(varfig)

        elif len(groups) > 2:

            # Non parametric ANOVA
            nonparam_ANOVA = spm1d.stats.nonparam.anova1(values, grouping)
            cont_comp[key]['np_ANOVA'] = nonparam_ANOVA.inference(alpha=0.05, iterations=500)

            # Vis
            varfig = plt.figure(figsize=(10, 4))

            # Average and std patterns by group
            plt.subplot(1, 2, 1)
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(group, linecolor=colour, facecolor=colour)
                plt.title(key)

            plt.subplot(1, 2, 2)
            cont_comp[key]['np_ANOVA'].plot()
            cont_comp[key]['np_ANOVA'].plot_threshold_label(fontsize=8)
            cont_comp[key]['np_ANOVA'].plot_p_values()
            plt.title(f'np_ANOVA {key}')
            plt.tight_layout()
            varfig.savefig(os.path.join(figdir, f'{title_kword}_{key}_np_ANOVA.png'))
            plt.close(varfig)

            if cont_comp[key]['np_ANOVA'].h0reject:

                # Adjust alpha for the number of comparisons to be performed
                ngroups = len(groups)
                alpha = 0.05 / ngroups * (ngroups - 1) / 2

                # Get unique pairwise comparisons
                paircomp = list(combinations(np.unique(grouping), 2))

                # Set number of subplots for comparison
                if len(paircomp) == 3:
                    fig, axes = plt.subplots(2, 3)
                    fig.set_size_inches(11, 6)

                elif len(paircomp) == 6:
                    fig, axes = plt.subplots(4, 3)
                    fig.set_size_inches(11, 12)

                else:
                    print('I am not ready for so many plots. Figure it out.')
                axes = axes.flat
                for pairi, pair in enumerate(paircomp):

                    # Get pair key word
                    pairkw = f'{str(pair[0])}_{str(pair[1])}'

                    # Run post-hoc analysis
                    cont_comp[key]['post_hoc_np_ttest2'] = {}
                    nonparam_ttest2 = spm1d.stats.nonparam.ttest2(groups[pair[0]], groups[pair[1]])
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw] = nonparam_ttest2.inference(alpha=alpha,
                                                                                             two_tailed=True,
                                                                                             iterations=500)

                    # Vis
                    if pairi <= 2:
                        axi = pairi
                    else:
                        axi = pairi + 6

                    # NOTE THIS ASSUMES THAT THE ORDER OF THE COLOURS MATCHES THE ORDER OF THE LABELS
                    spm1d.plot.plot_mean_sd(groups[pair[0]], ax=axes[axi],
                                            linecolor=colours[pair[0]],
                                            facecolor=colours[pair[0]])
                    spm1d.plot.plot_mean_sd(groups[pair[1]], ax=axes[axi],
                                            linecolor=colours[pair[0]],
                                            facecolor=colours[pair[1]])
                    axes[pairi].set_title(str(pair))

                    pairkw = f'{str(pair[0])}_{str(pair[1])}'

                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot(ax=axes[axi + 3])
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot_threshold_label(ax=axes[axi + 3], fontsize=8)
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot_p_values(ax=axes[axi + 3])

                fig.suptitle(f'{title_kword}_{key}')
                plt.tight_layout()
                plt.savefig(os.path.join(figdir, f'{title_kword}_{key}_posthoc.png'))
                plt.close(plt.gcf())

    return cont_comp


# 3D distance
def dist3D(a, b):
    # Calculate the Ukelele distance
    dist = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    return dist

#  # TODO. Create opensim utils file with all of these because it gives dll imort errors
# def list_to_osim_array_str(list_str):
#     """Convert Python list of strings to OpenSim::Array<string>.Taken from:
#         https://github.com/mitkof6/opensim_automated_pipeline/blob/7822e1520ceb4ce0943b613a58471b6614437b57/simple/scripts/utils.py"""
#     arr = osim.ArrayStr()
#     for element in list_str:
#         arr.append(element)
#
#     return arr
#
#
# def create_opensim_storage(time, data, column_names):
#     """Creates a OpenSim::Storage. Taken from:
#         https://github.com/mitkof6/opensim_automated_pipeline/blob/7822e1520ceb4ce0943b613a58471b6614437b57/simple/scripts/utils.py
#     Parameters
#     ----------
#     time: SimTK::Vector
#     data: SimTK::Matrix
#     column_names: list of strings
#     Returns
#     -------
#     sto: OpenSim::Storage
#     """
#     sto = osim.Storage()
#     sto.setColumnLabels(list_to_osim_array_str(['time'] + column_names))
#     for i in range(data.nrow()):
#         row = osim.ArrayDouble()
#         for j in range(data.ncol()):
#             row.append(data.getElt(i, j))
#
#         sto.append(time[i], row)
#
#     return sto
#
#
# def np_array_to_simtk_matrix(array):
#     """Convert numpy array to SimTK::Matrix"""
#     n, m = array.shape
#     M = osim.Matrix(n, m)
#     for i in range(n):
#         for j in range(m):
#             M.set(i, j, array[i, j])
#
#     return M
#
#
# def list_to_osim_array_str(list_str):
#     """Convert Python list of strings to OpenSim::Array<string>."""
#     arr = osim.ArrayStr()
#     for element in list_str:
#         arr.append(element)
#
#     return arr
#
#
# def create_clust_opensim_models(scalingdata, ptlabels, modelpath, genscalsetuppath, reportdir):
#
#
#     for clust in np.unique(ptlabels['clustlabel']):
#
#         # Get ptcodes from ptcode column for participants in given cluster
#         clustpts = ptlabels['ptcode'].loc[ptlabels['clustlabel'] == clust]
#
#         # Get average measurements
#         height = scalingdata.loc[clustpts]['Height'].mean()
#         exp_pelv_width = scalingdata.loc[clustpts]['PelvWidth'].mean()
#         exp_thighl_r = scalingdata.loc[clustpts]['ThiLgth_r'].mean()
#         exp_shankl_r = scalingdata.loc[clustpts]['ShaLgth_r'].mean()
#
#         # Initialise model
#         model = osim.Model(modelpath)
#         state = model.initSystem()
#
#         # Extract marker positions
#         markers = model.getMarkerSet()
#         markerpos = {}
#
#         for i in range(markers.getSize()):
#             markername = markers.get(i).getName()
#             markerpos[markername] = markers.get(i).getLocationInGround(state)
#
#         # Calculate ukelele distances between markers
#         mod_pelv_width = dist3D(markerpos['RHJC'], markerpos['LHJC'])
#         mod_thighl_r = dist3D(markerpos['RHJC'], markerpos['RKJC'])
#         mod_shankl_r = dist3D(markerpos['RKJC'], markerpos['RAJC'])
#
#         # Divide experimental by model
#         pelvwidth_sf = exp_pelv_width / mod_pelv_width
#         thighl_r_sf = exp_thighl_r / mod_thighl_r
#         shankl_r_sf = exp_shankl_r / mod_shankl_r
#
#         # Initialise scaler tool
#         scaleTool = osim.ScaleTool(genscalsetuppath)
#
#         # Scale height
#         # scaleTool.setSubjectHeight(height)
#
#         # Scale segments
#         scaleTool.getModelScaler().getScaleSet().get('pelvis').setScaleFactors(osim.Vec3(pelvwidth_sf))
#         scaleTool.getModelScaler().getScaleSet().get('femur_r').setScaleFactors(osim.Vec3(thighl_r_sf))
#         scaleTool.getModelScaler().getScaleSet().get('tibia_r').setScaleFactors(osim.Vec3(shankl_r_sf))
#
#         # Set path to generic model file
#         scaleTool.getGenericModelMaker().setModelFileName(r'gait2392_simbody_custom.osim')
#
#         # Set path to scaled model file
#         scaleTool.getModelScaler().setOutputModelFileName(os.path.join(reportdir, f'Clust_{clust}.osim'))
#         scaleTool.getModelScaler().processModel(model)
#
#         scaleTool.printToXML(r'C:\Users\arr43\Documents\OpenSim\4.3\Models\Gait2392_Simbody\Test_Scale_setup.xml')
#
#         # Run scaler tool
#         scaleTool.run()
#
#         # Get Cn colour as RGB
#         colour = ptlabels['colourcode'].loc[ptlabels['clustlabel'] == clust].values[0]
#         colour = mcolors.to_rgb(colour)
#
#         # Read in scaled model as a list of strings
#         with open(os.path.join(reportdir, f'Clust_{clust}.osim'), 'r') as file:
#             filedata = file.read().split('\n')
#
#         # Get model segment names
#         bodies = model.get_BodySet()
#         segnames = [bodies.get(i).getName() for i in range(bodies.getSize())]
#
#         # Get color lines
#         colorlines = [i for i, x in enumerate(filedata) if '<color>' in x]
#
#         for body in segnames:
#
#             # Find the line with the segment name
#             bodystart = [i for i, x in enumerate(filedata) if f'<Body name="{body}"' in x][0]
#             bodyend = [i for i, x in enumerate(filedata) if f'</Body>' in x and i > bodystart][0]
#
#             # Get colorlines within bodystart and bodyend
#             bodycolorlines = [i for i in colorlines if bodystart < i < bodyend]
#
#             for line in bodycolorlines:
#
#                 # Find where <color> starts in the line
#                 start = filedata[line].find('<color>')
#
#                 # Replace the line with the new colour
#                 filedata[line] = filedata[line][:start] + f'<color>{colour[0]} {colour[1]} {colour[2]}</color>'
#
#         # Merge all list items into a string with new lines
#         filedata = '\n'.join(filedata)
#         with open(os.path.join(reportdir, f'Clust_{clust}.osim'), 'w') as file:
#             file.write(filedata)