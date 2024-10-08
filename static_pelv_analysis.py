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

# Matplotlib style
matplotlib.use('Qt5Agg')
matplotlib.style.use('default')

# Project dir wherever this script is
projectdir = os.path.dirname(os.path.realpath(__file__))

# Data dir
datadir = os.path.join(projectdir, 'data')

# Report dir for saving figures
reportdir = os.path.join(projectdir, 'report')

# savingkw
savingkw = 'static_pelv'

# Load pelvic tilt data
data = pd.read_csv(os.path.join(datadir, 'Clust_multispeed_avgepelvictilt.csv'))

# Get cluster colours
clustcolours = np.unique(data['colour'])


# Compare pelvis tilt between clusters
pelvictilt_comparison = comparison_0D_contvar_indgroups({'pelvictilt': data['Session1'].values.astype(float)},
                                                               data['clustlabel'],
                                                               f'{savingkw}_multipeed',
                                                               reportdir,
                                                               clustcolours
                                                        )

# Make violin plot
plt.figure()
sns.violinplot(x=data['clustlabel'], y=data['Session1'].values.astype(float), palette=clustcolours)

# Decorate
plt.xlabel('Cluster')
plt.ylabel('${\Theta}$ (°) \n< Ant - Post >')
plt.title('Static pelvic tilt')

# Get rid of top and right spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Print out results
print('Pelvic tilt comparison:\n')
for key in pelvictilt_comparison['pelvictilt'].keys():
    print(f'{key}: {pelvictilt_comparison["pelvictilt"][key]}')

# Save
plt.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_.png'), dpi=300, bbox_inches='tight')

