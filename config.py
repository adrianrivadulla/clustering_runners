import matplotlib
import matplotlib.pyplot as plt
import os

# matplotlib backend
matplotlib.use("Qt5Agg")

# matplotlib style
plt.style.use("default")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Project directory
projdir = "."

# Data dir
datadir = os.path.join(projdir, "data")

# Fatigure report dir
reportdir = os.path.join(projdir, "report")

# Path to fatigue data file
datapath = os.path.join(datadir, 'AllCurves_ptavgs.npy')

# Master datasheet
masterdatapath = os.path.join(datadir, "MasterDataSheet.xlsx")

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
demoanthrophysvars_titles = {'Sex': 'Sex',
                             'Age': 'Age',
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
