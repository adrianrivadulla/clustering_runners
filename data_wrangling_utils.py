

import pandas as pd
import numpy as np
import itertools

def load_mastersheet_and_kinematics(masterdatapath, kindatapath, stages, speeds, wantedvars, discvars, contvars):

    # Load mastersheet
    master = pd.read_excel(masterdatapath, index_col=1, header=1)
    
    # Have VO2max called VO2peakkg for correctness
    master['VO2peakkg'] = master['VO2max']
    
    # EE is in kcal/min, normalise by mass
    for speed in speeds:
        master[f'EE{speed}kg'] = master[f'EEJW{speed}'] / master['Mass']
    
    # Get 10k times which are datetime.time in seconds
    master['Time10Ks'] = master['Time10K'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    
    # Get LT+1 speed for all participants
    ltplus1 = np.round(master['LT'])
    
    # Get participants who have LT+1 speed of at least 13 km/h
    pts_physok = ltplus1.loc[ltplus1 >= 13].index.to_list()
    sub13 = master.loc[pts_physok]
    
    # Load kinematic data
    data = np.load(kindatapath, allow_pickle=True).item()
    
    kindata = {}
    rawstridelen = []
    
    for stgi, (stage, speed) in enumerate(zip(stages, speeds)):
    
        # Get pt codes which are in both master and data
        pts = sorted(set(pts_physok) & set(data[stage].keys()))
    
        # Get average patterns
        kindata[stage] = {}
    
        # Stack each participant's avge pattern together
        for pti, pt in enumerate(pts):
    
            for key in wantedvars:
    
                # Preallocate array if not already in kindata
                if key not in kindata[stage].keys():
    
                    if key in discvars:
                        kindata[stage][key] = np.full((len(pts), 1), np.nan)
                    elif key in contvars:
                        kindata[stage][key] = np.full((len(pts), len(data[stage][pt][key])), np.nan)
    
                # Assign values
                kindata[stage][key][pti, :] = data[stage][pt][key]
    
            # Get raw length of stride
            rawstridelen.append(1 / (data[stage][pt]['STRIDEFREQ'] * master['LegLgth_r'].loc[pt]))
    
    # Create vartracker
    vartracker = {}
    for stage in stages:
        vartracker[stage] = [[key] * values.shape[1] for key, values in kindata[stage].items()]
        vartracker[stage] = list(itertools.chain(*vartracker[stage]))
    
    # Horizontally stack all stages in multispeed
    kindata['multispeed'] = {}
    vartracker['multispeed'] = []
    
    for key in kindata[stages[0]].keys():
        kindata['multispeed'][key] = np.hstack([kindata[stage][key] for stage in stages])
        vartracker['multispeed'].extend(
            [[f'{key}_{int(speeds[stgi])}'] * kindata[stage][key].shape[1] for stgi, stage in enumerate(stages)])
    
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

    return master, kindata, vartracker, pts