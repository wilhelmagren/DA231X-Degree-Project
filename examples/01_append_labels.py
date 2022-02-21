"""
Extend the labels file 'subjects.tsv' with the reaction times
of the subjects. Currently only extract the normal
reaction time (RT), 1/RT, and the diff of RT between
normal and sleep deprivation.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 21-02-2022
"""
import os
import numpy as np
import pandas as pd

cwd = os.getcwd()
LABELS_ = os.path.join(cwd, 'data/subjects_backup.tsv')
RT_LABELS_ = os.path.join(cwd, 'data/pvt_ind_data.tsv')

dframe = pd.read_csv(RT_LABELS_, sep='\t')
RTrecip = dframe['RTrecip']

# 64 - 10 = 54
tmp = []
for i, x in enumerate(RTrecip):
    if i == 54:
        tmp.append(1.0)
        tmp.append(1.0)
    
    tmp.append(x)

RTrecip = np.array(tmp).reshape((1, 66))

RT = 1 / RTrecip
RTdiffs = []
RTrecips = []
RTs = []
for idx in range(0, RTrecip.shape[1], 2):
    control = RTrecip[0, idx]
    psd = RTrecip[0, idx+1]
    RTdiffs.append(psd - control)
    RTrecips.append((control, psd))
    lol = RT[0, idx]
    xd = RT[0, idx+1]
    RTs.append((lol, xd))

print(RTdiffs)
print(RTrecips)
print(RTs)

BUFFER = []
with open(LABELS_, 'r') as f:
    for idx, line in enumerate(f.readlines()):
        if idx == 0:
            s = line.rstrip()
            s += '\tRTrecipControl\tRTrecipSleep\tRTControl\tRTSleep\tRTdiff'
            BUFFER.append(s)
        else:
            print(idx, len(RTrecips), line)
            control, psd = RTrecips[idx - 1]
            s = line.rstrip()
            s += f'\t{control}\t{psd}'
            control, psd = RTs[idx - 1]
            s += f'\t{control}\t{psd}'
            diff = RTdiffs[idx - 1]
            s += f'\t{diff}'
            BUFFER.append(s)

with open('test.tsv', 'w') as f:
    for line in BUFFER:
        f.write(line + '\n')
