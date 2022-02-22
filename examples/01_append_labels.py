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


__root__ = os.getcwd()
rtpath = __root__ + '/data/pvt_ind_data.tsv'
labels = __root__ + '/data/subjects_backup.tsv'

dframe = pd.read_csv(rtpath, sep='\t')
RTrecip = dframe['RTrecip']

RT = 1 / RTrecip
RTdiffs = []
RTrecips = []
RTs = []
for idx in range(0, RTrecip.shape[0], 2):
    control = RTrecip[idx]
    psd = RTrecip[idx+1]
    lol = RT[idx]
    xd = RT[idx+1]
    if any(np.isnan([control, psd, lol, xd])):
        control = psd = lol = xd = 0.0
    RTdiffs.append(psd - control)
    RTrecips.append((control, psd))
    RTs.append((lol, xd))


BUFFER = []
with open(labels, 'r') as f:
    for idx, line in enumerate(f.readlines()):
        if idx == 0:
            s = line.rstrip()
            s += '\tRTrecipControl\tRTrecipSleep\tRTControl\tRTSleep\tRTdiff'
            BUFFER.append(s)
        else:
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
