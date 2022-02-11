"""
Script for cleaning SLEMEG dataset of EOG and-or ECG artifacts.
(blinking, saccades, heart beats)

Number of components to fit ICA to is 25, otherwise too much information is lost.
When using this parameter, approximately >90% of information is retained.

MNE ICA is sensitive to low frequency oscillations, so we need to high pass
filter the data at 1Hz. This unfortunately decreases the quality of the data,
since signal becomes unstable close to 1Hz but we have to do this...

Rejection parameters based on peak-to-peak amplitude (PTP) in the continuous data.
Signal periods exceeding the thresholds in reject will be removed before 
fitting the ICA. 

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 31-01-2022
"""
import mne
import os
import matplotlib
import numpy as np

from collections import defaultdict
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs


plot = True
save = False
duration = 30
n_channels = 10
relative_dirty_MEG = 'data/data-ds-200Hz/'
relative_cleaned_MEG = 'data/data-cleaned/'

dfpaths = [os.path.join(relative_dirty_MEG, f) for f in os.listdir(relative_dirty_MEG)]
cfpaths = [os.path.join(relative_cleaned_MEG, f) for f in os.listdir(relative_dirty_MEG)]

stats = defaultdict(int)
for dfpath, cfpath in zip(dfpaths, cfpaths):
    raw = mne.io.read_raw_fif(dfpath, preload=True)

    if plot:
        raw.plot(duration=duration, n_channels=n_channels, remove_dc=False, block=True)

    raw.filter(1., None, n_jobs=1, fir_design='firwin')
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')

    # Set up and fit Independent Component Analysis
    decim = None
    kwargs = {'method': 'fastica', 'n_components': 25, 'random_state': 1}
    ica = ICA(**kwargs)
    ica.fit(raw, picks=picks_meg, decim=decim)
    print(ica)

    # find EOG components causing artifacts
    eog_epochs = create_eog_epochs(raw)
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
    stats['eog_inds'] += len(eog_inds)

    # find ECG comoponents causing artifacts
    ecg_epochs = create_ecg_epochs(raw)
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
    stats['ecg_inds'] += len(ecg_inds)

    # remove the artifacts from the raw data signals
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(ecg_inds)

    if eog_inds or ecg_inds:
        stats['cleaned'] += 1
        ica.apply(raw)

    if plot:
        raw.plot(duration=duration, n_channels=n_channels, remove_dc=False, block=True)

    if save:
        raw.save(cfpath, overwrite=True)

print(f'\nDone with SLEMEG cleaning!')
print(f'> cleaned recordings: {stats["cleaned"]}')
print(f'> percentage cleaned {100*stats["cleaned"]/132:.2f}%')
print(f'> bad components: {sum(stats.values())}')

