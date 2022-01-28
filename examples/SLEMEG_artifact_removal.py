"""
Artifact removal on MEG data, by means of ICA relating to EOG and ECG measurements.
NOTE: the rejection criterion for MAG and GRAD are somewhat arbitrarily picked, and 
vary depending on recording. So some recordings have not been cleaned with ICA,
because they had no valid epochs based on the criterion.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 28-01-2022
"""
import os
import mne
import time

from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

datafpath = 'data/data-ds-200Hz/'
fpaths = [os.path.join(datafpath, f) for f in os.listdir(datafpath)]
nfpaths = [f'data/data_cleaned/{f}' for f in os.listdir(datafpath)]
# mne ICA is sensitive to low frequencies, so high pass filter it at 1Hz.
# this might however ruin some of the slower oscilating brainwaves,
# such as delta and or theta.

decim = 3
method = 'fastica'
n_components = 25
random_state = 1

bads = 0
nskipped = 0
for idx, (fpath, nfpath) in enumerate(zip(fpaths, nfpaths)):
    ica = ICA(method=method, n_components=n_components, random_state=random_state)
    ica.exclude = []
    raw = mne.io.read_raw_fif(fpath, preload=True)

    raw.filter(1., None, n_jobs=1, fir_design='firwin')
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
    reject = {'mag': 5e-11, 'grad': 4e-11}
    
    try:
        ica.fit(raw, picks=picks, decim=decim, reject=reject, verbose=False)

        eog_epochs = create_eog_epochs(raw)
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)

        ecg_epochs = create_ecg_epochs(raw)
        ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
        
        bads += len(eog_scores)
        bads += len(ecg_scores)

        ica.exclude.extend(eog_inds)
        ica.exclude.extend(ecg_inds)
        ica.apply(raw)
    except:
        print(f'No clean segments found, rejection criteria too strict!..')
        nskipped += 1
    raw.save(nfpath, overwrite=True)

print('Done with all recordings!')
print(f'Skipped {nskipped} recordings due to rejection criteria...')
print(f'Found {bads} bad EOG+ECG components with:')
print(f'ICA, {method=}  {n_components=}')
