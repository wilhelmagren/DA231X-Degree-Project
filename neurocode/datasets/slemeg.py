"""
Dataset implementation for partial-sleep-deprivation MEG 2021 dataset.
Provided as part of the SLEMEG project, in collaboration with 
Karolinska Institutet (KI) and Stockholm University (SU).

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import mne
import os
import time

from collections import defaultdict
from torch.utils.data import Dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from ..utils import DEFAULT_MEG_CHANNELS, fetch_meg_data, load_raw_fif, create_epochs


class SLEMEG(Dataset):
    def __init__(self, *args, **kwargs):
        self.info = self._setup(**kwargs)
        self.metadata = self._load()

    def __len__(self):
        return self.n_epochs

    def __getitem__(self, item):
        recording_idx, i_epoch = item
        return (self.metadata['data'][recording_idx][i_epoch], self.metadata['labels'][recording_idx])

    def _setup(self, s_epoch_size=5., sfreq=200, channels=None, subjects=None, recordings=None, **kwargs):
       info = dict()
       info['subjects'] = subjects if subjects else list(range(2, 34))
       info['recordings'] = recordings if recordings else list(range(0, 4))
       info['channels'] = channels if channels else DEFAULT_MEG_CHANNELS
       info['s_epoch_size'] = s_epoch_size
       info['sfreq'] = sfreq
       return info

    def _load(self):
        metadata = dict(data=dict(), labels=dict())
        fpaths_and_labels = fetch_meg_data(self.info['subjects'], self.info['recordings'])
        for recording_idx, (subject_id, recording_id, gender, age, fpath) in enumerate(fpaths_and_labels):
            raw = load_raw_fif(fpath, self.info, drop_channels=True)
            epochs = create_epochs(raw, self.info, s_start_trials=5., s_end_trials=5.)
            metadata['data'][recording_idx] = epochs
            metadata['labels'][recording_idx] = (recording_id, gender, age)

        self.n_recordings = len(metadata['data'])
        self.n_epochs = sum(list(len(epochs) for epochs in metadata['data'].values()))
        
        print(f'loaded {self.n_recordings} MEG recordings')
        print(f'total number of epochs: {self.n_epochs}')
        return metadata

