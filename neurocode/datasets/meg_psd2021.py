"""
Dataset implementation for partial-sleep-deprivation MEG 2021 dataset.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 28-11-2021
"""
import mne
import os
import time

from collections import defaultdict
from torch.utils.data import Dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from ..utils import DEFAULT_MEG_CHANNELS, fetch_meg_data


class MEGDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.info = self._setup(**kwargs)
        self.metadata = self._load()

    def _setup(self, window_length=5., channels=None, subjects=None, recordings=None, **kwargs):
       info = dict()
       info['subjects'] = subjects if subjects else list(range(2, 34))
       info['recordings'] = recordings if recordings else list(range(0, 4))
       info['channels'] = channels if channels else DEFAULT_MEG_CHANNELS
       info['window_length'] = window_length
       return info
    
    def _fetch_data(self):
       subject_ids = list(map(lambda s: '0'+str(s) if len(str(x)) != 2 else str(x), self.info['subjects']))
       files = 

    def _load(self):
        fpaths_and_labels = fetch_meg_data(self.info['subjects'], self.info['recordings'])
        raise NotImplementedError


