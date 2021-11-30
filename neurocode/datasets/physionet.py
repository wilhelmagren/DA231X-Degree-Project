"""
yes.


Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""
import mne
import os
import time

from torch.utils.data import Dataset
from ..utils import fetch_eeg_data, load_raw_edf, create_epochs


class Physionet(Dataset):
    def __init__(self, *args, **kwargs):
        self.info = self._setup(**kwargs)
        self.metadata = self._load()

    def _setup(self, *args, **kwargs):
        raise NotImplementedError('svartpeppar')
    
    def _load(self):
        raise NotImplementedError('svartpeppera')

