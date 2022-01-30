"""
Dataset implementation for partial-sleep-deprivation MEG 2021 dataset.
Provided as part of the SLEMEG project, in collaboration with 
Karolinska Institutet (KI) and Stockholm University (SU).

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""
import mne
import os
import time

from collections import defaultdict
from torch.utils.data import Dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.datautil.preprocess import preprocess
from ..utils import DEFAULT_MEG_CHANNELS, fetch_meg_data, load_raw_fif, create_epochs


class SLEMEG(BaseConcatDataset):
    def __init__(self, subjects=None, recordings=None, preload=False,
            load_meg_only=True, preprocessors=None, cleaned=False):
        if subjects is None:
            subjects = list(range(2, 34))
        if recordings is None:
            recordings = list(range(0, 4))

        super().__init__(self._fetch_and_load(
            subjects, recordings, preload, load_meg_only, cleaned))
        
        if preprocessors:
            preprocess(self, preprocessors)

    def _fetch_and_load(self, subjects, recordings, preload, load_meg_only, cleaned):
        fpaths = fetch_meg_data(subjects, recordings, cleaned)
        all_base_ds = list()
        self.labels = list()
        for subj_id, reco_id, gender, age, path in fpaths:
            raw, desc = load_raw_fif(
                    path, subj_id, reco_id, preload, drop_channels=load_meg_only)
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
            self.labels.append((subj_id, reco_id, gender, age))

        return all_base_ds

