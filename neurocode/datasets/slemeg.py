"""
Dataset implementation for partial-sleep-deprivation MEG 2021 dataset.
Provided as part of the SLEMEG project, in collaboration with 
Karolinska Institutet (KI) and Stockholm University (SU).

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 27-01-2022
"""
import mne
import os
import time

from collections import defaultdict
from torch.utils.data import Dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from braindecode.datasets import BaseConcatDataset, BaseDataset
from ..utils import fetch_meg_data, load_raw_fif


class SLEMEG(BaseConcatDataset):
    def __init__(self, subjects=None, recordings=None,
            preload=False, load_meg_only=True, channels=None):
        if subjects is None:
            subjects = list(range(2, 34))
        if recordings is None:
            recordings = list(range(0, 4))

        super().__init__(self._fetch_and_load(
            subjects, recordings, preload, load_meg_only, channels))

    def _fetch_and_load(self, subjects, recordings, preload, load_meg_only, channels):
        fpaths = fetch_meg_data(subjects, recordings)
        all_base_ds = list()
        for subj_id, reco_id, gender, age, path in fpaths:
            raw, desc = load_raw_fif(
                    path, subj_id, reco_id, preload, drop_channels=load_meg_only, channels=channels)
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)

        return all_base_ds

