"""
Scalogram-Signal Temporal Ordering Sampler, for the
novel pretext task based on previously detailed
tasks from literature.

Authors: Wilhelm Ågren, wagren@kth.se
Last edited: 26-01-2022
"""
import torch
import numpy as np

from .base import PretextSampler


class SSTOSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, tau=4, **kwargs):
        self.tau = tau

    def _sample_pair(self):
        batch_anchors = list()
        batch_samples = list()
        batch_labels = list()
        reco_idx = self._sample_recording()
        for _ in range(self.batch_size):
            win_idx1 = self._sample_window(recording_idx=reco_idx)
            pair_type = self.rng.binomial(1, .5)
            win_idx2 = -1

            if pair_type == 0:
                while win_idx1 - self.tau <= 0:
                    win_idx1 = self._sample_window(recording_idx=reco_idx)

                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while win_idx2 >= win_idx1 or np.abs(win_idx2 - win_idx1) >= self.tau:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)
            
            elif pair_type == 1:
                while win_idx1 + self.tau >= self.info['lengths'][reco_idx]:
                    win_idx1 = self._sample_window(recording_idx=reco_idx)

                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while win_idx2 <= win_idx1 or np.abs(win_idx2 - win_idx1) >= self.tau:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)

            batch_anchors.append(self.data[reco_idx][win_idx1][0][None])
            batch_samples.append(self.data[reco_idx][win_idx2][0][None])
            batch_labels.append(float(pair_type))

        ANCHORS = np.concatenate(batch_anchors, axis=0)
        SAMPLES = np.concatenate(batch_samples, axis=0)
        LABELS = np.array(batch_labels)

        return (ANCHORS, SAMPLES, LABELS)

