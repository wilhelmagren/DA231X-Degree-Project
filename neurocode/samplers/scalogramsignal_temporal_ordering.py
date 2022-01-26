"""

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
            win_idx1 = self._sample_window(recording_idx=reco_dix)
            pair_type = self.rng.binomial(1, .5)
            win_idx2 = -1

            if pair_type == 0:
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while win_idx2 >= win_idx1:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)
            
            elif pair_type == 1:
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while win_idx2 <= win_idx1:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)

            batch_anchors.append(self.data[reco_idx][win_idx1][0][:1][None])
            batch_samples.append(self.data[reco_idx][win_idx2][0][:1][None])
            batch_labels.append(float(pair_type))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, SAMPLES, LABELS)

