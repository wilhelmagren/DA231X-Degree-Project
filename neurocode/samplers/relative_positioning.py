"""
Pretext task `Relative Positioning` (RP) implementation as an
extention to the BaseSampler class. Main hyperparameters which
dictate training and learning are: tau_neg & tau_pos. The batch
size indirectly affects learning as well, since it drives loss
down with varying amounts of steps.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""
import torch
import numpy as np

from .base import PretextSampler


class RelativePositioningSampler(PretextSampler):
    def __init__(self, data, info, **kwargs):
        super().__init__(data, info, **kwargs)

    def _parameters(self, tau_neg=30, tau_pos=2, **kwargs):
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos

    def _sample_window(self, recording_idx=None):
        if recording_idx is None:
            recording_idx = self._sample_recording()
        return self.rng.choice(self.info['lengths'][recording_idx])

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
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while np.abs(win_idx1 - win_idx2) < self.tau_neg or win_idx1 == win_idx2:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)
            elif pair_type == 1:
                win_idx2 = self._sample_window(recording_idx=reco_idx)
                while np.abs(win_idx1 - win_idx2) > self.tau_pos or win_idx1 == win_idx2:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)

            batch_anchors.append(self.data[reco_idx][win_idx1][0][:2][None])
            batch_samples.append(self.data[reco_idx][win_idx2][0][:2][None])
            batch_labels.append(float(pair_type))
        
        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))
        
        return (ANCHORS, SAMPLES, LABELS)

