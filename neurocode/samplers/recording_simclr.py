import torch
import numpy as np

from .base import PretextSampler


class RecordingSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _parameters(self, n_channels, n_views=2, **kwargs):
        self.n_channels = n_channels
        self.n_views = n_views

    def _sample_pair(self):
        batch_anchors = []
        batch_samples = []
        recordings = self.rng.choice(self.info['n_recordings'], size=(self.batch_size), replace=False)
        for reco_idx1 in recordings:
            win_idx1 = self._sample_window(recording_idx=reco_idx1)
            win_idx2 = self._sample_window(recording_idx=reco_idx1)

            batch_anchors.append(self.data[reco_idx1][win_idx1][0][None])
            batch_samples.append(self.data[reco_idx1][win_idx2][0][None])
        
        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
    
        return (ANCHORS, SAMPLES)
