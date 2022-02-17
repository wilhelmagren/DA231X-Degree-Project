"""
Temporal Shuffling pretext task, implemented as detailed in
Banville et al. 2020, 

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 17-02-2022
"""
import torch
import numpy as np

from .base import PretextSampler


class TSSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _parameters(self, tau_pos=5, tau_neg=5, **kwargs):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
    
    def _sample_pair(self):
        """
        """
        batch_anchors = list()
        batch_middles = list()
        batch_samples = list()
        batch_labels = list()

        for _ in range(self.batch_size):
            pair_type = self.rng.binomial(1, .5)
            reco_idx = self._sample_recording()
            anchor_idx = self._sample_window(recording_idx=reco_idx)
            sample_idx = self._sample_window(recording_idx=reco_idx)
            middle_idx = self._sample_window(recording_idx=reco_idx)

            # Resample the other anchor index until it is inside the positive 
            # context relative to the anchor. 
            while ((np.abs(anchor_idx - sample_idx) >= self.tau_pos)
                or (anchor_idx == sample_idx)
                or (np.abs(anchor_idx - sample_idx) < 2)):
                sample_idx = self._sample_window(recording_idx=reco_idx)
            
            if pair_type == 0:
                # Negative sample, they are not ordered
                while ((anchor_idx <= middle_idx <= sample_idx)
                    or (sample_idx <= middle_idx <= anchor_idx)):
                    middle_idx = self._sample_window(recording_idx=reco_idx)

            elif pair_type == 1:
                # Positive sample, they are ordered
                while ((not(anchor_idx <= middle_idx <= sample_idx) and
                        not(anchor_idx >= middle_idx >= sample_idx))
                    or (middle_idx == anchor_idx or middle_idx == sample_idx)):
                    middle_idx = self._sample_window(recording_idx=reco_idx)
            
            batch_anchors.append(self.data[reco_idx][anchor_idx][0][None])
            batch_middles.append(self.data[reco_idx][middle_idx][0][None])         
            batch_samples.append(self.data[reco_idx][sample_idx][0][None])
            batch_labels.append(float(pair_type))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        MIDDLES = torch.Tensor(np.concatenate(batch_middles, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, MIDDLES, SAMPLES, LABELS)