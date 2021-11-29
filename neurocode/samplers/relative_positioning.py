"""
Pretext task `Relative Positioning` (RP) implementation as an
extention to the BaseSampler class. Main hyperparameters which
dictate training and learning are: tau_neg & tau_pos. The batch
size indirectly affects learning as well, since it drives loss
down with varying amounts of steps.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch
import numpy as np

from .base import BaseSampler


class RelativePositioningSampler(BaseSampler):
    def __init__(self, metadata, info, tau_neg=10, tau_pos=4, **kwargs):
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos
        super().__init__(metadata, info, **kwargs)

    def _sample_pair(self, recording_idx=None):
        if not recording_idx:
            recording_idx = self._sample_recording()

        batch_anchors = list()
        batch_samples = list()
        batch_labels = list()
        for _ in range(self.info['batch_size']):
            anchor_idx = self.rng.randint(0, len(self.metadata['data'][recording_idx]))
            label = self.rng.binomial(1, .5)
            sample_idx = None

            if label == 0:
                # Sample from the negative context
                low = max(0, anchor_idx - self.tau_pos - self.tau_neg)
                high = min(anchor_idx + self.tau_pos + self.tau_neg + 1, len(self.metadata['data'][recording_idx]))
                sample_idx = self.rng.randint(low, high)
                while anchor_idx - self.tau_pos <= sample_idx <= anchor_idx + self.tau_pos:
                    sample_idx = self.rng.randint(low, high)
            elif label == 1:
                # Sample from the positive context
                sample_idx = self.rng.randint(max(0, anchor_idx - self.tau_pos), min(anchor_idx + self.tau_pos, len(self.metadata['data'][recording_idx])))
                while sample_idx == anchor_idx:
                    sample_idx = self.rng.randint(max(0, anchor_idx - self.tau_pos), min(anchor_idx + self.tau_pos, len(self.metadata['data'][recording_idx])))

            batch_anchors.append(self.metadata['data'][recording_idx][anchor_idx][None])
            batch_samples.append(self.metadata['data'][recording_idx][sample_idx][None])
            batch_labels.append(float(label))
        
        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, SAMPLES, LABELS)

