"""
TODO: handle when second recording is shorter than anchor!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch
import numpy as np

from .base import BaseSampler


class RecordingRelativePositioningSampler(BaseSampler):
    def __init__(self, metadata, info, tau=5, gamma=.5, **kwargs):
        self.tau = tau
        self.gamma = gamma
        super().__init__(metadata, info, **kwargs)

    def _sample_pair(self, recording_idx=None):
        if recording_idx is None:
            recording_idx = self._sample_recording()

        batch_anchors = list()
        batch_samples = list()
        batch_labels = list()
        for _ in range(self.info['batch_size']):
            anchor_idx = self.rng.randint(0, len(self.metadata['data'][recording_idx]))
            same_recording = self.rng.binomial(1, self.gamma)
            sample_idx = None
            sample_recording_idx = recording_idx

            if same_recording == 0:
                # Sample from the same recording
                sample_idx = self.rng.randint(max(0, anchor_idx - self.tau), min(anchor_idx + self.tau, len(self.metadata['data'][recording_idx])))
                while sample_idx == anchor_idx:
                    sample_idx = self.rng.randint(max(0, anchor_idx - self.tau), min(anchor_idx + self.tau, len(self.metadata['data'][recording_idx])))
                
            elif same_recording == 1:
                # Sample from another recording, either previous or next
                sample_recording_idx = recording_idx + 1
                if sample_recording_idx >= len(self.metadata['data']):
                    sample_recording_idx = 0

                if anchor_idx >= len(self.metadata['data'][sample_recording_idx]):
                    continue

                sample_idx = self.rng.randint(max(0, anchor_idx - self.tau), min(anchor_idx + self.tau, len(self.metadata['data'][sample_recording_idx])))
            
            batch_anchors.append(self.metadata['data'][recording_idx][anchor_idx][None])
            batch_samples.append(self.metadata['data'][sample_recording_idx][sample_idx][None])
            batch_labels.append(float(same_recording))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, SAMPLES, LABELS)

