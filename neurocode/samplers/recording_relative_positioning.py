"""
TODO: handle when second recording is shorter than anchor!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch
import numpy as np

from .base import PretextSampler


class RecordingRelativePositioningSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, tau=5, gamma=.5, **kwargs):
        self.tau = tau
        self.gamma = gamma

    def _sample_pair(self):
        batch_anchors = list()
        batch_samples = list()
        batch_labels = list()
        reco_idx1 = self._sample_recording()
        for _ in range(self.batch_size):
            win_idx1 = self._sample_window(recording_idx=reco_idx1)
            pair_type = self.rng.binomial(1, self.gamma)
            reco_idx2, win_idx2 = -1, -1

            if pair_type == 0:
                # sample from other recording
                reco_idx2 = self._sample_recording()
                while reco_idx1 == reco_idx2:
                    reco_idx2 = self._sample_recording()


                win_idx2 = self._sample_window(recording_idx=reco_idx2)
                while np.abs(win_idx1 - win_idx2) > self.tau or win_idx1 == win_idx2:
                    win_idx2 = self._sample_window(recording_idx=reco_idx2)
            elif pair_type == 1:
                # sample from the same recording
                reco_idx2 = reco_idx1
                win_idx2 = self._sample_window(recording_idx=reco_idx2)
                while np.abs(win_idx1 - win_idx2) > self.tau or win_idx1 == win_idx2:
                    win_idx2 = self._sample_window(recording_idx=reco_idx2)

            batch_anchors.append(self.data[reco_idx1][win_idx1][0][:2][None])
            batch_samples.append(self.data[reco_idx2][win_idx2][0][:2][None])
            batch_labels.append(float(pair_type))

        ANCHORS = torch.Tensor(np.concatenate(batch_anchors, axis=0))
        SAMPLES = torch.Tensor(np.concatenate(batch_samples, axis=0))
        LABELS = torch.Tensor(np.array(batch_labels))

        return (ANCHORS, SAMPLES, LABELS)

