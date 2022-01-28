"""
Temporal Shuffling pretext task sampler, implemented based
on H. Banville et al. 2020.

Authors: Wilhelm Ågren <wagren@kht.se>
Last edited: 28-01-2022
"""
import numpy as np

from . import PretextSampler


class TemporalShufflingSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, tau_neg=15, tau_pos=4, **kwargs):
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos

    def _valid_anchors(self, win_idx1, win_idx3):
        conditions = [
                win_idx1 < win_idx3,
                np.abs(win_idx1 - win_idx3) < self.tau_pos,
                np.abs(win_idx1 - win_idx3) > 1]

        return all(conditions)

    def _sample_pair(self):
        batch_lanchors = list()
        batch_ranchors = list()
        batch_samples = list()
        batch_labels = list()
        reco_idx = self._sample_recording()
        for _ in range(self.batch_size):
            win_idx1 = self._sample_window(recording_idx=reco_idx)
            win_idx3 = self._sample_window(recording_idx=reco_idx)
            
            while win_idx1 >= self.info['lengths'][reco_idx] - self.tau_pos:
                win_idx1 = self._sample_window(recording_idx=reco_idx)
            while not self._valid_anchors(win_idx1, win_idx3):
                win_idx3 = self._sample_window(recording_idx=reco_idx)

            pair_type = self.rng.binomial(1, .5)
            win_idx2 = self._sample_window(recording_idx=reco_idx)

            if pair_type == 0: # negative sample
                while win_idx1 < win_idx2 and win_idx2 < win_idx3:
                    win_idx2 = self._sample_window(recording_idx=reco_idx)
            elif pair_type == 1: # positive sample
                while not ((win_idx1 < win_idx2) and (win_idx2 < win_idx3)):
                    win_idx2 = self._sample_window(recording_idx=reco_idx)

            batch_lanchors.append(self.data[reco_idx][win_idx1][0][None])
            batch_ranchors.append(self.data[reco_idx][win_idx3][0][None])
            batch_samples.append(self.data[reco_idx][win_idx2][0][None])
            batch_labels.append(float(pair_type))

        LANCHORS = np.concatenate(batch_ranchors, axis=0)
        RANCHORS = np.concatenate(batch_lanchors, axis=0)
        SAMPLES = np.concatenate(batch_samples, axis=0)
        LABELS = np.array(batch_labels)

        return (LANCHORS, RANCHORS, SAMPLES, LABELS)

