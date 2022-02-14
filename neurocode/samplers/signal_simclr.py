"""
TODO: implement the stuff

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from .base import PretextSampler
from ..datautil import CropResize, Permutation


class ContrastiveViewGenerator(object):
    """
    """
    def __init__(self, T, n_views):
        self.transforms = T

        self.n_views = n_views
    
    def __call__(self, x, plot=False):
        first = self.transforms[0](x)
        second = self.transforms[1](x)
        
        if plot:
            fig, axs = plt.subplots(1, 3)
            axs[0].plot(x)
            axs[0].set_title('Original MEG signal')
            axs[1].plot(first)
            axs[1].set_title('Crop&Resize augmentation')
            axs[2].plot(second)
            axs[2].set_title('Permutation augmentation')
            fig.suptitle('MEG signal with two data augmentation equivalents')
            plt.show()

        return [torch.Tensor(self.transforms[t](x)[None]) for t in range(self.n_views)]


class SignalSampler(PretextSampler):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, n_channels, n_views=2, **kwargs):
        self.n_channels = n_channels
        self.n_views = n_views

        self._transforms = [
                CropResize,
                Permutation
                ]

        self.transformer = ContrastiveViewGenerator(
                self._transforms, n_views)

    def _sample_pair(self):
        """
        """
        batch_anchors = list()
        batch_samples = list()
        for _ in range(self.batch_size):
            reco_idx = self._sample_recording()
            wind_idx = self._sample_window(recording_idx=reco_idx)

            x = self.data[reco_idx][wind_idx][0]
            T1, T2 = self.transformer(x[0])
            

            batch_anchors.append(T1.unsqueeze(0))
            batch_samples.append(T2.unsqueeze(0))

        ANCHORS = torch.cat(batch_anchors)
        SAMPLES = torch.cat(batch_samples)

        return (ANCHORS, SAMPLES)

    def extract_features(self, model, device):
        """
        """
        X, Y = [], []
        model.eval()
        model._return_features = True
        with torch.no_grad():
            for recording in range(len(self.data)):
                for window in range(len(self.data[recording])):
                    if window % 10 == 0:
                        window = torch.Tensor(self.data[recording][window][0][None]).float().to(device)
                        feature = model(window.unsqueeze(0))
                        X.append(feature[0, :][None])
                        Y.append(self.labels[recording])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        model._return_features = False

        return (X, Y)
