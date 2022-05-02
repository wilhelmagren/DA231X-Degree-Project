"""
implementation of Pretext task sampler parent class for each specific sampler,
built using torch Sampler. 

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 31-01-2022
"""
import torch
import numpy as np

from torch.utils.data.sampler import Sampler
from collections import defaultdict
from ..datasets import RecordingDataset


class PretextSampler(Sampler):
    def __init__(self, data, labels, info, **kwargs):
        self.data = data
        self.labels = labels
        self.info = info
        self._parameters(**kwargs)
        self._setup(**kwargs)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for i in range(self.n_samples):
            yield self.samples[i] if self.presample else self._sample_pair()

    def _setup(self, seed=1, n_samples=256, batch_size=32, presample=False, **kwargs):
        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.presample = presample

        if presample:
            self._presample()
 
    def _extract_features(self, emb, device, n_samples_per_recording=None):
        X, Y = [], []
        emb.eval()
        emb._return_features = True
        with torch.no_grad():
            for reco_idx in range(len(self.data)):
                for idx, window in enumerate(self.data[reco_idx]):
                    window = torch.Tensor(window[0][None]).to(device)
                    embedding = emb(window)
                    X.append(embedding[0, :][None])
                    Y.append(self.labels[reco_idx])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        emb._return_features = False
        return (X, Y)
    
    def downstream_sample(self, emb, device):
        X, y = [], []
        emb.eval()
        emb.return_feats = True
        

    def _parameters(self, *args, **kwargs):
        raise NotImplementedError(
                'Please implement setup for parameters of pretext-task sampling!')

    def _presample(self):
        self.samples = list(self._sample_pair() for _ in range(self.n_samples))
 
    def _sample_recording(self):
        return self.rng.randint(0, high=self.info['n_recordings'])
    
    def _sample_window(self, recording_idx=None, **kwargs):
        if recording_idx is None:
            recording_idx = self._sample_recording()
        return self.rng.choice(self.info['lengths'][recording_idx])
   
    def _sample_pair(self, *args, **kwargs):
        raise NotImplementedError(
                'Please implement window-pair sampling!')
  
    def _split(self):
        X_train = defaultdict(list)
        Y_train = defaultdict(list)
        X_valid = defaultdict(list)
        Y_valid = defaultdict(list)
        for recording in range(self.info['n_recordings']):
            r_len = len(self.data[recording])
            split = np.ceil(r_len * 0.7).astype(int)
            for window in range(split):
                X_train[recording].append(self.data[recording][window])
            Y_train[recording].append(self.labels[recording])
            for window in range(split, r_len):
                X_valid[recording].append(self.data[recording][window])
            Y_valid[recording].append(self.labels[recording])
        
        train_dataset = RecordingDataset(X_train, Y_train, formatted=True, sfreq=self.info['sfreq'])
        valid_dataset = RecordingDataset(X_valid, Y_valid, formatted=True, sfreq=self.info['sfreq'])
        return (train_dataset, valid_dataset)
