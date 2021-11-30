"""
implementation of Pretext task sampler parent class for each specific sampler,
built using torch Sampler. 

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""

import numpy as np

from torch.utils.data.sampler import Sampler


class PretextSampler(Sampler):
    def __init__(self, data, info, **kwargs):
        self.data = data
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

 
    def _parameters(self, *args, **kwargs):
        raise NotImplementedError(
                'Please implement setup for parameters of pretext-task sampling!')

    def _presample(self):
        self.samples = list(self._sample_pair() for _ in range(self.n_samples))
 
    def _sample_recording(self):
        return self.rng.randint(0, high=self.info['n_recordings'])
    
    def _sample_window(self, *args, **kwargs):
        raise NotImplementedError(
                'Please implement window sampling!')
   
    def _sample_pair(self, *args, **kwargs):
        raise NotImplementedError(
                'Please implement window-pair sampling!')
  
