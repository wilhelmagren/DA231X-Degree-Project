"""
implementation of BaseSampler parent class for each specific sampler,
built using torch Sampler. 

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""

import numpy as np

from torch.utils.data.sampler import Sampler


class BaseSampler(Sampler):
    def __init__(self, metadata, info, **kwargs):
        self.metadata = metadata
        self._setup(info, **kwargs)

    def __len__(self):
        return self.info['n_samples']

    def __iter__(self):
        for i in range(self.info['n_samples']):
            yield self.sampled[i] if self.info['presample'] else self._sample_pair()
    
    def _setup(self, fullinfo, n_samples=None, batch_size=None, presample=False, seed=1):
        if n_samples is None:
            raise ValueError('number of samples to draw not given')

        if batch_size is None:
            raise ValueError('no batch size given')

        info = dict(sfreq=fullinfo['sfreq'],
                    s_epoch_size=fullinfo['s_epoch_size'],
                    channels=fullinfo['channels'],
                    n_samples=n_samples,
                    batch_size=batch_size,
                    presample=presample)
        rng = np.random.RandomState(seed=seed)
        
        self.info = info
        self.rng = rng

        if presample:
            self._presample()

    def _sample_recording(self):
        return self.rng.randint(0, high=len(self.metadata['data']))
 
    def _sample_pair(self, *args, **kwargs):
        raise NotImplementedError('no window sampling implemented')
   
    def _presample(self):
        self.sampled = list(self._sample_pair() for _ in range(self.info['n_samples']))

