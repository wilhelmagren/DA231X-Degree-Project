"""
"""

import numpy as np

from torch.utils.data.sampler import Sampler


class BaseSampler(Sampler):
    def __init__(self, metadata, *args, **kwargs):
        self.metadata = metadata
        self.info = self._setup()

    def __len__(self):
        raise NotImplementedError('no assigned length to the sampler!')

    def __iter__(self):
        raise NotImplementedError('no iteration implemented for sampler!')
    
    def _setup(self):
        raise NotImplementedError

    def _sample_recording(self):
        raise NotImplementedError 

