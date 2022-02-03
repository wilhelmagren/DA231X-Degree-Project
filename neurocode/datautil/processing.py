"""
Collection of processing and transformation functions for the
processor module.

Work in progress!!!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import numpy as np

from functools import partial
from joblib import Parallel, delayed
from braindecode.datasets.base import BaseConcatDataset


class Preprocessor(object):
    def __init__(self, fn, *, apply_on_array=True, **kwargs):
        if callable(fn) and apply_on_array:
            channel_wise = kwargs.pop('channel_wise', False)
            picks = kwargs.pop('picks', None)
            n_jobs = kwargs.pop('n_jobs', 1)
            kwargs = {'fun': partial(fn, **kwargs), 
                      'channel_wis': channel_wise,
                      'picks': picks, 'n_jobs': n_jobs}
            fn = 'apply_function'
        self.fn = fn
        self.kwargs = kwargs

    def apply(self, raw_or_epochs):
        try:
            self._try_apply(raw_or_epochs)
        except RuntimeError:
            # most likely data wasn't loaded yet, some MNE
            # functions require the data to be preloaded, 
            # but not preloading the raw data can substantially
            # increase speed of the preprocessing pipeline...
            raw_or_epochs.load_data()
            self._try_apply(raw_or_epochs)

    def _try_apply(self, raw_or_epochs):
        if callable(self.fn):
            self.fn(raw_or_epochs, **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(
                        f'MNE objects does not have a {self.fn} method.')
                getattr(raw_or_epochs, self.fn)(**self.kwargs)
    

def preprocess(concat_ds, preprocessors, n_jobs=None):
    """func applies all preprocessor to the provided dataset.
    Parallelization availble from `joblib`.
    """
    for func in preprocessors:
        assert hasattr(func, 'apply'), (
                'preprocessor object needs an `apply` method.')
    
    list_of_ds = Parallel(n_jobs=n_jobs)(
            delayed(_preprocess)(ds, i, preprocessors)
            for i, ds in enumerate(concat_ds.datasets))

    if n_jobs is None or n_jobs == 1:
        concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)
    else:
        _replace_inplace(concat_ds, BaseConcatDataset(list_of_ds))
    
    return concat_ds

def _replace_inplace(concat_ds, new_concat_ds):
    """func replaces the subdatasets and preprocessor kwargs 
    of a BaseConcatDatset, inplace
    """
    assert len(concat_ds) == len(new_concat_ds), (
            'Both inputs must have the same length.')
    for i in range(len(new_concat_ds.datasets)):
        concat_ds.datasets[i] = new_concat_ds.datasets[i]

    concat_kind = 'raw' if hasattr(concat_ds.datasets[0], 'raw') else 'window'
    preproc_kwargs_attr = concat_kind + '_preproc_kwargs'
    if hasattr(new_concat_ds, preproc_kwargs_attr):
        setattr(concat_ds, preproc_kwargs_attr,
                getattr(new_concat_ds, preproc_kwargs_attr))


def _preprocess(ds, ds_idx, preprocessors):
    """
    """
    def _preprocess_raw_or_epochs(raw_or_epochs, preprocessors):
        for preproc in preprocessors:
            preproc.apply(raw_or_epochs)

    if hasattr(ds, 'raw'):
        _preprocess_raw_or_epochs(ds.raw, preprocessors)
    elif hasattr(ds, 'windows'):
        _preprocess_raw_or_epochs(ds.windows, preprocessors)
    else:
        raise ValueError(
                'Can only preprocess ConcatDataset with'
                'either `raw` or `windows` attribute.')

    return ds


def zscore(X):
    """func applies zscoring/standard scoring on the provided data inplace.
    Works on both continuous and discrete data, expects X to not be too big,
    mostly performed on windowed data after making RecordingDatasets.

    Parameters
    ----------
    X: np.array
        The data to which we want to apply the standardization. Dimensionality
        and shape does not matter, as we are only interested in the mean and std.
     
    Returns
    -------
    zscored:
        Standardized version of the input argument data.
    
    """
    zscored = X - np.mean(X, keepdims=True, axis=-1)
    zscored = zscored / np.std(zscored, keepdims=True, axis=-1)
    return zscored
