"""
Base dataset class to wrap the concat datasets created when making 
fixed time length windows using braindecode. Access window in recording
by specifying recording_idx and window_idx.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""
import numpy as np

from torch.utils.data import Dataset


class RecordingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self._setup(*args, **kwargs)
    
    def __len__(self):
        return self.info['n_recordings']

    def __getitem__(self, indices):
        recording, window = indices
        return self.data[recording][window]

    def __iter__(self):
        for idx in range(len(self)):
            yield (self.data[idx], self.labels[idx])

    def _setup(self, datasets, labels, formatted=False, **kwargs):

        if not formatted:
            datasets = {recording: dataset for recording, dataset in enumerate(datasets)}
            labels = {recording: label for recording, label in enumerate(labels)}

        lengths = {recording: len(dataset) for recording, dataset in enumerate(datasets.values())}
        info = {'lengths': lengths, 'n_recordings': len(datasets)}
        info = {**info, **kwargs}

        self.data = datasets
        self.labels = labels
        self.info = info
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels

    def get_info(self):
        return self.info

    def split(self, split=.6, shuffle=True):
        split_idx = int(len(self) * split)
        indices = list(range(len(self)))
        
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        X_train = {idx: self.data[k] for idx, k in enumerate(train_indices)}
        Y_train = {idx: self.labels[k] for idx, k in enumerate(train_indices)}
        X_valid = {idx: self.data[k] for idx, k in enumerate(valid_indices)}
        Y_valid = {idx: self.labels[k] for idx, k in enumerate(valid_indices)}

        train_dataset = RecordingDataset(X_train, Y_train, formatted=True, sfreq=self.info['sfreq'])
        valid_dataset = RecordingDataset(X_valid, Y_valid, formatted=True, sfreq=self.info['sfreq'])
        return (train_dataset, valid_dataset)

