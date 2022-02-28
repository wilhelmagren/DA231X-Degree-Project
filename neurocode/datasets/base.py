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

    def split_fixed(self):
        train_indices = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99, 104, 105, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 52, 53, 54]
        valid_indices = [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 55, 84, 85, 86, 87, 92, 93, 94, 95, 100, 101, 102, 103]

        X_train = {idx: self.data[k] for idx, k in enumerate(train_indices)}
        Y_train = {idx: self.labels[k] for idx, k in enumerate(train_indices)}
        X_valid = {idx: self.data[k] for idx, k in enumerate(valid_indices)}
        Y_valid = {idx: self.labels[k] for idx, k in enumerate(valid_indices)}

        train_dataset = RecordingDataset(X_train, Y_train, formatted=True, sfreq=self.info['sfreq'])
        valid_dataset = RecordingDataset(X_valid, Y_valid, formatted=True, sfreq=self.info['sfreq'])
        return (train_dataset, valid_dataset)


