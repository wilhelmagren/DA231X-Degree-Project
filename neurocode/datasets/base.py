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
        """
        #train_indices = [73, 114, 38, 14, 48, 125, 113, 81, 15, 12, 122, 119, 68, 124, 78, 79, 49, 110, 59, 39, 118, 77, 62, 75, 3, 51, 61 ,130,  65, 106,  72,  32, 117,  83 ,127, 60,  66,  50, 111,  27,  26, 104,  70,  37,  63,  82, 115,  98,  90,  88, 108,   2,  25, 96,  58,  64,  13,  80,  36,  89,  34, 105, 126,  91, 128,  57,  56, 121,  33,  99,  30, 85,  43,   7, 102,  42,  47,   9,  10,  45,  53,  29,  52,   4,  54,   6,  46,  92,  11, 87,  22,  23, 100, 103,  17,  84,  95,  41,   8,  19,  40,   5,  20,  93,  18,  31]
        #valid_indices = [107, 112,  97, 123,  24,   1,  74,  35,  69,  76, 120,   0,  71, 109, 116, 129,  67,  28, 44,  16, 101,  21,  55,  86,  94]
        train_indices = [ 24,  61,  56, 104, 120,  97, 108,  79,  15,  58,   1,  26,  75, 113, 105,  78,  68,  66,
  99,  62,  90,  38,  35,  63,  36,  37, 116, 127,   3, 107, 115, 130,  60,  80,  65,  83,
  59, 122,  77,  12,  67, 129, 106, 117,  70,  98,  33,  89,  13,  82, 114,  27,  53,
 102,  43,  40,  41,  10,   6,  52,   7,  95,  31,   5,  20,  94,  42,  45,  18,  87,  19,
 101,  17,   9,  28,  21,  44,  54,  85]
        valid_indices = [ 69, 119,  96,  51,  74, 125,  25, 111,  32,  71,  88, 126,  91, 109,  39,  64,  81,  72,
  57, 112,  34, 110,   0,  49,  73, 128, 121,  14,   2, 123,  50, 124,  48, 118,  76,  11,
  47,  46,  22,  29,  30,   8,   4, 100,  16,  92,  55,  23,  86,  84, 103,  93]
        """
        train_indices = [117,  58,  89,  73,  69,   2, 118,  79,  90,  83, 106,  67, 104, 116, 122,  36,  12,  82,
  51, 115, 109,  78,  76,  57, 110,  48,  59,  37, 114,  25,  77, 108,  66, 125,  72, 127,
  14,  97,  81, 121, 105,  70,  15, 107,   1,  68,  63, 62,  61,  75,  64,  98, 112,
   3,  65, 123, 126, 119, 111,  32, 130,  94,  11,  86,   5,  29,   8,  47,  19,  52,  46,
  95,  55,  84,  23,  53,  93, 102,  45, 101,   4,   7,  30,  87,   6,  22,  20,  28,  10,
 100,  43,  85]
        valid_indices = [120,  26, 129,  60,  99, 113,  35,   0,  34,  88,  38,  96,  50, 128, 124,  33,  24,  56,
  91,  71,  13,  74,  80,  49,  27,  39,  18,  41,  31,  44,  17,  16, 103,  40,  42,  92,
   9,  54,  21]
        X_train = {idx: self.data[k] for idx, k in enumerate(train_indices)}
        Y_train = {idx: self.labels[k] for idx, k in enumerate(train_indices)}
        X_valid = {idx: self.data[k] for idx, k in enumerate(valid_indices)}
        Y_valid = {idx: self.labels[k] for idx, k in enumerate(valid_indices)}

        train_dataset = RecordingDataset(X_train, Y_train, formatted=True, sfreq=self.info['sfreq'])
        valid_dataset = RecordingDataset(X_valid, Y_valid, formatted=True, sfreq=self.info['sfreq'])
        return (train_dataset, valid_dataset)


    def split_signal(self):
        X_train = {idx: self.split_train(self.data[k]) for idx, k in enumerate(range(len(self.data)))}
        Y_train = {idx: self.labels[k] for idx, k in enumerate(range(len(self.data)))}
        X_valid = {idx: self.split_valid(self.data[k]) for idx, k in enumerate(range(len(self.data)))}
        Y_valid = {idx: self.labels[k] for idx, k in enumerate(range(len(self.data)))}

        train_dataset = RecordingDataset(X_train, Y_train, formatted=True, sfreq=self.info['sfreq'])
        valid_dataset = RecordingDataset(X_valid, Y_valid, formatted=True, sfreq=self.info['sfreq'])
        return (train_dataset, valid_dataset)

    def split_train(self, data):
        n_samples = len(data)
        ind = np.ceil(n_samples * 0.7).astype(int)
        return data[:ind]
    
    def split_valid(self, data):
        n_samples = len(data)
        ind = np.ceil(n_samples * 0.7).astype(int)
        return data[ind:]