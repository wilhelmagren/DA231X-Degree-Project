"""
Base dataset class to wrap the concat datasets created when making 
fixed time length windows using braindecode. Access window in recording
by specifying recording_idx and window_idx.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 30-11-2021
"""
from torch.utils.data import Dataset

class RecordingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.datasets, self.info = self._setup(*args, **kwargs)
    
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, indices):
        recording, window = indices
        return self.datasets[recording][window]

    def _setup(self, datasets, **kwargs):
        dataset = {recording: dataset for recording, dataset in enumerate(datasets)}
        lengths = {recording: len(dataset) for recording, dataset in enumerate(datasets)}
        info = dict(lengths=lengths, n_recordings=len(dataset))
        info = {**info, **kwargs}
        return dataset, info
    
    def get_data(self):
        return self.datasets

    def get_info(self):
        return self.info

