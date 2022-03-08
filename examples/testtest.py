import torch
import numpy as np

from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import SignalSampler
from neurocode.models import SignalNet
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore

torch.manual_seed(73)
np.random.seed(73)

load_model_ = False
subjects = list(range(30, 34))
recordings = [0,1,2,3]
n_samples = 20
batch_size = 32
window_size_s = 5.
n_channels = 3
n_views = 2
n_epochs = 50
temperature = .1
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)
emb_size = 256
latent_size = 100

preprocessors = [Preprocessor(lambda x: x*1e12)]
dset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows_dataset = create_fixed_length_windows(dset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dset.labels, sfreq=sfreq, channels='MEG')

sampler = SignalSampler(dataset.get_data(), dataset.get_labels(),
    dataset.get_info(), n_channels=n_channels, n_views=n_views, n_samples=n_samples, batch_size=batch_size)

for _ in sampler:
    pass