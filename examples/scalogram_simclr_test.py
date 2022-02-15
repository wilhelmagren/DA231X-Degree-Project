"""
SimCLR pipeline for use with the SLEMEG dataset.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import sys
import logging
import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import ScalogramSampler
from neurocode.models import ResNet18, SimCLR, ShallowSimCLR
from neurocode.training import SimCLR
from neurocode.datautil import tSNE_plot, history_plot
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from pytorch_metric_learning import losses


subjects = list(range(0, 33))
recordings = [0,1,2,3]
batch_size = 128
n_samples = 50
window_size_s = .5
input_shape = (1, 96, 96)
widths = 30
n_views = 2
n_epochs = 30
temperature = .5
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = dataset.split(split=.7)

samplers = {'train': ScalogramSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_views=n_views, widths=widths, shape=(96, 96), n_samples=n_samples,
    batch_size=batch_size),
           'valid': ScalogramSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_views=n_views, widths=widths, shape=(96, 96), n_samples=n_samples,
    batch_size=batch_size)}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ShallowSimCLR(input_shape, sfreq, n_filters=32, dropout=.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(samplers['train']),
        eta_min=0, last_epoch=-1)
criterion = losses.NTXentLoss(temperature=temperature)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=batch_size, epochs=n_epochs, temperature=temperature,
        n_views=n_views)

logging.info('Extracting pre-training features...')
tSNE_plot(samplers['valid'].extract_features(model, device), 'pre')

print(f'Training encoder with SimCLR on device={device} for {n_epochs} epochs')
print(f'   epoch       training loss       validation loss         training acc        validation acc')
print(f'------------------------------------------------------------------------------------------------')
history = simclr.fit(samplers, plot=False)
history_plot(history)

logging.info('Extracting post-training features...')
tSNE_plot(samplers['valid'].extract_features(model, device), 'post')

