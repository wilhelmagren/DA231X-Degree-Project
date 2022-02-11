"""
SimCLR pipeline for use with the SLEMEG dataset.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 10-02-2022
"""
import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import WaveletSimCLR
from neurocode.models import ResNet18, SimNet, ShallowSimCLR
from neurocode.training import SimCLR
from neurocode.datautil import tSNE_plot, history_plot
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from pytorch_metric_learning import losses


subjects = list(range(5,10))
recordings = [0,1,2,3]
batch_size = 128
window_size_s = .5
input_shape = (1, 96, 96)
widths = 50
n_views = 2
n_epochs = 20
temperature = .7
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
train_dataset, valid_dataset = dataset.split(split=.8)

loaders = {'train': WaveletSimCLR(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_views=n_views, widths=widths, shape=(96, 96), premake=False, transform=True),
           'valid': WaveletSimCLR(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_views=n_views, widths=widths, shape=(96, 96), premake=False, transform=True)}

for key in loaders:
    print(f'Remaking windows using CWT on {key} dataset...')
    loaders[key]._remake_dataset()

dataloaders = {'train': DataLoader(loaders['train'], batch_size=batch_size, shuffle=True,
                    num_workers=1, drop_last=True),
               'valid': DataLoader(loaders['valid'], batch_size=batch_size, shuffle=True,
                    num_workers=1, drop_last=True)}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ShallowSimCLR(input_shape, sfreq, n_filters=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders['train']),
        eta_min=0, last_epoch=-1)
criterion = losses.NTXentLoss(temperature=temperature)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=batch_size, epochs=n_epochs, temperature=temperature,
        n_views=n_views)

print(f'Extracting pre-training embeddings...')
tSNE_plot(loaders['train']._extract_embeddings(model, device), 'pre')

history = simclr.fit(dataloaders)
history_plot(history)

print(f'Extracting post-training embeddings...')
tSNE_plot(loaders['train']._extract_embeddings(model, device), 'post')

