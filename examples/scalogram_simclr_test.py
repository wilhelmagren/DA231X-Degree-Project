"""
SimCLR pipeline for use with the SLEMEG dataset.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 22-02-2022
"""
import logging
from sklearn import manifold
import torch
import numpy as np

from torch.utils.data import DataLoader
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import ScalogramSampler
from neurocode.models import ResNet18, ProjectionHead, load_model
from neurocode.training import SimCLR
from neurocode.datautil import manifold_plot, history_plot
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from pytorch_metric_learning import losses

torch.manual_seed(73)
np.random.seed(73)

manifold = 'tSNE'
load_model_ = False
subjects = list(range(0, 34))
recordings = [0,1,2,3]
batch_size = 64
n_samples = 10
window_size_s = .5
shape = (64, 128)
widths = 50
n_views = 2
n_epochs = 30
temperature = .1
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)
emb_size = 256
latent_size = 100

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)


windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = dataset.split()

samplers = {'train': ScalogramSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_views=n_views, widths=widths, shape=shape, n_samples=n_samples,
    batch_size=batch_size),
           'valid': ScalogramSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_views=n_views, widths=widths, shape=shape, n_samples=n_samples,
    batch_size=batch_size)}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder =  ResNet18()  ##load_model('params.pth')  # ResNet18(1)
model = ProjectionHead(encoder, emb_size, latent_size, dropout=.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(samplers['train']),
        eta_min=0, last_epoch=-1)
criterion = losses.NTXentLoss(temperature=temperature)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=batch_size, epochs=n_epochs, temperature=temperature,
        n_views=n_views)

manifold_plot(samplers['train']._extract_features(encoder, device), 'train-data_pre', technique=manifold)
manifold_plot(samplers['valid']._extract_features(encoder, device), 'valid-data_pre', technique=manifold)

print(f'Training encoder with SimCLR on device={device} for {n_epochs} epochs')
print(f'   epoch       training loss       validation loss         training acc        validation acc')
print(f'------------------------------------------------------------------------------------------------')
history = simclr.fit(samplers, plot=False, save_model=True)
history_plot(history)

manifold_plot(samplers['train']._extract_features(encoder, device), 'train-data_post', technique=manifold)
manifold_plot(samplers['valid']._extract_features(encoder, device), 'valid-data_post', technique=manifold)
