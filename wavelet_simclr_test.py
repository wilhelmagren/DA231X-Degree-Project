"""
SimCLR pipeline for use with the SLEMEG dataset.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 10-02-2022
"""
import torch
import numpy as np

from torch.utils.data import DataLoader
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import WaveletSimCLR
from neurocode.models import ResNetSimCLR
from neurocode.training import SimCLR
from neurocode.datautil import tSNE_plot
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore

window_size_s = .5
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=[2,3,4,5,6], recordings=[0,1,2,3], preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=200, channels='MEG')
dataset = WaveletSimCLR(dataset.get_data(), dataset.get_labels(), dataset.get_info(),
        n_views=2, widths=50, premake=False, transform=True)
print(f'Transforming windows using Continuous Wavelet Transform (CWT) on entire Recording Dataset...')
dataset._remake_dataset()
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1, drop_last=True) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNetSimCLR('resnet18', 100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader),
        eta_min=0, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss().to(device)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=256, epochs=5, temperature=.7, n_views=2)

print(f'Extracting pre-training embeddings...')
tSNE_plot(dataset._extract_embeddings(model, device), 'pre')

simclr.fit(dataloader)

print(f'Extracting post-training embeddings...')
tSNE_plot(dataset._extract_embeddings(model, device), 'post')
