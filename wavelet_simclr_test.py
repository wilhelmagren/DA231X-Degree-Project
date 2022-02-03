import torch
import numpy as np

from torch.utils.data import DataLoader
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import WaveletSimCLR
from neurocode.models import ResNetSimCLR
from neurocode.training import SimCLR
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore

window_size_s = .5
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=[3,4,5,6], recordings=[1,3], preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=200, channels='MEG')
dataset = WaveletSimCLR(dataset.get_data(), dataset.get_labels(), dataset.get_info(),
        n_views=2, widths=100, premake=False, transform=True)
print(f'Transforming windows using Continuous Wavelet Transform (CWT) on entire Recording Dataset...')
dataset._remake_dataset()

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNetSimCLR('resnet18', 100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader),
        eta_min=0, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss().to(device)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=64, epochs=100, temperature=.7, n_views=2)

simclr.fit(dataloader)

"""
import matplotlib.pyplot as plt
for images in dataloader:
    images = torch.cat(images, dim=0)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(torch.swapaxes(images[0, :], 0, 2))
    axs[1].imshow(torch.swapaxes(images[16, :], 0, 2))
    plt.show()
"""

