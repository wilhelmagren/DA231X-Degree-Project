import torch
import numpy as np

from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import ScalogramSampler
from neurocode.models import VGG
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
n_samples = 20
window_size_s = 0.5
shape = (1, 64, 64)
widths = 50
n_views = 2
n_epochs = 100
temperature = .1
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)
latent_space_dim = 100
dropout = 0.0

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

preprocess(dataset, [Preprocessor(zscore)])
windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = dataset.split_fixed()

samplers = {'train': ScalogramSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_views=n_views, widths=widths, shape=shape, n_samples=n_samples,
    batch_size=batch_size),
           'valid': ScalogramSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_views=n_views, widths=widths, shape=shape, n_samples=n_samples,
    batch_size=batch_size)}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =  VGG(shape, latent_space_dim, n_conv_chs=16, dropout=dropout).to(device)  ##load_model('params.pth')  # ResNet18(1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(samplers['train']),
        eta_min=0, last_epoch=-1)
criterion = losses.NTXentLoss(temperature=temperature)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=batch_size, epochs=n_epochs, temperature=temperature,
        n_views=n_views)

manifold_plot(samplers['train']._extract_features(model, device), 'train-data_pre', technique=manifold)
manifold_plot(samplers['valid']._extract_features(model, device), 'valid-data_pre', technique=manifold)

print(f'Training encoder with SimCLR on device={device} for {n_epochs} epochs')
print(f'   epoch       training loss       validation loss         training acc        validation acc')
print(f'------------------------------------------------------------------------------------------------')
history = simclr.fit(samplers, plot=False, save_model=True)
history_plot(history)
for key, value in history.items():
    if key in ['tloss', 'vloss']:
        with open(f'{key}_scalogram.scores', 'w') as fpath:
            for val in value:
                fpath.write(str(val) + '\n')

manifold_plot(samplers['train']._extract_features(model, device), 'train-data_post', technique=manifold)
manifold_plot(samplers['valid']._extract_features(model, device), 'valid-data_post', technique=manifold)
