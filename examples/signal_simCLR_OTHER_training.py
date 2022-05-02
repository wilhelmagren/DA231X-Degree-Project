#
#   imports
#
import torch
import numpy as np

from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import SignalSampler
from neurocode.models import SignalNet, load_model
from neurocode.training import SimCLR
from neurocode.datautil import manifold_plot, history_plot
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from pytorch_metric_learning import losses

torch.manual_seed(73)
np.random.seed(73)


#
#   hyperparameter settings
#
manifold = 'tSNE'
load_model_ = False
subjects = list(range(0, 34))
recordings = [0,1,2,3]
batch_size = 128
n_samples = 20
window_size_s = 5.0
n_views = 2
n_channels = 3
n_epochs = 200
temperature = 0.1
sfreq = 200
n_conv_chs = 50
dropout = 0.0
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)


#
#   load the MEG data from cleaned .fif files
#
preprocessors = [Preprocessor(lambda x: x*1e12)]
recording_dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)


#
#   create windowed samples, stride is window side; no overlapping
#   split into training,validation, predefined split. create samplers
#
preprocess(recording_dataset, [Preprocessor(zscore)])
windows_dataset = create_fixed_length_windows(recording_dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

# preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, recording_dataset.labels, sfreq=sfreq, channels='MEG')

sampler = SignalSampler(dataset.get_data(), dataset.get_labels(),
    dataset.get_info(), n_channels=n_channels, n_views=n_views, n_samples=n_samples, batch_size=batch_size)

train, valid = sampler._split()

samplers = {'train': SignalSampler(train.get_data(), train.get_labels(),
    train.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size),
           'valid': SignalSampler(valid.get_data(), valid.get_labels(),
    valid.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size)}

#
#   setup model, optimizer, scheduler, criterion, and training routine
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SignalNet(n_channels, sfreq, input_size_s=window_size_s, 
                  n_filters=n_conv_chs, apply_batch_norm=True, dropout=dropout).to(device)
#model = load_model('params.pth').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(samplers['train']),
        eta_min=0, last_epoch=-1)
criterion = losses.NTXentLoss(temperature=temperature)

simclr = SimCLR(model, device, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, batch_size=batch_size, epochs=n_epochs, 
        temperature=temperature, n_views=n_views)


#
#   extract features prior to training, plot tSNE
#   traing model and save the loss scores
#   extract features post training and plot tSNE based on subject labels
#
manifold_plot(samplers['train']._extract_features(model, device), 'train-data_pre', technique=manifold)
manifold_plot(samplers['valid']._extract_features(model, device), 'valid-data_pre', technique=manifold)

print(f'Training encoder with SimCLR on device={device} for {n_epochs} epochs')
print(f'   epoch       training loss       validation loss         training acc        validation acc')
print(f'------------------------------------------------------------------------------------------------')
history = simclr.fit(samplers, plot=False, save_model=True)
history_plot(history)
for key, value in history.items():
    if key in ['tloss', 'vloss']:
        with open(f'{key}_cropresize.scores', 'w') as fpath:
            for val in value:
                fpath.write(str(val) + '\n')

manifold_plot(samplers['train']._extract_features(model, device), 'train-data_post', technique=manifold)
manifold_plot(samplers['valid']._extract_features(model, device), 'valid-data_post', technique=manifold)
