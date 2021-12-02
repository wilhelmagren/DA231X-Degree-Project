import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.utils import BCEWithLogitsAccuracy, recording_train_valid_split
from neurocode.datasets import RecordingDataset, SLEMEG
from neurocode.samplers import RecordingRelativePositioningSampler
from neurocode.models import ContrastiveNet, StagerNet
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from braindecode.datautil.windowers import create_fixed_length_windows


# parameters for the example pipeline
seed=1
n_jobs=1
window_size_s = 5
sfreq = 200
window_size_samples = window_size_s * sfreq
subjects = list(range(2,12))
recordings = [1,3]
tau_pos = 10
tau_neg = 30
gamma = .7
n_samples = 100
batch_size = 32
n_channels = 2
emb_size = 100
split = .7

# fetch the dataset
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True, load_meg_only=True)


# preprocess the raw files
preprocessor = [
        Preprocessor(lambda x: x*1e12)
        ]
preprocess(dataset, preprocessor)

# create 5 second windows of data from the preprocessed SLEMEG dataset
windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=1, 
        stop_offset_samples=None, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

# normalize all windows channel-wise using zscoring
preprocess(windows_dataset, [Preprocessor(zscore)])

# split full dataset into train/valid, and reformat to RecordingDatasets
print(f'splitting dataset into train/valid {int(100*split)}/{int(100 - 100*split)}')
recordings_train, recordings_valid = recording_train_valid_split(windows_dataset.datasets, split=split)
datasets = dict(train=RecordingDataset(recordings_train, sfreq=sfreq, channels='MEG'),
                valid=RecordingDataset(recordings_valid, sfreq=sfreq, channels='MEG'))

# set up relative-positioning sampler with MEG recording dataset
samplers = dict(train=RecordingRelativePositioningSampler(datasets['train'].get_data(), datasets['train'].get_info(),
                tau=tau_pos, gamma=gamma, n_samples=n_samples, batch_size=batch_size),
                valid=RecordingRelativePositioningSampler(datasets['valid'].get_data(), datasets['valid'].get_info(),
                tau=tau_pos, gamma=gamma, n_samples=n_samples, batch_size=batch_size))


# Setup pytorch training, move models etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb = StagerNet(n_channels, sfreq, dropout=.25, input_size_s=5.).to(device)
model = ContrastiveNet(emb, emb_size, dropout=.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()

epochs = 30
history = dict(tloss=list(), tacc=list())
emb.train()
model.train()

print(f'starting model training for {epochs} epochs on device={device}')
print(f'  epoch    training loss   validation loss   training acc   validation acc')
print(f'----------------------------------------------------------------------------')
for epoch in range(epochs):
    tloss, tacc = 0., 0.
    vloss, vacc = 0., 0.
    for batch, (anchors, samples, labels) in enumerate(samplers['train']):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, samples))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item() / outputs.shape[0]
        tacc += BCEWithLogitsAccuracy(outputs, labels) / outputs.shape[0]
    emb.eval()
    model.eval()
    with torch.no_grad():
        for batch, (anchors, samples, labels) in enumerate(samplers['valid']):    
            anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
            outputs = model((anchors, samples))
            loss = criterion(outputs, labels)
            vloss += loss.item() / outputs.shape[0]
            vacc += BCEWithLogitsAccuracy(outputs, labels) / outputs.shape[0]
    tloss /= len(samplers['train'])
    vloss /= len(samplers['valid'])
    tacc /= len(samplers['train'])
    vacc /= len(samplers['valid'])
    history['tloss'].append(tloss)
    history['vloss'].append(vloss)
    history['tacc'].append(tacc)
    history['vacc'].append(vacc)
    print(f'   {epoch+1:02d}       {tloss:.5f}       {vloss:.5f}       {100*tacc:.2f}%      {100*vacc:.2f}%')

fig, ax1 = plt.subplots(figsize=(8,3))
ax2 = ax1.twinx()
ax1.plot(history['tloss'], ls='-', marker='d', ms=5, alpha=.7, color='tab:blue', label='training loss')
ax1.plot(history['vloss'], ls=':'. marker='d'. ms=5, alpha=.7, color='tab:blue', label='validation loss')
ax2.plot(history['tacc'], ls='-', marker='d', ms=5, alpha=.7, color='tab:orange', label='training acc')
ax2.plot(history['vacc'], ls=':', marker='d', ms=5, alpha=.7, color='tab:orange', label='validation acc')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylabel('Loss', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylabel('Accuracy [%]', color='tab:orange')
ax1.set_xlabel('Epoch')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2)
plt.tight_layout()
plt.savefig('sleepstaging_relative-positioning_training.png')

