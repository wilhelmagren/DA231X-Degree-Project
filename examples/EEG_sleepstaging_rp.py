import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.utils import BCEWithLogitsAccuracy, recording_train_valid_split
from neurocode.datasets import RecordingDataset
from neurocode.samplers import RelativePositioningSampler
from neurocode.models import ContrastiveRP, StagerNet
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_fixed_length_windows


# parameters for the example pipeline
seed=1
n_jobs=1
high_cut_hz = 30
window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq
picks = 'eeg'
subjects = list(range(11))
recordings = [1]
tau_pos =  2  # 2*30 seconds = 60 seconds = 1 minute makes up the positive context
tau_neg = 30  # 30*30 seconds = 900 seconds = 15 minutes makes up the negative context
n_samples = 100
batch_size = 32
n_channels = 2
emb_size = 100
split = .7

# fetch the dataset
dataset = SleepPhysionet(subject_ids=subjects, recording_ids=recordings, crop_wake_mins=30, load_eeg_only=True)

# preprocess the raw files
preprocessor = [
        Preprocessor(lambda x: x * 1e6),
        Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)
        ]
preprocess(dataset, preprocessor)

# create 30 second windows of data from the preprocessed dataset
windows_dataset = create_fixed_length_windows(dataset,
        start_offset_samples=0, stop_offset_samples=0, drop_last_window=True,
        window_size_samples=window_size_samples, window_stride_samples=window_size_samples,
        preload=True, picks=picks)

# Reformat the BaseConcatDataset created by braindecode to a recording-window relative dataset
recordings_train, recordings_valid = recording_train_valid_split(windows_dataset.datasets, split=split)
datasets = dict(train=RecordingDataset(recordings_train, sfreq=sfreq, channels='MEG'),
                valid=RecordingDataset(recordings_valid, sfreq=sfreq, channels='MEG'))
#recording_dataset = RecordingDataset(windows_dataset.datasets, sfreq=sfreq, channels='eeg')

# Set up a Relative Positioning sampler to use for training the Contrastive Network
samplers = dict(train=RelativePositioningSampler(datasets['train'].get_data(), datasets['train'].get_info(), tau_neg=tau_neg, tau_pos=tau_pos, n_samples=n_samples, batch_size=batch_size),
                valid=RelativePositioningSampler(datasets['valid'].get_data(), datasets['valid'].get_info(), tau_neg=tau_neg, tau_pos=tau_pos, n_samples=n_samples, batch_size=batch_size, presample=True))

# Setup pytorch training, move models etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb = StagerNet(n_channels, sfreq, dropout=.25).to(device)
model = ContrastiveRP(emb, emb_size, dropout=.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()

epochs = 30
history = dict(tloss=list(), tacc=list(), vloss=list(), vacc=list())
emb.train()
model.train()

print(f'starting model training for {epochs} epochs on device={device}')
print(f'  epoch      training loss      training acc')
print(f'--------------------------------------------------')
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
ax1.plot(history['vloss'], ls=':', marker='d', ms=5, alpha=.7, color='tab:blue', label='validation loss')
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

