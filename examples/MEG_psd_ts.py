"""
Self-supervised Learning pipeline for SLEMEG dataset project.
Utilizes the auxiliary task 'Recording Relative Positioning' 
and samples there-after. See results in subdir 'results/RRP/' 


TODO: formalize training in a neurocode module.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.utils import BCEWithLogitsAccuracy
from neurocode.datasets import RecordingDataset, SLEMEG
from neurocode.samplers import TSSampler
from neurocode.models import ContrastiveTSNet, StagerNet
from neurocode.datautil import tSNE_plot, history_plot
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from braindecode.datautil.windowers import create_fixed_length_windows


# parameters for the example pipeline
seed=1
n_jobs=1
window_size_s = 5.
sfreq = 200
window_size_samples = np.ceil(window_size_s * sfreq).astype(int)
subjects = list(range(10,12))
recordings = [0,1,2,3]
tau_pos = 6
tau_neg = 10
n_samples = 100
batch_size = 64
n_channels = 3
emb_size = 100

preprocessors = [Preprocessor(lambda x: x*1e12)]

# fetch the dataset
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

# create 5 second windows of data from the preprocessed SLEMEG dataset
windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0, 
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

# "normalize" each window and channel repsectively, using zscoring for scaling
preprocess(windows_dataset, [Preprocessor(zscore)])

# reformat from BaseConcatDataset to RecordingDataset and perform 70/30 split into train/valid
recording_dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = recording_dataset.split(split=.7)

# set up recording-relative-positioning sampler with MEG recording dataset
samplers = {'train': TSSampler(train_dataset.get_data(), train_dataset.get_labels(),
                train_dataset.get_info(), tau_pos=tau_pos, tau_neg=tau_neg, 
                n_samples=n_samples, batch_size=batch_size),
            'valid': TSSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
                valid_dataset.get_info(), tau_pos=tau_pos, tau_neg=tau_neg,
                n_samples=n_samples, batch_size=batch_size)}


# Setup pytorch training, move models etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb = StagerNet(n_channels, sfreq, n_conv_chs=32, dropout=.5, 
    emb_size=emb_size, input_size_s=window_size_s).to(device)
model = ContrastiveTSNet(emb, emb_size, dropout=.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()


print(f'extracting embeddings before training...')
#embeddings = {'pre': samplers['valid']._extract_embeddings(emb, device)}
#tSNE_plot(embeddings['pre'], 'before')

epochs = 5
history = {'tloss': [], 'vloss': [], 'tacc': [], 'vacc': []}
print(f'starting model training for {epochs} epochs on device={device}')
print(f'  epoch    training loss   validation loss   training acc   validation acc')
print(f'----------------------------------------------------------------------------')
for epoch in range(epochs):
    tloss, tacc = 0., 0.
    vloss, vacc = 0., 0.
    emb.train()
    model.train()
    for batch, (anchors, middles, samples, labels) in enumerate(samplers['train']):
        anchors, middles, samples, labels = anchors.to(device), middles.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, middles, samples))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item() / outputs.shape[0]
        tacc += BCEWithLogitsAccuracy(outputs, labels) / outputs.shape[0]
    emb.eval()
    model.eval()
    with torch.no_grad():
        for batch, (anchors, middles, samples, labels) in enumerate(samplers['valid']):
            anchors, middles, samples, labels = anchors.to(device), middles.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
            outputs = model((anchors, middles, samples))
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
    print(f'    {epoch+1:02d}         {tloss:.5f}         {vloss:.5f}          {100*tacc:.2f}%          {100*vacc:.2f}%')

print(f'extracting embeddings after training...')
#embeddings['post'] = samplers['valid']._extract_embeddings(emb, device)
#tSNE_plot(embeddings['post'], 'after')

history_plot(history)
