import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.utils import BCEWithLogitsAccuracy, recording_train_valid_split
from neurocode.datasets import RecordingDataset, SLEMEG
from neurocode.samplers import SSTOSampler
from neurocode.models import SignalNet, WaveletNet, ContrastiveSSTO
from neurocode.datautil import CWT, SSTOLoader
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from braindecode.datautil.windowers import create_fixed_length_windows
from scipy import signal
from tqdm import tqdm
from sklearn.manifold import TSNE


# Parameters for example
seed=1
n_jobs=1
window_size_s=3
sfreq=200
window_size_samples=window_size_s*sfreq
subjects=list(range(2, 12))
recordings=[1, 3]
channel_picks=['MEG0731']
n_channels=len(channel_picks)
emb_size=100
n_frequency_bands = 100
dropout=.25
n_conv_chs=32
tau=3
n_samples=100
batch_size=32
epochs=5
split=.7

# Load the dataset, preloading
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, channels=channel_picks)

# Scale the MEG data to mT ranges
preprocessor = [
        Preprocessor(lambda x: x*1e12)
        ]
preprocess(dataset, preprocessor)

# Create 3s windows of the data, no overlapping, i.e. stride=3s
windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=1,
        stop_offset_samples=None, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

# normalize all windows channel-wise using zscoring, retaining the oscillatory sufficient statistics
preprocess(windows_dataset, [Preprocessor(zscore)])

recordings_train, recordings_valid = recording_train_valid_split(windows_dataset.datasets, split=split)
datasets = dict(train=RecordingDataset(recordings_train, sfreq=sfreq, channels='MEG'),
                valid=RecordingDataset(recordings_valid, sfreq=sfreq, channels='MEG'))

samplers = dict(train=SSTOSampler(datasets['train'].get_data(), datasets['train'].get_info(),
    tau=tau, n_samples=n_samples, batch_size=batch_size),
                valid=SSTOSampler(datasets['valid'].get_data(), datasets['valid'].get_info(),
    tau=tau, n_samples=n_samples, batch_size=batch_size))

# Setup, load, move embedders and contrastive model to device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

scalogram_embedder = WaveletNet(n_frequency_bands,
                   sfreq,
                   n_conv_chs=n_conv_chs,
                   emb_size=emb_size,
                   input_size_s=window_size_s,
                   dropout=dropout).to(device)

signal_embedder = SignalNet(n_channels, sfreq,
        n_conv_chs=n_conv_chs,
        emb_size=emb_size,
        input_size_s=window_size_s,
        dropout=dropout).to(device)

model = ContrastiveSSTO(signal_embedder,
        scalogram_embedder, emb_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()

history = dict(tloss=list(), tacc=list(), vloss=list(), vacc=list())

# Train the models
print(f'starting model training for {epochs} epochs on device={device}')
print(f'  epoch    training loss   validation loss   training acc   validation acc')
print(f'----------------------------------------------------------------------------')
for epoch in range(epochs):
    signal_embedder.train()
    scalogram_embedder.train()
    model.train()
    tloss = 0
    vloss = 0
    tacc = 0
    vacc = 0
    for batch, (anchors, samples, labels) in enumerate(samplers['train']):
        signals, scalograms, labels = SSTOLoader(anchors, samples, labels, device, unsqueeze_labels=True)
        optimizer.zero_grad()
        outputs = model((signals, scalograms))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item() / outputs.shape[0]
        tacc += BCEWithLogitsAccuracy(outputs, labels) / outputs.shape[0]
    signal_embedder.eval()
    scalogram_embedder.eval()
    with torch.no_grad():
        for batch, (anchors, samples, labels) in enumerate(samplers['valid']):
            signals, scalograms, labels = SSTOLoader(anchors, samples, labels, device, unsqueeze_labels=True)
            outputs = model((signals, scalograms))
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
    print(f'   {epoch+1:02d}         {tloss:.5f}         {vloss:.5f}         {100*tacc:.2f}%        {100*vacc:.2f}%')

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
plt.clf()

# Extract features
X, sampler = list(), samplers['train']
model.return_features=True
with torch.no_grad():
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='Extracting embeddings'):
        signals, scalograms, _ = SSTOLoader(anchors, samples, labels, device, unsqueeze_labels=False)
        features = model((signals, scalograms))
        X.append(features[0, :][None])
X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)
tsne = TSNE(n_components=2, perplexity=30)
components = tsne.fit_transform(X)
for idx, (x,y) in enumerate(components):
    plt.scatter(x, y, alpha=.5, color='blue')
plt.savefig('t-SNE_SSTOSampler_post-training.png')





