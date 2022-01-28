import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.utils import BCEWithLogitsAccuracy, recording_train_valid_split
from neurocode.datasets import RecordingDataset, SLEMEG
from neurocode.samplers import SSTOSampler
from neurocode.models import WaveletNet, ContrastiveRP
from neurocode.datautil import WaveletRPLoader, ExtractRPEmbeddings, plot_history_acc_loss
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from braindecode.datautil.windowers import create_fixed_length_windows
from tqdm import tqdm


# Parameters for example
seed=1
n_jobs=1
window_size_s=2
sfreq=200
window_size_samples=window_size_s*sfreq
subjects=list(range(2, 8))
recordings=[0,1,2,3]
channel_picks=['MEG2123']
n_channels=len(channel_picks)
emb_size=100
n_frequency_bands = 70
dropout=.25
n_conv_chs=16
tau_neg=10
tau_pos=2
n_samples_train=100
n_samples_valid=50
batch_size=32
epochs=10
split=.7

# Load the dataset, preloading
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, channels=channel_picks, cleaned=True)

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
# preprocess(windows_dataset, [Preprocessor(zscore)])

recordings_train, recordings_valid = recording_train_valid_split(windows_dataset.datasets, split=split)
datasets = dict(train=RecordingDataset(recordings_train, sfreq=sfreq, channels='MEG'),
                valid=RecordingDataset(recordings_valid, sfreq=sfreq, channels='MEG'))

samplers = dict(train=SSTOSampler(datasets['train'].get_data(), datasets['train'].get_info(),
    tau=tau_pos, n_samples=n_samples_train, batch_size=batch_size),
                valid=SSTOSampler(datasets['valid'].get_data(), datasets['valid'].get_info(),
    tau=tau_pos, n_samples=n_samples_valid, batch_size=batch_size))

# Setup, load, move embedders and contrastive model to device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

emb = WaveletNet(n_frequency_bands,
                   sfreq,
                   n_conv_chs=n_conv_chs,
                   emb_size=emb_size,
                   input_size_s=window_size_s,
                   dropout=dropout).to(device)

model = ContrastiveRP(emb, emb_size, dropout=dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()

history = dict(tloss=list(), tacc=list(), vloss=list(), vacc=list())

# Train the models
print(f'starting model training for {epochs} epochs on device={device}')
print(f'  epoch    training loss   validation loss   training acc   validation acc')
print(f'----------------------------------------------------------------------------')
for epoch in range(epochs):
    emb.train() 
    model.train()
    tloss = 0
    vloss = 0
    tacc = 0
    vacc = 0
    for batch, data in tqdm(enumerate(samplers['train']), total=len(samplers['train'])):
        anchors, samples, labels = WaveletRPLoader(data, device, unsqueeze_labels=True, widths=n_frequency_bands)
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
        for batch, data in enumerate(samplers['valid']):
            anchors, samples, labels = WaveletRPLoader(data, device, unsqueeze_labels=True, widths=n_frequency_bands)
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
    print(f'   {epoch+1:02d}         {tloss:.5f}         {vloss:.5f}           {100*tacc:.2f}%          {100*vacc:.2f}%')

plot_history_acc_loss(history)

# Extract features
X = ExtractRPEmbeddings(emb, samplers['train'], device, n_frequency_bands)
components = TSNE_transform(X)
for idx, (x,y) in enumerate(components):
    plt.scatter(x, y, alpha=.5, color='blue')
plt.savefig('t-SNE_SSTOSampler_post-training.png')

