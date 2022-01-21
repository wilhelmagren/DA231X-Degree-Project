import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurocode.datasets import RecordingDataset, SLEMEG
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore
from braindecode.datautil.windowers import create_fixed_length_windows


window_size_s = 3
sfreq = 200
window_size_samples = window_size_s * sfreq
subjects = list(range(4, 5))
recordings = [3]
n_samples = 100
n_channels = 2
emb_size = 100

dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True, load_meg_only=True)

preprocessor = [
        Preprocessor(lambda x: x*1e12)
        ]
preprocess(dataset, preprocessor)

windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=1,
        stop_offset_samples=None, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

#preprocess(windows_dataset, [Preprocessor(zscore)])

# Extract time-windows
from scipy import signal

dataset = RecordingDataset(windows_dataset.datasets, sfreq=sfreq, channels='MEG')
print(dataset.get_info())
fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)

for channel_idx in range(0, 2):
    for window_idx in range(5, 10):
        npdata = dataset[0, window_idx][0]
        # f, t, Zxx = signal.stft(npdata[channel_idx, :], sfreq, noverlap=0)
        f, t, Sxx = signal.spectrogram(npdata[channel_idx, :],
                sfreq, noverlap=0, nperseg=sfreq)
        print(Sxx)
        print(f'{f=}, {t=}')
        exit()
        axs[channel_idx, window_idx-5].pcolormesh(t, f, Sxx,
                vmin=0, shading='gouraud', cmap='hot')
        axs[channel_idx, window_idx-5].set_title(f'{window_idx*window_size_s}s to {(window_idx+1)*window_size_s}s')
fig.suptitle(f'Spectrogram of MEG channels, in {window_size_s} second windows.')
fig.text(.04, .5, 'Frequency [Hz]', ha='center', rotation='vertical')
fig.text(.5, .04, 'Time [s]', ha='center')
plt.show()

