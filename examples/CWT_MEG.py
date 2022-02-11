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

# Extract time-windows
from scipy import signal

dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=sfreq, channels='MEG')
print(dataset.get_info())

widths = 50

fig, axs = plt.subplots(2, 3)
for widx in range(5, 8):
    npdata = dataset[0, widx][0]
    cwtmatr = signal.cwt(npdata[0, :], signal.ricker, np.arange(1, widths + 1))
    axs[0, widx-5].plot(list(range(1 + 600*(widx-5), 1 + 600*(widx-4))), npdata[0, :], label='Signal')
    axs[1, widx-5].imshow(cwtmatr, cmap='hot', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), label='Wavelet transform')
fig.suptitle(f'Signal and Continuous Wavelet Transform scalogram, in {window_size_s}s windows')
fig.text(.04, .2, 'Frequency [Hz]                        Amplitude [mT]', ha='center', rotation='vertical')
fig.text(.5, .04, 'Time [s]', ha='center')
plt.legend()
plt.show()

