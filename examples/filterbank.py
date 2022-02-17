import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from sklearn import preprocessing
from neurocode.datasets import RecordingDataset, SLEMEG
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore


# Sampling frequency 200 Hz, with window size of 80ms
# we get 0.08 x 200 = 16 samples, quite small amount
# and resolution of scalogram is going to be poor.
subjects = [1]
recordings = [0]
window_size_s = 1.
sfreq = 200
window_size_samples = np.ceil(window_size_s * sfreq).astype(int)
widths = np.arange(1, 51)

preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows, [Preprocessor(zscore)])
dataset = RecordingDataset(windows.datasets, dataset.labels, sfreq=sfreq, channels='MEG')

widths = [np.arange(1, 31 + i*20) for i in range(3)]
data = dataset[0, 8][0][0, :]

# Frequency band       Frequency     Brain states
# -------------------------------------------------
#    Gamma                35 Hz      Concentration
#    Beta               12-35 Hz     Anxiety dominant, active, external attention, relaxed
#    Alpha               8-12 Hz     Very relaxed, passive attention
#    Theta                4-8 Hz     Deeply relaxed, inward focused
#    Delta               .5-4Hz      Sleep
gamma_range = (35., 100.)
beta_range = (12., 35.)
alpha_range = (8., 12.)
theta_range = (4., 8.)


for width in widths:
    fig, axs = plt.subplots(1, 3)
    cwt = signal.cwt(data, signal.ricker, width)
    print(cwt.shape)


    axs[0].plot(np.arange(len(data)), data)
    axs[0].set_title('MEG')
    axs[1].imshow(cwt, extent=[-1, 1, 1, width[-1]], aspect='auto')
    axs[1].set_title('CWT')
    cwt = cwt[:(width[-1] // 2), :]
    print(cwt)
    print(cwt.shape)
    axs[2].imshow(cwt, extent=[-1, 1, 1, width[-1]], aspect='auto')
    axs[2].set_title('Cropped CWT')

    plt.suptitle(f'{window_size_s} seconds and {width[-1]} scales')
    plt.show()