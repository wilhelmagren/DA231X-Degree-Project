import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
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
window_size_s = .5
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

shape = (64, 128)
widths = [np.arange(1, 31 + i*20) for i in range(3)]
fig, axs = plt.subplots(4, 3)

resizecrop = transforms.Compose([
    transforms.RandomResizedCrop(size=shape, scale=(.3, .8)),
    transforms.ToTensor()])

posterize = transforms.Compose([
    transforms.RandomPosterize(bits=3, p=0.9999),
    transforms.Resize(size=shape),
    transforms.ToTensor()])

for win in [10, 11, 12]:
    data = dataset[0, win][0][0, :]
    axs[0, win-10].plot(data)
    cwt = signal.cwt(data, signal.ricker, widths[1])
    cwt = cwt[:(widths[1][-1] // 2), :]
    cwt = ((cwt - cwt.min()) * (1/(cwt.max() - cwt.min()) * 255)).astype('uint8')
    axs[1, win-10].imshow(cwt, extent=[-1, 1, 1, widths[1][-1]], aspect='auto')

    image = Image.fromarray(cwt)

    t1 = resizecrop(image)
    t2 = posterize(image)

    arr1 = torch.swapaxes(torch.swapaxes(t1, 0, 2), 0, 1).numpy().squeeze(2)
    axs[2, win-10].imshow(arr1, extent=[-1, 1, 1, widths[1][-1]], aspect='auto')
    arr2 = torch.swapaxes(torch.swapaxes(t2, 0, 2), 0, 1).numpy().squeeze(2)
    axs[3, win-10].imshow(arr2, extent=[-1, 1, 1, widths[1][-1]], aspect='auto')

    print(cwt)
    print(arr1)
    print(arr2)
    print(cwt.shape)
    print(arr1.shape)
    print(arr2.shape)
    print(arr1 - arr2)

fig.suptitle('MEG signal with corresponding CWT + augmentations')
fig.text(.04, .15, 'Posterized                           CropResized                                  CWT                                  Signal', ha='center', rotation='vertical')
fig.text(.52, .04, 'Consecutive windows [.5 s]', ha='center') 
plt.show()

