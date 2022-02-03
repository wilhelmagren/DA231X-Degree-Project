"""
 Pytorch resnet implementation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from torchvision import transforms
from PIL import Image
from neurocode.datasets import SLEMEG, RecordingDataset
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.preprocess import preprocess, Preprocessor, zscore

wsize=100
preprocessors = [Preprocessor(lambda x: x*1e12)]
dataset = SLEMEG(subjects=[1,2,3], recordings=[0], preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

windows_dataset = create_fixed_length_windows(dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=wsize,
        window_stride_samples=wsize, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, dataset.labels, sfreq=200, channels='MEG')

preprocess = transforms.Compose([
    transforms.RandomResizedCrop(100, scale=(.7, .95)),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomVerticalFlip(p=.5),
    transforms.ToTensor()])


for idx in range(100, 103):
    
    npdata = dataset[0, idx][0][0, :]
    cwtmatrix = signal.cwt(npdata, signal.ricker, np.arange(1, 101))

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(list(range(100)), npdata, label='Signal')
    axs[0].set_title('MEG Signal')
    axs[1].imshow(cwtmatrix, extent=[-1, 1, 1, 101], cmap='hot', aspect='auto',
            vmax=abs(cwtmatrix).max(), vmin=-abs(cwtmatrix).max(), label='CWT')
    axs[1].set_title('CWT')

    img = Image.fromarray(cwtmatrix)
    input_tensor = preprocess(img)
    axs[2].imshow(torch.swapaxes(input_tensor, 0, 2).numpy(), extent=[-1, 1, 1, 101], cmap='hot',
            aspect='auto', vmax=abs(input_tensor).max(), vmin=-abs(input_tensor).max(), label='CWT augmentation')
    axs[2].set_title('CWT transformed*') 
    fig.suptitle('SimCLR data augmentation module')
    fig.text(.35, .03, '*by RandomResizedCrop, RandomHorizontalFlip, and RandomVerticalFlip')
    plt.show()


