from scipy.signal import ricker
from scipy import signal as sig
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

xx = np.linspace(-3, 3, num=100)
vec = np.sin(xx**2)
widths = np.arange(1, 51)
cwt = sig.cwt(vec, ricker, widths)

for _ in range(10):
    cwt = ((cwt - cwt.min()) * (1/(cwt.max() - cwt.min()) * 255)).astype('uint8')
    imge = Image.fromarray(cwt)
    trans = transforms.Compose([
    transforms.RandomResizedCrop((64, 64), scale=(0.2, 0.9)),
    transforms.ToTensor()])
    T = trans(imge)
    T = (T - T.mean()) / T.std()

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(vec)
    axs[1].imshow(cwt, extent=[-1, 1, 1, widths[-1]], aspect='auto', cmap='viridis')
    arr = torch.swapaxes(torch.swapaxes(T, 0, 2), 0, 1).numpy().squeeze(2)
    axs[2].imshow(arr, extent=[-1, 1, 1, widths[-1]], aspect='auto', cmap='viridis')
    plt.show()