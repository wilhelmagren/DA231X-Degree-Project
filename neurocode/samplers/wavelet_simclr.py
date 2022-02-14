"""
TODO: add docs!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class ScalogramSampler(PretextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, widths=50, signal='ricker', 
            n_views=2, shape=None, **kwargs):
        
        self.widths = np.arange(1, widths + 1)
        self.signal = signal.ricker
        self.n_views = n_views
        self.shape = shape

        self._transforms = [
                transforms.RandomResizedCrop(shape, (.7, .95)),
                transforms.RandomVerticalFlip(p=.999)]

        self.transformer = ContrastiveViewGenerator(
                self._transforms, n_views)

    def _sample_pair(self):
        batch_anchors = list()
        batch_samples = list()
        for _ in range(self.batch_size):
            reco_idx = self._sample_recording()
            wind_idx = self._sample_window(recording_idx=reco_idx)

            x = self.data[reco_idx][wind_idx][0]
            scalogram = signal.cwt(x, self.signal, self.widths)
            T1, T2 = self.transforms(scalogram)



class ContrastiveViewGenerator(object):
    def __init__(self, transforms, n_views):
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms(x) for _ in range(self.n_views)]


class WaveletSimCLR(Dataset):
    """
    """
    def __init__(self, data, labels, info, **kwargs):
        self.data = data
        self.labels = labels
        self.info = info
        self._setup(**kwargs)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.transform:
            return self.transforms(self.data[index])

        return (self.data[index], self.labels[index])

    def _setup(self, n_views=2, widths=100, shape=(128, 128), premake=True, transform=False, **kwargs):
        self.shape = shape
        self.n_views = n_views
        self.widths = widths
        self.premake = premake
        self.transform = transform
        self.preprocess = transforms.Compose([
            transforms.RandomInvert(p=.25),
            transforms.RandomHorizontalFlip(p=.3),
            transforms.RandomVerticalFlip(p=.3),
            transforms.RandomRotation((-20, 20)),
            transforms.RandomResizedCrop(self.shape, scale=(.5, .95))
            ])

        self.transforms = ContrastiveViewGenerator(
                self.preprocess, n_views)

        if premake:
            self._remake_dataset()

    def _remake_dataset(self):
        remade_data = []
        remade_labels  = []
        resizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.shape)])
        for recording, windows in self.data.items():
            for (window, _, _) in windows:
                cwt_matrix = signal.cwt(window.squeeze(0), signal.ricker, np.arange(1, self.widths + 1))
                cwt_matrix = resizer(cwt_matrix)
                remade_data.append(cwt_matrix)
                remade_labels.append(self.labels[recording])
        
        assert len(remade_data) == sum(self.info['lengths'].values()), (
                'remade data doesn`t have the same amount of windows as previous data.')
        
        self.data = remade_data
        self.labels = remade_labels

    def _extract_embeddings(self, emb, device):
        X, Y = [], []
        emb.eval()
        emb._return_features = True
        with torch.no_grad():
            for index in tqdm(range(len(self)), total=len(self)):
                if index % 30 != 0:
                    continue
                scalogram = self.data[index].to(device)
                embedding = emb(scalogram.unsqueeze(0).float())
                X.append(embedding[0, :][None])
                Y.append(self.labels[index])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        emb._return_features = False
        return (X, Y)


