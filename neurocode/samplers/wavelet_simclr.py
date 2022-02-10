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


class ContrastiveViewGenerator(object):
    def __init__(self, transforms, n_views):
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, x):
        x = Image.fromarray(x)
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

    def _setup(self, n_views=2, widths=100, premake=True, transform=False, **kwargs):
        self.n_views = n_views
        self.widths = widths
        self.premake = premake
        self.transform = transform
        self.preprocess = transforms.Compose([
            transforms.RandomResizedCrop(widths, scale=(.5, .95)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])

        self.transforms = ContrastiveViewGenerator(
                self.preprocess, n_views)

        if premake:
            self._remake_dataset()

    def _remake_dataset(self):
        remade_data = []
        remade_labels  = []
        for recording, windows in self.data.items():
            for (window, _, _) in windows:
                cwt_matrix = signal.cwt(window.squeeze(0), signal.ricker, np.arange(1, self.widths + 1))
                remade_data.append(cwt_matrix)
                remade_labels.append(self.labels[recording])
        
        assert len(remade_data) == sum(self.info['lengths'].values()), (
                'remade data doesn`t have the same amount of windows as previous data.')
        
        self.data = remade_data
        self.labels = remade_labels

    def _extract_embeddings(self, emb, device):
        X, Y = [], []
        emb.eval()
        grayscaler = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])
        with torch.no_grad():
            for index in tqdm(range(len(self)), total=len(self)):
                if index % 20 != 0:
                    continue
                scalogram = grayscaler(Image.fromarray(self.data[index])).to(device)
                embedding = emb(scalogram.unsqueeze(0))
                X.append(embedding[0, :][None])
                Y.append(self.labels[index])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        return (X, Y)


