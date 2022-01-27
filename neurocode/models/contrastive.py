"""
Siamese network architecture for performing logistic regression 
on pretext task sampling labels. The net requires an embedder
model which is used for feature extraction.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 27-01-2022
"""
import torch

from torch import nn
from ..datautil import CWT

class ContrastiveRP(nn.Module):
    def __init__(self, emb, emb_size, dropout=.5):
        super(ContrastiveRP, self).__init__()

        self.emb = emb
        self.clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(emb_size, 1)
                )

    def __str__(self):
        return 'ContrastiveRP'

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1-z2))


class ContrastiveSSTO(nn.Module):
    def __init__(self, signal_emb, scalogram_emb, emb_size, dropout=.5):
        super(ContrastiveSSTO, self).__init__()

        self.return_features = False
        self.signal_emb = signal_emb
        self.scalogram_emb = scalogram_emb
        self.clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2*emb_size, 1)
                )
    
    def __str__(self):
        return 'ContrastiveSSTO'

    def forward(self, x):
        signals, scalograms = x

        signal_anchors, signal_samples = signals
        scalogram_anchors, scalogram_samples = scalograms

        sig_z1, sig_z2 = self.signal_emb(signal_anchors), self.signal_emb(signal_samples)
        sca_z1, sca_z2 = self.scalogram_emb(scalogram_anchors), self.scalogram_emb(scalogram_samples)

        features = torch.cat((torch.abs(sig_z1 - sig_z2), torch.abs(sca_z1 - sca_z2)), dim=1)
        if self.return_features:
            return features

        return self.clf(features)
