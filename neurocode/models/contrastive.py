"""
Siamese network architecture for performing linear regression 
on pretext task sampling labels. The net requires an embedder
model which is used for feature extraction.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch

from torch import nn


class ContrastiveNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=.5):
        super(ContrastiveNet, self).__init__()
        self.emb = emb
        self.clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(emb_size, 1)
                )

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1-z2))

