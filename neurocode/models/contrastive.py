"""
Siamese network architecture for performing linear regression 
on pretext task sampling labels. The net requires an embedder
model which is used for feature extraction.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, encoder, emb_size, latent_size, dropout=.25):
        super(ProjectionHead, self).__init__()
        self.f = encoder
        self.g = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, latent_size)
        )

    def forward(self, x):
        features = self.f(x)
        latent = self.g(features)

        return latent


class ContrastiveRPNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=.5):
        super(ContrastiveRPNet, self).__init__()
        self.emb = emb
        self.clf = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(emb_size, 1)
                )

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1-z2))


class ContrastiveTSNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=.5):
        super(ContrastiveTSNet, self).__init__()
        self.emb = emb
        self.clf = nn.Sequential(
            nn.Linear(2*emb_size, emb_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
    
    def forward(self, x):
        z1, z2, z3 = [self.emb(t) for t in x]
        return self.clf(torch.cat((torch.abs(z1 - z2), torch.abs(z2 - z3)), dim=1))
    