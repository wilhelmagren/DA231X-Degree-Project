"""
implementation of the sleep staging CNN presented in
Hubert Banville et al. `Uncovering the structure of clinical
EEG signals with self-supervised learning` 2020.
model is based on that presented by Chambon et al. in 2018.


Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import torch
import numpy as np

from torch import nn


class StagerNet(nn.Module):
    def __init__(self, n_channels, sfreq, n_conv_chs=16, emb_size=100,
                input_size_s=30., time_conv_size_s=.5, max_pool_size_s=.125,
                pad_size_s=.25, dropout=.5, apply_batch_norm=False,
                return_feats=False, **kwargs):
        super(StagerNet, self).__init__()
        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * sfreq).astype(int)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * sfreq).astype(int)
        
        self.n_channels = n_channels

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
                batch_norm(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, max_pool_size)),
                nn.Conv2d(n_conv_chs, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
                batch_norm(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, max_pool_size))
                )

        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self.return_feats = return_feats
        if not return_feats:
            self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(self.len_last_layer, emb_size)
                    )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)
        
        feats = self.feature_extractor(x).flatten(start_dim=1)
        
        if self.return_feats:
            return feats
        else:
            return self.fc(feats)

