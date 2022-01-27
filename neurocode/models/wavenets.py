"""
Embedder architectures to be used together with SSTO pretext task.
SignalNet applies spatial- and temporal convolutions to the input signal,
whereas the WaveletNet applies traditional symmetric kernel convolutions to
the Scalogram input. These embeddings should later be fed to a contrastive
module, combining them by their differences.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 26-01-2022
"""
import torch
import numpy as np

from torch import nn
from torchsummary import summary


class SignalNet(nn.Module):
    def __init__(self, n_channels, sfreq, 
                 n_conv_chs=16,
                 emb_size=100,
                 input_size_s=5.,
                 dropout=.25,
                 return_features=False):

        super(SignalNet, self).__init__()
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.return_features = return_features
        input_size = np.ceil(input_size_s * sfreq).astype(int)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, n_conv_chs, (1, 10)),
                nn.MaxPool2d((1, 4), stride=2),
                nn.Conv2d(n_conv_chs, 2*n_conv_chs, (1, 8)),
                nn.MaxPool2d((1, 4), stride=2),
                nn.Conv2d(2*n_conv_chs, 3*n_conv_chs, (1, 6)),
                nn.Dropout(dropout),
                nn.Conv2d(3*n_conv_chs, 4*n_conv_chs, (1, 4)),
                nn.Conv2d(4*n_conv_chs, 4*n_conv_chs, (1, 4)),
            )

        resulting_conv = self._resulting_conv(n_channels, input_size)
        self._shape_last_conv = resulting_conv.shape
        self._len_last_layer = len(resulting_conv.flatten())

        self.global_max_pooling = nn.Sequential(
                nn.MaxPool2d((self._shape_last_conv[2], self._shape_last_conv[3]))
            )

        self.affine_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4*n_conv_chs, emb_size),
            )


    def _resulting_conv(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            resulting_conv = self.feature_extractor(torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return resulting_conv

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        features = self.feature_extractor(x)
        features = self.global_max_pooling(features)

        if self.return_features:
            return features.flatten(start_dim=1)

        embeddings = self.affine_layer(features)
        return embeddings


class WaveletNet(nn.Module):
    def __init__(self, n_frequency_bands,
                 sfreq,
                 n_conv_chs=16,
                 emb_size=100,
                 input_size_s=3.,
                 dropout=.25,
                 return_features=False):

        super(WaveletNet, self).__init__()
        self.sfreq = sfreq
        self.return_features = return_features
        input_size = np.ceil(input_size_s * sfreq).astype(int)

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, n_conv_chs, (5, 5)),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(n_conv_chs, 2*n_conv_chs, (4, 4)),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(2*n_conv_chs, 3*n_conv_chs, (3, 3)),
                nn.Dropout(dropout),
                nn.Conv2d(3*n_conv_chs, 4*n_conv_chs, (3, 3)),
                nn.Conv2d(4*n_conv_chs, 4*n_conv_chs, (3, 3))
            )
        
        resulting_conv = self._resulting_conv(n_frequency_bands, input_size)
        self._shape_last_conv = resulting_conv.shape
        self._len_last_layer = len(resulting_conv.flatten())

        self.global_max_pooling = nn.Sequential(
                nn.MaxPool2d((self._shape_last_conv[2], self._shape_last_conv[3]))
            )

        self.affine_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4*n_conv_chs, emb_size)
            )

    def _resulting_conv(self, n_frequency_bands, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            resulting_conv = self.feature_extractor(torch.Tensor(1, 1, n_frequency_bands, input_size))
            self.feature_extractor.train()
            return resulting_conv

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)
        features = self.global_max_pooling(features)

        if self.return_features:
            return features.flatten(start_dim=1)

        embeddings = self.affine_layer(features)
        return embeddings


    
if __name__ == '__main__':
    inputsize_s = 3.
    n_conv_chs = 32
    sfreq = 200
    n_frequency_bands = 100
    device = 'cuda'
    model = WaveletNet(n_frequency_bands, sfreq,
            n_conv_chs=n_conv_chs, input_size_s=inputsize_s).to(device)
    #summary(model, (1, n_frequency_bands, int(inputsize_s*sfreq)))
    model2 = SignalNet(1, sfreq, n_conv_chs=n_conv_chs,
            input_size_s=inputsize_s).to(device)

    optimizer = torch.optim.Adam(model2.parameters(), lr=5e-4, weight_decay=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()

    tensor = torch.Tensor(1, 1, n_frequency_bands,  int(inputsize_s*sfreq)).to(device)
    tensor2 = torch.Tensor(1, 1, 1, int(inputsize_s*sfreq)).to(device)
    output = model(tensor)
    output2 = model2(tensor2)
    label = torch.Tensor(1, 100).to(device)

    loss = criterion(output2, label)
    loss.backward()
    optimizer.step()

