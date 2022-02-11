"""
TODO: implement shallow model for SimCLR, and add Docs!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary

class ResNet18(models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=200):
        super(ResNet18, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


class SimNet(nn.Module):
    def __init__(self, encoder, embedding_size=200, projection_size=100,
            dropout=.25, return_features=False, **kwargs):
        
        super(SimNet, self).__init__()
        self._return_features = return_features
        self._projection_size = projection_size
        self._dropout = dropout

        self.f = encoder
        self.g = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_size, projection_size)
                )

    def forward(self, x):
        features = self.f(x)

        if self._return_features:
            return features

        return self.g(features)



class ShallowSimCLR(nn.Module):
    """neural network pytorch module implementing the encoder f()
    and projection head g() for the SimCLR framework. The given
    hyperparameter and architecture setup yields a model with 
    approximately 32k parameters.

    All in all, f() + g() constitues of the same architectural ideas
    as AlexNet (Krizhevsky, Sutskever, Hinton 2010), but the encoder
    f() is regarded as only the convolutional layers of the model, 
    and g() is the MLP projection head with fully connected layers.
    Given an input shape (1, 128, 128) and 32 filters in the 
    first layer the the model consists of 340k tunable parameters.
    Reducing the number of filters and size of input greatly reduces
    number of parameters. 


    Parameters
    ----------
    input_shape: tuple | list
        Broadcastable struct, consisting of (channel, height, width)
    sfreq: int | float
        The sampling frequency of the data, this is unnecessary...
    """
    def __init__(self, input_shape, sfreq, n_filters=16, emb_size=256,
            projection_head=100, dropout=.25, apply_batch_norm=False,
            return_features=False, **kwargs):

        super(ShallowSimCLR, self).__init__()
        n_channels, height, width = input_shape
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.sfreq = sfreq
        self.return_features = return_features
        self.batch_norm = apply_batch_norm

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.encoder = nn.Sequential(
                nn.Conv2d(n_channels, n_filters, (11, 11), stride=(4, 4)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3), stride=(2, 2)),

                nn.Conv2d(n_filters, n_filters*2, (5, 5), stride=(1, 1), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3), stride=(2, 2)),

                nn.Conv2d(n_filters*2, n_filters*3, (3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(n_filters*3, n_filters*3, (3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(n_filters*3, n_filters*2, (3, 3), stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d((3, 3), stride=(2, 2))
                )
        
        encoder_shape = self._encoder_output_shape(n_channels, height, width)
        self.real_emb_size = len(encoder_shape.flatten())

        self.projection = nn.Sequential(
                nn.Linear(self.real_emb_size, self.real_emb_size),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(self.real_emb_size, projection_head)
                )


    def _encoder_output_shape(self, n_channels, height, width):
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(torch.Tensor(1, n_channels, height, width))
        self.encoder.train()
        return out

    def forward(self, x):
        features = self.encoder(x).flatten(start_dim=1)
        
        if self.return_features:
            return features

        return self.projection(features)


if __name__ == '__main__':
    model = ShallowSimCLR((1, 96, 96), 200, n_filters=32).to('cuda')
    summary(model, (1, 96, 96))

