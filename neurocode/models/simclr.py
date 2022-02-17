"""
SimCLR torch.nn.Module(s) for ScalogramSampler. Currently, AlexNet and ResNet18
implementations are available. ResNet18 is larger, deeper, requires more VRAM
to properly traing with suitable batch sizes, AlexNet is easier to train.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary

class ResNet18(models.resnet.ResNet):
    """neural network pytorch module, specifically ResNet18,
    but modify first convolution so that 1 channel images can be
    used, i.e. the scalograms. this model is large, thus, causes
    potential VRAM allocation failures when training on CUDA.

    Parameters
    ----------
    block: torch.nn.Module
        The resdiual blocks to use in the model. See pytorch documentation
        for detailed information.
    layers: tuple | list
        Collection specifying how many convolutions each layer in the model
        should have. For ResNet18 it is something like [2,2,2,2], again see
        documentation on pytorch.
    num_classes: int
        The dimensionality of the encoder f() output, i.e. the latent space
        z. For larger ResNet this is something like 1000, but we use 200 here.
    """
    def __init__(self, block, layers, num_classes=200):
        super(ResNet18, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


class SimCLR(nn.Module):
    """pytorch moduel constituting f() + g(), implementing g() but taking
    f() as parameter. g() maps g: embedding_size -> projection_size in a 
    mathematical sense, by implementing an MLP, be aware of overfitting...
    
    Parameters
    ----------
    encoder: torch.nn.Module
        f() that performs the real feature extraction
    embedding_size: int
        The dimensionality of the latent space z.
    projection_size: int
        The dimensionality of the space in which to perform similarity measuremnts
        and subsequentially applying the NTXent loss.
    dropout: float
        the amount of dropout to apply in the g() MLP. should most certainly 
        be a rather large amount, as to avoid overfitting.
    return_features: bool
        this is a settable attribute, and defaults to False when initializing
        the module. invert this when performing feature extraction using f().

    """
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
        The sampling frequency of the data, this is unnecessary if the
        padding and stride is not dependent on the size of the input...
    n_filters: int
        The number of convolutional filters, channels, to apply in the 
        first layer. This number is multiplied linearly as we go
        deeper in the model.
    emb_size: int
        The wanted latent space z size, this is ignored currently.
    projection_head: int
        The size of resulting g() mapping, where the loss is applied.
    dropout: float
        The amount of dropout to apply, set to large amount when MLP
        g() is big.
    apply_batch_norm: bool
        Currently, no batch normalization is applied. But this parameter
        specifies whether or not it should be applied after convolutions.
    return_features: bool
        settable attribute, invert this when extracting features
        with f(), defaults to False since expecting training 
        to be first mode the model should be in.

    """
    def __init__(self, input_shape, sfreq, n_filters=16, emb_size=256,
            projection_head=100, dropout=.25, apply_batch_norm=False,
            return_features=False, **kwargs):

        super(ShallowSimCLR, self).__init__()
        height, width = input_shape
        self.height = height
        self.width = width
        self.sfreq = sfreq
        self.return_features = return_features
        self.batch_norm = apply_batch_norm

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.encoder = nn.Sequential(
                nn.Conv2d(1, n_filters, (11, 11), stride=(3, 3)),
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
        
        encoder_shape = self._encoder_output_shape(height, width)
        self.real_emb_size = len(encoder_shape.flatten())

        self.projection = nn.Sequential(
                nn.Linear(self.real_emb_size, self.real_emb_size),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(self.real_emb_size, self.real_emb_size),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(self.real_emb_size, projection_head)
                )


    def _encoder_output_shape(self, height, width):
        """calculate the output shape of f(), such that we can
        connect the flattened output to the projection head g().
        makes sure that no gradients are collected, speed++.

        Parameters
        ----------
        height: int
            The height of the input image x.
        width: int
            The width of the input image x. Thus, there is no requirement
            of symmetric image. Could be dis-proportional, i.e. more 
            time contra frequency in the scalogram.

        Returns
        -------
        torch.tensor
            the output tensor of the encoder f(), such that flattened length
            can be calculated and nn.Linear layers set accordingly.

        """
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(torch.Tensor(1, 1, height, width))
        self.encoder.train()
        return out

    def forward(self, x):
        features = self.encoder(x).flatten(start_dim=1)
        
        if self.return_features:
            return features

        return self.projection(features)



class SignalNet(nn.Module):
    def __init__(self, n_channels, sfreq, n_filters=16, projection_size=100,
            input_size_s=5., time_conv_size_s=.1, max_pool_size_s=.05,
            pad_size_s=.05, dropout=.5, apply_batch_norm=False, return_features=False,
            **kwargs):

        super(SignalNet, self).__init__()
        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * sfreq).astype(int)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * sfreq).astype(int)

        self.n_channels = n_channels
        if n_channels > 1:
            self.f_spatial = nn.Conv2d(1, n_channels, (n_channels, 1))

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.f_temporal = nn.Sequential(
                nn.Conv2d(1, n_filters, (1, time_conv_size), padding=(0, pad_size)),
                batch_norm(n_filters),
                nn.ReLU(),
                nn.MaxPool2d((1, max_pool_size)),

                nn.Conv2d(n_filters, n_filters*2, (1, time_conv_size), padding=(0, pad_size)),
                batch_norm(n_filters*2),
                nn.ReLU(),
                nn.MaxPool2d((1, max_pool_size))
                )
        
        encoder_output = self._encoder_output_shape(n_channels, input_size)
        self._encoder_output_size = len(encoder_output.flatten())
        self._return_features = return_features

        self.g = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self._encoder_output_size, self._encoder_output_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self._encoder_output_size, projection_size)
                )

    def _encoder_output_shape(self, n_channels, input_size):
        self.f_temporal.eval()
        with torch.no_grad():
            out = self.f_temporal(torch.Tensor(1, 1, n_channels, input_size))
        self.f_temporal.train()
        return out
    
    def forward(self, x):
        if x.ndim == 3:
            x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.f_spatial(x)
            x = x.transpose(1, 2)

        features = self.f_temporal(x).flatten(start_dim=1)

        if self._return_features:
            return features
        
        return self.g(features)



if __name__ == '__main__':
    if torch.cuda.is_available():
        # model = SignalNet(1, 200, n_filters=32, input_size_s=5).to('cuda')
        model = ShallowSimCLR((64, 128), 200, n_filters=32, dropout=.5).to('cuda')
        summary(model, (1, 64, 128))
    else:
        print('Device `CUDA` is not available, can`t see summary of model...')

