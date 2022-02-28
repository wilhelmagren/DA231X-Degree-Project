"""
Scalogram SimCLR sampler implementation, inherits from the base 
PretextSampler, and depends on the ContrastiveViewGenerator
object to perform data augmentations.

Recap; the SimCLR pipeline constitutes of four main modules:
    1: data augmentation T
    2: encoder model f()
    3: projection head g()
    4: contrastive loss function NTXentLoss

I tried to perform the Scalogram SimCLR pipeline by means of a 
regular Dataset, but this had some dire performance issues, since
all data had to lie in RAM, this sampling method keeps the raw signal
data in RAM, but not the CWT images which takes up a lot more data...

But, stochastically sampling windows yields batches that should contain
a lot more contrastive information, compared to the Dataset alternative,
so I believe that this approach could be beneficial... remains to be 
proven by the results. Initial results of the tSNE plots indicate that 
the model is not 'overfitting' as much when doing this sampling, most
likely because the model is most likely not shown all datapoints.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal as sig
from torchvision import transforms
from .base import PretextSampler


class ContrastiveViewGenerator(object):
    """callable object that generates n_views
    of the given input data x. currently
    is lowkey hardcoded for two views, modify
    the transforms attribute of the object
    if you want more views.

    Attributes
    ----------
    T: tuple | list
        A collection holding the torchvision.transforms, length of T
        should be equal to n_views parameter.
    n_views: int
        Number dictating the amount of different augmentations
        to apply to x, and is the length of the resulting list.
    shape: tuple | list
       The shape of the resulting image after augmentations,
       passed from the Sampler module.

    """
    def __init__(self, T, n_views, shape):
        self.transforms = T

        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms[t](x) for t in range(self.n_views)]


class ScalogramSampler(PretextSampler):
    """pretext task sampler for the SimCLR pipeline,
    applying data augmentations to scalogram (CWT) of
    the MEG signal. currently applies two augmentations:
    RandomResizeCrop() and RandomVerticalFlip(), the 
    strength of these two together is not too good, 
    see literature in thesis, but should present 
    the model with two images that are contrastive
    in nature.

    Attributes
    ----------
    widths: int
        Specifies the upper bound for number of scales
        to use in CWT, assumes incremental increase
        of +1. Look how this results when having
        low widths, since CWT scales are exponential.
    signal: str | signal
        Decides what mother wavelet transform to use
        for CWT. Defaults to `ricker`, mexican hat,
        because literature uses this. And, it is a 
        derived from a Gaussian, so its good for
        natural data.
    n_views: int
        Number of different views to generate,
        SimCLR wants 2 generally, so don't change
        this please.
    shape: tuple | list | None
        Specify the shape of the resulting image
        after augmenting it. Small CNNs generally
        want close to 96x96, but larger nets can
        have up to 255x255.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parameters(self, widths=50, signal='ricker',
            n_views=2, shape=None, **kwargs):

        if not shape:
            shape = (96, 96)
        
        self.widths = np.arange(1, widths + 1)
        self.signal = sig.ricker
        self.n_views = n_views
        self.shape = shape

        self._transforms = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=shape, scale=(.5, .8)),
                transforms.ToTensor()]),
            transforms.Compose([
                transforms.RandomPosterize(bits=3, p=.9999),
                transforms.Resize(size=shape),
                transforms.ToTensor()])]
                

        self.transformer = ContrastiveViewGenerator(
                self._transforms, n_views, shape)

    def _sample_pair(self):
        """sample two augmented views (t1, t2) of the same original datapoint xt,
        as per the SimCLR framework suggestions. randomly picks a recording and
        corresponding window to transform. the augmentation is performed on the
        scalogram of the sampled signal, meaning, the Continuous Wavelet Transform (CWT).
        it is parametrized by the scale/widths, which together with the time-length
        of the sampled window determines the scalogram dimensions. all scalograms
        are before being transformed resized to attribute self.shape, such that the
        encoder f() can operate on the images.
        
        Returns
        -------
        ANCHORS: torch.tensor
            The scalograms which has had the first transform applied to them. 
            It should hvae shape (N, C, H, W) where N is the batch size, C 
            is the number of color channels, H is height and W is width of
            the provided scalogram image.
        SAMPLES: torch.tensor
            Same as above, but scalograms which has had the second transform
            applied to them.

        """
        batch_anchors = list()
        batch_samples = list()
        for _ in range(self.batch_size):
            reco_idx = self._sample_recording()
            wind_idx = self._sample_window(recording_idx=reco_idx)

            x = self.data[reco_idx][wind_idx][0]
            scalogram = sig.cwt(x.squeeze(0), self.signal, self.widths)
            scalogram = scalogram[:(self.widths[-1] // 2), :]
            scalogram = ((scalogram - scalogram.min()) * (1/(scalogram.max() - scalogram.min()) * 255)).astype('uint8')
            image = Image.fromarray(scalogram)
            T1, T2 = self.transformer(image)

            """
            fig, axs = plt.subplots(1, 4)
            axs[0].plot(x.squeeze(0))
            axs[1].imshow(scalogram, extent=[-1, 1, 1, self.widths[-1]], aspect='auto')
            arr1 = torch.swapaxes(torch.swapaxes(T1, 0, 2), 0, 1).numpy().squeeze(2)
            axs[2].imshow(arr1, extent=[-1, 1, 1, self.widths[-1]], aspect='auto')
            arr2 = torch.swapaxes(torch.swapaxes(T2, 0, 2), 0, 1).numpy().squeeze(2)
            axs[3].imshow(arr2, extent=[-1, 1, 1, self.widths[-1]], aspect='auto')
            plt.show()
            """

            batch_anchors.append(T1.unsqueeze(0))
            batch_samples.append(T2.unsqueeze(0))

        ANCHORS = torch.cat(batch_anchors)
        SAMPLES = torch.cat(batch_samples)

        return (ANCHORS, SAMPLES)

    def _extract_features(self, model, device):
        """sample every 200th window from each recording
        and extract the f() features of the window.
        make sure that the corresponding labels are 'sampled'
        pairwise to the features, such that the tSNE plots 
        can be class-colored accordingly.

        Parameters
        ----------
        model: torch.nn.Module
            the neural network model, encoder f(), which is yet to
            be or has already been trained and performs feature 
            extraction. make sure to set model in evaluation mode,
            disabling dropout, BN etc., and enable returning features
            which in practice means that the f() output is not 
            fed to g().
        device: str | torch.device
            the device on which to perform feature extraction,
            should be the same as that of the model.

        Returns
        -------
        X: np.array
            The extracted features, casted to numpy arrays and forced to 
            move to the CPU if they were on another device. The amount
            of features to extract is a bit arbitrary, and depends
            on the window_size_s of the pipeline and the amount of
            recordings provided to the sampler instance.
        Y: np.array
            The corresponding labels for the extracted features. Used
            such that the tSNE plots can be labeled accordingly, and 
            has to be the same length as X.

        """
        X, Y = [], []
        model.eval()
        model._return_features = True
        resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.shape)])

        with torch.no_grad():
            for recording in range(len(self.data)):
                for window in range(len(self.data[recording])):
                    if window % 50 == 0:
                        window = self.data[recording][window][0]
                        matrix = sig.cwt(window.squeeze(0), self.signal, self.widths)
                        matrix = resize_transform(matrix).to(device)
                        feature = model(matrix.unsqueeze(0).float())
                        X.append(feature[0, :][None])
                        Y.append(self.labels[recording])
        X = np.concatenate([x.cpu().detach().numpy() for x in X], axis=0)
        model._return_features = False

        return (X, Y)

