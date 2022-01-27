"""
Some signal transformation methods.
See scipy documentation for more information.

Authors: Wilhelm Ågren
Last edited: 26-01-2022
"""
import numpy as np

from scipy import signal


def CWT(data,
        wavelet=signal.ricker,
        widths=100,
        dtype=np.float64,
        batch=False,
        **kwargs):

    if batch:
        batch_scalograms = list()
        for image in data:
            batch_scalograms.append(
                    signal.cwt(image[0, :], wavelet, np.arange(1, widths + 1), 
                        dtype=dtype, **kwargs)[None])

        return np.concatenate(batch_scalograms, axis=0)

    cwtmatrix = signal.cwt(data, wavelet, np.arange(1, widths + 1), dtype=dtype, **kwargs)
    return cwtmatrix

def STFT(data,
        sfreq,
        window='hann',
        nperseg=8,
        noverlap=None,
        return_onesided=True,
        padded=True,
        **kwargs):

    f, t, Sxx = signal.stft(data, sfreq, nperseg=nperseg, noverlap=noverlap, 
            return_onesided=return_onesided, padded=padded, **kwargs)
    return (f, t, Sxx)

