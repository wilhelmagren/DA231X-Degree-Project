"""

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import numpy as np

from scipy import signal


def CropResize(X, n_partitions=10):
    """
    """
    slice_size = X.shape[0] // n_partitions

    if X.shape[0] % n_partitions != 0:
        raise ValueError(
                f'can`t properly partition X with shape {X.shape} into {n_partitions} partitions ...')
    
    indices = np.arange(n_partitions) 
    choice = np.random.choice(indices)
    
    start, end = int(choice*slice_size), int((choice+1)*slice_size)
    resampled = signal.resample(X[start:end], X.shape[0])

    return resampled[None]


def Permutation(X, n_partitions=10):
    """
    """
    slice_size = X.shape[0] // n_partitions

    if X.shape[0] % n_partitions != 0:
        raise ValueError(
                f'can`t properly partition X with shape {X.shape} into {n_partitions} partitions ...')

    indices = np.random.permutation(n_partitions)
    samples = [(int(i*slice_size), int((i+1)*slice_size)) for i in indices]
    permutated = np.zeros(X.shape).astype(X.dtype)

    for idx, (start, end) in enumerate(samples):
        nstart, nend = int(idx*slice_size), int((idx+1)*slice_size)
        permutated[nstart:nend] = X[start:end]

    return permutated[None]



if __name__ == '__main__':
    x = np.linspace(0, 20, 200)
    y = np.cos(-x**2/10.)

    cropresized = CropResize(y, n_partitions=5)
    permutated = Permutation(y, n_partitions=5)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(y)
    axs[1].plot(cropresized)
    axs[2].plot(permutated)
    plt.show()
       
