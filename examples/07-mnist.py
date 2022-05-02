import requests
import gzip
import os
import sys
import numpy as np

from pathlib import Path


def _fetch(url):
    root = Path(os.getcwd())
    datafolder = Path(root, '.data/')

    if not datafolder.exists():
        datafolder.mkdir(parents=True)

    filepath = url.replace('/', '-')
    filepath = Path(root, f'.data/{filepath}')
    if filepath.exists():
        sys.stdout.write(f'{url=} already exists, reading it... ')
        with open(filepath, 'rb') as f:
            data = f.read()
    else:
        sys.stdout.write(f'{url=} was not found, downloading it... ')
        filepath.touch()
        with open(filepath, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    sys.stdout.write('done!\n')
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def _remove(url):
    root = Path(os.getcwd())
    datafolder = Path(root, '.data/')

    if not datafolder.exists():
        raise FileNotFoundError(
        f'trying to remove downloaded file but no data folder exists...')

    filepath = url.replace('/', '-')
    filepath = Path(root, f'.data/{filepath}')
    if filepath.exists():
        sys.stdout.write(f'removing {url=} ')
        filepath.unlink(filepath)
        sys.stdout.write('done!\n')
        return True
    raise FileNotFoundError(
    f'{url=} was not found in .data/ directory... skipping')
    return False


def fetch_mnist(version='digits', clean=False):
    urls = {}
    urls['digits'] = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    urls['fashion'] = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']

    if version not in urls:
        raise ValueError(
        f'provided MNIST version does not exist, {version=}')

    datasets = [_fetch(url) for url in urls[version]]
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[8:] if i % 2 else dataset[0x10:].reshape((-1, 28, 28))

    if clean:
        cleaned = [_remove(url) for url in urls[version]]
        if not all(cleaned):
            print('could not remove all downloaded files, manually inspect this')

    return datasets


_, _, X_test, Y_test = fetch_mnist()
X_test = X_test.reshape((-1, 784))
X_test = X_test[:10000]
Y_test = Y_test[:10000]
print(X_test.shape)
print(Y_test.shape)

from sklearn.manifold import TSNE
manifold = TSNE(n_components=2, perplexity=30)
components = manifold.fit_transform(X_test).astype(np.float32)
print(components.shape)
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, 10)]
for k, col in zip(range(10), colors):
    mask = Y_test == k
    xy = components[mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=10, label=f'Digit: {k}')
plt.legend()
#plt.savefig('mnist_t-SNE.png')
plt.show()