import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(87)
# plt.style.use('seaborn-deep')

centers = [[1.5, 1], [-1.5, -1], [1, -1.5], [-1, 1.5]]
X, labels_true = make_blobs(
    n_samples=500, centers=centers, cluster_std=.3, random_state=0
)


colormap = plt.cm.Spectral
kkk = [2,3,4,5,6,7,8,9]
fig, axs = plt.subplots(1, len(kkk))
for kk in kkk:
    kmeans = KMeans(init='k-means++', n_clusters=kk, n_init=4, random_state=87)
    cobj = kmeans.fit(X)
    labels = cobj.labels_

    kk -= 2

    uniques = set(labels)
    colors = [colormap(each) for each in np.linspace(0, 1, len(uniques))]
    for k, col in zip(uniques, colors):
        class_member_mask = labels == k

        xy = X[class_member_mask]
        axs[kk].plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6
        )


    axs[kk].plot(
        cobj.cluster_centers_[:, 0],
        cobj.cluster_centers_[:, 1],
        "o",
        markerfacecolor="cyan",
        markeredgecolor="k",
        markersize=6
    )

    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, .01), np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, .01))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    axs[kk].imshow(
        Z,
        interpolation='nearest',
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=colormap,
        aspect='auto',
        origin='lower',
    )   

    fig.suptitle("k-Means clustering, k={2, ..., 5}")
plt.show()