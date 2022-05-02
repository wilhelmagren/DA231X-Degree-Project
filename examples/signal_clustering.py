import torch
import numpy as np

from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from neurocode.models import load_model, SignalNet
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import SignalSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

np.random.seed(73)
torch.manual_seed(73)

n_channels = 3 
sfreq = 200
input_size_s = 5.0
n_conv_chs = 50
dropout= 0.0
subjects = list(range(0, 34))
recordings = [0,1,2,3]
batch_size = 1
n_samples = 1
n_views = 2
window_size_samples = np.ceil(sfreq * input_size_s).astype(int)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessors = [Preprocessor(lambda x: x*1e12)]
recording_dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)

preprocess(recording_dataset, [Preprocessor(zscore)])
windows_dataset = create_fixed_length_windows(recording_dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

dataset = RecordingDataset(windows_dataset.datasets, recording_dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = dataset.split_fixed()

samplers = {'train': SignalSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size),
           'valid': SignalSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size)}


model = load_model('params.pth').to(device)
for param in model.parameters():
    param.requires_grad = False

model._return_features = True
model.eval() 
X_features = []
with torch.no_grad():
    for recording in range(len(samplers['valid'].data)):
        for window in range(len(samplers['valid'].data[recording])):
            w = torch.Tensor(samplers['valid'].data[recording][window][0][None]).float().to(device)
            feature = model(w.unsqueeze(0))
            X_features.append(feature.cpu().detach().numpy())

X_features = np.concatenate(X_features)

from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
manifold = TSNE(n_components=2, perplexity=30.0)
components = manifold.fit_transform(X_features)
plt.scatter(components[:, 0], components[:, 1], alpha=.8, s=10.)
plt.savefig('t-SNE_features_signal.png')
plt.clf()

num_clusters = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
best = -1
bestnc = -1
sscores = []
for nc in num_clusters:
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(X_features)
    labels = kmeans.predict(X_features)
    sscores.append(silhouette_score(X_features, labels))
    print(f' {nc} clusters, score: {sscores[-1]:.4f}')
    if sscores[-1] > best:
        bestnc = nc
        best = sscores[-1]
import matplotlib.pyplot as plt
plt.title('Linear search for number of clusters in k-Means')
plt.ylabel('Average Silhouette Coefficient')
plt.xlabel('Number of clusters')
plt.plot([x*2 for x in range(1, 16)], sscores)
plt.savefig('clustering_scores_signal.png')
plt.clf()


"""
VISUALIZE THE CLUSTERS WITH TSNE FOR THE NUMBER OF CLUSTERS WITH HIGHEST AVERAGE SILHOUETTE COEFFICIENT
"""
kmeans = KMeans(n_clusters=bestnc)
kmeans.fit(X_features)
labels_ = kmeans.predict(X_features)
uniques = set(labels_)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(uniques))]
for k, col in zip(uniques, colors):
    mask = labels_ == k
    xy = components[mask]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        color=col,
        alpha=.8, s=10.
    )
plt.title(f'k-Means {bestnc} clusters validation data')
plt.savefig('kmeans_clustering_signal.png')

"""
PERFORM N-FOLD CROSS VALIDATION ON CLUSTER
"""
n = 5
n_samples = X_features.shape[0]
crosscore = []
size = np.ceil(n_samples / 5).astype(int)
#indices = np.arange(X_features.shape[0])
#np.random.shuffle(indices)
#X_features = X_features[indices]
for i in range(n):
    lidx = size * i
    ridx = size * (i+1)
    lx = X_features[:lidx]
    rx = X_features[ridx:]
    fitte = np.concatenate((lx, rx))
    kmeans = KMeans(n_clusters=bestnc)
    kmeans.fit(fitte)
    labels = kmeans.predict(fitte)
    crosscore.append(silhouette_score(fitte, labels))
    print(f'score: {crosscore[-1]:.4f}')

plt.clf()
plt.plot(crosscore)
ax = plt.gca()
ax.set_ylim((-1, 1))
plt.savefig('nfold.png')
scores = np.array(crosscore)
mean = scores.mean()
std = scores.std()
print(f'mean: {mean},  std: {std}')

plt.clf()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X_features) + (bestnc + 1) * 10])

# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = KMeans(n_clusters=bestnc, random_state=10)
cluster_labels = clusterer.fit_predict(X_features)

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X_features, cluster_labels)
print(
    "For n_clusters =",
    bestnc,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_features, cluster_labels)

y_lower = 10
for i in range(bestnc):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.Accent(float(i) / bestnc)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The Silhouette plot for the clusters.")
ax1.set_xlabel("Silhouette Coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = plt.cm.Accent(cluster_labels.astype(float) / bestnc)
ax2.scatter(
    components[:, 0], components[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
)

# Labeling the clusters
#centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
#ax2.scatter(
#    centers[:, 0],
#    centers[:, 1],
#    marker="o",
#    c="white",
#    alpha=1,
#    s=200,
#    edgecolor="k",
#)

#for i, c in enumerate(centers):
#    ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

ax2.set_title("t-SNE visualization of the clustered features.")
#ax2.set_xlabel("Feature space for the 1st feature")
#ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(
    "Silhouette analysis for k-Means clustering on validation data with k=%d"
    % bestnc,
    fontsize=14,
    fontweight="bold",
)

plt.savefig('silhouette_stuffs_signal.png')