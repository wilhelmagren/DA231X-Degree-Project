import torch
import numpy as np

from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from neurocode.models import load_model, VGG
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import ScalogramSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import signal as sig
from torchvision import transforms
from scipy.stats import ks_2samp

np.random.seed(73)
torch.manual_seed(73)

manifold = 'tSNE'
load_model_ = False
subjects = list(range(0, 34))
recordings = [0,1,2,3]
batch_size = 64
n_samples = 20
window_size_s = 0.5
shape = (1, 64, 64)
widths = 50
n_views = 2
n_epochs = 100
temperature = .1
sfreq = 200
window_size_samples = np.ceil(sfreq * window_size_s).astype(int)
latent_space_dim = 100
dropout = 0.0
widths = np.arange(1, 51)


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

samplers = {'train': ScalogramSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_views=n_views, widths=50, shape=shape, n_samples=n_samples,
    batch_size=batch_size),
           'valid': ScalogramSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_views=n_views, widths=50, shape=shape, n_samples=n_samples,
    batch_size=batch_size)}

resize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))])
    
model = load_model('params.pth').to(device)
for param in model.parameters():
    param.requires_grad = False

model._return_features = True
model.eval() 
X_features = []
with torch.no_grad():
    for recording in range(len(samplers['valid'].data)):
        for window in range(len(samplers['valid'].data[recording])):
            if window % 10 == 0:
                w = samplers['valid'].data[recording][window][0]
                mat = sig.cwt(w.squeeze(0), sig.ricker, widths)
                mat = resize_transform(mat).to(device)
                feature = model(mat.unsqueeze(0).float())
                X_features.append(feature.cpu().detach().numpy())
X_features = np.concatenate(X_features)

baseline_encoder = VGG(shape, latent_space_dim, n_conv_chs=16, dropout=dropout).to(device) 
baseline_encoder._return_features = True
for parameter in baseline_encoder.parameters():
    parameter.requires_grad = False
baseline_encoder.eval()
X_baseline_test = []
with torch.no_grad():
    for recording in range(len(samplers['valid'].data)):
        for window in range(len(samplers['valid'].data[recording])):
            if window % 10 == 0:
                w = samplers['valid'].data[recording][window][0]
                mat = sig.cwt(w.squeeze(0), sig.ricker, widths)
                mat = resize_transform(mat).to(device)
                labels = samplers['valid'].labels[recording]
                baseline_feature = baseline_encoder(mat.unsqueeze(0).float())
                X_baseline_test.append(baseline_feature.cpu().detach().numpy())
X_baseline_test = np.concatenate(X_baseline_test)

print('[*]  calculating ks-stats and pvalues for you...')
stats, pvals = [], []
for idx in range(X_features.shape[0]):
    stat, pval = ks_2samp(X_features[idx, :], X_baseline_test[idx, :])
    stats.append(stat)
    pvals.append(pval)

print('[*]  done!')
avg_stats = sum(stats) / X_features.shape[0]
avg_pvals = sum(pvals) / X_features.shape[0]
print(f' --> {avg_stats=}')
print(f' --> {avg_pvals=}')

import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.hist(pvals, bins=40, range=(0, 1))
plt.show()