import torch
import torch.nn as nn
import numpy as np

from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from neurocode.models import load_model, SignalNet
from neurocode.datasets import SLEMEG, RecordingDataset
from neurocode.samplers import SignalSampler
from neurocode.datautil import manifold_plot
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

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

baseline_encoder = SignalNet(n_channels, sfreq, input_size_s=input_size_s, n_filters=n_conv_chs, dropout=dropout).to(device)
baseline_encoder._return_features = True
for parameter in baseline_encoder.parameters():
    parameter.requires_grad = False


preprocessors = [Preprocessor(lambda x: x*1e12)]
recording_dataset = SLEMEG(subjects=subjects, recordings=recordings, preload=True,
        load_meg_only=True, preprocessors=preprocessors, cleaned=True)


windows_dataset = create_fixed_length_windows(recording_dataset, start_offset_samples=0,
        stop_offset_samples=0, drop_last_window=True, window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True)

preprocess(windows_dataset, [Preprocessor(zscore)])
dataset = RecordingDataset(windows_dataset.datasets, recording_dataset.labels, sfreq=sfreq, channels='MEG')
train_dataset, valid_dataset = dataset.split_fixed()

samplers = {'train': SignalSampler(train_dataset.get_data(), train_dataset.get_labels(),
    train_dataset.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size),
           'valid': SignalSampler(valid_dataset.get_data(), valid_dataset.get_labels(),
    valid_dataset.get_info(), n_channels=n_channels, 
    n_views=n_views, n_samples=n_samples, batch_size=batch_size)}

model = load_model('params.pth').to(device)
manifold='tSNE'


manifold_plot(samplers['train']._extract_features(model, device), 'train-data_pre', technique=manifold)
manifold_plot(samplers['valid']._extract_features(model, device), 'valid-data_pre', technique=manifold)