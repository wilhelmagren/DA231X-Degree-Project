from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

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

# subj_id, reco_id, gender, age, RTrecipControl, RTrecipSleep, 
# RTControl, RTSleep, RTdiff, minor_lapses_control, minor_lapses_sleep

X_baseline_train, Y_baseline_train = [], []
sleep, eyes, gender, age, rtctr, rtpsd = [], [], [], [], [], []
X_baseline_train_psd = []
X_baseline_train_ctr = []
baseline_encoder.eval()
with torch.no_grad():
    for recording in range(len(samplers['train'].data)):
        for window in range(len(samplers['train'].data[recording])):
            w = torch.Tensor(samplers['train'].data[recording][window][0][None]).float().to(device)
            labels = samplers['train'].labels[recording]
            baseline_feature = baseline_encoder(w.unsqueeze(0))
            X_baseline_train.append(baseline_feature.cpu().detach().numpy())
            sleep.append(labels[1] // 2)
            eyes.append(labels[1] % 2)
            gender.append(labels[2])
            age.append(labels[3])
            if labels[1] // 2:
                rtpsd.append(labels[7])
                X_baseline_train_psd.append(baseline_feature.cpu().detach().numpy())
            else:
                rtctr.append(labels[6])
                X_baseline_train_ctr.append(baseline_feature.cpu().detach().numpy())

X_baseline_train_ctr = np.concatenate(X_baseline_train_ctr)
X_baseline_train_psd = np.concatenate(X_baseline_train_psd)
X_baseline_train = np.concatenate(X_baseline_train)
age = (np.array(age) - 21 ) / 30
rtpsd = np.array(rtpsd)
rtpsd = (rtpsd - rtpsd.max()) / (rtpsd.max() - rtpsd.min())
rtctr = np.array(rtctr)
rtctr = (rtctr - rtctr.max()) / (rtctr.max() - rtctr.min())
baseline_train = {'sleep': (X_baseline_train, np.array(sleep)), 'eyes': (X_baseline_train, np.array(eyes)), 'gender': (X_baseline_train, np.array(gender)), 
'age': (X_baseline_train, age), 'RTctr': (X_baseline_train_ctr, rtctr), 'RTpsd': (X_baseline_train_psd, rtpsd)}
print('training shape: ', X_baseline_train.shape)

X_baseline_test, Y_baseline_test = [], []
sleep, eyes, gender, age, rtctr, rtpsd = [], [], [], [], [], []
X_baseline_test_psd = []
X_baseline_test_ctr = []
with torch.no_grad():
    for recording in range(len(samplers['valid'].data)):
        for window in range(len(samplers['valid'].data[recording])):
            w = torch.Tensor(samplers['valid'].data[recording][window][0][None]).float().to(device)
            labels = samplers['valid'].labels[recording]
            baseline_feature = baseline_encoder(w.unsqueeze(0))
            X_baseline_test.append(baseline_feature.cpu().detach().numpy())
            sleep.append(labels[1] // 2)
            eyes.append(labels[1] % 2)
            gender.append(labels[2])
            age.append(labels[3])
            if labels[1] // 2:
                rtpsd.append(labels[7])
                X_baseline_test_psd.append(baseline_feature.cpu().detach().numpy())
            else:
                rtctr.append(labels[6])
                X_baseline_test_ctr.append(baseline_feature.cpu().detach().numpy())

X_baseline_test_ctr = np.concatenate(X_baseline_test_ctr)
X_baseline_test_psd = np.concatenate(X_baseline_test_psd)
X_baseline_test = np.concatenate(X_baseline_test)
age = (np.array(age) - 21 ) / 30
rtpsd = np.array(rtpsd)
rtpsd = (rtpsd - rtpsd.max()) / (rtpsd.max() - rtpsd.min())
rtctr = np.array(rtctr)
rtctr = (rtctr - rtctr.max()) / (rtctr.max() - rtctr.min())
baseline_test = {'sleep': (X_baseline_test, np.array(sleep)), 'eyes': (X_baseline_test, np.array(eyes)), 'gender': (X_baseline_test, np.array(gender)), 
'age': (X_baseline_test, age), 'RTctr': (X_baseline_test_ctr, rtctr), 'RTpsd': (X_baseline_test_psd, rtpsd)}
print('testing shape: ', X_baseline_test.shape)

models = [LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
    KNeighborsRegressor(n_neighbors=30), KNeighborsRegressor(n_neighbors=30), KNeighborsRegressor(n_neighbors=30)
]
svrs = ['age', 'RTctr', 'RTpsd']
for (label_type, (X_baseline_train, Y_baseline_train)), (X_baseline_test, Y_baseline_test), model in zip(baseline_train.items(), baseline_test.values(), models):
    if label_type in svrs:
        pipline = make_pipeline(StandardScaler(), model)
        pipline.fit(X_baseline_train, Y_baseline_train)
        train_score = mean_squared_error(Y_baseline_train, pipline.predict(X_baseline_train), squared=False)
        test_score = mean_squared_error(Y_baseline_test, pipline.predict(X_baseline_test), squared=False)
        if label_type == 'age':
            ytr_pred = (pipline.predict(X_baseline_train) * 30) + 21
            yte_pred = (pipline.predict(X_baseline_test) * 30) + 21
            ytr = (Y_baseline_train*30) + 21
            yte = (Y_baseline_test*30) + 21
            train_score = mean_squared_error(ytr, ytr_pred, squared=False)
            test_score = mean_squared_error(yte, yte_pred, squared=False)
        print(f'!!!!! Subject {label_type} performance with k-NN:')
        print(f'Train RMSE score: {train_score:.4f}')
        print(f'Test RMSE score: {test_score:.4f}')
    else:
        model.fit(X_baseline_train, Y_baseline_train)
        baseline_train_y_pred = model.predict(X_baseline_train)
        baseline_test_y_pred = model.predict(X_baseline_test)
        baseline_train_balacc = balanced_accuracy_score(Y_baseline_train, baseline_train_y_pred)
        baseline_test_balacc = balanced_accuracy_score(Y_baseline_test, baseline_test_y_pred)

        print(f'!!!!!! Subject {label_type} baseline performance with logistic regression:')
        print(f'Train balanced acc: {baseline_train_balacc:.4f}')
        print(f'Test balanced acc: {baseline_test_balacc:.4f}')

print('\n\n==================================================================')
model = load_model('params.pth').to(device)
model._return_features = True
X_train, Y_train = [], []
sleep, eyes, gender, age, rtctr, rtpsd = [], [], [], [], [], []
X_train_psd = []
X_train_ctr = []
model.eval()
with torch.no_grad():
    for recording in range(len(samplers['train'].data)):
        for window in range(len(samplers['train'].data[recording])):
            w = torch.Tensor(samplers['train'].data[recording][window][0][None]).float().to(device)
            labels = samplers['train'].labels[recording]
            feature = model(w.unsqueeze(0))
            X_train.append(feature.cpu().detach().numpy())
            sleep.append(labels[1] // 2)
            eyes.append(labels[1] % 2)
            gender.append(labels[2])
            age.append(labels[3])
            if labels[1] // 2:
                rtpsd.append(labels[7])
                X_train_psd.append(feature.cpu().detach().numpy())
            else:
                rtctr.append(labels[6])
                X_train_ctr.append(feature.cpu().detach().numpy())
X_train_ctr = np.concatenate(X_train_ctr)
X_train_psd = np.concatenate(X_train_psd)
X_train = np.concatenate(X_train)
age = (np.array(age) - 21 ) / 30
rtpsd = np.array(rtpsd)
rtpsd = (rtpsd - rtpsd.max()) / (rtpsd.max() - rtpsd.min())
rtctr = np.array(rtctr)
rtctr = (rtctr - rtctr.max()) / (rtctr.max() - rtctr.min())
train = {'sleep': (X_train, np.array(sleep)), 'eyes': (X_train, np.array(eyes)), 'gender': (X_train, np.array(gender)), 
'age': (X_train, age), 'RTctr': (X_train_ctr, rtctr), 'RTpsd': (X_train_psd, rtpsd)}
print('training shape: ', X_train.shape)

X_test, Y_test = [], []
sleep, eyes, gender, age, rtctr, rtpsd = [], [], [], [], [], []
X_test_psd = []
X_test_ctr = []
with torch.no_grad():
    for recording in range(len(samplers['valid'].data)):
        for window in range(len(samplers['valid'].data[recording])):
            w = torch.Tensor(samplers['valid'].data[recording][window][0][None]).float().to(device)
            labels = samplers['valid'].labels[recording]
            feature = model(w.unsqueeze(0))
            X_test.append(feature.cpu().detach().numpy())
            sleep.append(labels[1] // 2)
            eyes.append(labels[1] % 2)
            gender.append(labels[2])
            age.append(labels[3])
            if labels[1] // 2:
                rtpsd.append(labels[7])
                X_test_psd.append(feature.cpu().detach().numpy())
            else:
                rtctr.append(labels[6])
                X_test_ctr.append(feature.cpu().detach().numpy())
X_test_ctr = np.concatenate(X_test_ctr)
X_test_psd = np.concatenate(X_test_psd)
X_test = np.concatenate(X_test)
age = (np.array(age) - 21 ) / 30
rtpsd = np.array(rtpsd)
rtpsd = (rtpsd - rtpsd.max()) / (rtpsd.max() - rtpsd.min())
rtctr = np.array(rtctr)
rtctr = (rtctr - rtctr.max()) / (rtctr.max() - rtctr.min())
test = {'sleep': (X_test, np.array(sleep)), 'eyes': (X_test, np.array(eyes)), 'gender': (X_test, np.array(gender)), 
'age': (X_test, age), 'RTctr': (X_test_ctr, rtctr), 'RTpsd': (X_test_psd, rtpsd)}
print('testing shape: ', X_test.shape)

models = [LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', multi_class='ovr'),
    KNeighborsRegressor(n_neighbors=30), KNeighborsRegressor(n_neighbors=30), KNeighborsRegressor(n_neighbors=30)
]
svrs = ['age', 'RTctr', 'RTpsd']
for (label_type, (xtrain, ytrain)), (xtest, ytest), model in zip(train.items(), test.values(), models):

    if label_type in svrs:
        pipline = make_pipeline(StandardScaler(), model)
        pipline.fit(xtrain, ytrain)
        train_score = mean_squared_error(ytrain, pipline.predict(xtrain), squared=False)
        test_score = mean_squared_error(ytest, pipline.predict(xtest), squared=False)
        if label_type == 'age':
            ytr_pred = (pipline.predict(xtrain) * 30) + 21
            yte_pred = (pipline.predict(xtest) * 30) + 21
            ytr = (ytrain*30) + 21
            yte = (ytest*30) + 21
            train_score = mean_squared_error(ytr, ytr_pred, squared=False)
            test_score = mean_squared_error(yte, yte_pred, squared=False)
        print(f'!!!!! Subject {label_type} performance with k-NN:')
        print(f'Train RMSE score: {train_score:.4f}')
        print(f'Test RMSE score: {test_score:.4f}')
    else:
        model.fit(xtrain, ytrain)
        train_y_pred = model.predict(xtrain)
        test_y_pred = model.predict(xtest)
        train_balacc = balanced_accuracy_score(ytrain, train_y_pred)
        test_balacc = balanced_accuracy_score(ytest, test_y_pred)

        print(f'!!!!!! Subject {label_type} performance with logistic regression:')
        print(f'Train balanced acc: {train_balacc:.4f}')
        print(f'Test balanced acc: {test_balacc:.4f}')