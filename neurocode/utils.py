"""
utility functions et al.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 29-11-2021
"""
import mne
import os
import torch

from enum import Enum

CWD = os.getcwd()
RELATIVE_LABEL_PATH = os.path.join(CWD, 'data/subjects.tsv')
RELATIVE_MEG_PATH = os.path.join(CWD, 'data/data-ds-200Hz/')
DEFAULT_MEG_CHANNELS = ['MEG0811', 'MEG0812']
RECORDING_ID_MAP = {
        0: 'ses-con_task-rest_ec',
        1: 'ses-con_task-rest_eo',
        2: 'ses-psd_task-rest_ec',
        3: 'ses-psd_task-rest_eo'}

def BCEWithLogitsAccuracy(outputs, labels):
    outputs, labels = torch.flatten(outputs), torch.flatten(labels)
    outputs = outputs > 0.
    return (outputs == labels).sum().item()

def load_raw_fif(fpath, info, drop_channels=False):
    raw = mne.io.read_raw_fif(fpath, preload=True)   # we need to preload, otherwise can't access data
    
    if raw.info['sfreq'] != info['sfreq']:
        raise ValueError('specified sampling frequency does not equal that of the data!')

    if drop_channels:
        exclude = list(ch for ch in list(map(lambda ch: None if ch in info['channels'] else ch, raw.info['ch_names'])) if ch)
        raw.drop_channels(exclude)
    return raw

def create_epochs(raw, info, s_start_trials=5., s_end_trials=5.):
    sfreq = info['sfreq']
    s_epoch_size = info['s_epoch_size']
    n_samples_in_epoch = int(sfreq * s_epoch_size)
    offset_start = int(sfreq * s_start_trials)
    offset_end = int(sfreq * s_end_trials)
    arr_raw = raw.get_data()
    n_channels, n_samples = arr_raw.shape

    if (n_samples - offset_start - offset_end) <= 0:
        raise ValueError(f'Not enough samples in recording to trim start or end, s_start_trials={s_start_trials}s, s_end_trials={s_end_trials}s')

    n_epochs = int((n_samples - offset_start - offset_end) // n_samples_in_epoch)
    recording_epochs = list()
    for epoch in range(n_epochs):
        i_end_epoch = offset_start + (epoch + 1) * n_samples_in_epoch 
        i_start_epoch = offset_start + epoch * n_samples_in_epoch
        arr_epoch = arr_raw[:, i_start_epoch:i_end_epoch]
        recording_epochs.append(arr_epoch)
    return recording_epochs

def pad_and_stringify(iterable, num):
    return list(map(lambda s: '0'+str(s) if len(str(s)) != num else str(s), iterable))

def get_subject_id(f):
    return f.split('_')[0].split('-')[-1]

def get_recording_id(f):
    for key, val in RECORDING_ID_MAP.items():
        if val in f:
            return key
    raise ValueError

def get_subject_gender(f):
    id = get_subject_id(f)
    with open(RELATIVE_LABEL_PATH, 'r') as f:
        for line in f.readlines():
            if id in line:
                return 0. if line.split('\t')[2] == 'F' else 1.
    raise ValueError

def get_subject_age(f):
    id = get_subject_id(f)
    with open(RELATIVE_LABEL_PATH, 'r') as f:
        for line in f.readlines():
            if id in line:
                return float(line.split('\t')[1])
    raise ValueError

def fetch_meg_data(subjects, recordings):
    included_files = list()
    subject_ids = pad_and_stringify(subjects, 2)
    files = list(os.path.join(RELATIVE_MEG_PATH, f) for f in list(os.listdir(RELATIVE_MEG_PATH)) if get_subject_id(f) in subject_ids)
    for f in files:
        for recording in recordings:
            if RECORDING_ID_MAP[recording] in f:
                included_files.append(f)
    return list((get_subject_id(f), get_recording_id(f), get_subject_gender(f), get_subject_age(f), f) for f in included_files)


