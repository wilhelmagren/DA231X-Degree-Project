"""
utility functions et al.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 22-02-2022
"""
import mne
import os
import torch
import pandas as pd


__rightfrontal__ = [
    'MEG0811',
    'MEG0812',
    'MEG0813',
    'MEG0911',
    'MEG0912',
    'MEG0913',
    'MEG0921',
    'MEG0922',
    'MEG0923',
    'MEG0931',
    'MEG0932',
    'MEG0933',
    'MEG0941',
    'MEG0942',
    'MEG0943',
    'MEG1011',
    'MEG1012',
    'MEG1013',
    'MEG1021',
    'MEG1022',
    'MEG1023',
    'MEG1031',
    'MEG1032',
    'MEG1033',
    'MEG1211',
    'MEG1212',
    'MEG1213',
    'MEG1221',
    'MEG1222',
    'MEG1223',
    'MEG1231',
    'MEG1232',
    'MEG1233',
    'MEG1241',
    'MEG1242',
    'MEG1243',
    'MEG1411',
    'MEG1412',
    'MEG1413'
]

__leftfrontal__ = [
    'MEG0121',
    'MEG0122',
    'MEG0123',
    'MEG0311',
    'MEG0312',
    'MEG0313',
    'MEG0321',
    'MEG0322',
    'MEG0323',
    'MEG0331',
    'MEG0332',
    'MEG0333',
    'MEG0341',
    'MEG0342',
    'MEG0343',
    'MEG0511',
    'MEG0512',
    'MEG0513',
    'MEG0521',
    'MEG0522',
    'MEG0523',
    'MEG0531',
    'MEG0532',
    'MEG0533',
    'MEG0541',
    'MEG0542',
    'MEG0543',
    'MEG0611',
    'MEG0612',
    'MEG0613',
    'MEG0621',
    'MEG0622',
    'MEG0623',
    'MEG0641',
    'MEG0642',
    'MEG0643',
    'MEG0821',
    'MEG0822',
    'MEG0823'
]

__righttemporal__ = [
    'MEG1311',
    'MEG1312',
    'MEG1313',
    'MEG1321',
    'MEG1322',
    'MEG1323',
    'MEG1331',
    'MEG1332',
    'MEG1333',
    'MEG1341',
    'MEG1342',
    'MEG1343',
    'MEG1421',
    'MEG1422',
    'MEG1423',
    'MEG1431',
    'MEG1432',
    'MEG1433',
    'MEG1441',
    'MEG1442',
    'MEG1443',
    'MEG2411',
    'MEG2412',
    'MEG2413',
    'MEG2421',
    'MEG2422',
    'MEG2423',
    'MEG2611',
    'MEG2612',
    'MEG2613',
    'MEG2621',
    'MEG2622',
    'MEG2623',
    'MEG2631',
    'MEG2632',
    'MEG2633',
    'MEG2641',
    'MEG2642',
    'MEG2643'
]

__lefttemporal__ = [
    'MEG0111',
    'MEG0112',
    'MEG0113',
    'MEG0131',
    'MEG0132',
    'MEG0133',
    'MEG0141',
    'MEG0142',
    'MEG0143',
    'MEG0211',
    'MEG0212',
    'MEG0213',
    'MEG0221',
    'MEG0222',
    'MEG0223',
    'MEG0231',
    'MEG0232',
    'MEG0233',
    'MEG0241',
    'MEG0242',
    'MEG0243',
    'MEG1511',
    'MEG1512',
    'MEG1513',
    'MEG1521',
    'MEG1522',
    'MEG1523',
    'MEG1531',
    'MEG1532',
    'MEG1533',
    'MEG1541',
    'MEG1542',
    'MEG1543',
    'MEG1611',
    'MEG1612',
    'MEG1613',
    'MEG1621',
    'MEG1622',
    'MEG1623'
]

__rightparietal__ = [
    'MEG0721',
    'MEG0722',
    'MEG0723',
    'MEG0731',
    'MEG0732',
    'MEG0733',
    'MEG1041',
    'MEG1042',
    'MEG1043',
    'MEG1111',
    'MEG1112',
    'MEG1113',
    'MEG1121',
    'MEG1122',
    'MEG1123',
    'MEG1131',
    'MEG1132',
    'MEG1133',
    'MEG1141',
    'MEG1142',
    'MEG1143',
    'MEG2021',
    'MEG2022',
    'MEG2023',
    'MEG2211',
    'MEG2212',
    'MEG2213',
    'MEG2221',
    'MEG2222',
    'MEG2223',
    'MEG2231',
    'MEG2232',
    'MEG2233',
    'MEG2241',
    'MEG2242',
    'MEG2243',
    'MEG2441',
    'MEG2442',
    'MEG2443'
]

__leftparietal__ = [
    'MEG0411',
    'MEG0412',
    'MEG0413',
    'MEG0421',
    'MEG0422',
    'MEG0423',
    'MEG0431',
    'MEG0432',
    'MEG0433',
    'MEG0441',
    'MEG0442',
    'MEG0443',
    'MEG0631',
    'MEG0632',
    'MEG0633',
    'MEG0711',
    'MEG0712',
    'MEG0713',
    'MEG0741',
    'MEG0742',
    'MEG0743',
    'MEG1811',
    'MEG1812',
    'MEG1813',
    'MEG1821',
    'MEG1822',
    'MEG1823',
    'MEG1831',
    'MEG1832',
    'MEG1833',
    'MEG1841',
    'MEG1842',
    'MEG1843',
    'MEG1631',
    'MEG1632',
    'MEG1633',
    'MEG2011',
    'MEG2012',
    'MEG2013'
]

__root__ = os.getcwd()
RELATIVE_LABEL_PATH = os.path.join(__root__, 'data/subjects.tsv')
RELATIVE_MEG_PATH = os.path.join(__root__, 'data/data-ds-200Hz/')
RELATIVE_CLEANED_MEG_PATH = os.path.join(__root__, 'data/data-cleaned/')
DEFAULT_MEG_CHANNELS =  ['MEG2121', 'MEG2131', 'MEG2141'] #, 'MEG2342', 'MEG2343']
RECORDING_ID_MAP = {
        0: 'ses-con_task-rest_ec',
        1: 'ses-con_task-rest_eo',
        2: 'ses-psd_task-rest_ec',
        3: 'ses-psd_task-rest_eo'}

def recording_train_valid_split(recordings, split=.6):
    split_idx = int(len(recordings) * split)
    train_indices = list(range(split_idx))
    valid_indices = list(range(split_idx, len(recordings)))

    recordings_train = {k: recordings[k] for k in train_indices}
    recordings_valid = {k: recordings[k] for k in valid_indices}
    return (recordings_train, recordings_valid)

def BCEWithLogitsAccuracy(outputs, labels):
    outputs, labels = torch.flatten(outputs), torch.flatten(labels)
    outputs = outputs > 0.
    return (outputs == labels).sum().item()

def load_raw_fif(fpath, subj_id, reco_id, preload, drop_channels=False):
    raw = mne.io.read_raw_fif(fpath, preload=True)   # we need to preload, otherwise can't access data
    
    if drop_channels:
        exclude = list(ch for ch in list(map(lambda ch: None if ch in DEFAULT_MEG_CHANNELS else ch, raw.info['ch_names'])) if ch)
        raw.drop_channels(exclude)
    
    desc = pd.Series({'subject': subj_id, 'recording': reco_id}, name='')
    return raw, desc

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
            if ('sub-'+id) in line:
                return 0. if line.split('\t')[2] == 'F' else 1.
    raise ValueError

def get_subject_age(f):
    id = get_subject_id(f)
    with open(RELATIVE_LABEL_PATH, 'r') as f:
        for line in f.readlines():
            if ('sub-'+id) in line:
                return float(line.split('\t')[1])
    raise ValueError

def get_subject_RT(f):
    id = get_subject_id(f)
    with open(RELATIVE_LABEL_PATH, 'r') as f:
        for line in f.readlines():
            if ('sub-'+id) in line:
                return [float(item) for item in line.split('\t')[5:10]]
    raise ValueError(
        f'filepath doesn`t exist, {f}')

def fetch_meg_data(subjects, recordings, cleaned):
    megpath = RELATIVE_CLEANED_MEG_PATH if cleaned else RELATIVE_MEG_PATH
    included_files = []
    subject_ids = pad_and_stringify(subjects, 2)
    files = [os.path.join(megpath, f) for f in list(os.listdir(megpath)) if get_subject_id(f) in subject_ids]
    for f in files:
        for recording in recordings:
            if RECORDING_ID_MAP[recording] in f:
                included_files.append(f)
    return [(get_subject_id(f), get_recording_id(f), get_subject_gender(f), 
    get_subject_age(f), *get_subject_RT(f), f) for f in included_files]


