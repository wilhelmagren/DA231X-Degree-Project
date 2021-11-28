"""
utility functions etc.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 28-11-2021
"""
import os

from enum import Enum


RELATIVE_LABEL_PATH = '../data/subjects.tsv'
RELATIVE_MEG_PATH = '../data/data-ds-200HZ/'
DEFAULT_MEG_CHANNELS = ['MEG0811', 'MEG0812']
RECORDING_ID_MAP = {
        0: 'ses-con_task-rest_ec',
        1: 'ses-con_task-rest_eo',
        2: 'ses-psd_task-rest_ec',
        3: 'ses-psd_task-rest_eo'}

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
    with open(RELATIVE_LABEL:PATH, 'r') as f:
        for line in f.readlines():
            if id in line:
                return float(line.split('\t')[1])
    raise ValueError

def fetch_meg_data(subjects, recordings):
    included_files = list()
    subject_ids = pad_and_stringify(subjects, 2)
    files = list(os.path.join(RELATIVE_DIRPATH, f) for f in list(os.listdir(RELATIVE_DIRPATH)) if get_subject_id(f) in subject_ids)
    for f in files:
        for recording in recordings:
            if RECORDING_ID_MAP[recording] in file:
                included_files.append(f)
    return list((get_subject_id(f), get_recording_id(f), get_subject_gender(f), get_subject_age(f), f) for f in included_files))


