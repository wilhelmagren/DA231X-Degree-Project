"""
Module for plotting information, such as 
manifold projection of embeddings 
or training history. Work in progress.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 31-01-2022
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from ..utils import RECORDING_ID_MAP

def tSNE_plot(X, title, n_components=2, perplexity=30.0):
    """
    labels are of the form:
        (subj_id, reco_id, gender, age)

    """
    embeddings, Y = X
    tSNE = TSNE(n_components=n_components, perplexity=perplexity)
    components = tSNE.fit_transform(embeddings)

    # format the labels to numpy array, for masking on class labels
    labels = {'sleep': np.ones((len(Y), )),
            'eyes': np.ones((len(Y), )),
            'recording': np.ones((len(Y), )),
            'gender': np.ones((len(Y), )),
            'age': np.ones((len(Y), ))}
    for idx, (subj_id, reco_id, gender, age) in enumerate(Y):
        labels['sleep'][idx] = int(reco_id // 2)
        labels['eyes'][idx] = int(reco_id % 2)
        labels['recording'][idx] = int(reco_id)
        labels['gender'][idx] = int(gender)
        labels['age'][idx] = int(age)

    fig, ax = plt.subplots()
    # plot based on gender first
    gender_labels = [0, 1]
    gender_colors = ['red', 'blue']
    gender_idx2label = {0: 'Female', 1: 'Male'}
    for k, col in zip(gender_labels, gender_colors):
        class_member_mask = labels['gender'] == k
        xy = components[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], alpha=.6, color=col, label=gender_idx2label[k])
    handles, lbls = ax.get_legend_handles_labels()
    uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
    ax.legend(*zip(*uniques))
    fig.suptitle(f'tSNE of embeddings, subject gender, {title} training')
    plt.show()

    fig, ax = plt.subplots()
    # plot psd/con
    sleep_labels = [0, 1]
    sleep_colors = ['red', 'blue']
    sleep_idx2label = {0: 'con', 1: 'psd'}
    for k, col in zip(sleep_labels, sleep_colors):
        class_member_mask = labels['sleep'] == k
        xy = components[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], alpha=.6, color=col, label=sleep_idx2label[k])
    handles, lbls = ax.get_legend_handles_labels()
    uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
    ax.legend(*zip(*uniques))
    fig.suptitle(f'tSNE of embeddings, subject con/psd, {title} training')
    plt.show()


    fig, ax = plt.subplots()
    # plot eyes-open/closed
    eyes_labels = [0, 1]
    eyes_colors = ['red', 'blue']
    eyes_idx2label = {0: 'closed', 1: 'open'}
    for k, col in zip(eyes_labels, eyes_colors):
        class_member_mask = labels['eyes'] == k
        xy = components[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], alpha=.6, color=col, label=eyes_idx2label[k])
    handles, lbls = ax.get_legend_handles_labels()
    uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
    ax.legend(*zip(*uniques))
    fig.suptitle(f'tSNE of embeddings, subject eyes open/closed, {title} training')
    plt.show()

    fig, ax = plt.subplots()
    recording_labels = [0, 1, 2, 3]
    recording_colors = ['red', 'orange', 'blue', 'green']
    recording_idx2label = {0: 'con-ec', 1: 'con-eo', 2: 'psd-ec', 3: 'psd-eo'}
    for k, col in zip(recording_labels, recording_colors):
        class_member_mask = labels['recording'] == k
        xy = components[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], alpha=.6, color=col, label=recording_idx2label[k])
    handles, lbls = ax.get_legend_handles_labels()
    uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
    ax.legend(*zip(*uniques))
    fig.suptitle(f'tSNE of embeddings, subject state/recording, {title} training')
    plt.show()

    fig, ax = plt.subplots()
    age_labels = np.unique(labels['age'])
    print(age_labels)
    age_colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(age_labels))]
    for k, col in zip(age_labels, age_colors):
        class_member_mask = labels['age'] == k
        xy = components[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], alpha=.6, color=col, label=f'Age {k}')
    handles, lbls = ax.get_legend_handles_labels()
    uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
    ax.legend(*zip(*uniques))
    fig.suptitle(f'tSNE of embeddings, subject age, {title} training')
    plt.show()




