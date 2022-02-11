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

def tSNE_plot(X, title, n_components=2, perplexity=30.0, savefig=True):
    """func applies the non-linear dimensionality reduction technique t-SNE
    to the provided embeddings. X is a tuple containing both the list of
    embddings and list of labels, which are of the form:
    
    >>> embeddings, labels = X
    >>> labels = [(subj_id, reco_id, gender, age), ...]

    After applying t-SNE and transforming the embeddings to cartesian space,
    clamps ALL labels accordingly. Specify saving the plots with
    argument 'savefig'.

    For information about t-SNE, see sklearn.manifold documentation.
    Some parameters are soon to be deprecated, look into it.

    Currently this function is hardcoded for the SLEMEG dataset, 
    modify labels accordingly if you are planning on using this
    visualization with another dataset.

    Parameters
    ----------
    X: tuple
        Contains two items (np.array, list) where the np.array is 
        a collection of extracted embeddings and the list are the
        respective embedding labels. 
    title: str
        Used for the plots, usually either `before` or `after` to 
        specify if embeddings where extracted prior- or post-
        training. 
    n_components: int | float
        Specifier for number of dimensions to reduce embeddings to,
        by means of t-SNE. Since we want a 2D plot, almost always,
        leave this set at 2 as default.
    perplexity: float
        Sets hyperparameter for t-SNE, determines complexity of 
        resulting visualization.
    savefig: bool
       Saves the produced plots if true, filenames are based on 
       subject labels that the plot represents and based on title
       arg as well. 

    For more information on manifold learning, see sklearn.manifold 
    documentation for t-SNE, or see the original paper.
    """
    embeddings, Y = X
    tSNE = TSNE(n_components=n_components, perplexity=perplexity)
    components = tSNE.fit_transform(embeddings)

    # set up the clamping of labels, requires the labels to be stored in numpy
    # arrays from now one, since we want to do masking on the transformed 
    # embedding components. this ultimately makes it so we only have to iterate
    # over the different classes instead of each point.
    n_samples = len(Y)
    labels = {
            'sleep': np.ones((n_samples, )),
            'eyes': np.ones((n_samples, )),
            'recording': np.ones((n_samples, )),
            'gender': np.ones((n_samples, )),
            'age': np.ones((n_samples, ))
            }

    for idx, (subj_id, reco_id, gender, age) in enumerate(Y):
        labels['sleep'][idx] = int(reco_id // 2)
        labels['eyes'][idx] = int(reco_id % 2)
        labels['recording'][idx] = int(reco_id)
        labels['gender'][idx] = int(gender)
        labels['age'][idx] = int(age)

    unique_labels = {
            'sleep': [0, 1],
            'eyes': [0, 1],
            'recording': [0, 1, 2, 3],
            'gender': [0, 1],
            'age': np.unique(labels['age'])
            }

    unique_ll = {
            'sleep': ['control', 'psd'],
            'eyes': ['closed', 'open'],
            'recording': ['control eyes-closed', 'control eyes-open', 'psd eyes-closed', 'psd eyes-open'],
            'gender': ['female', 'male'],
            'age': np.unique(labels['age'])
            }

    for cls in labels:
        fig, ax = plt.subplots()
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels[cls]))]
        for idx, (k, col) in enumerate(zip(unique_labels[cls], colors)):
            class_mask = labels[cls] == k
            xy = components[class_mask]
            ax.scatter(xy[:, 0], xy[:, 1], alpha=.5, color=col, label=unique_ll[cls][idx])
        handles, lbls = ax.get_legend_handles_labels()
        uniques = [(h, l) for i, (h, l) in enumerate(zip(handles, lbls)) if l not in lbls[:i]]
        ax.legend(*zip(*uniques))
        fig.suptitle(f'tSNE of embeddings, subject {cls}, {title} training')
        if savefig:
            plt.savefig(f'tSNE_{cls}_{title}-training.png')
        plt.show()

def history_plot(history, savefig=True):
    """func takes lists of training metrics and visualizes them in a combined plot.
    If you want more customizability then use your own plotting. 

    Parameters
    ----------
    history: dict
        Dictionary containing training/testing metrics, valid keys
        are: `tloss`, `vloss`, `tacc`, `vacc`.
    savefig: bool
        Saves the produces plot to the curent working directory of
        the user. 
    """
    fig, ax1 = plt.subplots(figsize=(8,3))
    ax2 = None

    if 'tacc' in history or 'vacc' in history:
        ax2 = ax1.twinx()
    
    ax1.plot(history['tloss'], ls='-', marker='1', ms=5, alpha=.7,
            color='tab:blue', label='training loss')

    if 'vloss' in history:
        ax1.plot(history['vloss'], ls=':', marker='1', ms=5, alpha=.7,
                color='tab:blue', label='validation loss')

    if 'tacc' in history:
        ax2.plot(history['tacc'], ls='-', marker='2', ms=5, alpha=.7,
                color='tab:orange', label='training acc')

    if 'vacc' in history:
        ax2.plot(history['vacc'], ls=':', marker='2', ms=5, alpha=.7,
                color='tab:orange', label='validation acc')
    
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    lines1, labels1 = ax1.get_legend_handles_labels()

    if ax2:
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylabel('Accuracy [%]', color='tab:orange')
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2)
    else:
        ax1.legend(lines1, labels1)

    plt.tight_layout()
    plt.show()
    
    if savefig:
        plt.savefig(f'training_history.png')

