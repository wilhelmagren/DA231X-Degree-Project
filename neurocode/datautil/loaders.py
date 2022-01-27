"""
Functions for loading, preparing, and moving batches of data before training on,
respectivelly for each pretext task...

Authors: Wilhelm Ågren, wagren@kth.se
Last edited: 27-01-2022
"""
import torch

from .transforms import CWT


def SSTOLoader(anchors, samples, labels, device, unsqueeze_labels=False, **kwargs):
    signal_anchors = torch.Tensor(anchors).to(device)
    signal_samples = torch.Tensor(samples).to(device)

    scalogram_anchors = torch.Tensor(CWT(anchors, batch=True)).to(device)
    scalogram_samples = torch.Tensor(CWT(samples, batch=True)).to(device)
    
    if unsqueeze_labels:
        labels = torch.unsqueeze(labels, dim=1)
    
    labels = labels.to(device)
    
    return ((signal_anchors, signal_samples),
            (scalogram_anchors, scalogram_samples), labels)

