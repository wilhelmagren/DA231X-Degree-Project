"""
Functions for loading, preparing, and moving batches of data before training on,
respectivelly for each pretext task...

Authors: Wilhelm Ågren, wagren@kth.se
Last edited: 28-01-2022
"""
import torch

from .transforms import CWT_transform


def WaveletTSLoader(data, device, unsqueeze_labels=True, widths=50):
    lanchors, ranchors, samples, labels = data

    cwt_lanchors = torch.Tensor(CWT_transform(lanchors, batch=True, widths=widths)).to(device)
    cwt_ranchors = torch.Tensor(CWT_transform(ranchors, batch=True, widths=widths)).to(device)
    cwt_samples = torch.Tensor(CWT_transform(samples, batch=True, widths=widths)).to(device)
    labels = torch.Tensor(labels).to(device)

    if unsqueeze_labels:
        labels = torch.unsqueeze(labels, dim=1)

    return (cwt_lanchors, cwt_ranchors, cwt_samples, labels)

def WaveletRPLoader(data, device, unsqueeze_labels=True, widths=70):
    anchors, samples, labels = data
    cwt_anchors = torch.Tensor(CWT_transform(anchors, batch=True, widths=widths)).to(device)
    cwt_samples = torch.Tensor(CWT_transform(samples, batch=True, widths=widths)).to(device)
    labels = torch.Tensor(labels).to(device)

    if unsqueeze_labels:
        labels = torch.unsqueeze(labels, dim=1)

    return (cwt_anchors, cwt_samples, labels)

def ExtractTSEmbeddings(emb, sampler, device):
    X = list()
    emb.eval()
    with torch.no_grad():
        for batch, data in enumerate(sampler):
            lanchors, _, _, _ = WaveletTSLoader(data, device)
            embedding = emb(lanchors)
            X.append(embedding[0, :][None])
    X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)

    return X

def ExtractRPEmbeddings(emb, sampler, device, widths):
    X = list()
    emb.eval()
    with torch.no_grad():
        for batch, data in enumerate(sampler):
            anchors, _, _ = WaveletRPLoader(data, device, widths=widths)
            embedding = emb(anchors)
            X.append(embedding[0, :][None])
    X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)

    return X
            
