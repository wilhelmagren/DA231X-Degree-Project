"""
SimCLR training module, implementing Info NCE Loss.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import torch
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

torch.manual_seed(0)


class SimCLR(object):
    """
    """
    def __init__(self, model, device,
            optimizer=None,
            scheduler=None,
            criterion=None,
            batch_size=256,
            epochs=100,
            temperature=.7,
            n_views=2,
            **kwargs):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        self.temperature = temperature
        self.n_views = n_views
        
    def info_nce_loss(self, features):
        """
        """
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(self.device)

        # to use cosine-similarity it is required that the features are normalized, and non-zero
        features = F.normalize(features, dim=1)
        sim_mat = torch.matmul(features, features.T)

        assert sim_mat.shape == labels.shape, (
                'Labels and similarity matrix doesn`t have matching shapes.')

        # discard the diagonal from both labels and similarity matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1)

        # select and combine the multiple positives, and select negatives
        positives = sim_mat[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_mat[~labels.bool()].view(sim_mat.shape[0], -1)

        # placing all positive logits first results in all correct index
        # labels ending up at position 0, so the zero matrix and the
        # corresponding logits matrix can now simply be sent to
        # the criterion to calculate loss.
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return (logits, labels)

    def fit(self, dataloader):
        scaler = GradScaler()
        for epoch in range(self.epochs):
            tloss = 0
            for images in tqdm(dataloader):
                images = torch.cat(images, dim=0).to(self.device)

                with autocast():
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                tloss += loss.item() / images.shape[0]

            print(f'{epoch=}   {tloss=}')

