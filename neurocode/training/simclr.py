"""
SimCLR training module, implementing Info NCE Loss.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 14-02-2022
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

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

    def fit(self, samplers, plot=False):
        print(f'Training encoder with SimCLR on device={self.device} for {self.epochs} epochs')
        print(f'   epoch       training loss       validation loss         training acc        validation acc')
        print(f'------------------------------------------------------------------------------------------------')
        history =  defaultdict(list)
        for epoch in range(self.epochs):
            self.model.train()
            tloss, tacc = 0., 0.
            vloss, vacc = 0., 0.
            for images in samplers['train']:
                anchors, samples = images
                images = torch.cat((anchors, samples)).float().to(self.device)
                embeddings = self.model(images)

                if plot:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(torch.swapaxes(images[0, :].cpu(), 0, 2).numpy())
                    axs[1].imshow(torch.swapaxes(images[8, :].cpu(), 0, 2).numpy())
                    plt.show()

                indices = torch.arange(0, anchors.size(0), device=anchors.device)
                labels = torch.cat((indices, indices)).to(self.device)

                loss = self.criterion(embeddings, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tloss += loss.item() / embeddings.shape[0]

            with torch.no_grad():
                self.model.eval()
                for images in samplers['valid']:
                    anchors, samples = images
                    images = torch.cat((anchors, samples)).float().to(self.device)
                    embeddings = self.model(images)

                    indices = torch.arange(0, anchors.size(0), device=anchors.device)
                    labels = torch.cat((indices, indices)).to(self.device)

                    loss = self.criterion(embeddings, labels)
                    vloss += loss.item() / embeddings.shape[0]

            self.scheduler.step()
            tloss /= len(samplers['train'])
            vloss /= len(samplers['valid'])
            history['tloss'].append(tloss)
            history['vloss'].append(vloss)
            print(f'     {epoch + 1:02d}            {tloss:.4f}              {vloss:.4f}                  {tacc:.2f}%                 {vacc:.2f}%')

        return history


