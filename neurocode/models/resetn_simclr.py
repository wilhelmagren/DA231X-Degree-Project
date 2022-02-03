"""
SimCLR resnset implementation, adapted from
sthalles github, SimCLR implementation.

Work in progress!!!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from exceptions.exceptions import InvalidBackboneError
from tqdm impor tqdm


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        self.resnet_dict = {'resnet18': models.resnet18(pretrained=False, num_classes=out_dim),
                            'resnet50': models.resnet50(pretrained=False, num_classes=out_dim)}
        
        self.backbone = self._get_basemodel(base_model)
        dim_proj_head = self.backbone.fc.in_features

        # add the MLP projection head
        self.backbone.fc = nn.Sequential(
                nn.Linear(dim_proj_head, dim_proj_head),
                nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                    'Invalid backbone architecture. Check the config file,'
                    'pass either resnet18 or resnet 50')

    def forward(self, x):
        return self.backbone(x)


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives as negatives... obviously
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature

        return logits, labels

    def fit(self, train_loader):
        scaler = GradScaler()
        n_iter = 0
        for epoch in range(self.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1

