"""
TODO: implement shallow model for SimCLR, and add Docs!

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 03-02-2022
"""
import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {'resnet18': models.resnet18(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        projection_head = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
                nn.Linear(projection_head, projection_head),
                nn.ReLU(), self.backbone.fc)
        
    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x):
        return self.backbone(x)


class ShallowSimCLR(nn.Module):
    def __init__(self, *args, **kwargs):
        pass
