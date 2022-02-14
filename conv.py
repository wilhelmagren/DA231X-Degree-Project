import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def normalize(x):
    return x.astype(float) / 255.0


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 3, (7, 7), stride=2)

    def forward(self, x):
        return self.conv(x)


image = mpimg.imread('cat.jpg')
image = normalize(image)
imaged = torch.swapaxes(torch.Tensor(image), 0, 2)
model =  Model()
featuremap = model(imaged[None])[0, :].detach()
featuremap = torch.swapaxes(featuremap, 0, 2).numpy()
print(featuremap)
print(featuremap.shape)

plt.imshow(featuremap[:, :, 0])
fig, axs = plt.subplots(1, 3)
axs[0].imshow(featuremap[:, :, 0])
axs[1].imshow(featuremap[:, :, 1])
axs[2].imshow(featuremap[:, :, 2])
plt.savefig('featuremaps.png')
