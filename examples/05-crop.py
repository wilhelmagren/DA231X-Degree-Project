import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

img = Image.open('images/MEG_sensor_topology.png')
trans = transforms.Compose([transforms.RandomResizedCrop((500, 500), scale=(0.2, 0.99)), transforms.ToTensor()])

img = trans(img)
arr = torch.swapaxes(torch.swapaxes(img, 0, 2), 0, 1).numpy().squeeze(2)
plt.imshow(arr)
plt.show()
