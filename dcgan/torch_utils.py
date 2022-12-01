import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn as nn
import torchvision
import tqdm
import torch.nn.functional as F
from torch._C import device
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from IPython.display import Image
from torchvision.utils import save_image

latent_size = 128
device = torch.device('cpu')
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def apply_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                # in: 3 x 64 x 64
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 64 x 32 x 32

                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 128 x 16 x 16

                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 256 x 8 x 8

                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 512 x 4 x 4

                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
                # out: 1 x 1 x 1

                nn.Flatten(),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
                nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                

                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                

                nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                

                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                

                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                

                nn.Tanh()
            )

    def forward(self, x):
        return self.model(x)


model = Generator()
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
model.load_state_dict(torch.load('G.pth' , map_location= device))
model.eval()

