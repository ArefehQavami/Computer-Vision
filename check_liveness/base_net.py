import os
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, pretrained=True):
        try:
            # if f'django-insecure-{os.getenv("SECRET_KEY")}' != settings.SECRET_KEY:
            #     raise Exception('SECRET KEY NOT FOUND!!!!!')

            super(Network, self).__init__()
            dense = models.densenet161(pretrained=pretrained)
            features = list(dense.features.children())
            self.enc = nn.Sequential(*features[0:8])
            self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
            self.linear = nn.Linear(14 * 14, 1)

        except Exception as e:
            print(e)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        dec = self.linear(out_map.view(-1, 14 * 14))
        dec = F.sigmoid(dec)
        return out_map, dec
