import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from .layer import Konvolution2d as Konv2d
# from .layer import KonvR2d as Konv2d
from cuLegKan.layer import LegendreKANLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops as ein


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.konv1 = Konv2d(3, 16, kernel_size=5, padding=2, degree=4)
        self.konv2 = Konv2d(16, 32, kernel_size=5, padding=2, degree=4)
        self.konv3 = Konv2d(32, 32, kernel_size=3, padding=2, degree=4)
        self.konv4 = Konv2d(32, 10, kernel_size=3, padding=2, degree=4)

    def forward(self, x):

        x = ein.repeat(x, 'b c h w -> b (3 c) h w')

        x = self.konv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.konv2(x)
        x = self.konv3(x)
        x = self.konv4(x)
        x = torch.mean(x, dim=(2, 3))
        
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.konv1 = Konv2d(1, 10, kernel_size=5, degree=4)
        self.konv2 = Konv2d(10, 20, kernel_size=5, degree=4)
        self.layer1 = LegendreKANLayer(320, 10, polynomial_order=4)

    def forward(self, x):

        x = self.konv1(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)

        x = self.konv2(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        
        x = x.view(-1, 320)

        x = self.layer1(x)

        return x