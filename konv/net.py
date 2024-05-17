from .layer import Konvolution2d
from cuLegKan.layer import LegendreKANLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.konv1 = Konvolution2d(1, 10, kernel_size=5, degree=4)
        self.konv2 = Konvolution2d(10, 20, kernel_size=5, degree=4)
        self.layer1 = LegendreKANLayer(320, 50, polynomial_order=4)
        self.layer2 = LegendreKANLayer(50, 10, polynomial_order=4)

    def forward(self, x):
        x = self.konv1(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)

        x = self.konv2(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        
        x = x.view(-1, 320)

        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)

        return x