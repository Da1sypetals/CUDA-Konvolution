import torch
import torch.nn as nn
import torch.nn.functional as F

from konv.layer import Konvolution2d as KonvR2d


class CIFAR_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # (32,32,3)
        self.in_conv = KonvR2d(3, 64, 7, padding=3)
        self.in_pool = nn.MaxPool2d(2)

        self.block1 = nn.Sequential(
            KonvR2d(64, 64, kernel_size=3, padding=1),
            KonvR2d(64, 64, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            KonvR2d(64, 64, kernel_size=3, padding=1),
            KonvR2d(64, 64, kernel_size=3, padding=1),
        )

        self.pool1 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            KonvR2d(64, 128, kernel_size=3, padding=1),
            KonvR2d(128, 128, kernel_size=3, padding=1),
        )
        self.skip3 = KonvR2d(64, 128, kernel_size=3, padding=1)

        self.block4 = nn.Sequential(
            KonvR2d(128, 128, kernel_size=3, padding=1),
            KonvR2d(128, 128, kernel_size=3, padding=1),
        )
        
        # self.pool2 = nn.MaxPool2d(2)

        # self.block5 = nn.Sequential(
        #     KonvR2d(128, 256, kernel_size=3, padding=1),
        #     KonvR2d(256, 256, kernel_size=3, padding=1),
        # )
        # self.skip5 = KonvR2d(128, 256, kernel_size=3, padding=1)

        # self.block6 = nn.Sequential(
        #     KonvR2d(256, 256, kernel_size=3, padding=1),
        #     KonvR2d(256, 256, kernel_size=3, padding=1),
        # )

        self.last = nn.Linear(128, 10)


    def forward(self, x):

        x = self.in_conv(x)
        x = self.in_pool(x)

        identity = x
        x = self.block1(x)
        x = x + identity

        identity = x
        x = self.block2(x)
        x = x + identity

        x = self.pool1(x)

        skip = self.skip3(x)
        x = self.block3(x)
        x = x + skip

        identity = x
        x = self.block4(x)
        x = x + identity

        # x = self.pool2(x)

        # skip = self.skip5(x)
        # x = self.block5(x)
        # x = x + skip

        # identity = x
        # x = self.block6(x)
        # x = x + identity

        x = torch.mean(x, dim=(2, 3))

        x = self.last(x)

        return F.softmax(x, dim=-1)