import torch
import torch.nn as nn
import torch.nn.functional as F

from konv.layer import Konvolution2d as Konv2d
# from konv.original_layer import OKonv2d as Konv2d
# from konv.layer import KonvR2d as Konv2d

from original.LegKanLayer import KAL_Layer



class CIFAR_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,32,32)
        self.in_conv = Konv2d(3, 64, kernel_size=7, padding=3)
        self.in_pool = nn.MaxPool2d(2)

        self.block1 = nn.Sequential(
            Konv2d(64, 64, kernel_size=3, padding=1),
            Konv2d(64, 64, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            Konv2d(64, 64, kernel_size=3, padding=1),
            Konv2d(64, 64, kernel_size=3, padding=1),
        )

        self.pool1 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            Konv2d(64, 128, kernel_size=3, padding=1),
            Konv2d(128, 128, kernel_size=3, padding=1),
        )
        self.skip3 = Konv2d(64, 128, kernel_size=3, padding=1)

        self.block4 = nn.Sequential(
            Konv2d(128, 128, kernel_size=3, padding=1),
            Konv2d(128, 128, kernel_size=3, padding=1),
        )
        
        self.pool2 = nn.MaxPool2d(2)

        self.block5 = nn.Sequential(
            Konv2d(128, 256, kernel_size=3, padding=1),
            Konv2d(256, 256, kernel_size=3, padding=1),
        )
        self.skip5 = Konv2d(128, 256, kernel_size=3, padding=1)

        self.block6 = nn.Sequential(
            Konv2d(256, 256, kernel_size=3, padding=1),
            Konv2d(256, 256, kernel_size=3, padding=1),
        )

        self.pool3 = nn.MaxPool2d(2)

        self.block7 = nn.Sequential(
            Konv2d(256, 512, kernel_size=3, padding=1),
            Konv2d(512, 512, kernel_size=3, padding=1),
        )
        self.skip7 = Konv2d(256, 512, kernel_size=3, padding=1)

        self.block8 = nn.Sequential(
            Konv2d(512, 512, kernel_size=3, padding=1),
            Konv2d(512, 512, kernel_size=3, padding=1),
        )

        # self.last = nn.Linear(256, 10)
        self.last = KAL_Layer(256, 10)


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

        x = self.pool2(x)

        skip = self.skip5(x)
        x = self.block5(x)
        x = x + skip

        identity = x
        x = self.block6(x)
        x = x + identity

        # skip = self.skip7(x)
        # x = self.block7(x)
        # x = x + skip

        # identity = x
        # x = self.block8(x)
        # x = x + identity

        x = torch.mean(x, dim=(2, 3))

        x = self.last(x)

        return x