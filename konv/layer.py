import torch
import torch.nn as nn

import torch.nn.functional as F

import einops as ein

from cuLegKan.legendre import legendre_function
from cuLegKan2d.legendre import legendre_2d



class Konvolution2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding=0,
                 stride=1,
                 bias=True,
                 degree=4, 
                 base_activation=nn.SiLU):

        super().__init__()

        self.degree = degree
        self.conv_in_channels = in_channels * (degree + 1)

        self.conv = nn.Conv2d(self.conv_in_channels, 
                              out_channels, 
                              kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias,)
        
        self.tanh_scale = nn.Parameter(torch.ones((1, in_channels, 1, 1)))
        self.tanh_bias = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        
    def forward(self, x):

        b, c, h, w = x.size()

        x = torch.tanh(self.tanh_scale * x + self.tanh_bias)
        
        # expand
        x_in = ein.rearrange(x, 'b c h w -> (b h w) c')

        x_in = legendre_function(x_in, self.degree)

        # print(x_in.size())
    
        x_in = ein.rearrange(x_in, 'd (b h w) c -> b (c d) h w', b=b, h=h, w=w)
        
        x_out = self.conv(x_in)

        return x_out
    



class KonvR2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding=0,
                 stride=1,
                 bias=True,
                 degree=4, 
                 base_activation=nn.SiLU):

        super().__init__()

        self.degree = degree
        self.conv_in_channels = in_channels * (degree + 1)

        self.conv = nn.Conv2d(self.conv_in_channels, 
                              out_channels, 
                              kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias,)
        
        self.tanh_scale = nn.Parameter(torch.ones((1, in_channels, 1, 1)))
        self.tanh_bias = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))


        
    def forward(self, x):

        x_in = torch.tanh(self.tanh_scale * x + self.tanh_bias)
        
        x_in = legendre_2d(x, self.degree)

        # print(x_in)

        x_out = self.conv(x_in)

        return x_out



















