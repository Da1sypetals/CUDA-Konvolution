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
                 out_norm=True,
                 padding=0,
                 stride=1,
                 bias=True,
                 degree=4, 
                 base_activation=nn.SiLU):

        super().__init__()

        self.degree = degree
        self.conv_in_channels = in_channels * (degree + 1)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.conv_in_channels, 
                              out_channels, 
                              kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias,)
        
        self.tanh_scale = nn.Parameter(torch.ones((1, in_channels, 1, 1)))
        self.tanh_bias = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        self.out_norm = out_norm
        if self.out_norm:
            self.norm = nn.LayerNorm(self.out_channels)

        
    def forward(self, x):

        b, c, h, w = x.size()

        x = torch.tanh(self.tanh_scale * x + self.tanh_bias)
        # x = torch.tanh(x)
        
        # expand
        x = ein.rearrange(x, 'b c h w -> (b h w) c')
        # x = 2 * (x - x.min()) / (x.max() - x.min()) - 1

        x_in = legendre_function(x, self.degree)

        # print(x_in.size())
    
        x_in = ein.rearrange(x_in, 'd (b h w) c -> b (c d) h w', b=b, h=h, w=w)
        
        x_out = self.conv(x_in)

        if self.out_norm:
            b_out, c_out, h_out, w_out = x_out.size()
            x_out = ein.rearrange(x_out, 'b c h w -> (b h w) c')
            x_out = self.norm(x_out)
            x_out = ein.rearrange(x_out, '(b h w) c -> b c h w', b=b_out, h=h_out, w=w_out)

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



















