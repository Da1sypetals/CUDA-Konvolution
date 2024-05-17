import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import legendre_2d_ops


class Legendre2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, degree):
        """
        returns: cheby
        Note: degree does not require grad
        """
        # ctx.save_for_backward(x)

        legendre = legendre_2d_ops.forward(x, degree)

        ctx.save_for_backward(x, legendre, torch.tensor(degree, dtype=torch.int32))

        return legendre


    @staticmethod
    def backward(ctx, grad_output): 
        # print(f'{grad_output.size()=}')
        x, legendre, degree = ctx.saved_tensors

        grad_x = legendre_2d_ops.backward(grad_output, x, legendre, degree.item())

        return grad_x, None # None for degree

def legendre_2d(x, degree):
    return Legendre2d.apply(x, degree)















