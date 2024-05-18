import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

import einops as ein


@lru_cache(maxsize=128)
def compute_legendre_polynomials(x, order):
    P0 = x.new_ones(x.shape)
    if order == 0:
        return ein.rearrange(P0, 'n -> 1 n')
    P1 = x
    legendre_polys = [P0, P1]
    for d in range(1, order):
        Pd = ((2.0 * d + 1.0) * x * legendre_polys[-1] - d * legendre_polys[-2]) / (d + 1.0)
        legendre_polys.append(Pd)
    return torch.stack(legendre_polys, dim=-1)