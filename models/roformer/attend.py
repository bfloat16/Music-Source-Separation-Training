from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

class Attend(nn.Module):
    def __init__(self, dropout = 0., flash = False, scale = None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = [SDPBackend.FLASH_ATTENTION]
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if there is a compatible device for flash attention

        backends = self.cuda_config if q.is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        scale = default(self.scale, q.shape[-1] ** -0.5)
        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out