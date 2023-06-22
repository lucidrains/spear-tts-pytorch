import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

from spear_tts_pytorch.attend import Attend

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# class

class TextToSemantic(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(self, x):
        return x
