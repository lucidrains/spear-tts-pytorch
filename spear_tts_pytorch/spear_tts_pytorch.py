import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

from spear_tts_pytorch.attend import Attend

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans

from beartype import beartype
from beartype.typing import Optional, Union, Callable

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# t5 relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position

        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)

        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')

        return bias * self.scale

# class

class TextToSemantic(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        num_text_token_ids,
        tokenizer_encode: Optional[Callable] = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        semantic_pad_id = -1
    ):
        super().__init__()
        self.dim = dim
        self.wav2vec = wav2vec

        self.tokenizer_encode = tokenizer_encode

        num_semantic_token_ids = wav2vec.codebook_size if exists(wav2vec) else num_semantic_token_ids
        assert exists(num_semantic_token_ids), 'you need to either pass in a wav2vec model from audiolm-pytorch, or specify the number of semantic token ids with num_semantic_token_ids'

        self.num_semantic_token_ids = num_semantic_token_ids
        self.semantic_pad_id = semantic_pad_id
        self.semantic_token_emb = nn.Embedding(num_semantic_token_ids, dim)

        self.text_token_emb = nn.Embedding(num_text_token_ids, dim)

    def forward(
        self,
        x
    ):
        return x
