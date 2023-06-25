import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

from spear_tts_pytorch.attend import Attend

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans

from beartype import beartype
from beartype.typing import Optional, Union

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# class

class TextToSemantic(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        num_text_token_ids,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        semantic_pad_id = -1
    ):
        super().__init__()
        self.dim = dim
        self.wav2vec = wav2vec

        num_semantic_token_ids = wav2vec.codebook_size if exists(wav2vec) else num_semantic_token_ids
        assert exists(num_semantic_token_ids), 'you need to either pass in a wav2vec model from audiolm-pytorch, or specify the number of semantic token ids with num_semantic_token_ids'

        self.num_semantic_token_ids = num_semantic_token_ids
        self.semantic_token_emb = nn.Embedding(num_semantic_token_ids, dim)
        self.semantic_pad_id = semantic_pad_id

        self.text_token_emb = nn.Embedding(num_text_token_ids, dim)

    def forward(self, x):
        return x
