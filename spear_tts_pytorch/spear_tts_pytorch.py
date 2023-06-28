import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans

from beartype import beartype
from beartype.typing import Optional, Union, Callable, Literal, Tuple

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# t5 relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        *,
        heads,
        scale = 1,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
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

    def forward(self, n):
        device = self.device
        pos = torch.arange(n, dtype = torch.long, device = device)

        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')

        return bias * self.scale

# rmsnorm

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.causal = causal

        self.norm = RMSNorm(dim)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        attn_bias = None,
        context = None,
        mask = None
    ):
        h = self.heads
        x = self.norm(x)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q = q * self.scale
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        causal = False,
        ff_mult = 4,
        cross_attend = False
    ):
        super().__init__()

        self.rel_pos_bias = RelativePositionBias(
            scale = dim_head ** 0.5,
            causal = causal,
            heads = heads
        )

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads),
                Attention(dim = dim, dim_head = dim_head, heads = heads) if cross_attend else None,
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None
    ):
        seq_len = x.shape[-2]
        has_context = exists(context)

        attn_bias = self.rel_pos_bias(seq_len)

        for attn, maybe_cross_attn, ff in self.layers:
            x = attn(x, mask = mask, attn_bias = attn_bias) + x

            if exists(maybe_cross_attn):
                assert has_context
                x = maybe_cross_attn(x, context = context, mask = context_mask) + x

            x = ff(x) + x

        return self.final_norm(x)

# class

SpeechOrTextLiteral = Union[
    Literal['speech'],
    Literal['text']
]

class TextToSemantic(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        num_text_token_ids,
        source_depth,
        target_depth,
        tokenizer_encode: Optional[Callable] = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        semantic_pad_id = -1,
        text_pad_id = 0
    ):
        super().__init__()
        self.dim = dim
        self.wav2vec = wav2vec

        self.tokenizer_encode = tokenizer_encode

        num_semantic_token_ids = wav2vec.codebook_size if exists(wav2vec) else num_semantic_token_ids
        assert exists(num_semantic_token_ids), 'you need to either pass in a wav2vec model from audiolm-pytorch, or specify the number of semantic token ids with num_semantic_token_ids'

        self.num_semantic_token_ids = num_semantic_token_ids

        # padding id, for deriving attention mask automatically if not passed in

        semantic_pad_id = semantic_pad_id
        text_pad_id = text_pad_id

        self.pad_id = dict(
            speech = semantic_pad_id,
            text = text_pad_id
        )

        # embedding

        semantic_token_emb = nn.Embedding(num_semantic_token_ids, dim)
        text_token_emb = nn.Embedding(num_text_token_ids, dim)

        self.semantic_token_emb = semantic_token_emb

        self.token_emb = nn.ModuleDict(dict(
            speech = semantic_token_emb,
            text = text_token_emb
        ))

        # respective start tokens

        self.start_token = nn.ParameterDict(dict(
            speech = nn.Parameter(torch.randn(dim)),
            text = nn.Parameter(torch.randn(dim))
        ))

        # projection to logits

        to_semantic_logit = nn.Linear(dim, num_semantic_token_ids, bias = False)
        to_text_logit = nn.Linear(dim, num_text_token_ids, bias = False)

        to_semantic_logit.weight = semantic_token_emb.weight
        to_text_logit.weight = to_text_logit.weight

        self.to_logits = nn.ModuleDict(dict(
            speech = to_semantic_logit,
            text = to_text_logit
        ))

        # source and target attention layers

        self.source_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = source_depth,
            causal = False
        )

        self.target_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = source_depth,
            causal = True,
            cross_attend = True
        )

    @beartype
    def forward(
        self,
        source: Tensor,
        target: Tensor,
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        return_loss = False
    ):
        batch = source.shape[0]

        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # if source mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id

        # all target modules and parameters

        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]

        # embedding

        source_emb = source_token_emb(source)

        target_emb = target_token_emb(target)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        target_emb = torch.cat((start_token, target_emb), dim = 1)

        # source attention

        source_emb = self.source_transformer(source_emb, mask = source_mask)

        # target attention

        target_emb = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask)

        # decoder logits

        logits = target_to_logit(target_emb)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits[:, :-1], 'b n c -> b c n'),
            target,
            ignore_index = target_pad_id
        )

        return loss
