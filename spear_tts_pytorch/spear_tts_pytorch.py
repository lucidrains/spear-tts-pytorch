import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Union, Callable, Literal, Tuple, List

from x_clip.tokenizer import tokenizer

from tqdm import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def empty(t: Tensor):
    return t.numel() == 0

def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    batch_range = rearrange(batch_range, '... -> ... 1')

    t = F.pad(t, (0, 1), value = pad_id)
    t[batch_range, eos_indices] = eos_id
    return t

# sampling helpers

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

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

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
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
        causal = False,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.causal = causal

        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

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
        attn = self.attn_dropout(attn)

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
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
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
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout) if cross_attend else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
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

SemanticModelType = Union[
    FairseqVQWav2Vec,
    HubertWithKmeans
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
        use_openai_tokenizer = False,
        wav2vec: Optional[SemanticModelType] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        semantic_pad_id = -1,
        text_pad_id = 0,
        autoset_semantic_eos_id = True,
        autoset_text_eos_id = True,
    ):
        super().__init__()
        self.dim = dim
        self.wav2vec = wav2vec

        self.tokenizer_encode = tokenizer_encode

        if use_openai_tokenizer:
            assert not exists(tokenizer_encode)
            self.tokenizer_encode = tokenizer.tokenize

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

        # eos id

        self.autoset_eos_id = dict(
            speech = autoset_semantic_eos_id,
            text = autoset_text_eos_id
        )

        self.eos_id = dict(
            speech = num_semantic_token_ids,
            text = num_text_token_ids
        )

        # embedding

        semantic_token_emb = nn.Embedding(num_semantic_token_ids + int(autoset_semantic_eos_id), dim)
        text_token_emb = nn.Embedding(num_text_token_ids + int(autoset_text_eos_id), dim)

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
        to_text_logit.weight = text_token_emb.weight

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
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            causal = False
        )

        self.target_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = source_depth,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            causal = True,
            cross_attend = True
        )

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg, strict = strict)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @eval_decorator
    @beartype
    def generate(
        self,
        source: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        source_mask: Optional[Tensor] = None,
        max_length = 2048
    ):
        if is_bearable(source, List[str]):
            assert exists(self.tokenizer_encode)
            source = self.tokenizer_encode(source)
            source = source.to(self.device)

        batch = source.shape[0]

        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters

        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]
        target_eos_id = self.eos_id[target_type]

        # auto set eos id

        if self.autoset_eos_id[source_type]:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)

        # if source mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id

        # source embedding

        source_emb = source_token_emb(source)

        source_emb = self.source_transformer(source_emb, mask = source_mask)

        # decode target

        target = torch.empty((batch, 0), dtype = torch.long, device = self.device)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        # loop to decode

        for _ in tqdm(range(max_length)):
            target_emb = target_token_emb(target)
            target_emb = torch.cat((start_token, target_emb), dim = 1)

            # target attention

            target_emb = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask)

            # decoder logits

            logits = target_to_logit(target_emb)

            logits = logits[:, -1]
            logits = filter_logits_fn(logits, thres = filter_thres)

            sampled = gumbel_sample(logits, temperature = temperature)
            target, _ = pack((target, sampled), 'b *')

            if not self.autoset_eos_id[target_type]:
                continue

            is_eos = target == target_eos_id

            if not is_eos.any(dim = -1).all():
                continue

            mask = is_eos.cumsum(dim = -1) == 0
            target = target.masked_fill(~mask, target_pad_id)
            break

        return target

    @beartype
    def forward(
        self,
        source: Union[List[str], Tensor],
        target: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        return_loss = False
    ):
        if is_bearable(source, List[str]):
            assert exists(self.tokenizer_encode)
            source = self.tokenizer_encode(source)
            source = source.to(self.device)

        if is_bearable(target, List[str]):
            assert exists(self.tokenizer_encode)
            target = self.tokenizer_encode(target)
            target = target.to(self.device)

        assert source.shape[0] == target.shape[0]
        batch = source.shape[0]

        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters

        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]

        # auto set eos id

        if self.autoset_eos_id[source_type]:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)

        if self.autoset_eos_id[target_type] and return_loss:
            target_eos_id = self.eos_id[target_type]
            target = set_eos_id(target, target_eos_id, pad_id = target_pad_id)

        # if source mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id

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

        assert not empty(target)

        logits = rearrange(logits[:, :-1], 'b n c -> b c n')

        loss = F.cross_entropy(
            logits,
            target,
            ignore_index = target_pad_id
        )

        return loss

# pretraining modules

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

class SpeechSpeechPretrainWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic,
        wav2vec: Optional[SemanticModelType] = None,
        deletion_prob: float = 0.6,
        reconstruct_seq: bool = False
    ):
        super().__init__()

        self.model = model
        self.wav2vec = default(wav2vec, model.wav2vec)
        assert exists(self.wav2vec)

        self.deletion_prob = deletion_prob
        self.reconstruct_seq = reconstruct_seq # whether to reconstruct the entire sequence, or just output the deleted ones in order

    def forward(
        self,
        x
    ):
        is_raw_audio = x.dtype == torch.float

        if is_raw_audio:
            with torch.no_grad():
                self.wav2vec.eval()
                x = self.wav2vec(x, flatten = False)

        batch = x.shape[0]

        mask = torch.ones_like(x, dtype = torch.bool, device = self.model.device)

        delete_mask = get_mask_subset_prob(mask, self.deletion_prob)

        source = rearrange(x[~delete_mask], '(b n) -> b n', b = batch)

        if self.reconstruct_seq:
            target = x
        else:
            target = rearrange(x[delete_mask], '(b n) -> b n', b = batch)

        loss = self.model(
            source, target,
            source_type = 'speech',
            target_type = 'speech',
            return_loss = True
        )

        return loss
