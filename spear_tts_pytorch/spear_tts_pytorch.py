import math
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn, einsum, FloatTensor, IntTensor, LongTensor
from torch.nn import Module, ModuleList

from torch.utils.data import Dataset

from einops import rearrange, repeat, pack

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans
from audiolm_pytorch.data import get_dataloader

from rotary_embedding_torch import RotaryEmbedding

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

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

def mask_after_eos(target, eos_id, pad_id):
    mask = (target == eos_id).cumsum(dim = -1) > 0
    mask = F.pad(mask, (1, -1), value = False)
    return target.masked_fill(mask, pad_id)

# freezing and unfreezing helpers

def set_requires_grad_(module: Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def freeze(module: Module):
    set_requires_grad_(module, False)

def unfreeze(module: Module):
    set_requires_grad_(module, True)

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
        dim_context = None,
        dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_emb = rotary_emb

        self.causal = causal

        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        has_context = exists(context)
        h = self.heads
        x = self.norm(x)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(self.rotary_emb):
            assert not has_context
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

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

        rotary_emb = RotaryEmbedding(dim_head)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
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
        has_context = exists(context)

        for attn, maybe_cross_attn, ff in self.layers:
            x = attn(x, mask = mask) + x

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
        ff_mult = 4,
        ff_dropout = 0.,
        semantic_pad_id = -1,
        text_pad_id = 0,
        autoset_semantic_eos_id = True,
        autoset_text_eos_id = True
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
        self.num_text_token_ids = num_text_token_ids

        # padding id, for deriving attention mask automatically if not passed in

        self.semantic_pad_id = semantic_pad_id
        self.text_pad_id = text_pad_id

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
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            causal = False
        )

        self.target_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = target_depth,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            causal = True,
            cross_attend = True
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # a set of freezing / unfreezing utils
    # then rely on get_optimizer to filter out the parameters that do not require grad from being exposed to optimizer

    def unfreeze_all(self):
        unfreeze(self)

    def freeze_encoder(self):
        freeze(self.source_transformer)

    def freeze_encoder_below_layer(self, layer: int):
        """
        for the final training of text-to-semantic on pseudo-labelled dataset
        they freeze the encoder part way up to a certain layer
        """
        unfreeze(self.source_transformer)

        for ind, module in enumerate(self.source_transformer.layers):
            current_layer = ind + 1

            if current_layer <= layer:
                freeze(module)

    def freeze_decoder(self):
        freeze(self.target_transformer)

    def freeze_speech_emb(self):
        freeze(self.token_emb['speech'])
        self.start_token['speech'].requires_grad = False

    def freeze_text_emb(self):
        freeze(self.token_emb['text'])
        self.start_token['text'].requires_grad = False

    # sampling function

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
        max_length = 2048,
        beam_search_decode = False,
        beam_size = 4,
        return_source = False
    ):
        if isinstance(source, (FloatTensor)) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'
            source = self.wav2vec(source)

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

        if not beam_search_decode:
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
                all_eos = is_eos.any(dim = -1).all()

                if not all_eos:
                    continue

                target = mask_after_eos(target, target_eos_id, target_pad_id)
                break
        else:
            beam = [(target, 0.0)]

            batch_range = torch.arange(batch, device = self.device, dtype = torch.long)
            batch_range = rearrange(batch_range, 'b -> b 1')

            for _ in tqdm(range(max_length)):
                all_candidates = []
                
                for sentence, sentence_prob in beam:
                    target_emb = target_token_emb(sentence)
                    target_emb = torch.cat((start_token, target_emb), dim = 1)

                    # target attention
                    
                    target_emb = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask)

                    # decoder logits

                    logits = target_to_logit(target_emb)
                    logits = logits[:, -1]

                    log_probs = torch.log_softmax(logits / max(temperature, 1e-10), dim = -1)
                    topk_log_probs, topk_ids = log_probs.topk(beam_size, dim = -1)
                    
                    for i in range(beam_size):
                        candidate = torch.cat([sentence, topk_ids[..., i:i + 1]], dim = -1)
                        candidate_prob = sentence_prob + topk_log_probs[..., i]
                        all_candidates.append((candidate, candidate_prob))

                # concat into shape (beam, batch, seq), (beam, batch)

                candidates, candidate_probs = map(partial(torch.stack, dim = 1), zip(*all_candidates))

                # sort by candidate scores across beams

                sorted_indices = candidate_probs.sort(dim = 1, descending = True).indices

                sorted_candidates = candidates[batch_range, sorted_indices]
                sorted_candidate_probs = candidate_probs[batch_range, sorted_indices]

                # reconstitute ordered List[Tuple[Tensor, Tensor]]

                ordered = list(zip(*map(partial(torch.unbind, dim = 1), (sorted_candidates, sorted_candidate_probs))))

                beam = ordered[:beam_size]
                
                # check if we've hit eos for all sequences

                all_eos = all([((sentence == target_eos_id).any(dim = -1)).all() for sentence, _ in beam])

                if all_eos:
                    break

            target = beam[0][0]

            if exists(target_eos_id):
                target = mask_after_eos(target, target_eos_id, target_pad_id)

        if not return_source:
            return target

        return source, target

    @beartype
    def forward(
        self,
        source: Union[List[str], Tensor],
        target: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        return_loss = False,
        return_logits = False
    ):
        if isinstance(source, FloatTensor) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'
            source = self.wav2vec(source)

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

        # if source/target mask is not passed in
        # automatically derive by the padding id of the modality

        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id

        if not exists(target_mask) and target.dtype == torch.long:
            target_mask = target != target_pad_id

            # attend to bos
            target_mask = F.pad(target_mask, (1, 0), value = True)

        # embedding

        source_emb = source_token_emb(source)

        target_emb = target_token_emb(target)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        target_emb = torch.cat((start_token, target_emb), dim = 1)

        # source attention

        source_emb = self.source_transformer(source_emb, source_mask)

        # target attention

        target_emb = self.target_transformer(target_emb, mask = target_mask, context = source_emb, context_mask = source_mask)

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

        if return_logits:
            return loss, logits
        else:
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
        reconstruct_seq: bool = False,
        mask_id = None
    ):
        super().__init__()

        self.model = model
        self.wav2vec = default(wav2vec, model.wav2vec)

        self.deletion_prob = deletion_prob
        self.reconstruct_seq = reconstruct_seq # whether to reconstruct the entire sequence, or just output the deleted ones in order
        self.mask_id = mask_id

    def forward(
        self,
        x
    ):
        is_raw_audio = x.dtype == torch.float

        if is_raw_audio:
            assert exists(self.wav2vec)
            
            with torch.no_grad():
                self.wav2vec.eval()
                x = self.wav2vec(x, flatten = False)

        batch = x.shape[0]

        mask = torch.ones_like(x, dtype = torch.bool, device = self.model.device)

        if exists(self.mask_id):
            assert self.reconstruct_seq, 'reconstruct_seq must be true if mask id is provided'
            
            mask = mask.masked_fill(x == self.model.semantic_pad_id, False)
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)

            source = x.masked_fill(delete_mask, self.mask_id)
        else:
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)

            source = rearrange(x[~delete_mask], '(b n) -> b n', b = batch)

        if self.reconstruct_seq:
            target = x
        else:
            target = rearrange(x[delete_mask], '(b n) -> b n', b = batch)

        loss, logits = self.model(
            source, target,
            source_type = 'speech',
            target_type = 'speech',
            return_loss = True,
            return_logits = True
        )

        return loss, logits

# wrapper for backtranslation task

class SemanticToTextWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        semantic_token_ids,
        grapheme_token_ids,
    ):
        source = semantic_token_ids
        target = grapheme_token_ids

        loss, logits = self.model(
            source, target,
            source_type = 'speech',
            target_type = 'text',
            return_loss = True,
            return_logits = True
        )

        return loss, logits

# wrapper for text to semantic task

class TextToSemanticWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        grapheme_token_ids,
        semantic_token_ids,
    ):
        source = grapheme_token_ids
        target = semantic_token_ids

        loss, logits = self.model(
            source, target,
            source_type = 'text',
            target_type = 'speech',
            return_loss = True,
            return_logits = True
        )

        return loss, logits

# wrapper for generating the pseudo-labelled audio to text dataset

class SemanticToTextDatasetGenerator(nn.Module):
    @beartype
    def __init__(
        self,
        model,
        *,
        dataset: Dataset,
        folder = './generated-audio-text-pairs',
        batch_size = 4,
        delimiter_id: int = -1,
        audio_pad_id = None,
        text_pad_id = 0
    ):
        super().__init__()
        self.model = model

        self.dataset = dataset
        self.dl = get_dataloader(dataset, batch_size = batch_size)
        self.delimiter_id = delimiter_id

        self.audio_pad_id = audio_pad_id
        self.text_pad_id = text_pad_id

        self.folder = Path(folder)
        self.folder.mkdir(exist_ok = True, parents = True)

    def forward(
        self,
        max_length = 2048,
        beam_search_decode = False,
        **generate_kwargs
    ):
        delimiter = torch.tensor([self.delimiter_id], device = self.model.device)

        counter = 0

        for audio, in self.dl:
            audio_semantic_ids, text_ids = self.model.generate(
                source = audio,
                source_type = 'speech',
                target_type = 'text',
                return_source = True,
                max_length = max_length,
                beam_search_decode = beam_search_decode,
                **generate_kwargs
            )

            for audio_semantic_id, text_id in zip(audio_semantic_ids, text_ids):

                if exists(self.audio_pad_id):
                    audio_pad_mask = audio_semantic_id == self.audio_pad_id
                    audio_semantic_id = audio_semantic_id[~audio_pad_mask]

                if exists(self.text_pad_id):
                    text_pad_mask = text_id == self.text_pad_id
                    text_id = text_id[~text_pad_mask]

                row, _ = pack([audio_semantic_id, delimiter, text_id], '*')
                path = str(self.folder / f'{counter}.pt')

                torch.save(row, path)
                counter += 1
