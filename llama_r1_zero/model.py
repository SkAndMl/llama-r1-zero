import torch
from torch import nn
from typing import Optional
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    max_seq_len: int = 256
    vocab_size: int = -1
    norm_eps: float = 1e-6
    max_batch_size: int = 16
    hidden_dim: int = 2048
    ffn_dim_multiplier: Optional[int] = None
    multiple_of: int = 4
    rope_theta: int = 10000
    use_scaled_rope: bool = True
    device: str = "cpu"


def precompute_freq_cis(seq_len: int=512, theta: int=10000, dim: int=384):

    freqs = 1/(theta**(torch.arange(0, dim, 2)[:dim//2].float()/dim))
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freq_cis

def reshape_for_broadcast(freq_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i==1 or i==ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freq_cis: torch.Tensor):
    # b, s, h, head_dim -> b, s, h, head_dim/2, 2 -> b, s, h, head_dim/2
    x_c = torch.view_as_complex(x.float().reshape([*x.shape[:-1], -1, 2]))
    freq_cis = reshape_for_broadcast(freq_cis, x_c)
    x_c_out = torch.view_as_real(x_c*freq_cis).flatten(3)
    # b, s, h, head_dim/2, 2
    return x_c_out.type_as(x)


class RMSNorm(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(args.dim))
        self.eps = args.norm_eps
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        out = self.weight*self._norm(x.float()).type_as(x)
        return out
    

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads//args.n_kv_heads
        self.head_dim = args.dim//self.n_heads
        self.wq = nn.Linear(args.dim, self.head_dim*self.n_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim*self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim*self.n_kv_heads, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
    
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))
    
    def forward(self,
                x: torch.Tensor, 
                freq_cis: torch.Tensor,
                start_pos: int,
                mask: Optional[torch.Tensor]=None):
        bs, seq_len, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bs, seq_len, self.n_heads, self.head_dim)
        k = k.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        q = apply_rotary_emb(q, freq_cis)
        k = apply_rotary_emb(k, freq_cis)
        
        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(v)

        self.cache_k[:bs, start_pos:start_pos+seq_len] = k
        self.cache_v[:bs, start_pos:start_pos+seq_len] = v

        new_k = self.cache_k[:bs, :start_pos+seq_len]
        new_v = self.cache_v[:bs, :start_pos+seq_len]

        new_k = torch.repeat_interleave(input=new_k, repeats=self.n_rep, dim=2)
        new_v = torch.repeat_interleave(input=new_v, repeats=self.n_rep, dim=2)
        q = q.transpose(1, 2)
        new_k = new_k.transpose(1, 2)
        new_v = new_v.transpose(1, 2)
        attn = q @ new_k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None: attn += mask
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        out = attn @ new_v
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4*args.dim
        hidden_dim = int(2*hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim*args.ffn_dim_multiplier)
        
        hidden_dim = args.multiple_of*((hidden_dim+args.multiple_of-1)//args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.ffn_norm = RMSNorm(args)
        self.attention_norm = RMSNorm(args)
    
    def forward(self, 
                x: torch.Tensor, 
                freq_cis: torch.Tensor,
                start_pos: int,
                mask: Optional[torch.Tensor]=None):
        
        h = x + self.attention(self.attention_norm(x), freq_cis, start_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freq_cis = precompute_freq_cis(
            seq_len=args.max_seq_len*2,
            theta=args.rope_theta,
            dim=args.dim//args.n_heads
        ).to(args.device)
    
    def forward(self, x: torch.Tensor, start_pos: int):
        bs, seq_len = x.shape
        h = self.tok_embeddings(x)
        freq_cis = self.freq_cis[start_pos:start_pos+seq_len]
        
        mask = None
        if seq_len>1:
            mask = torch.full(size=(seq_len, seq_len), fill_value=float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=x.device), mask
            ]).type_as(h)
        
        for layer in self.layers:
            h = layer(h, freq_cis, start_pos, mask)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output