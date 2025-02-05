import torch
from torch import nn
from typing import Optional
from torch.nn import functional as F
import math


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
    theta: int = 10000
    device: str = "cuda"


def precompute_freq_cis(seq_len: int=512, theta: int=10000, dim: int=384):

    thetas = 1/(theta*(torch.arange(0, dim, 2)[:dim//2]/dim))
    print(f"thetas: {thetas.shape}")
    pos = torch.arange(seq_len)
    pos_thetas = torch.outer(pos, thetas)
    print(f"pos_thetas: {pos_thetas.shape}")
    freq_cis = torch.polar(torch.ones_like(pos_thetas), pos_thetas)
    print(f"freq_cis: {freq_cis.shape}")
    return freq_cis


def apply_rotary_emb(x: torch.Tensor, freq_cis: torch.Tensor):
    # b, s, h, head_dim -> b, s, h, head_dim/2, 2 -> b, s, h, head_dim/2
    x_c = torch.view_as_complex(x.reshape([*x.shape[:-1], -1, 2]))
    print(f"x_c: {x_c.shape}")
    freq_cis_ = freq_cis.unsqueeze(0).unsqueeze(2)
    print(f'freq_cis_: {freq_cis_.shape}')
    x_c = x_c*freq_cis_
    print(f'x_c: {x_c.shape}')
    x_real = torch.view_as_real(x_c)
    # b, s, h, head_dim/2, 2
    print(f"x_real: {x_real.shape}")
    x_rotated = x_real.reshape(*x.shape)
    print(f'x_rotated: {x_rotated.shape}')
    return x_rotated


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
    def __init__(self, n_heads, n_kv_heads, dim, max_seq_len, max_batch_size):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = n_heads//n_kv_heads
        self.head_dim = dim//self.n_heads
        self.wq = nn.Linear(dim, self.head_dim*self.n_heads)
        self.wk = nn.Linear(dim, self.head_dim*self.n_kv_heads)
        self.wv = nn.Linear(dim, self.head_dim*self.n_kv_heads)
        self.wo = nn.Linear(dim, dim)
    
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, self.head_dim))
    
    def forward(self,
                x: torch.Tensor, 
                freq_cis: torch.Tensor,
                start_pos: int,
                mask: Optional[torch.Tensor]=None):
        
        bs, seq_len, dim = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(v)
        q = q.view((bs, seq_len, self.n_heads, self.head_dim))
        k = k.view((bs, seq_len, self.n_kv_heads, self.head_dim))
        v = v.view((bs, seq_len, self.n_kv_heads, self.head_dim)) 

        k = apply_rotary_emb(k, freq_cis)
        v = apply_rotary_emb(v, freq_cis)

        self.cache_k[:bs, start_pos:start_pos+seq_len] = k
        self.cache_v[:bs, start_pos:start_pos+seq_len] = v

        new_k = self.cache_k[:bs, :start_pos+seq_len]
        new_v = self.cache_v[:bs, :start_pos+seq_len]

        new_k = torch.repeat_interleave(input=new_k, repeats=self.n_rep, dim=2)
        new_v = torch.repeat_interleave(input=new_v, repeats=self.n_rep, dim=2)

        q = q.transpose((1, 2))
        new_k = new_k.transpose((1, 2))
        new_v = new_v.transpose((1, 2))

        attn = q @ new_k.T
        if mask is not None: attn += mask
        attn = F.softmax(attn/math.sqrt(self.head_dim), dim=-1)
        out = attn @ v
        out = out.transpose((1, 2)).contiguous().view((bs, seq_len, -1))
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = (2*args.hidden_dim)//3
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim*args.ffn_dim_multiplier)
        
        hidden_dim = ((hidden_dim+args.multiple_of-1)//args.multiple_of)*args.multiple_of
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x))*self.w3(x))


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
            theta=args.theta,
            dim=args.dim
        ).to(args.device)
    
    def forward(self, x: torch.Tensor, start_pos: int):
        bs, seq_len = x.shape
        h = self.tok_embeddings(x)
        freq_cis = self.freq_cis[start_pos:start_pos+seq_len]
        
        mask = None
        if seq_len>1:
            mask = torch.full(size=(seq_len, seq_len), fill_value=float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.full((seq_len, start_pos)), mask
            ]).type_as(h)
        
        for layer in self.layers:
            h = layer(h, freq_cis, start_pos, mask)
        
        h = self.norm(h)
        output = self.output(h)
        return output