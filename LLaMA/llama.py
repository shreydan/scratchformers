import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

class RMSNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = nn.Parameter(
            torch.ones(self.embed_dim,dtype=torch.float32),
            requires_grad=True
        )
        
    def forward(self, x):
        # x [B, S, D]
        mean = x.pow(2).mean(dim=-1,keepdim=True)
        r_sqrt = x * torch.rsqrt(mean + 1e-5) # [B, S, 1]
        y = r_sqrt * self.gamma
        return y.to(x.dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x [B S D]
        return x * F.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.intermediate_dim = config.intermediate_dim
        self.linear1 = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=config.dtype)
        self.linear2 = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=config.dtype)
        self.linear3 = nn.Linear(self.intermediate_dim, self.embed_dim, bias=False, dtype=config.dtype)
        self.act = SiLU()
        
    def forward(self, x):
        # x [B S D]
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = self.act(x1) * x2
        x = self.linear3(x)
        return x
        

def precompute_rope(head_dim, base_theta=10_000, context_length=4096):
    k = torch.arange(0,head_dim,2)[:head_dim//2].float()
    inv_freq = 1 / (base_theta ** (k/head_dim))
    
    positions = torch.arange(context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # [S, H/2]
    angles = torch.cat([angles, angles],dim=-1) # [S, H]
    
    cos = torch.cos(angles) # [S, H]
    sin = torch.sin(angles) # [S, H]
    
    
    return cos, sin

def apply_rope(x, cos, sin):
    B, nH, S, H = x.shape
    x1 = x[...,:H//2] # [B, nH, S, H/2]
    x2 = x[...,H//2:] # [B, nH, S, H/2]
    cos_values = cos[:S,:].unsqueeze(0).unsqueeze(1) # [1,1,S,H]
    sin_values = sin[:S,:].unsqueeze(0).unsqueeze(1) # [1,1,S,H]
    rotated = torch.cat([-x2,x1],dim=-1)
    x_rope = (x * cos_values) + (rotated * sin_values)
    return x_rope.to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config.embed_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_q_heads = config.num_q_heads
        
        assert self.embed_dim % self.num_q_heads == 0, 'embed_dim should be div. by num. of query heads'
        assert self.num_q_heads % self.num_kv_heads ==0, 'num. query heads should be div. by num. key-value heads'
        
        self.head_dim = self.embed_dim // self.num_q_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_q_heads, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_kv_heads, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_kv_heads, bias=False, dtype=config.dtype)
        
        self.drop = nn.Dropout(config.attn_dropout)
        
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, dtype=config.dtype)
        
        self.register_buffer('causal_mask',
                             torch.triu(torch.ones(
                                 config.max_position_embeddings,
                                 config.max_position_embeddings
                             ),diagonal=1))
        
        cos, sin = precompute_rope(self.head_dim,base_theta=config.base_theta,context_length=config.max_position_embeddings)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
    def forward(self, x):
        # x [B S D]
        
        B,S,D = x.shape
        
        q = self.q_proj(x) # [B S H*nQ]
        k = self.k_proj(x) # [B S H*nKV]
        v = self.v_proj(x) # [B S H*nKV]
        
        q = q.view(B, S, self.num_q_heads, self.head_dim).transpose(1,2) # [B nQ S H]
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1,2) # [B nKV S H]
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1,2) # [B nKV S H]
        
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        
        k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1) # [B nQ S H]
        v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1) # [B nQ S H]
        
        attn = q @ k.transpose(2,3) # [B nQ S H] @ [B nQ H S] = [B nQ S S]
        
        # apply mask, mul with v, reshape, return
        mask = self.causal_mask[:S,:S].bool()
        attn.masked_fill_(mask,-torch.inf)
        
        attn = F.softmax(attn / (self.head_dim ** 0.5), dim=-1)
        
        attn = self.drop(attn)
        
        out = attn @ v # [B nQ S S] @ [B nQ S H] = [B nQ S H]
        out = out.transpose(1,2) # [B S nQ H]
        out = out.reshape(B, S, D)
        
        proj = self.o_proj(out)
        
        return proj

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = RMSNorm(config.embed_dim)
        self.self_attention = GroupedQueryAttention(config)

        self.ln2 = RMSNorm(config.embed_dim)
        self.mlp = FeedForward(config)
        
    def forward(self, x):
        # x [B S D]
        skip = x
        x = self.ln1(x)
        x = self.self_attention(x)
        x = x + skip
        
        skip = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + skip
        
        return x


class LLaMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.embed_dim, dtype=self.config.dtype)
        self.transformer_layers = nn.Sequential(*[
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        ])
        self.ln = RMSNorm(self.config.embed_dim)
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)
        
    def forward(self, input_ids):
        # input_ids [B S]
        token_embeddings = self.token_embedding(input_ids)
        x = self.transformer_layers(token_embeddings)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


if __name__ == '__main__':
    config = SimpleNamespace(
        embed_dim = 576,
        intermediate_dim = 1536,
        max_position_embeddings = 8192,
        base_theta = 100000,
        num_q_heads = 9,
        num_kv_heads = 3,
        attn_dropout = 0.,
        num_layers = 30,
        vocab_size = 49152,
        dtype = torch.float32,
    )
    model = LLaMA(config)
    x = torch.randint(0,config.vocab_size,(1,10)).long()
    out = model(x)
    print(out.shape)
    assert out.shape == (1, 10, config.vocab_size)