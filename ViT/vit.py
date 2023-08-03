import torch
import torch.nn as nn
from einops import rearrange

from types import SimpleNamespace

__all__ = ['ViT']

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention Module
    -------------------------

    input: torch.Tensor(batch_size x sequence_length x embedding_dim)

    for ViT, sequence_length = patch_embedding dim

    output: torch.Tensor(batch_size x sequence_length x embedding_dim)
    """
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension must be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.embed_dim
        
        self.qkv = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=False)
        self.scale = self.head_size ** -0.5
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.qkv(x).chunk(3,dim=-1)
        q = rearrange(q,'b t (h n) -> b n t h',n=self.n_heads) # h = head_size
        k = rearrange(k,'b t (h n) -> b n t h',n=self.n_heads)
        v = rearrange(v,'b t (h n) -> b n t h',n=self.n_heads)
        
        # qk_t = einsum(q,k,'b n t1 h, b n t2 h -> b n t1 t2') * self.scale
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        
        weights = self.attention_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x seq_len x head_size
        attention = rearrange(attention,'b n t h -> b t (n h)') # batch x n_heads x seq_len x embed_dim
        
        out = self.proj(attention)
        out = self.residual_dropout(out)
        
        return out
    

class TransformerBlock(nn.Module):
    """
    Transformer Block
    -----------------

    input: torch.Tensor(batch_size x sequence_length x embedding_dim)

    for ViT, sequence_length = patch_embedding dim

    output: torch.Tensor(batch_size x sequence_length x embedding_dim)

    consists of:
    - LayerNormalization 1: pre-norm for attention
    - MultiHeadAttention Block
    - LayerNormalization 2: pre-norm for feed-forward
    - MLP: feed forward layer: 
        - embedding_dim
        - embedding_dim * mlp_ratio
        - embedding_dim
        - mlp dropout

    """
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim,config.embed_dim*config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.embed_dim*config.mlp_ratio,config.embed_dim),
            nn.Dropout(config.mlp_dropout)
        )
        
    def forward(self,x):
        x = x+self.attn(self.ln1(x))
        x = x+self.mlp(self.ln2(x))
        return x
    

class ViT(nn.Module):
    """
    Vision Transformer model
    ------------------------
    
    input: torch.tensor(batch_size x num_channels x image_width x image_height)

    output: torch.tensor(batch_size x num_classes)

    - doesn't use CLS token, instead mean pooling is used.
    """
    def __init__(self,config):
        
        super().__init__()
        
        config.num_patches = (config.img_size // config.patch_size) ** 2
        config.patch_dim = config.num_channels * config.patch_size ** 2
        
        self.config = config
        
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(self.config.patch_dim),
            nn.Linear(self.config.patch_dim, self.config.embed_dim, bias=False),
            nn.LayerNorm(self.config.embed_dim)
        )
        self.pos_embedding = nn.Sequential(
            nn.Linear(
                self.config.embed_dim,
                self.config.embed_dim,
                bias=False
            ),
            nn.Dropout(self.config.pos_dropout)
        )
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.config.depth)])
        
        self.head = nn.Linear(self.config.embed_dim,self.config.num_classes)
        
    def forward(self,x):
        
        x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.config.patch_size,
                      p2=self.config.patch_size
                     )
        x = self.patch_embedding(x)
        x += self.pos_embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x
        

if __name__ == '__main__':
    config_vit_pico = SimpleNamespace(
        embed_dim = 128,
        num_heads = 4,
        depth = 6,
        pool = 'mean',
        img_size = 224,
        num_channels = 3,
        patch_size = 16,
        attention_dropout = 0.,
        residual_dropout = 0.,
        mlp_ratio = 4,
        mlp_dropout = 0.,
        pos_dropout = 0.,
        num_classes = 1000
    )
    model = ViT(config_vit_pico)
    x = torch.rand(1,config_vit_pico.num_channels,config_vit_pico.img_size,config_vit_pico.img_size)
    out = model(x)
    assert out.shape == (1,config_vit_pico.num_classes)
    print(out.shape)