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
        # q,k,v shape individually (1 head): batch_size x seq_len x embed_dim
        # qkv: batch_size x seq_len x embed_dim*3
        # q,k,v are chunked into 3 equal parts: batch_size x seq_len x embed_dim 
        q,k,v = self.qkv(x).chunk(3,dim=-1)
        # embed_dim = head_size x n_heads
        # q,k,v for n_heads: batch_size x num_heads x seq_len x head_size
        q = rearrange(q,'b t (h n) -> b n t h',n=self.n_heads) # h = head_size
        k = rearrange(k,'b t (h n) -> b n t h',n=self.n_heads)
        v = rearrange(v,'b t (h n) -> b n t h',n=self.n_heads)
        
        # we know that qk_t = q x k_t, where q=b x tx head_dim, k_t= b x head_size x t
        # qk_t = batch_size x num_heads x seq_len x seq_len
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        
        weights = self.attention_dropout(qk_t)
        
        attention = weights @ v # batch x num_heads x seq_len x head_size
        attention = rearrange(attention,'b n t h -> b t (n h)') # batch x seq_len x embed_dim
        
        # batch x seq_len x embed_dim
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
        x = x+self.attn(self.ln1(x)) # batch x seq_len x embed_dim
        x = x+self.mlp(self.ln2(x)) # batch x seq_len x embed_dim
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
        self.pos_embed = nn.Parameter(torch.randn(1,self.config.num_patches,self.config.embed_dim),requires_grad=True)
        self.pos_dropout = nn.Dropout(self.config.pos_dropout)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.config.depth)])
        
        self.head = nn.Linear(self.config.embed_dim,self.config.num_classes)
        
    def forward(self,x):
        
        # seq_len = height * weight
        # patch_dim = patch_size ** 2 * num_channels
        x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.config.patch_size,
                      p2=self.config.patch_size
                     )
        x = self.patch_embedding(x) # batch x seq_len x embed_dim
        x += self.pos_embed  # batch x seq_len x embed_dim
        
        for block in self.transformer_blocks:
            x = block(x) # batch x seq_len x embed_dim
            
        x = x.mean(dim=1) # batch x embed_dim
        x = self.head(x) # batch x num_classes
        
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