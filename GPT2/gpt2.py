import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from types import SimpleNamespace


class GPT2Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        self.scale = self.head_size ** -0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        q = rearrange(q,'b t (h n) -> b n t h',n=self.n_heads) # h = head_size
        k = rearrange(k,'b t (h n) -> b n t h',n=self.n_heads)
        v = rearrange(v,'b t (h n) -> b n t h',n=self.n_heads)
        
        # qk_t = einsum(q,k,'b n t1 h, b n t2 h -> b n t1 t2') * self.scale
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        # fun fact, limit mask to [:,:,:t,:t] else short prompts will not work
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = rearrange(attention,'b n t h -> b t (n h)') # batch x n_heads x t x embed_dim
        
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out
    

class GPT2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout
        
        self.c_fc = nn.Linear(self.embed_dim,self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio,self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class GPT2Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        
    def forward(self,x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x
    

class GPT2Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self,x):
        
        token_embeddings = self.transformer.wte(x) # batch x seq_len
        pos_embs = torch.arange(0,x.size(1)).to(x.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        x = self.transformer.drop(token_embeddings+positional_embeddings)
        for h in self.transformer.h:
            x = h(x) # batch_size x seq_len x embed_dim
        x = self.transformer.ln_f(x)[:,[-1],:] # get last hidden state: batch_size x 1 x embed_dim
        x = self.lm_head(x) # batch_size x vocab_size
        
        return x
    
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=5,temperature=1.0):
        
        for _ in range(max_new_tokens):
            
            inp = idx if idx.size(1) <= self.config.seq_len else inp[:,-self.config.seq_len:]
            out = self(inp)
            out = out[:, -1, :] / temperature
            probs = F.softmax(out, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx[:,-max_new_tokens:]
    

if __name__ == '__main__':
    gpt2_small = SimpleNamespace(
        vocab_size = 50_257,
        embed_dim = 768,
        num_heads = 12,
        seq_len = 1024,
        depth = 12,
        attention_dropout = 0.1,
        residual_dropout = 0.1,
        mlp_ratio = 4,
        mlp_dropout = 0.1,
        emb_dropout = 0.1,
    )
    model = GPT2Model(gpt2_small)
    x = torch.randint(0,gpt2_small.vocab_size,size=(1,gpt2_small.seq_len))
    out = model.generate(x,max_new_tokens=128)
    print(out.shape)