import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, 'dim should be div by n_heads'
        self.head_dim = self.dim // self.n_heads
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim,dim)
        
    def forward(self,q,k,v,mask=None):
        b,t,c = q.shape
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = q.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        k = k.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        
        qkT = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        qkT = self.attn_dropout(qkT)
        
        if mask is not None:
            mask = mask.to(dtype=qkT.dtype,device=qkT.device)
            qkT = qkT.masked_fill(mask==0,float('-inf'))
            
        qkT = F.softmax(qkT,dim=-1)
            
        attn = torch.matmul(qkT,v)
        attn = attn.permute(0,2,1,3).contiguous().view(b,t,c)
        out = self.out_proj(attn)
        
        return out
    


class FeedForward(nn.Module):
    def __init__(self,dim,dropout=0.):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        
    def forward(self, x):
        return self.feed_forward(x)
    


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0., mlp_dropout=0.):
        super().__init__()
        self.attn = MultiheadAttention(dim,n_heads,attn_dropout)
        self.ffd = FeedForward(dim,mlp_dropout)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        
    def forward(self,x,mask=None):
        x = self.ln_1(x)
        x = x + self.attn(x,x,x,mask)
        x = self.ln_2(x)
        x = x + self.ffd(x)
        return x
    


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0., mlp_dropout=0.):
        super().__init__()
        self.self_attn = MultiheadAttention(dim,n_heads,attn_dropout)
        self.cross_attn = MultiheadAttention(dim,n_heads,attn_dropout)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.ln_3 = nn.LayerNorm(dim)
        self.ffd = FeedForward(dim,mlp_dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.ln_1(x)
        x = x + self.self_attn(x,x,x,tgt_mask)
        x = self.ln_2(x)
        x = x + self.cross_attn(x,enc_out,enc_out,src_mask) # decoder: q, encoder: k,v
        x = self.ln_3(x)
        x = x + self.ffd(x)
        
        return x
    


class Embedding(nn.Module):
    def __init__(self,vocab_size,max_len,dim):
        super().__init__()
        self.max_len = max_len
        self.class_embedding = nn.Embedding(vocab_size,dim)
        self.pos_embedding = nn.Embedding(max_len,dim)
    def forward(self,x):
        x = self.class_embedding(x)
        pos = torch.arange(0,self.max_len,device=x.device)
        x = x + self.pos_embedding(pos)
        return x
    


class Transformer(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        
        self.enc_embedding = Embedding(config['encoder_vocab_size'],config['encoder_max_len'],config['dim'])
        self.dec_embedding = Embedding(config['decoder_vocab_size'],config['decoder_max_len'],config['dim'])
        
        self.depth = config['depth']
        self.encoders = nn.ModuleList([
            EncoderBlock(
                dim=config['dim'],
                n_heads=config['n_heads'],
                attn_dropout=config['attn_dropout'],
                mlp_dropout=config['mlp_dropout']
            ) for _ in range(self.depth)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(
                dim=config['dim'],
                n_heads=config['n_heads'],
                attn_dropout=config['attn_dropout'],
                mlp_dropout=config['mlp_dropout']
            ) for _ in range(self.depth)
        ])
        
        self.src_pad_token_id = config['src_pad_token_id']
        self.register_buffer('tgt_mask',torch.tril(torch.ones(1,1,config['decoder_max_len'],config['decoder_max_len'])))
    
    def create_src_mask(self,src):
        return (src != self.src_pad_token_id).unsqueeze(1).unsqueeze(2) # N, 1, 1, src_len
    
    def forward(self, src, tgt):
        
        src_mask = self.create_src_mask(src)
        
        src = self.enc_embedding(src)
        tgt = self.dec_embedding(tgt)
        
        
        for i in range(self.depth):
            enc_out = self.encoders[i](src,mask=src_mask)
            dec_out = self.decoders[i](tgt,enc_out,src_mask=src_mask,tgt_mask=self.tgt_mask)
            
        return dec_out
    

if __name__ == '__main__':
    config = {
        'dim': 512,
        'n_heads': 8,
        'attn_dropout': 0.1,
        'mlp_dropout': 0.1,
        'depth': 6,
        'encoder_vocab_size': 20_000,
        'encoder_max_len': 128,
        'decoder_vocab_size': 25_000,
        'decoder_max_len': 128,
        'src_pad_token_id': -1
    }
    model = Transformer(config)
    src = torch.randint(0,config['encoder_vocab_size'],size=(1,config['encoder_max_len']))
    tgt = torch.randint(0,config['decoder_vocab_size'],size=(1,config['decoder_max_len']))
    # src.shape, tgt.shape: (torch.Size([1, 128]), torch.Size([1, 128]))
    # model(src,tgt).shape: 1 x 128 x 512
    # we can further add a LM head for the decoder