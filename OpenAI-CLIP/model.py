import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict



class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.in_proj = nn.Linear(dim,dim*3)
        self.out_proj = nn.Linear(dim,dim)
        
    def forward(self,x, mask=None):
        # x: batch x seq x dim
        B,S,D = x.shape
        q, k, v = self.in_proj(x).chunk(3,dim=-1) # q,k,v: batch x seq x dim
        # to reshape into: batch x num_heads x seq x head_size
        q = q.view(B, S, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(B, S, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        # k.T: batch x num_heads x head_size x seq
        # attn: batch x num_heads x seq x seq
        attn = (q @ k.transpose(-1,-2)) * self.scale
        
        if mask is not None:
            mask = mask.to(dtype=attn.dtype,device=attn.device)
            attn = attn.masked_fill(mask==0,float('-inf'))
            attn = F.softmax(attn,dim=-1)
        
        # attn: batch x num_heads x seq x head_size
        attn = attn @ v
        # attn: batch x seq x (num_heads x head_size=dim)
        attn = attn.permute(0,2,1,3).contiguous().view(B,S,D)
        out = self.out_proj(attn)
        return out
    


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mask=None
    ):
        super().__init__()
        
        self.mask = mask
        self.attn = MultiheadAttention(
            dim = dim,
            num_heads = num_heads
        )
        
        self.ln_1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc',nn.Linear(dim,dim * 4)), # 4 : mlp ratio
            ('gelu',nn.GELU()),
            ('c_proj',nn.Linear(dim * 4,dim))
        ]))
        self.ln_2 = nn.LayerNorm(dim)
        
    def forward(self,x):
        x = x + self.attn(self.ln_1(x),mask=self.mask)
        x = x + self.mlp(self.ln_2(x))
        
        return x
    


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        depth,
        mask=None
    ):
        super().__init__()
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mask=mask
            ) for _ in range(depth)
        ])
        
    def forward(self, x):
        return self.resblocks(x)
    


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        dim,
        num_heads,
        depth,
        out_dim
    ):
        super().__init__()
        
        num_patches = (img_size // patch_size)** 2
        
        self.conv1 = nn.Conv2d(3,dim,patch_size,patch_size,bias=False)
        self.ln_pre = nn.LayerNorm(dim)
        
        self.transformer = Transformer(
            dim=dim,
            num_heads=num_heads,
            depth=depth
        )
        
        self.ln_post = nn.LayerNorm(dim)
        
        self.class_embedding = nn.Parameter(torch.randn(dim))
        self.positional_embedding = nn.Parameter(torch.randn(num_patches+1,dim))
        
        self.proj = nn.Parameter(torch.randn(dim,out_dim))
        
    def forward(self, x):
        
        x = self.conv1(x) # batch x dim x patch_num_cols x patch_num_rows
        x = x.reshape(x.size(0),x.size(1),-1) # batch x dim x num_patches
        x = x.permute(0,2,1) # batch x num_patches x dim
        
        # batch x 1 x dim
        cls_dim = torch.zeros((x.size(0),1,x.size(2)),dtype=x.dtype,device=x.device)
        
        # batch x 1 x dim
        emb = self.class_embedding + cls_dim
        
        # batch x num_patches + 1 x dim ; + 1 for [CLS]
        x = torch.cat([emb,x],dim=1)
        x += self.positional_embedding
        
        x = self.ln_pre(x)
        x = self.transformer(x)
        
        cls_out = x[:,0,:] # batch x dim
        
        out = self.ln_post(cls_out)
        out = out @ self.proj # batch x dim
        
        return out
    


class CLIP(nn.Module):
    def __init__(self,config):
        
        super().__init__()
        
        self.config = config
        
        self.vocab_size = self.config['vocab_size']
        self.context_length = self.config['context_length']
        
        self.visual = VisionTransformer(
            img_size=self.config['img_size'],
            patch_size=self.config['patch_size'],
            dim=self.config['vis_dim'],
            depth=self.config['depth'],
            num_heads=self.config['vis_num_heads'],
            out_dim=self.config['out_dim'],
        )
        
        self.transformer = Transformer(
            dim=self.config['text_dim'],
            num_heads=self.config['text_num_heads'],
            depth=self.config['depth'],
            mask=torch.tril(torch.ones(1,1,self.context_length,self.context_length))
        )
        
        self.token_embedding = nn.Embedding(self.vocab_size,self.config['text_dim'])
        self.positional_embedding = nn.Parameter(torch.rand(self.context_length,self.config['text_dim']))
        
        self.ln_final = nn.LayerNorm(self.config['text_dim'])
        
        self.text_projection = nn.Parameter(torch.rand(self.config['text_dim'], self.config['out_dim']))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def encode_image(self,x):
        x = self.visual(x)
        return x
    
    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    


if __name__=='__main__':
    config = {
        'img_size': 224,
        'patch_size': 32,
        'vis_num_heads': 12,
        'depth': 12,
        'vis_dim': 768,
        'out_dim': 512,
        'vocab_size': 49408,
        'context_length': 77,
        'text_dim': 512,
        'text_num_heads':8
    }

    model = CLIP(config)