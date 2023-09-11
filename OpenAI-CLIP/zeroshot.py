from model import CLIP
import torch
import numpy as np
from PIL import Image
import clip

_,preprocess = clip.load('ViT-B/32')

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cl = CLIP(config).to(device)

# load state dict ....

img = Image.open('path_to_image').convert('RGB')
img = preprocess(img).to(device).unsqueeze(0)
labels = ['image of a cat','image of a dog']
labels = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = cl.encode_image(img).float()
    text_features = cl.encode_text(labels).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)


text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)
print(top_probs, top_labels)