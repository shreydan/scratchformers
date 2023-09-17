# ScratchFormers

implementing transformers from scratch (and training them)*!

_* if I get necessary compute_

##### einops [starter](einops.ipynb)

## Models

- **simple Vision Transformer**
  - for process, check [building_ViT.ipynb](./ViT/building_ViT.ipynb)
  - model [implementation](./ViT/vit.py)
  - used `mean` pooling instead of `[class]` token
- **GPT2**
  - for process, check [buildingGPT2.ipynb](./GPT2/buildingGPT2.ipynb)
  - model [implementation](./GPT2/gpt2.py)
  - built in such a way that it supports loading pretrained openAI/huggingface weights [gpt2-load-via-hf.ipynb](./GPT2/gpt2-load-via-hf.ipynb)
  - TODON'T: emulate hf text-generation to get good outputs
- **OpenAI CLIP**
  - implemented `ViT-B/32` variant
  - for process, check [building_clip.ipynb](./OpenAI-CLIP/building_clip.ipynb)
  - inference req: install clip for tokenization and preprocessing: `pip install git+https://github.com/openai/CLIP.git`
  - model [implementation](./OpenAI-CLIP/model.py)
  - zero-shot inference [code](./OpenAI-CLIP/zeroshot.py)
  - built in such a way that it supports loading pretrained openAI weights and IT WORKS!!!
  
### Requirements
```
torch
torchvision
numpy
matplotlib
einops
```

Here's my puppy's picture:
![sumo](sumo.jpg)