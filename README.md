# ScratchFormers
### implementing transformers from scratch.

> Attention is all you need.


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


- **Encoder Decoder Transformer**
  - for process, check [building_encoder-decoder.ipynb](./encoder-decoder/building_encoder-decoder.ipynb)
  - model [implementation](./encoder-decoder/model.py)
  - src_mask for encoder is optional but is nice to have since it is used to mask out the pad tokens so attention is not considered for those tokens.
  - used learned embeddings for position instead of sin/cos as per the OG


- **BERT - MLM**
  - for process of masked language modeling, check [masked-language-modeling.ipynb](./BERT-MLM/masked-language-modeling.ipynb)
  - model [implementation](./BERT-MLM/model.py)
  - simplification: for pre-training no use of [CLS] & [SEP] tokens since I only built the model for masked language modeling and not for next sentence prediction. 
  - I trained an entire model on the wikipedia dataset, more info in [shreydan/mased-language-modeling](https://github.com/shreydan/masked-language-modeling) repo.
  - once, pretrained the MLM head can be replaced with any other downstream task head.

- **ViT MAE**
  - Paper: [Masked autoencoders are scalable vision learners](https://arxiv.org/abs/2111.06377)  
  - model [implementation](./vitmae/model.py)
  - for process, check: [building-vitmae.ipynb](./vitmae/building-vitmae.ipynb)
  - Quite reliant on the original code released by authors.
  - Only simplification: No [CLS] token so used mean pooling
  - The model can be trained 2 ways:
    - For pretraining: the decoder can be thrown away and the encoder can be used for downstream tasks
    - For visualization: can be used to reconstruct masked images.
  - I trained a smaller model for reconstruction visualization: [ViTMAE on Animals Dataset](./vitmae/animals-vitmae.ipynb)


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