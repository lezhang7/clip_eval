import sys
sys.path.append("/home/mila/l/le.zhang/scratch/SAIL")

from typing import Optional, Union
import torch
from model import create_model
from functools import partial

def load_sail(
       model_name: str = "SAIL-B", pretrained: str = None, cache_dir: str = None, device="cpu", sharelock=False
    ):

    vision_model_name = model_name.split("_")[0]
    text_model_name = model_name.split("_")[1]

    if "dinov2l" in vision_model_name.lower():
        vision_model_name = "facebook/dinov2-large"
    elif "dinov2b" in vision_model_name.lower():
        vision_model_name = "facebook/dinov2-base"
    else:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")
    
    if "gte" in text_model_name.lower():
        text_model_name = "Alibaba-NLP/gte-large-en-v1.5"
    elif "nv" in text_model_name.lower():
        text_model_name = "nvidia/NV-Embed-v2"
    else:
        raise ValueError(f"Unsupported text model: {text_model_name}")

    model = create_model(
        text_model_name=text_model_name, 
        vision_model_name=vision_model_name, 
        head_weights_path=pretrained,
        target_dimension=1024,
        sharelock=sharelock,
    )
    model = model.to(device=device)
    tokenizer = model.text_model.tokenizer
    transform = model.vision_model.image_processor
    return model, transform, tokenizer

