
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import open_clip

import sys
sys.path.append("/home/z/zhangle7/links/scratch/colxlip/src")

from colxlip.factory import create_model_and_transforms

def load_colxlip(
    model_name: str,
    pretrained: str,
    cache_dir: str,
    device: Union[str, torch.device] = "cuda",
):
    base_model_name = model_name.replace("-colxlip", "")
    model, _ , transform = create_model_and_transforms(model_name, pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(base_model_name)
    return model, transform, tokenizer
