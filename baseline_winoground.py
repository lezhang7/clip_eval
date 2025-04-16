import open_clip
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
from datasets import load_dataset
import torch
winoground = load_dataset("facebook/winoground")

from tqdm import tqdm
winoground_clip_scores = []
winoground_colbert_scores = []
winoground_combined_scores = []

for example in tqdm(winoground['test']):
    # Process images and text
    image1 = preprocess(example["image_0"].convert("RGB")).unsqueeze(0)
    image2 = preprocess(example["image_1"].convert("RGB")).unsqueeze(0)
    images = torch.cat([image1, image2], dim=0)
    
    text = tokenizer([example["caption_0"], example["caption_1"]])
    
    # Get model outputs
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.T
        # Extract the four scores for CLIP
        clip_score_c0_i0 = logits_per_image[0][0].item()
        clip_score_c1_i0 = logits_per_image[0][1].item()
        clip_score_c0_i1 = logits_per_image[1][0].item()
        clip_score_c1_i1 = logits_per_image[1][1].item()
        
   
    # Store CLIP scores
    winoground_clip_scores.append({
        "id": example["id"], 
        "c0_i0": clip_score_c0_i0, 
        "c0_i1": clip_score_c0_i1, 
        "c1_i0": clip_score_c1_i0, 
        "c1_i1": clip_score_c1_i1
    })

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

text_correct_count = 0
image_correct_count = 0
group_correct_count = 0

for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct(result) else 0
  image_correct_count += 1 if image_correct(result) else 0
  group_correct_count += 1 if group_correct(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)
print("image score:", image_correct_count/denominator)
print("group score:", group_correct_count/denominator)