import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_colbert_similarity(token_image_features, token_text_features):
    """
    Compute token-level similarity. Given relative information between image and text tokens,
    we only compute similarity from text tokens to image tokens, without considering the reverse.
    This is based on the assumption that the image tokens are more informative than the text tokens, 
    and we assume each text token is associated with image tokens while not vice versa.
    
    Args:
        token_image_features: Token-level features from images [batch_size_img, n_img_tokens, embed_dim]
        token_text_features: Token-level features from text [batch_size_txt, n_txt_tokens, embed_dim]
        
    Returns:
        Token-level similarity matrix [batch_size_txt, batch_size_img], similar to global similarity, each entry with value in [-1, 1]
    """
    sim_matrix = torch.einsum('mnd,kqd->mknq', token_text_features, token_image_features)
    max_sim_per_txt_token = torch.max(sim_matrix, dim=3)[0]  # [batch_size_txt, batch_size_img, n_txt_tokens]
    
    # Create a mask for non-zero values
    mask = (max_sim_per_txt_token != 0).float()
    # Sum of non-zero values
    sum_sim = torch.sum(max_sim_per_txt_token, dim=2)
    # Count of non-zero values (adding small epsilon to avoid division by zero)
    count = torch.sum(mask, dim=2) + 1e-8
    # Average of non-zero values
    logits_per_text_token = sum_sim / count  # [batch_size_txt, batch_size_img]
  
    return logits_per_text_token

def evaluate(model, dataloader, tokenizer, device, amp=True, recall_k_list=[5], colbert_weight=1.0):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
        
    colbert_weight: float
        Weight for ColBERT similarity (only used for ColXLIP model)
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    batch_token_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    batch_token_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):

        if 'clip' not in model.__class__.__name__.lower() and 'colxlip' not in model.__class__.__name__.lower():
            if isinstance(batch_images, list):
                # food 101
                batch_images = torch.stack([img['pixel_values'][0] for img in batch_images])
            else:
                batch_images = batch_images['pixel_values'][0]

        batch_images = batch_images.to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            if 'colxlip' in model.__class__.__name__.lower():
                # Handle ColXLIP model which returns both global and token features
                batch_images_emb, batch_token_images_emb = model.encode_image(batch_images)
                batch_images_emb = F.normalize(batch_images_emb, dim=-1)
                batch_token_images_emb = F.normalize(batch_token_images_emb, dim=-1)
                
                texts = [text for i, texts in enumerate(batch_texts) for text in texts]
                batch_texts_tok = tokenizer(texts).to(device)
                batch_texts_emb, batch_token_texts_emb = model.encode_text(batch_texts_tok)
                batch_texts_emb = F.normalize(batch_texts_emb, dim=-1)
                batch_token_texts_emb = F.normalize(batch_token_texts_emb, dim=-1)
                
                batch_token_images_emb_list.append(batch_token_images_emb.cpu())
                batch_token_texts_emb_list.append(batch_token_texts_emb.cpu())
            elif 'clip' in model.__class__.__name__.lower():
                # implementation of the original openclip
                batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
            else:
                # implementation of the SAIL
                batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                texts = [text for i, texts in enumerate(batch_texts) for text in texts]
                batch_texts_tok = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)  # tokenize
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok, text_list=texts), dim=-1)
            

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    
    # get the score for each text and image pair
    if 'colxlip' in model.__class__.__name__.lower():
        # For ColXLIP, combine global and token-level similarities
        token_images_emb = torch.cat(batch_token_images_emb_list)
        token_texts_emb = torch.cat(batch_token_texts_emb_list)
        
        # Global similarity
        global_scores = texts_emb @ images_emb.t()
        
        # Token-level similarity using ColBERT approach
        # Process in batches to avoid OOM
        token_scores = torch.zeros_like(global_scores)
        batch_size = 128
        for i in range(0, len(token_texts_emb), batch_size):
            i_end = min(i + batch_size, len(token_texts_emb))
            for j in range(0, len(token_images_emb), batch_size):
                j_end = min(j + batch_size, len(token_images_emb))
                
                # Compute ColBERT similarity for this batch
                with torch.no_grad():
                    batch_token_scores = compute_colbert_similarity(
                        token_images_emb[j:j_end].to(device), 
                        token_texts_emb[i:i_end].to(device)
                    )
                token_scores[i:i_end, j:j_end] = batch_token_scores.cpu()
        
        # Combine scores
        scores = (1 - colbert_weight) * global_scores + colbert_weight * token_scores
    else:
        scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(y)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
