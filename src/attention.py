import torch


def dot_product_attention_weights(x):
    attention_scores = x @ x.T
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return attention_weights
