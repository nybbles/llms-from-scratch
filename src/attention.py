import torch


def dot_product_attention_weights(inputs):
    attention_scores = inputs @ inputs.transpose(-1, -2)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return attention_weights
